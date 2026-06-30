import numpy as np
import torch
import torch.nn as nn

# ============================================================================
# Graph Construction for 75 Landmarks
# ============================================================================

def get_hop_distance(num_node, edges, center_node=11):
    # Construct adjacency list
    adj = {i: [] for i in range(num_node)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS to find distance from center_node to all nodes
    dist = {i: float('inf') for i in range(num_node)}
    dist[center_node] = 0
    queue = [center_node]
    while queue:
        curr = queue.pop(0)
        for neighbor in adj[curr]:
            if dist[neighbor] == float('inf'):
                dist[neighbor] = dist[curr] + 1
                queue.append(neighbor)
    return dist

class Graph:
    def __init__(self):
        self.num_node = 75
        self.edges = []
        
        # 1. Pose skeleton edges (33 joints, index 0 to 32)
        pose_edges = [
            # face/head
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10),
            # shoulders/torso
            (11, 12), (11, 23), (12, 24), (23, 24),
            # arms
            (11, 13), (13, 15), (12, 14), (14, 16),
            (15, 17), (15, 19), (15, 21), (17, 19),
            (16, 18), (16, 20), (16, 22), (18, 20),
            # legs
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
        ]
        self.edges.extend(pose_edges)
        
        # 2. Left Hand edges (21 joints, index 33 to 53, offset 33)
        left_hand_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # index
            (0, 9), (9, 10), (10, 11), (11, 12),  # middle
            (0, 13), (13, 14), (14, 15), (15, 16),# ring
            (0, 17), (17, 18), (18, 19), (19, 20),# pinky
            # base connections
            (5, 9), (9, 13), (13, 17)
        ]
        self.edges.extend([(u + 33, v + 33) for u, v in left_hand_edges])
        
        # 3. Right Hand edges (21 joints, index 54 to 74, offset 54)
        right_hand_edges = left_hand_edges
        self.edges.extend([(u + 54, v + 54) for u, v in right_hand_edges])
        
        # 4. Wrist cross-connections to join hand with pose
        # Left wrist in pose is 15. Left hand wrist is 33.
        # Right wrist in pose is 16. Right hand wrist is 54.
        self.edges.append((15, 33))
        self.edges.append((16, 54))
        
        self.A = self.get_adjacency_matrix()
        
    def get_adjacency_matrix(self):
        hop_dis = get_hop_distance(self.num_node, self.edges, center_node=11)
        
        A_self = np.eye(self.num_node)
        A_close = np.zeros((self.num_node, self.num_node))
        A_further = np.zeros((self.num_node, self.num_node))
        
        for u, v in self.edges:
            # Centripetal vs Centrifugal partition
            if hop_dis[v] <= hop_dis[u]:
                A_close[u, v] = 1
                A_further[v, u] = 1
            else:
                A_further[u, v] = 1
                A_close[v, u] = 1
                
        # Normalize
        def normalize(A):
            row_sum = np.sum(A, axis=1)
            Dn = np.zeros_like(A)
            for i in range(A.shape[0]):
                if row_sum[i] > 0:
                    Dn[i, i] = 1.0 / row_sum[i]
            return np.dot(Dn, A)
            
        A_close = normalize(A_close)
        A_further = normalize(A_further)
        
        # Combine into (3, 75, 75) adjacency matrix
        A = np.stack([A_self, A_close, A_further], axis=0)
        return A

# ============================================================================
# ST-GCN Network Blocks
# ============================================================================

class GCNUnit(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32), requires_grad=True)
        self.num_subsets = A.shape[0]  # 3 subsets
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * self.num_subsets,
            kernel_size=1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Input shape: (B, in_channels, T, V)
        B, C, T, V = x.size()
        x = self.conv(x)  # (B, out_channels * 3, T, V)
        x = x.view(B, self.num_subsets, -1, T, V)  # (B, 3, out_channels, T, V)
        
        # Matrix multiplication of A with joint dimension V
        y = torch.einsum('bsctv,svw->bctw', x, self.A).contiguous()
        y = self.bn(y)
        return self.relu(y)

class TCNUnit(nn.Module):
    def __init__(self, channels, kernel_size=9, stride=1, dropout=0.0):
        super().__init__()
        padding = ((kernel_size - 1) // 2, 0)
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=padding
        )
        self.bn = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.dropout(x)

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, dropout=0.0):
        super().__init__()
        self.gcn = GCNUnit(in_channels, out_channels, A)
        self.tcn = TCNUnit(out_channels, kernel_size=9, stride=stride, dropout=dropout)
        self.relu = nn.ReLU(inplace=True)
        
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)

class TemporalAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x shape: (B, C, T)
        x_trans = x.transpose(1, 2)  # (B, T, C)
        scores = self.attn(x_trans)  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        pooled = torch.sum(x_trans * weights, dim=1)  # (B, C)
        return pooled

# ============================================================================
# Main ST-GCN Network
# ============================================================================

class UniSignDownstreamModel(nn.Module):
    """
    ST-GCN classifier model for 75 landmarks (33 Pose + 21 Left + 21 Right).
    Input shape: (B, C, T, V) = (B, 3, 60, 75)
    """
    def __init__(self, num_classes=13):
        super().__init__()
        
        # Initialize the graph
        self.graph = Graph()
        A = self.graph.A
        
        # 10 ST-GCN Blocks
        # Blocks 1-3: 64 channels
        self.block1 = STGCNBlock(3, 64, A, stride=1)
        self.block2 = STGCNBlock(64, 64, A, stride=1)
        self.block3 = STGCNBlock(64, 64, A, stride=1)
        
        # Blocks 4-7: 128 channels (with stride 2 downsampling in block 4)
        self.block4 = STGCNBlock(64, 128, A, stride=2)
        self.block5 = STGCNBlock(128, 128, A, stride=1)
        self.block6 = STGCNBlock(128, 128, A, stride=1)
        self.block7 = STGCNBlock(128, 128, A, stride=1)
        
        # Blocks 8-10: 256 channels (with stride 2 downsampling in block 8)
        self.block8 = STGCNBlock(128, 256, A, stride=2)
        self.block9 = STGCNBlock(256, 256, A, stride=1)
        self.block10 = STGCNBlock(256, 256, A, stride=1)
        
        # Temporal Attention Pooling
        self.temp_attn = TemporalAttention(256)
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # If input shape is (B, T, V, C), permute to (B, C, T, V)
        if x.dim() == 4 and x.shape[3] == 3:
            x = x.permute(0, 3, 1, 2)  # (B, 3, T, V)
            
        # ST-GCN block operations
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        
        # Spatial pooling (Average pooling over V dimension)
        # x is (B, 256, T, V) -> mean(dim=-1) -> (B, 256, T)
        x = x.mean(dim=-1)
        
        # Temporal pooling using learnable attention
        x = self.temp_attn(x)  # (B, 256)
        
        # Classification logits
        logits = self.classifier(x)
        return logits
