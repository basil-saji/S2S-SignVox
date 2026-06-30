"""
model.py - Seq2Seq sign language model for the Zora inference pipeline.

Architecture:
    UniSignEncoderOnly: (dict of body/left/right/face_all) -> ST-GCN -> (B, T, 768)
    MeetixSeq2Seq: Encoder -> Proj(768->256) -> BiGRU -> Autoregressive GRU Decoder -> Logits
"""

import math
import copy
import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Graph and GCN Utilities (from Uni-Sign stgcn_layers)
# ============================================================================

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


class Graph:
    """The Graph to model the skeletons."""

    def __init__(self, layout='custom', strategy='uniform', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return str(self.A)

    def get_edge(self, layout):
        if layout == 'left' or layout == 'right':
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 1], [1, 2], [2, 3], [3, 4],
                [0, 5], [5, 6], [6, 7], [7, 8],
                [0, 9], [9, 10], [10, 11], [11, 12],
                [0, 13], [13, 14], [14, 15], [15, 16],
                [0, 17], [17, 18], [18, 19], [19, 20],
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 0
        
        elif layout == 'body':
            self.num_node = 9
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 1], [0, 2], [0, 3], [0, 4],
                [3, 5], [5, 7], [4, 6], [6, 8],
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 0
            
        elif layout == 'face_all':
            self.num_node = 9 + 8 + 1
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [[i, i + 1] for i in range(9 - 1)] + \
                             [[i, i + 1] for i in range(9, 9 + 8 - 1)] + \
                             [[9 + 8 - 1, 9]] + \
                             [[17, i] for i in range(17)]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = self.num_node - 1

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if (
                                self.hop_dis[j, self.center]
                                == self.hop_dis[i, self.center]
                            ):
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif (
                                self.hop_dis[j, self.center]
                                > self.hop_dis[i, self.center]
                            ):
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


# ============================================================================
# STGCN Blocks (from Uni-Sign stgcn_layers)
# ============================================================================

class GCN_unit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        A,
        adaptive=True,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        assert A.size(0) == self.kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )
        self.adaptive = adaptive
        if self.adaptive:
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer('A', A)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, len_x=None):
        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, self.A)).contiguous()
        y = self.bn(x)
        y = self.relu(y)
        return y


class STGCN_block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        A,
        adaptive=True,
        stride=1,
        dropout=0.0,
        residual=True,
    ):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = GCN_unit(
            in_channels,
            out_channels,
            kernel_size[1],
            A,
            adaptive=adaptive,
        )
        if kernel_size[0] > 1:
            self.tcn = nn.Sequential(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    (kernel_size[0], 1),
                    (stride, 1),
                    padding,
                ),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout, inplace=True),
            )
        else:
            self.tcn = nn.Identity()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, len_x=None):
        res = self.residual(x)
        x = self.gcn(x, len_x)
        x = self.tcn(x) + res
        return self.relu(x)


class STGCNChain(nn.Sequential):
    def __init__(self, in_dim, block_args, kernel_size, A, adaptive):
        super(STGCNChain, self).__init__()
        last_dim = in_dim
        for i, [channel, depth] in enumerate(block_args):
            for j in range(depth):
                self.add_module(f'layer{i}_{j}', STGCN_block(last_dim, channel, kernel_size, A.clone(), adaptive))
                last_dim = channel


def get_stgcn_chain(in_dim, level, kernel_size, A, adaptive):
    if level == 'spatial':
        block_args = [[64, 1], [128, 1], [256, 1]]
    elif level == 'temporal':
        block_args = [[256, 3]]
    else:
        raise NotImplementedError
    return STGCNChain(in_dim, block_args, kernel_size, A, adaptive), block_args[-1][0]


# ============================================================================
# Seq2Seq Encoder: UniSignEncoderOnly
# ============================================================================

class UniSignEncoderOnly(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.modes = ['body', 'left', 'right', 'face_all']
        self.graph, A = {}, []
        
        self.proj_linear = nn.ModuleDict()
        for mode in self.modes:
            self.graph[mode] = Graph(layout=f'{mode}', strategy='distance', max_hop=1)
            A.append(torch.tensor(self.graph[mode].A, dtype=torch.float32, requires_grad=False))
            self.proj_linear[mode] = nn.Linear(3, 64)

        self.gcn_modules = nn.ModuleDict()
        self.fusion_gcn_modules = nn.ModuleDict()
        spatial_kernel_size = A[0].size(0)
        
        for index, mode in enumerate(self.modes):
            self.gcn_modules[mode], final_dim = get_stgcn_chain(64, 'spatial', (1, spatial_kernel_size), A[index].clone(), True)
            self.fusion_gcn_modules[mode], _ = get_stgcn_chain(final_dim, 'temporal', (5, spatial_kernel_size), A[index].clone(), True)
        
        # Share weights: left reuses right's modules
        self.gcn_modules['left'] = self.gcn_modules['right']
        self.fusion_gcn_modules['left'] = self.fusion_gcn_modules['right']
        self.proj_linear['left'] = self.proj_linear['right']

        self.part_para = nn.Parameter(torch.zeros(hidden_dim * len(self.modes)))
        self.pose_proj = nn.Linear(256 * 4, 768)

    def forward(self, src_input):
        features = []
        body_feat = None
        
        for part in self.modes:
            proj_feat = self.proj_linear[part](src_input[part]).permute(0, 3, 1, 2)
            gcn_feat = self.gcn_modules[part](proj_feat)
            
            if part == 'body':
                body_feat = gcn_feat
            else:
                if part == 'left':
                    gcn_feat = gcn_feat + body_feat[..., -2][..., None].detach()
                elif part == 'right':
                    gcn_feat = gcn_feat + body_feat[..., -1][..., None].detach()
                elif part == 'face_all':
                    gcn_feat = gcn_feat + body_feat[..., 0][..., None].detach()
            
            gcn_feat = self.fusion_gcn_modules[part](gcn_feat)
            pool_feat = gcn_feat.mean(-1).transpose(1, 2)
            features.append(pool_feat)
        
        inputs_embeds = torch.cat(features, dim=-1) + self.part_para
        inputs_embeds = self.pose_proj(inputs_embeds)
        
        return inputs_embeds


# ============================================================================
# Seq2Seq Decoder: MeetixSeq2Seq
# ============================================================================

class MeetixSeq2Seq(nn.Module):
    def __init__(self, encoder, vocab_size):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Linear(768, 256)
        self.bigru = nn.GRU(256, 128, batch_first=True, bidirectional=True)
        self.embedding = nn.Embedding(vocab_size, 128)
        self.decoder = nn.GRU(128 + 256, 256, batch_first=True)
        self.out = nn.Linear(256, vocab_size)

    def forward(self, src, tgt):
        enc = self.encoder(src)
        enc = self.proj(enc)
        enc, _ = self.bigru(enc)
        context = enc.mean(1)
        emb = self.embedding(tgt[:, :-1])
        context = context.unsqueeze(1).repeat(1, emb.shape[1], 1)
        dec_in = torch.cat([emb, context], dim=-1)
        out, _ = self.decoder(dec_in)
        logits = self.out(out)
        return logits

    @torch.no_grad()
    def inference(self, src, sos_id, eos_id, max_len=10):
        enc = self.encoder(src)
        enc = self.proj(enc)
        enc, _ = self.bigru(enc)
        context = enc.mean(1)
        
        batch_size = context.shape[0]
        token = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=context.device)
        hidden = None
        output_tokens = []
        confidences = []
        
        for _ in range(max_len):
            emb = self.embedding(token)
            ctx = context.unsqueeze(1)
            dec_in = torch.cat([emb, ctx], dim=-1)
            out, hidden = self.decoder(dec_in, hidden)
            logits = self.out(out)
            probs = torch.softmax(logits, dim=-1)
            # logits: (B, 1, vocab_size) -> probs: (B, 1, vocab_size)
            # probs.max(dim=-1) -> values=(B,1), indices=(B,1)
            max_prob, token = probs.max(dim=-1)
            confidences.append(max_prob.squeeze(1))
            output_tokens.append(token.squeeze(1))
            if (token.squeeze(1) == eos_id).all():
                break
        
        token_ids = torch.stack(output_tokens, dim=1)
        avg_conf = torch.stack(confidences, dim=1).mean(dim=1)
        return token_ids, avg_conf


# ============================================================================
# MediaPipe to UniSign Adapter
# ============================================================================

def mediapipe_to_unisign(sequence):
    """
    Convert raw MediaPipe Holistic landmarks (T, 1662) to the dict format
    expected by UniSignEncoderOnly.

    MediaPipe layout (1662 floats per frame):
        face:  468 landmarks * 3 = 1404
        pose:   33 landmarks * 4 = 132  (x, y, z, visibility)
        left:   21 landmarks * 3 = 63
        right:  21 landmarks * 3 = 63

    Returns:
        dict with keys body, left, right, face_all — each (1, T, N, 3).
    """
    T = sequence.shape[0]
    face = sequence[:, :1404].reshape(T, 468, 3)
    pose = sequence[:, 1404:1536].reshape(T, 33, 4)[:, :, :3]
    left = sequence[:, 1536:1599].reshape(T, 21, 3)
    right = sequence[:, 1599:1662].reshape(T, 21, 3)
    
    body_idx = [0, 11, 12, 13, 14, 15, 16, 23, 24]
    body = pose[:, body_idx]
    
    face_idx = np.linspace(0, 467, 18).astype(int)
    face18 = face[:, face_idx]
    
    # Normalize relative to root joints
    left = left - left[:, 0:1]
    right = right - right[:, 0:1]
    face18 = face18 - face18[:, -1:]
    root = (body[:, 1:2] + body[:, 2:3]) / 2
    body = body - root
    
    src = {
        "body": torch.tensor(body).float().unsqueeze(0),
        "left": torch.tensor(left).float().unsqueeze(0),
        "right": torch.tensor(right).float().unsqueeze(0),
        "face_all": torch.tensor(face18).float().unsqueeze(0),
    }
    return src
