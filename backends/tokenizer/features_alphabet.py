import numpy as np

def angle(a, b, c, eps=1e-8):
    ba = a - b
    bc = c - b
    ba /= (np.linalg.norm(ba) + eps)
    bc /= (np.linalg.norm(bc) + eps)
    return np.arccos(np.clip(np.dot(ba, bc), -1.0, 1.0))

def featurize_pose(x63):
    pts = x63.reshape(21, 3).copy()

    pts -= pts[0]
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale

    normal = np.cross(pts[5], pts[17])
    normal /= (np.linalg.norm(normal) + 1e-6)

    tip_d = [np.linalg.norm(pts[i]) for i in [4,8,12,16,20]]

    tips = [pts[i] for i in [8,12,16,20]]
    inter = [np.linalg.norm(a-b) for i,a in enumerate(tips) for b in tips[i+1:]]

    angs = [angle(pts[a], pts[b], pts[c]) for (a,b,c) in
            [(5,6,7),(9,10,11),(13,14,15),(17,18,19)]]

    thumb = pts[4]
    index = pts[5]
    middle = pts[9]

    thumb_features = [
        np.linalg.norm(thumb - index),
        np.linalg.norm(thumb - middle),
        np.linalg.norm(thumb - middle) - np.linalg.norm(thumb - index)
    ]

    return np.concatenate([
        pts.reshape(-1),
        normal,
        tip_d,
        inter,
        angs,
        thumb_features
    ]).astype(np.float32)
