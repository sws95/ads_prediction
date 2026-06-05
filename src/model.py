"""
ESMM (Entire Space Multi-Task Model)
- Ma et al., SIGIR 2018
- CTR tower와 CVR tower가 임베딩 layer를 공유
- Loss = L_CTR + L_CTCVR (pCVR은 직접 supervise 안 됨)
"""

import torch
import torch.nn as nn


class Tower(nn.Module):
    """단일 task용 MLP tower."""
    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.mlp(x)).squeeze(-1)


class ESMM(nn.Module):
    def __init__(self, sparse_dims, dense_dim, embed_dim=16,
                 hidden_dims=(256, 128, 64), dropout=0.2):
        """
        sparse_dims: 각 sparse feature의 vocab size 리스트
        dense_dim:   dense feature 개수
        """
        super().__init__()
        # 공유 임베딩 layer (CTR/CVR tower가 같이 씀)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=d, embedding_dim=embed_dim)
            for d in sparse_dims
        ])
        input_dim = len(sparse_dims) * embed_dim + dense_dim

        # 두 개의 분리된 tower
        self.ctr_tower = Tower(input_dim, hidden_dims, dropout)
        self.cvr_tower = Tower(input_dim, hidden_dims, dropout)

    def forward(self, sparse, dense):
        """
        sparse: (B, num_sparse) long tensor
        dense:  (B, num_dense) float tensor
        """
        # 각 sparse feature를 임베딩하고 concat
        embs = [emb(sparse[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [dense], dim=-1)  # (B, input_dim)

        pCTR = self.ctr_tower(x)            # (B,)
        pCVR = self.cvr_tower(x)            # (B,)
        pCTCVR = pCTR * pCVR                # (B,)

        return pCTR, pCVR, pCTCVR


def esmm_loss(pCTR, pCTCVR, click_label, ctcvr_label, eps=1e-8):
    """
    ESMM은 pCTR과 pCTCVR에만 loss 적용.
    pCVR은 직접 supervise 안 됨 (그래서 SSB 해결).
    """
    pCTR = pCTR.clamp(eps, 1 - eps)
    pCTCVR = pCTCVR.clamp(eps, 1 - eps)

    loss_ctr = nn.functional.binary_cross_entropy(pCTR, click_label)
    loss_ctcvr = nn.functional.binary_cross_entropy(pCTCVR, ctcvr_label)
    return loss_ctr + loss_ctcvr, loss_ctr, loss_ctcvr