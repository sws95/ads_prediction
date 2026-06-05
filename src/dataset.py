"""
Ali-CCP 데이터 로더.
받으신 ali_ccp_train.csv / val.csv / test.csv를 PyTorch Dataset으로 변환.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset


SPARSE_COLS = [
    '101', '121', '122', '124', '125', '126', '127', '128', '129',
    '109_14', '110_14', '127_14', '150_14',
    '205', '206', '207', '210', '216',
    '508', '509', '702', '853',
    '301'
]  # 23개

DENSE_COLS = [
    'D109_14', 'D110_14', 'D127_14', 'D150_14',
    'D508', 'D509', 'D702', 'D853'
]  # 8개

LABEL_COLS = ['click', 'purchase']


class AliCCPDataset(Dataset):
    def __init__(self, df):
        # Sparse: long tensor (임베딩 입력용)
        self.sparse = torch.tensor(
            df[SPARSE_COLS].values, dtype=torch.long
        )
        # Dense: float tensor
        self.dense = torch.tensor(
            df[DENSE_COLS].values, dtype=torch.float
        )
        # Labels
        self.click = torch.tensor(df['click'].values, dtype=torch.float)
        self.purchase = torch.tensor(df['purchase'].values, dtype=torch.float)
        # CTCVR 라벨: click ∧ purchase
        self.ctcvr = self.click * self.purchase

    def __len__(self):
        return len(self.click)

    def __getitem__(self, idx):
        return {
            'sparse': self.sparse[idx],
            'dense': self.dense[idx],
            'click': self.click[idx],
            'ctcvr': self.ctcvr[idx],
        }


def load_data(data_dir):
    """CSV 3개 로드. 메모리 부담되면 chunk로 바꿔야 함."""
    train_df = pd.read_csv(f'{data_dir}/ali_ccp_train.csv')
    val_df = pd.read_csv(f'{data_dir}/ali_ccp_val.csv')
    test_df = pd.read_csv(f'{data_dir}/ali_ccp_test.csv')

    print(f'Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}')
    print(f'Train CTR: {train_df["click"].mean():.4f}')
    print(f'Train CVR (clicked): {train_df[train_df["click"]==1]["purchase"].mean():.4f}')
    print(f'Train CTCVR: {(train_df["click"]*train_df["purchase"]).mean():.6f}')

    return train_df, val_df, test_df


def get_sparse_dims(*dfs):
    """각 sparse feature의 vocab size 계산 (전체 데이터에서 max+1)."""
    sparse_dims = []
    for col in SPARSE_COLS:
        max_val = max(df[col].max() for df in dfs)
        sparse_dims.append(int(max_val) + 1)  # 0부터 시작하니까 +1
    return sparse_dims