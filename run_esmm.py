"""
ESMM 학습 실행 스크립트.
사용: python run_esmm.py
"""

import torch
from src.dataset import (
    AliCCPDataset, load_data, get_sparse_dims,
    SPARSE_COLS, DENSE_COLS
)
from src.model import ESMM
from src.train import train


CONFIG = {
    'data_dir': './ali_ccp',
    'batch_size': 4096,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'embed_dim': 16,
    'hidden_dims': (256, 128, 64),
    'dropout': 0.2,
    'epochs': 5,
    'num_workers': 0,
    'seed': 2026,
}


def main():
    torch.manual_seed(CONFIG['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 1. 데이터 로드
    train_df, val_df, test_df = load_data(CONFIG['data_dir'])

    # 2. Sparse feature의 vocab size 계산
    sparse_dims = get_sparse_dims(train_df, val_df, test_df)
    print(f'Sparse vocab sizes: {dict(zip(SPARSE_COLS, sparse_dims))}')

    # 3. Dataset 변환
    train_ds = AliCCPDataset(train_df)
    val_ds = AliCCPDataset(val_df)
    test_ds = AliCCPDataset(test_df)

    # 4. 모델 생성
    model = ESMM(
        sparse_dims=sparse_dims,
        dense_dim=len(DENSE_COLS),
        embed_dim=CONFIG['embed_dim'],
        hidden_dims=CONFIG['hidden_dims'],
        dropout=CONFIG['dropout'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')

    # 5. 학습
    test_metrics = train(model, train_ds, val_ds, test_ds, CONFIG, device)

    # 6. 결과 저장
    torch.save({
        'model_state': model.state_dict(),
        'config': CONFIG,
        'test_metrics': test_metrics,
        'sparse_dims': sparse_dims,
    }, 'esmm_checkpoint.pt')
    print('\nSaved to esmm_checkpoint.pt')


if __name__ == '__main__':
    main()