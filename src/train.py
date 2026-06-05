"""
ESMM 학습 루프.
- Train: pCTR, pCTCVR 두 loss로 학습
- Eval: CTR AUC, CTCVR AUC 측정 (pCVR은 라벨 없어서 직접 평가 불가)
"""

import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.model import esmm_loss


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_ctr_loss = 0.0
    total_ctcvr_loss = 0.0
    n_batches = 0

    for batch in loader:
        sparse = batch['sparse'].to(device)
        dense = batch['dense'].to(device)
        click = batch['click'].to(device)
        ctcvr = batch['ctcvr'].to(device)

        optimizer.zero_grad()
        pCTR, pCVR, pCTCVR = model(sparse, dense)
        loss, l_ctr, l_ctcvr = esmm_loss(pCTR, pCTCVR, click, ctcvr)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ctr_loss += l_ctr.item()
        total_ctcvr_loss += l_ctcvr.item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'ctr_loss': total_ctr_loss / n_batches,
        'ctcvr_loss': total_ctcvr_loss / n_batches,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_ctr_pred, all_ctr_label = [], []
    all_ctcvr_pred, all_ctcvr_label = [], []
    all_cvr_pred = []  # 진단용 (라벨 없어서 AUC 측정은 클릭된 샘플에서만)
    all_click = []
    all_purchase = []

    for batch in loader:
        sparse = batch['sparse'].to(device)
        dense = batch['dense'].to(device)
        click = batch['click']
        ctcvr = batch['ctcvr']

        pCTR, pCVR, pCTCVR = model(sparse, dense)
        all_ctr_pred.append(pCTR.cpu().numpy())
        all_ctr_label.append(click.numpy())
        all_ctcvr_pred.append(pCTCVR.cpu().numpy())
        all_ctcvr_label.append(ctcvr.numpy())
        all_cvr_pred.append(pCVR.cpu().numpy())
        all_click.append(click.numpy())
        # purchase 라벨 = ctcvr / click (click=1일 때만 의미)
        all_purchase.append((ctcvr / click.clamp(min=1)).numpy())

    ctr_pred = np.concatenate(all_ctr_pred)
    ctr_label = np.concatenate(all_ctr_label)
    ctcvr_pred = np.concatenate(all_ctcvr_pred)
    ctcvr_label = np.concatenate(all_ctcvr_label)
    cvr_pred = np.concatenate(all_cvr_pred)
    click_arr = np.concatenate(all_click)

    ctr_auc = roc_auc_score(ctr_label, ctr_pred)
    ctcvr_auc = roc_auc_score(ctcvr_label, ctcvr_pred)

    # CVR AUC: 클릭된 샘플에서만 측정 (논문 평가 방식)
    mask = click_arr == 1
    if mask.sum() > 0 and ctcvr_label[mask].sum() > 0:
        # purchase 라벨 복원: click=1인 샘플의 ctcvr 값이 곧 purchase
        purchase_label = ctcvr_label[mask]
        cvr_auc = roc_auc_score(purchase_label, cvr_pred[mask])
    else:
        cvr_auc = float('nan')

    return {'ctr_auc': ctr_auc, 'cvr_auc': cvr_auc, 'ctcvr_auc': ctcvr_auc}


def train(model, train_ds, val_ds, test_ds, config, device):
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True, num_workers=config.get('num_workers', 0),
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'] * 4,
                            shuffle=False, num_workers=config.get('num_workers', 0),
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'] * 4,
                             shuffle=False, num_workers=config.get('num_workers', 0),
                             pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    best_val_auc = 0.0
    best_state = None

    for epoch in range(config['epochs']):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        dt = time.time() - t0

        avg_auc = (val_metrics['ctr_auc'] + val_metrics['ctcvr_auc']) / 2
        print(
            f"Epoch {epoch+1:2d} | "
            f"loss={train_metrics['loss']:.4f} "
            f"(ctr={train_metrics['ctr_loss']:.4f}, ctcvr={train_metrics['ctcvr_loss']:.4f}) | "
            f"val CTR AUC={val_metrics['ctr_auc']:.4f}, "
            f"CVR AUC={val_metrics['cvr_auc']:.4f}, "
            f"CTCVR AUC={val_metrics['ctcvr_auc']:.4f} | "
            f"{dt:.1f}s"
        )

        if avg_auc > best_val_auc:
            best_val_auc = avg_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 베스트 모델로 test 평가
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    print(f"\n[Test] CTR AUC={test_metrics['ctr_auc']:.4f}, "
          f"CVR AUC={test_metrics['cvr_auc']:.4f}, "
          f"CTCVR AUC={test_metrics['ctcvr_auc']:.4f}")
    return test_metrics