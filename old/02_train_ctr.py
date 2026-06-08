"""
02_train_ctr.py  (final: tag 제외, negative downsampling + recalibration + calibration plot)
1458 CTR 예측 모델 (pCTR 생산).

설정 확정:
- UserTags 제외. 이유: 11278(in-market 의류) 등 in-market 태그가 클릭과 과결합돼
  AUC가 0.99까지 치솟아 입찰 전략 비교의 변별력이 사라짐. 논문 LR 베이스라인(1458 ≈ 0.71)과
  직접 비교 가능하고, 분포가 변하는 현실 난이도에 더 가까운 설정으로 태그를 제외함.
  (태그는 정상 피처이며 누수가 아님 — 실험 설계상의 선택)
- negative downsampling(NEG_SAMPLE_RATE) + 재보정 p/(p+(1-p)/w)
- calibration plot 저장

평가 분할: 6/06~6/11 학습, 6/12 평가 (시간순)
출력: sim_input_1458.parquet (보정된 pCTR + 시장가 + 클릭)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

BASE = Path(r"C:\Users\swson23\Desktop\test\ads_prediction\ipinyou.contest.dataset")
PROC = BASE / "processed"
IN_PATH = PROC / "imp_1458_season2_full.parquet"

TRAIN_DATES = ["20130606", "20130607", "20130608", "20130609", "20130610", "20130611"]
TEST_DATES = ["20130612"]
N_FEATURES = 2 ** 20
NEG_SAMPLE_RATE = 0.1   # negative 중 10%만 사용 (= 보정식의 w)
RANDOM_STATE = 42


def parse_timestamp(ts: str):
    try:
        dt = datetime.strptime(ts[:14], "%Y%m%d%H%M%S")
        return str(dt.weekday()), f"{dt.hour:02d}"
    except Exception:
        return "na", "na"


def parse_useragent(ua: str):
    ua_l = ua.lower()
    if "windows" in ua_l:
        os_ = "windows"
    elif "mac" in ua_l:
        os_ = "mac"
    elif "android" in ua_l:
        os_ = "android"
    elif "iphone" in ua_l or "ipad" in ua_l:
        os_ = "ios"
    elif "linux" in ua_l:
        os_ = "linux"
    else:
        os_ = "other"
    if "chrome" in ua_l:
        br = "chrome"
    elif "firefox" in ua_l:
        br = "firefox"
    elif "msie" in ua_l or "trident" in ua_l:
        br = "ie"
    elif "safari" in ua_l:
        br = "safari"
    else:
        br = "other"
    return os_, br


def floor_bucket(v):
    try:
        v = float(v)
    except Exception:
        return "na"
    if v <= 0:
        return "0"
    if v <= 10:
        return "1-10"
    if v <= 50:
        return "11-50"
    if v <= 100:
        return "51-100"
    return "100+"


def row_to_features(r):
    """UserTags 제외. 문맥/게재면/디바이스 피처만 사용."""
    wd, hr = parse_timestamp(str(r["Timestamp"]))
    os_, br = parse_useragent(str(r["UserAgent"]))
    feats = {
        f"wd={wd}": 1, f"hr={hr}": 1,
        f"region={r['Region']}": 1, f"city={r['City']}": 1,
        f"adx={r['AdExchange']}": 1, f"domain={r['Domain']}": 1,
        f"slot={r['AdSlotID']}": 1,
        f"w={r['AdSlotWidth']}": 1, f"h={r['AdSlotHeight']}": 1,
        f"vis={r['AdSlotVisibility']}": 1, f"fmt={r['AdSlotFormat']}": 1,
        f"floor={floor_bucket(r['AdSlotFloorPrice'])}": 1,
        f"creative={r['CreativeID']}": 1,
        f"os={os_}": 1, f"br={br}": 1,
    }
    return feats


def build_matrix(df, hasher):
    dicts = [row_to_features(r) for _, r in df.iterrows()]
    return hasher.transform(dicts)


def recalibrate(p, w):
    """downsampling으로 부풀려진 확률 p를 원래 분포로 보정. w = negative sample rate."""
    return p / (p + (1.0 - p) / w)


def calibration_table(y_true, p_pred, n_bins=10):
    bins = np.linspace(0, p_pred.max() + 1e-12, n_bins + 1)
    idx = np.digitize(p_pred, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append((p_pred[m].mean(), y_true[m].mean(), m.sum()))
    return np.array(rows)


def main():
    df = pd.read_parquet(IN_PATH)
    print(f"loaded: {len(df):,} rows")

    train_df = df[df["date"].isin(TRAIN_DATES)].reset_index(drop=True)
    test_df = df[df["date"].isin(TEST_DATES)].reset_index(drop=True)

    pos = train_df[train_df["click"] == 1]
    neg = train_df[train_df["click"] == 0].sample(frac=NEG_SAMPLE_RATE, random_state=RANDOM_STATE)
    train_ds = pd.concat([pos, neg]).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"train full: {len(train_df):,} (pos {len(pos):,})")
    print(f"train down: {len(train_ds):,} (pos {len(pos):,}, neg {len(neg):,}, "
          f"pos rate {train_ds['click'].mean()*100:.2f}%)")
    print(f"test      : {len(test_df):,} (pos {test_df['click'].sum():,})")

    hasher = FeatureHasher(n_features=N_FEATURES, input_type="dict", alternate_sign=False)
    print("building matrices...")
    X_train = build_matrix(train_ds, hasher)
    y_train = train_ds["click"].values
    X_test = build_matrix(test_df, hasher)
    y_test = test_df["click"].values

    print("training LR (tag 제외, downsampled)...")
    clf = LogisticRegression(C=1.0, max_iter=300, solver="saga")
    clf.fit(X_train, y_train)

    p_test_raw = clf.predict_proba(X_test)[:, 1]
    p_test_cal = recalibrate(p_test_raw, NEG_SAMPLE_RATE)

    actual_ctr = y_test.mean()
    print("\n=== CTR 모델 평가 (test = 6/12, tag 제외) ===")
    print(f"AUC: {roc_auc_score(y_test, p_test_raw):.4f}  (논문 LR 1458 ≈ 0.71 기준)")
    print(f"실제 CTR        : {actual_ctr*100:.4f}%")
    print(f"avg pCTR (raw)  : {p_test_raw.mean()*100:.4f}%")
    print(f"avg pCTR (cal)  : {p_test_cal.mean()*100:.4f}%   <- 실제 CTR에 근접해야 정상")
    print(f"logloss (raw)   : {log_loss(y_test, p_test_raw):.5f}")
    print(f"logloss (cal)   : {log_loss(y_test, p_test_cal):.5f}")

    tab_raw = calibration_table(y_test, p_test_raw)
    tab_cal = calibration_table(y_test, p_test_cal)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, tab, title in [(axes[0], tab_raw, "Before recalibration (raw)"),
                           (axes[1], tab_cal, "After recalibration")]:
        lim = max(tab[:, 0].max(), tab[:, 1].max()) * 1.1
        ax.plot([0, lim], [0, lim], "--", color="gray", label="perfect")
        ax.plot(tab[:, 0], tab[:, 1], "o-", color="#D85A30", label="model")
        ax.set_xlabel("mean predicted pCTR")
        ax.set_ylabel("actual CTR")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"Calibration (1458, tag 제외, w={NEG_SAMPLE_RATE})")
    fig.tight_layout()
    out_png = PROC / "calibration.png"
    fig.savefig(out_png, dpi=120)
    print(f"\nSaved calibration plot: {out_png}")

    sim = test_df[["Timestamp", "PayingPrice", "click"]].copy()
    sim["pCTR"] = p_test_cal
    sim["pCTR_raw"] = p_test_raw
    sim["avg_train_pCTR"] = actual_ctr
    sim = sim.sort_values("Timestamp").reset_index(drop=True)
    out_path = PROC / "sim_input_1458.parquet"
    sim.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"Saved sim input: {out_path}")


if __name__ == "__main__":
    main()