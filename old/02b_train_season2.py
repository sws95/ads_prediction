"""
02b_train_season2.py
season 2 전체 CTR 모델. AdvertiserID 넣고 vs 빼고 비교 + 캠페인별 AUC.

- 시간순 분할: 6/06~6/11 학습, 6/12 평가
- negative downsampling + 재보정
- 두 설정(with_adv / without_adv)을 각각 학습해 AUC 비교
- 전체 AUC와 캠페인별 AUC를 함께 출력
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE = Path(r"C:\Users\swson23\Desktop\test\ads_prediction\ipinyou.contest.dataset")
PROC = BASE / "processed"
IN_PATH = PROC / "imp_season2_all.parquet"
TRAIN_DATES = ["20130606","20130607","20130608","20130609","20130610","20130611"]
TEST_DATES = ["20130612"]
N_FEATURES = 2**20
NEG_RATE = 0.1
RS = 42


def parse_ts(ts):
    try:
        dt = datetime.strptime(str(ts)[:14], "%Y%m%d%H%M%S")
        return str(dt.weekday()), f"{dt.hour:02d}"
    except Exception:
        return "na", "na"


def parse_ua(ua):
    u = str(ua).lower()
    os_ = ("windows" if "windows" in u else "mac" if "mac" in u else
           "android" if "android" in u else "ios" if ("iphone" in u or "ipad" in u) else
           "linux" if "linux" in u else "other")
    br = ("chrome" if "chrome" in u else "firefox" if "firefox" in u else
          "ie" if ("msie" in u or "trident" in u) else "safari" if "safari" in u else "other")
    return os_, br


def floor_bucket(v):
    try:
        v = float(v)
    except Exception:
        return "na"
    return ("0" if v <= 0 else "1-10" if v <= 10 else "11-50" if v <= 50
            else "51-100" if v <= 100 else "100+")


def row_feats(r, use_adv):
    wd, hr = parse_ts(r["Timestamp"])
    os_, br = parse_ua(r["UserAgent"])
    f = {
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
    if use_adv:
        f[f"adv={r['AdvertiserID']}"] = 1
    t = str(r["UserTags"])
    if t and t not in ("null", "nan"):
        for x in t.split(","):
            f[f"tag={x}"] = 1
    return f


def build(df, use_adv, hasher):
    return hasher.transform([row_feats(r, use_adv) for _, r in df.iterrows()])


def recalibrate(p, w):
    return p / (p + (1.0 - p) / w)


def train_eval(train_ds, test_df, use_adv, label):
    h = FeatureHasher(n_features=N_FEATURES, input_type="dict", alternate_sign=False)
    Xtr = build(train_ds, use_adv, h); ytr = train_ds["click"].values
    Xte = build(test_df, use_adv, h); yte = test_df["click"].values
    clf = LogisticRegression(C=1.0, max_iter=300, solver="saga")
    clf.fit(Xtr, ytr)
    p_raw = clf.predict_proba(Xte)[:, 1]
    p_cal = recalibrate(p_raw, NEG_RATE)

    overall = roc_auc_score(yte, p_raw)
    print(f"\n[{label}] 전체 AUC: {overall:.4f}")
    print("  캠페인별 AUC:")
    tmp = test_df.copy()
    tmp["p"] = p_raw
    for adv, grp in tmp.groupby("AdvertiserID"):
        if grp["click"].nunique() < 2:
            print(f"    {adv}: (클릭 0 또는 전부 클릭, AUC 계산 불가)")
            continue
        a = roc_auc_score(grp["click"].values, grp["p"].values)
        print(f"    {adv}: AUC {a:.4f}  (imp {len(grp):,}, clk {int(grp['click'].sum())})")
    return overall


def main():
    df = pd.read_parquet(IN_PATH)
    print(f"loaded: {len(df):,} rows")
    train_df = df[df["date"].isin(TRAIN_DATES)].reset_index(drop=True)
    test_df = df[df["date"].isin(TEST_DATES)].reset_index(drop=True)

    pos = train_df[train_df["click"] == 1]
    neg = train_df[train_df["click"] == 0].sample(frac=NEG_RATE, random_state=RS)
    train_ds = pd.concat([pos, neg]).sample(frac=1.0, random_state=RS).reset_index(drop=True)
    print(f"train down: {len(train_ds):,} (pos {len(pos):,})  test: {len(test_df):,} (pos {test_df['click'].sum():,})")

    a_with = train_eval(train_ds, test_df, use_adv=True,  label="with AdvertiserID")
    a_without = train_eval(train_ds, test_df, use_adv=False, label="without AdvertiserID")

    print("\n=== 요약 ===")
    print(f"with    AdvertiserID: 전체 AUC {a_with:.4f}")
    print(f"without AdvertiserID: 전체 AUC {a_without:.4f}")
    print(f"차이: {a_with - a_without:+.4f}")


if __name__ == "__main__":
    main()
