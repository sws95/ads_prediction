"""
diag_leakage.py
AUC 0.99 원인 진단: 피처를 하나씩 제거(ablation)하며 test AUC 변화를 본다.
AUC가 크게 떨어지는 피처가 누수/과결합의 주범.
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
IN_PATH = PROC / "imp_1458_season2_full.parquet"
TRAIN_DATES = ["20130606","20130607","20130608","20130609","20130610","20130611"]
TEST_DATES = ["20130612"]
N_FEATURES = 2**20
NEG_RATE = 0.1
RS = 42

# 끌 수 있는 피처 그룹
ALL_FIELDS = ["time","region","city","adx","domain","slot","size",
              "vis","fmt","floor","creative","ua","tag"]

def parse_ts(ts):
    try:
        dt = datetime.strptime(str(ts)[:14], "%Y%m%d%H%M%S")
        return str(dt.weekday()), f"{dt.hour:02d}"
    except Exception:
        return "na","na"

def parse_ua(ua):
    u=str(ua).lower()
    os_=("windows" if "windows" in u else "mac" if "mac" in u else
         "android" if "android" in u else "ios" if ("iphone" in u or "ipad" in u) else
         "linux" if "linux" in u else "other")
    br=("chrome" if "chrome" in u else "firefox" if "firefox" in u else
        "ie" if ("msie" in u or "trident" in u) else "safari" if "safari" in u else "other")
    return os_,br

def floor_bucket(v):
    try: v=float(v)
    except: return "na"
    return "0" if v<=0 else "1-10" if v<=10 else "11-50" if v<=50 else "51-100" if v<=100 else "100+"

def row_feats(r, drop):
    f={}
    if "time" not in drop:
        wd,hr=parse_ts(r["Timestamp"]); f[f"wd={wd}"]=1; f[f"hr={hr}"]=1
    if "region" not in drop: f[f"region={r['Region']}"]=1
    if "city" not in drop: f[f"city={r['City']}"]=1
    if "adx" not in drop: f[f"adx={r['AdExchange']}"]=1
    if "domain" not in drop: f[f"domain={r['Domain']}"]=1
    if "slot" not in drop: f[f"slot={r['AdSlotID']}"]=1
    if "size" not in drop: f[f"w={r['AdSlotWidth']}"]=1; f[f"h={r['AdSlotHeight']}"]=1
    if "vis" not in drop: f[f"vis={r['AdSlotVisibility']}"]=1
    if "fmt" not in drop: f[f"fmt={r['AdSlotFormat']}"]=1
    if "floor" not in drop: f[f"floor={floor_bucket(r['AdSlotFloorPrice'])}"]=1
    if "creative" not in drop: f[f"creative={r['CreativeID']}"]=1
    if "ua" not in drop:
        os_,br=parse_ua(r["UserAgent"]); f[f"os={os_}"]=1; f[f"br={br}"]=1
    if "tag" not in drop:
        t=str(r["UserTags"])
        if t and t not in ("null","nan"):
            for x in t.split(","): f[f"tag={x}"]=1
    return f

def build(df, drop, hasher):
    return hasher.transform([row_feats(r, drop) for _,r in df.iterrows()])

def run(train_ds, test_df, drop):
    h=FeatureHasher(n_features=N_FEATURES,input_type="dict",alternate_sign=False)
    Xtr=build(train_ds,drop,h); ytr=train_ds["click"].values
    Xte=build(test_df,drop,h); yte=test_df["click"].values
    clf=LogisticRegression(C=1.0,max_iter=300,solver="saga")
    clf.fit(Xtr,ytr)
    p=clf.predict_proba(Xte)[:,1]
    return roc_auc_score(yte,p)

def main():
    df=pd.read_parquet(IN_PATH)
    tr=df[df["date"].isin(TRAIN_DATES)].reset_index(drop=True)
    te=df[df["date"].isin(TEST_DATES)].reset_index(drop=True)
    pos=tr[tr["click"]==1]; neg=tr[tr["click"]==0].sample(frac=NEG_RATE,random_state=RS)
    tr_ds=pd.concat([pos,neg]).sample(frac=1.0,random_state=RS).reset_index(drop=True)

    base=run(tr_ds,te,drop=set())
    print(f"[baseline] 전체 피처 AUC: {base:.4f}\n")
    print("피처 하나씩 제거 시 AUC (많이 떨어질수록 그 피처가 핵심/누수):")
    res=[]
    for fld in ALL_FIELDS:
        a=run(tr_ds,te,drop={fld})
        res.append((fld, a, base-a))
        print(f"  -{fld:<9}: {a:.4f}  (Δ {base-a:+.4f})")
    res.sort(key=lambda x:-x[2])
    print(f"\n가장 영향 큰 피처: {res[0][0]} (제거 시 AUC {res[0][1]:.4f})")

if __name__=="__main__":
    main()
