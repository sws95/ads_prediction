"""
diag_11278.py
1458 캠페인에서 11278 태그의 기여 분해.
세 설정의 test AUC 비교:
  (A) 태그 전부 유지         -> 0.99 예상
  (B) 11278만 제거, 나머지 태그 유지 -> 0.90 근처면 "누적" 가설 지지
  (C) 태그 전부 제거         -> 0.71 예상 (논문 LR)
"""
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

def row_feats(r, tag_mode):
    # tag_mode: "all"=전부, "drop_11278"=11278만 제외, "none"=태그 전부 제외
    wd,hr=parse_ts(r["Timestamp"]); os_,br=parse_ua(r["UserAgent"])
    f={f"wd={wd}":1,f"hr={hr}":1,f"region={r['Region']}":1,f"city={r['City']}":1,
       f"adx={r['AdExchange']}":1,f"domain={r['Domain']}":1,f"slot={r['AdSlotID']}":1,
       f"w={r['AdSlotWidth']}":1,f"h={r['AdSlotHeight']}":1,f"vis={r['AdSlotVisibility']}":1,
       f"fmt={r['AdSlotFormat']}":1,f"floor={floor_bucket(r['AdSlotFloorPrice'])}":1,
       f"creative={r['CreativeID']}":1,f"os={os_}":1,f"br={br}":1}
    if tag_mode != "none":
        t=str(r["UserTags"])
        if t and t not in ("null","nan"):
            for x in t.split(","):
                if tag_mode=="drop_11278" and x=="11278":
                    continue
                f[f"tag={x}"]=1
    return f

def build(df, tag_mode, h):
    return h.transform([row_feats(r, tag_mode) for _,r in df.iterrows()])

def run(tr, te, tag_mode):
    h=FeatureHasher(n_features=N_FEATURES,input_type="dict",alternate_sign=False)
    Xtr=build(tr,tag_mode,h); ytr=tr["click"].values
    Xte=build(te,tag_mode,h); yte=te["click"].values
    clf=LogisticRegression(C=1.0,max_iter=300,solver="saga"); clf.fit(Xtr,ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:,1])

def main():
    df=pd.read_parquet(IN_PATH)
    df=df[df["AdvertiserID"]=="1458"].reset_index(drop=True)
    tr=df[df["date"].isin(TRAIN_DATES)].reset_index(drop=True)
    te=df[df["date"].isin(TEST_DATES)].reset_index(drop=True)
    pos=tr[tr["click"]==1]; neg=tr[tr["click"]==0].sample(frac=NEG_RATE,random_state=RS)
    tr_ds=pd.concat([pos,neg]).sample(frac=1.0,random_state=RS).reset_index(drop=True)
    print(f"1458  train_ds {len(tr_ds):,} (pos {len(pos)})  test {len(te):,} (pos {te['click'].sum()})\n")

    a=run(tr_ds,te,"all")
    b=run(tr_ds,te,"drop_11278")
    c=run(tr_ds,te,"none")
    print(f"(A) 태그 전부 유지        : AUC {a:.4f}")
    print(f"(B) 11278만 제거          : AUC {b:.4f}")
    print(f"(C) 태그 전부 제거        : AUC {c:.4f}")
    print(f"\n11278 단독 기여 (A-B): {a-b:+.4f}")
    print(f"나머지 태그 기여 (B-C): {b-c:+.4f}")

if __name__=="__main__":
    main()
