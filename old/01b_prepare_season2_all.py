"""
01b_prepare_season2_all.py
iPinYou season 2 전체(1458, 3358, 3386, 3427, 3476) 데이터 준비.

- imp 로그를 advertiser 필터 없이 다 읽되, season 2 캠페인만 유지
- clk 로그의 (BidID) 집합으로 click 라벨 부착
- AdvertiserID를 컬럼으로 보존 (피처 넣고/빼고 비교용)
- 7일치 합쳐 parquet 저장
"""

import bz2
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\swson23\Desktop\test\ads_prediction\ipinyou.contest.dataset\training2nd")
OUT_DIR = BASE.parent / "processed"
SEASON2 = {"1458", "3358", "3386", "3427", "3476"}
DATES = ["20130606", "20130607", "20130608", "20130609",
         "20130610", "20130611", "20130612"]

COLUMNS = [
    "BidID", "Timestamp", "LogType", "iPinYouID", "UserAgent", "IP",
    "Region", "City", "AdExchange", "Domain", "URL", "AnonymousURLID",
    "AdSlotID", "AdSlotWidth", "AdSlotHeight", "AdSlotVisibility",
    "AdSlotFormat", "AdSlotFloorPrice", "CreativeID", "BiddingPrice",
    "PayingPrice", "KeyPageURL", "AdvertiserID", "UserTags",
]


def read_imp(path):
    """season 2 캠페인만 유지하고 전체 행 반환."""
    rows = []
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 24:
                continue
            if cols[22] not in SEASON2:
                continue
            rows.append(cols)
    return rows


def read_clk_bids(path):
    """clk 로그에서 season 2 (BidID, AdvertiserID) 집합 -> BidID만."""
    bids = set()
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 24:
                continue
            if cols[22] not in SEASON2:
                continue
            bids.add(cols[0])
    return bids


def main():
    OUT_DIR.mkdir(exist_ok=True)
    dfs = []

    for date in DATES:
        imp_rows = read_imp(BASE / f"imp.{date}.txt.bz2")
        df = pd.DataFrame(imp_rows, columns=COLUMNS)
        clk_bids = read_clk_bids(BASE / f"clk.{date}.txt.bz2")
        df["click"] = df["BidID"].isin(clk_bids).astype("int8")
        df["date"] = date

        by_adv = df.groupby("AdvertiserID")["click"].agg(["size", "sum"])
        line = "  ".join(f"{a}:{r['size']:,}/{r['sum']}" for a, r in by_adv.iterrows())
        print(f"{date}: {line}")
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)

    num_cols = ["AdSlotWidth", "AdSlotHeight", "AdSlotVisibility", "AdSlotFormat",
                "AdSlotFloorPrice", "BiddingPrice", "PayingPrice"]
    for c in num_cols:
        full[c] = pd.to_numeric(full[c], errors="coerce")
    for c in full.select_dtypes(include="object").columns:
        full[c] = full[c].astype(str)

    print("\n=== season 2 전체 ===")
    print(f"imp: {len(full):,}")
    print("캠페인별 노출/클릭/CTR:")
    g = full.groupby("AdvertiserID")["click"].agg(["size", "sum", "mean"])
    for a, r in g.iterrows():
        print(f"  {a}: imp {r['size']:>9,}  clk {int(r['sum']):>5,}  CTR {r['mean']*100:.4f}%")

    out_path = OUT_DIR / "imp_season2_all.parquet"
    full.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
