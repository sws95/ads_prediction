"""
01_prepare_data.py
iPinYou season 2, advertiser 1458 데이터 준비.

- imp 로그에서 1458만 필터
- clk 로그의 BidID로 click 라벨 부착 (conversion은 1458에 존재하지 않으므로 제외)
- 7일치(6/06~6/12) 합쳐 하나의 parquet로 저장
"""

import bz2
import pandas as pd
from pathlib import Path

# ---- 경로/설정 ----
BASE = Path(r"C:\Users\swson23\Desktop\test\ads_prediction\ipinyou.contest.dataset\training2nd")
OUT_DIR = BASE.parent / "processed"
ADVERTISER = "1458"
DATES = ["20130606", "20130607", "20130608", "20130609",
         "20130610", "20130611", "20130612"]

COLUMNS = [
    "BidID", "Timestamp", "LogType", "iPinYouID", "UserAgent", "IP",
    "Region", "City", "AdExchange", "Domain", "URL", "AnonymousURLID",
    "AdSlotID", "AdSlotWidth", "AdSlotHeight", "AdSlotVisibility",
    "AdSlotFormat", "AdSlotFloorPrice", "CreativeID", "BiddingPrice",
    "PayingPrice", "KeyPageURL", "AdvertiserID", "UserTags",
]


def read_bz2_tsv(path, advertiser_filter=None, bidid_only=False):
    """bz2 압축 상태 그대로 스트리밍으로 읽고 필터링."""
    rows = []
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 24:
                continue
            if advertiser_filter and cols[22] != advertiser_filter:
                continue
            rows.append(cols[0] if bidid_only else cols)
    return rows


def main():
    OUT_DIR.mkdir(exist_ok=True)
    dfs = []

    for date in DATES:
        imp_rows = read_bz2_tsv(BASE / f"imp.{date}.txt.bz2", advertiser_filter=ADVERTISER)
        df = pd.DataFrame(imp_rows, columns=COLUMNS)

        clk_bids = set(read_bz2_tsv(BASE / f"clk.{date}.txt.bz2",
                                    advertiser_filter=ADVERTISER, bidid_only=True))
        df["click"] = df["BidID"].isin(clk_bids).astype("int8")
        df["date"] = date

        print(f"{date}: imp {len(df):>7,}  clk {df['click'].sum():>4,}  "
              f"CTR {df['click'].mean()*100:.4f}%")
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)

    # 수치 컬럼 변환
    num_cols = ["AdSlotWidth", "AdSlotHeight", "AdSlotVisibility", "AdSlotFormat",
                "AdSlotFloorPrice", "BiddingPrice", "PayingPrice"]
    for c in num_cols:
        full[c] = pd.to_numeric(full[c], errors="coerce")

    # object 컬럼 string 강제 (parquet 직렬화 안정화)
    for c in full.select_dtypes(include="object").columns:
        full[c] = full[c].astype(str)

    print("\n=== 1458 전체 (7일) ===")
    print(f"imp:        {len(full):,}")
    print(f"clicks:     {full['click'].sum():,}  (CTR {full['click'].mean()*100:.4f}%)")
    print(f"PayingPrice mean {full['PayingPrice'].mean():.1f} / "
          f"median {full['PayingPrice'].median():.0f} / "
          f"90% {full['PayingPrice'].quantile(0.9):.0f}")

    out_path = OUT_DIR / f"imp_{ADVERTISER}_season2_full.parquet"
    full.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
