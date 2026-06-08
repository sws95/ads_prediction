"""
01c_prepare_bidlog.py
1458 bid 로그 추출 + imp 대조로 win/lose 라벨 부착 (진짜 censored 데이터 생성).

- bid 로그: 입찰한 모든 경매 (이긴 것 + 진 것)
- imp 로그: 이긴 경매만 (PayingPrice 관측됨)
- bid의 BidID가 imp에 있으면 win, 없으면 lose(censored)

win  : market_price = PayingPrice 관측 (event=1)
lose : market_price 미관측, "BiddingPrice보다 큼" (event=0, censored at BiddingPrice)

KM 입력으로 쓸 (bidprice, win, market_price) 만 저장 -> 용량 최소화.
bid 로그가 크므로(7일 ~3GB bz2) 스트리밍 처리.
"""

import bz2
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\swson23\Desktop\test\ads_prediction\ipinyou.contest.dataset\training2nd")
OUT_DIR = BASE.parent / "processed"
ADV = "1458"
DATES = ["20130606","20130607","20130608","20130609","20130610","20130611","20130612"]

# 컬럼 인덱스
# imp 로그: 24컬럼, AdvertiserID=22, PayingPrice=20, BidID=0
# bid 로그: 21컬럼, AdvertiserID=19, BiddingPrice=18, BidID=0 (PayingPrice 없음)
I_BIDID = 0
IMP_PAYPRICE = 20
IMP_ADV = 22
BID_BIDPRICE = 18
BID_ADV = 19


def read_imp_winset(path):
    """imp 로그(24컬럼, 이긴 경매)에서 1458의 {BidID: PayingPrice} 반환."""
    win = {}
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) != 24 or c[IMP_ADV] != ADV:
                continue
            try:
                win[c[I_BIDID]] = int(c[IMP_PAYPRICE])
            except ValueError:
                continue
    return win


def process_day(date):
    """bid 로그(21컬럼)를 스트리밍하며 1458만, imp 대조해 win/lose 라벨."""
    win_map = read_imp_winset(BASE / f"imp.{date}.txt.bz2")
    rows = []
    n_bid = n_win = 0
    with bz2.open(BASE / f"bid.{date}.txt.bz2", "rt", encoding="utf-8") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) != 21 or c[BID_ADV] != ADV:
                continue
            n_bid += 1
            bidid = c[I_BIDID]
            try:
                bidprice = int(c[BID_BIDPRICE])
            except ValueError:
                continue
            if bidid in win_map:
                rows.append((bidprice, 1, win_map[bidid]))  # win, market 관측
                n_win += 1
            else:
                rows.append((bidprice, 0, -1))               # lose, censored
    print(f"{date}: bid {n_bid:>8,}  win {n_win:>7,}  "
          f"win_rate {n_win/n_bid*100:.1f}%" if n_bid else f"{date}: no bids")
    return rows


def main():
    OUT_DIR.mkdir(exist_ok=True)
    all_rows = []
    for d in DATES:
        all_rows.extend(process_day(d))

    df = pd.DataFrame(all_rows, columns=["bidprice", "win", "market_price"])
    print(f"\n총 입찰: {len(df):,}")
    print(f"win: {df['win'].sum():,}  ({df['win'].mean()*100:.1f}%)")
    print(f"lose(censored): {(df['win']==0).sum():,}")
    print(f"win 시 평균 시장가: {df[df['win']==1]['market_price'].mean():.1f}")
    print(f"평균 입찰가: {df['bidprice'].mean():.1f}")

    out = OUT_DIR / "bidlog_1458.parquet"
    df.to_parquet(out, index=False, engine="pyarrow")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()