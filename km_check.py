"""
06_km_real.py
진짜 censored 데이터(bid 로그)로 KM win rate 추정.

01c가 만든 bidlog_1458.parquet:
- win=1: market_price 관측 (event, 시장가 = PayingPrice)
- win=0: censored, "시장가 > bidprice" (censor at bidprice)

KM:
- event는 그 시장가에서 '사망'(시장가 확정)
- censor는 그 bidprice에서 위험집합 이탈(사망 아님)
- S(b)=P(market>b), win rate W(b)=1-S(b)

비교: 진짜 KM vs (이긴 것만으로 만든) naive 경험적 CDF.
naive는 진 경매를 무시해 시장가를 과소추정 -> win rate 과대추정 편향을 보임.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(r"C:\Users\swson23\Desktop\test\ads_prediction\ipinyou.contest.dataset")
PROC = BASE / "processed"
IN = PROC / "bidlog_1458.parquet"


def km_fit(price, event):
    """
    price: 관측 가격 (win이면 market_price, lose면 bidprice에서 censor)
    event: 1=사망(시장가 관측), 0=censored
    반환: (times, S) — S(t)=P(market>t)
    """
    order = np.argsort(price)
    p = np.asarray(price)[order]
    e = np.asarray(event)[order]
    n = len(p)
    at_risk = n
    S = 1.0
    times, surv = [], []
    i = 0
    while i < n:
        t = p[i]
        d = c = 0
        j = i
        while j < n and p[j] == t:
            if e[j] == 1: d += 1
            else: c += 1
            j += 1
        if at_risk > 0 and d > 0:
            S *= (1 - d / at_risk)
        times.append(t); surv.append(S)
        at_risk -= (d + c)
        i = j
    return np.array(times), np.array(surv)


def winrate_at(times, surv, grid):
    W = []
    for b in grid:
        idx = np.searchsorted(times, b, side="right") - 1
        s = surv[idx] if idx >= 0 else 1.0
        W.append(1 - s)
    return np.array(W)


def main():
    df = pd.read_parquet(IN)
    print(f"입찰 {len(df):,}, win {df['win'].sum():,} ({df['win'].mean()*100:.1f}%)")

    # KM 입력: win은 market_price에서 event=1, lose는 bidprice에서 censor(event=0)
    price = np.where(df["win"] == 1, df["market_price"], df["bidprice"]).astype(float)
    event = df["win"].values.astype(int)
    times, surv = km_fit(price, event)

    grid = np.arange(1, 301)
    W_km = winrate_at(times, surv, grid)

    # naive: 이긴 경매의 market_price만으로 경험적 CDF
    win_mp = df[df["win"] == 1]["market_price"].values.astype(float)
    W_naive = np.array([(win_mp <= b).mean() for b in grid])

    # 비교 출력 (몇 개 지점)
    print("\nbid별 win rate 비교 (KM=진짜, naive=이긴것만):")
    print(f"{'bid':>5} {'KM':>8} {'naive':>8}")
    for b in [20, 40, 60, 80, 100, 150, 200]:
        i = b - 1
        print(f"{b:>5} {W_km[i]:>8.3f} {W_naive[i]:>8.3f}")

    # 시각화
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, W_km, "-", lw=2, label="KM (real censored, bid log)")
    ax.plot(grid, W_naive, "--", lw=2, label="naive empirical (imp only)")
    ax.set_xlabel("bid"); ax.set_ylabel("win rate W(b)")
    ax.set_title("iPinYou 1458 — real KM vs naive (selection bias)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = PROC / "km_real.png"
    fig.savefig(out, dpi=120)

    pd.DataFrame({"bid": grid, "W_km": W_km, "W_naive": W_naive}).to_csv(
        PROC / "km_real.csv", index=False)
    print(f"\nSaved: {out}")
    print(f"Saved: {PROC/'km_real.csv'}")


if __name__ == "__main__":
    main()
