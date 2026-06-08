"""
05_ortb2_csensitivity.py
(1) ORTB1 vs ORTB2 비교
(2) c 민감도 분석 (c는 win rate model에서 오는 시장특성 파라미터)

ORTB1 (Zhang 2014, w(b)=b/(b+c) 가정):
    bid = sqrt(c/λ * pCTR + c²) - c
ORTB2 (w(b)=(b/(b+c))² 가정에 해당하는 닫힌형):
    let A = pCTR + sqrt(c²λ² + pCTR²)
    bid = c * ( (A/(cλ))^(1/3) - (cλ/A)^(1/3) )

c 민감도:
- c는 곡선 '모양', λ는 '높이'. λ를 예산에 맞춰 재탐색하면 c의 영향이 작아야 함(robust).
- 여러 c에 대해 예산별 최적 클릭을 구해, c에 결과가 얼마나 둔감한지 확인.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(r"C:\Users\swson23\Desktop\test\ads_prediction\ipinyou.contest.dataset")
PROC = BASE / "processed"
SIM_PATH = PROC / "sim_input_1458.parquet"


def simulate(bids, market, click, budget):
    spend = 0.0; won = clicks = 0
    for i in range(len(bids)):
        b, mp = bids[i], market[i]
        if b >= mp and spend + mp <= budget:
            spend += mp; won += 1; clicks += click[i]
    return {"clicks": int(clicks), "won": won, "spend": round(spend,1),
            "win_rate": won/len(bids), "eCPC": (spend/clicks) if clicks else None}


def ortb1(p, c, lam):
    return np.sqrt(c / lam * p + c * c) - c

def ortb2(p, c, lam):
    A = p + np.sqrt(c*c*lam*lam + p*p)
    term = (A / (c*lam)) ** (1.0/3.0)
    return c * (term - 1.0/term)


def best_over_lambda(bid_fn, p, c, market, click, budget, lam_grid):
    best = None
    for lam in lam_grid:
        bids = bid_fn(p, c, lam)
        r = simulate(bids, market, click, budget)
        if best is None or r["clicks"] > best[1]["clicks"]:
            best = (lam, r)
    return best


def main():
    df = pd.read_parquet(SIM_PATH)
    market = df["PayingPrice"].values.astype(float)
    click = df["click"].values.astype(int)
    p = df["pCTR"].values.astype(float)
    full_cost = market.sum()

    lam_grid = [1e-4,5e-5,2e-5,1e-5,5e-6,2e-6,1e-6,5e-7]
    budget_fracs = [1/64, 1/32, 1/16, 1/8, 1/4, 1/2]
    c_default = 55  # win rate 적합값(c0)에 맞춤

    # ---------- (1) ORTB1 vs ORTB2 ----------
    print("=== ORTB1 vs ORTB2 (예산별, c=55, λ 재탐색) ===")
    print(f"{'budget':>14} {'ORTB1':>8} {'ORTB2':>8}")
    rows = []
    for frac in budget_fracs:
        bud = full_cost*frac
        b1 = best_over_lambda(ortb1, p, c_default, market, click, bud, lam_grid)
        b2 = best_over_lambda(ortb2, p, c_default, market, click, bud, lam_grid)
        rows.append({"budget":round(bud),"frac":f"1/{int(1/frac)}",
                     "ortb1":b1[1]["clicks"],"ortb2":b2[1]["clicks"]})
        print(f"1/{int(1/frac):<3} ({bud:>11,.0f}) {b1[1]['clicks']:>8} {b2[1]['clicks']:>8}")

    # ---------- (2) c 민감도 ----------
    print("\n=== c 민감도 (ORTB1, 예산별 최적 클릭) ===")
    c_list = [20, 35, 55, 80, 120]
    print(f"{'budget':>14} " + " ".join(f"c={c:>3}" for c in c_list))
    csens = []
    for frac in budget_fracs:
        bud = full_cost*frac
        line = {}
        cells = []
        for c in c_list:
            b = best_over_lambda(ortb1, p, c, market, click, bud, lam_grid)
            line[f"c{c}"] = b[1]["clicks"]
            cells.append(f"{b[1]['clicks']:>5}")
        line["frac"] = f"1/{int(1/frac)}"; line["budget"] = round(bud)
        csens.append(line)
        print(f"1/{int(1/frac):<3} ({bud:>11,.0f}) " + " ".join(cells))

    # c 민감도 요약: 각 예산에서 (max-min)/max
    print("\nc 변화에 따른 클릭 변동폭 (작을수록 robust):")
    for line in csens:
        vals = [line[f"c{c}"] for c in c_list]
        spread = (max(vals)-min(vals))/max(vals) if max(vals)>0 else 0
        print(f"  {line['frac']:>5}: {min(vals)}~{max(vals)} (변동 {spread*100:.1f}%)")

    pd.DataFrame(rows).to_csv(PROC/"ortb1_vs_ortb2.csv", index=False)
    pd.DataFrame(csens).to_csv(PROC/"c_sensitivity.csv", index=False)

    # ---------- 시각화 ----------
    fig, axes = plt.subplots(1, 3, figsize=(18,5))

    # (a) bid 곡선: c별 모양
    ax = axes[0]
    pgrid = np.linspace(0, 0.01, 200)
    lam_fix = 1e-5
    for c in c_list:
        ax.plot(pgrid*100, ortb1(pgrid, c, lam_fix), label=f"c={c}")
    ax.set_xlabel("pCTR (%)"); ax.set_ylabel("bid"); ax.set_title("ORTB1 bid curve by c (λ fixed)")
    ax.legend(); ax.grid(alpha=0.3)

    # (b) ORTB1 vs ORTB2
    ax = axes[1]
    x = [r["budget"] for r in rows]
    ax.plot(x, [r["ortb1"] for r in rows], "^-", label="ORTB1")
    ax.plot(x, [r["ortb2"] for r in rows], "v--", label="ORTB2")
    ax.set_xscale("log"); ax.set_xlabel("budget (log)"); ax.set_ylabel("clicks")
    ax.set_title("ORTB1 vs ORTB2"); ax.legend(); ax.grid(alpha=0.3)

    # (c) c 민감도: 예산별 c별 클릭
    ax = axes[2]
    for c in c_list:
        ax.plot([l["budget"] for l in csens], [l[f"c{c}"] for l in csens], "o-", label=f"c={c}")
    ax.set_xscale("log"); ax.set_xlabel("budget (log)"); ax.set_ylabel("clicks")
    ax.set_title("c sensitivity (λ re-tuned per budget)"); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("iPinYou 1458 — ORTB2 & c sensitivity")
    fig.tight_layout()
    out = PROC/"ortb2_csens.png"; fig.savefig(out, dpi=120)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
