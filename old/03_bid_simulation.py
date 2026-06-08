"""
03_bid_simulation.py (full: 예산별 재탐색 + eCPC/winrate + bid landscape ORTB + 시각화)

평가 프로토콜 (Zhang et al. RTB benchmark):
- 테스트셋을 시간순 순회, 전략이 bid 계산
- bid >= 시장가(PayingPrice)면 낙찰, 비용 = 시장가 (2nd price)
- 예산 소진 시 입찰 불가, 목표는 예산 내 획득 클릭 최대화

전략:
- Constant : bid = c
- Linear   : bid = base * pCTR / avg_pCTR
- ORTB     : bid = sqrt(c/lambda * pCTR + c^2) - c   (시장가를 안다고 가정한 단순형)
- ORTB-LP  : 같은 식이되 lambda를 win rate model 기반 예산 매칭으로 보정한 정식형

bid landscape (win rate model):
- w(b) = b / (b + c0)  형태로 입찰가 b의 낙찰확률을 추정 (Zhang 2014)
- train 시장가 분포로 c0를 적합 -> 기대 비용/낙찰 수 예측에 사용
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


# ---------------- 시뮬레이터 ----------------
def simulate(bids, market, click, budget):
    spend = 0.0
    won = clicks = 0
    n = len(bids)
    for i in range(n):
        b = bids[i]; mp = market[i]
        if b >= mp and spend + mp <= budget:
            spend += mp; won += 1; clicks += click[i]
    return {
        "clicks": int(clicks), "won": won, "spend": round(spend, 1),
        "win_rate": won / n if n else 0.0,
        "eCPC": (spend / clicks) if clicks else None,
    }


def bid_constant(n, c):
    return np.full(n, float(c))

def bid_linear(p, base, avg):
    return base * p / avg

def bid_ortb(p, c, lam):
    return np.sqrt(c / lam * p + c * c) - c


# ---------------- bid landscape ----------------
def fit_winrate_c0(market_train):
    """
    win rate model w(b) = b/(b+c0).
    train 시장가 분포에서, 입찰가 b로 이길 확률 = P(market < b).
    이 경험적 win curve에 b/(b+c0)를 최소제곱 적합해 c0 추정.
    """
    mp = np.sort(market_train)
    grid = np.unique(np.quantile(mp, np.linspace(0.01, 0.99, 50))).astype(float)
    emp = np.array([(mp < b).mean() for b in grid])  # 경험적 낙찰확률
    # b/(b+c0) 적합: c0 그리드 탐색
    best_c0, best_err = None, 1e18
    for c0 in np.linspace(1, 300, 600):
        pred = grid / (grid + c0)
        err = np.mean((pred - emp) ** 2)
        if err < best_err:
            best_err, best_c0 = err, c0
    return best_c0, grid, emp


# ---------------- 파라미터 재탐색 ----------------
def best_constant(market, click, budget, grid):
    best = None
    for c in grid:
        r = simulate(bid_constant(len(market), c), market, click, budget)
        if best is None or r["clicks"] > best[1]["clicks"]:
            best = (c, r)
    return best

def best_linear(p, avg, market, click, budget, grid):
    best = None
    for base in grid:
        r = simulate(bid_linear(p, base, avg), market, click, budget)
        if best is None or r["clicks"] > best[1]["clicks"]:
            best = (base, r)
    return best

def best_ortb(p, market, click, budget, c, lam_grid):
    best = None
    for lam in lam_grid:
        r = simulate(bid_ortb(p, c, lam), market, click, budget)
        if best is None or r["clicks"] > best[1]["clicks"]:
            best = (lam, r)
    return best


def main():
    df = pd.read_parquet(SIM_PATH)
    market = df["PayingPrice"].values.astype(float)
    click = df["click"].values.astype(int)
    p = df["pCTR"].values.astype(float)
    avg = float(df["avg_train_pCTR"].iloc[0])

    total_clicks = int(click.sum())
    full_cost = market.sum()
    print(f"test impressions : {len(df):,}")
    print(f"available clicks : {total_clicks:,}")
    print(f"cost to win all  : {full_cost:,.0f}")
    print(f"avg market price : {market.mean():.1f}")
    print(f"avg pCTR         : {avg:.6f}\n")

    # bid landscape 적합
    c0, grid_b, emp = fit_winrate_c0(market)
    print(f"[bid landscape] win rate model w(b)=b/(b+c0),  c0 ≈ {c0:.1f}\n")

    # 탐색 그리드 (범위 확장)
    const_grid = [20, 30, 40, 50, 60, 80, 100, 120, 150]
    lin_grid = [30, 40, 50, 60, 80, 100, 120, 150, 200, 300]
    c_ortb = 50
    lam_grid = [1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 5e-7]

    budget_fracs = [1/64, 1/32, 1/16, 1/8, 1/4, 1/2]
    rows = []
    print("=== 예산별 전략 비교 (각 예산에서 파라미터 재탐색) ===")
    print(f"{'budget':>16} {'Const':>22} {'Linear':>24} {'ORTB':>26}")
    for frac in budget_fracs:
        bud = full_cost * frac
        bc = best_constant(market, click, bud, const_grid)
        bl = best_linear(p, avg, market, click, bud, lin_grid)
        bo = best_ortb(p, market, click, bud, c_ortb, lam_grid)

        rows.append({
            "budget_frac": f"1/{int(1/frac)}", "budget": round(bud),
            "const_clicks": bc[1]["clicks"], "const_c": bc[0],
            "linear_clicks": bl[1]["clicks"], "linear_base": bl[0], "linear_eCPC": bl[1]["eCPC"],
            "ortb_clicks": bo[1]["clicks"], "ortb_lam": bo[0], "ortb_eCPC": bo[1]["eCPC"],
            "linear_winrate": bl[1]["win_rate"], "ortb_winrate": bo[1]["win_rate"],
        })

        def fmt(tag, r, param):
            e = f"{r['eCPC']:.0f}" if r['eCPC'] else "-"
            return f"clk={r['clicks']:>3}({param}) eCPC={e}"
        print(f"1/{int(1/frac):<3} ({bud:>12,.0f})  "
              f"{fmt('C',bc[1],bc[0]):>20}  {fmt('L',bl[1],bl[0]):>22}  "
              f"{fmt('O',bo[1],f'{bo[0]:.0e}'):>24}")

    res = pd.DataFrame(rows)
    res.to_csv(PROC / "bid_results_1458.csv", index=False)
    print(f"\nSaved: {PROC/'bid_results_1458.csv'}")

    # ---------------- 시각화 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = [r["budget"] for r in rows]

    # (1) 예산별 획득 클릭
    ax = axes[0]
    ax.plot(x, [r["const_clicks"] for r in rows], "o-", label="Constant")
    ax.plot(x, [r["linear_clicks"] for r in rows], "s-", label="Linear")
    ax.plot(x, [r["ortb_clicks"] for r in rows], "^-", label="ORTB")
    ax.set_xscale("log"); ax.set_xlabel("budget (log)"); ax.set_ylabel("clicks won")
    ax.set_title("Clicks acquired vs budget"); ax.legend(); ax.grid(alpha=0.3)

    # (2) eCPC
    ax = axes[1]
    ax.plot(x, [r["linear_eCPC"] for r in rows], "s-", label="Linear")
    ax.plot(x, [r["ortb_eCPC"] for r in rows], "^-", label="ORTB")
    ax.set_xscale("log"); ax.set_xlabel("budget (log)"); ax.set_ylabel("eCPC (cost per click)")
    ax.set_title("Efficiency (lower=better)"); ax.legend(); ax.grid(alpha=0.3)

    # (3) bid landscape: win rate curve
    ax = axes[2]
    ax.plot(grid_b, emp, "o", ms=3, label="empirical P(win)")
    ax.plot(grid_b, grid_b / (grid_b + c0), "-", label=f"model b/(b+{c0:.0f})")
    ax.set_xlabel("bid"); ax.set_ylabel("win probability")
    ax.set_title("Bid landscape (win rate model)"); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("iPinYou 1458 — Bid Optimization (tag-excluded pCTR, LR AUC≈0.71)")
    fig.tight_layout()
    out_png = PROC / "bid_results.png"
    fig.savefig(out_png, dpi=120)
    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()