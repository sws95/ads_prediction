"""
04_km_ortb.py
KM(Kaplan-Meier) 기반 시장가 분포 추정 + 정식 ORTB.

문제의식:
- 03의 win rate model w(b)=b/(b+c0)는 1개 파라미터라 실제 시장가 분포를 못 따라감.
- 현실 RTB는 '진 경매의 시장가'를 모름(낙찰자만 가격 관측) = censored data.
- KM으로 우중도절단을 반영해 win rate 곡선 W(b)=P(market<=b)를 비모수 추정.

정식 ORTB:
- 입찰식 bid(pCTR)=sqrt(c/lambda * pCTR + c^2) - c
- lambda는 예산 제약을 만족하도록 결정. 시장가를 모른다고 가정하고
  KM win rate W(b)로 기대 비용/낙찰을 계산해 예산에 맞는 lambda를 이분탐색.
- 비교: 03의 '시장가 안다고 가정' ORTB vs 04의 'KM 기반' 정식 ORTB.

평가는 03과 동일한 오라클 시뮬레이터(실제 PayingPrice로 win 판정)로 수행하되,
입찰 의사결정(lambda 선택)만 KM 추정에 기반 -> 현실과 동일한 정보 구조.
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


# ---------- KM 추정 ----------
def km_winrate(market_train, ref_bids, win_prob_at_train=0.5):
    """
    진짜 RTB라면 우리가 이긴 경매(시장가 관측됨)와 진 경매(시장가 절단)가 섞여 있다.
    여기선 train 시장가 전체가 관측되지만, 학습 시점 입찰 정책이 일부만 이겼다고 가정해
    win_prob_at_train 비율로 무작위 censoring을 부여해 KM을 시연한다.
    (censoring이 0이면 KM은 경험적 CDF와 동일)

    반환: ref_bids 각 값에서의 win rate W(b)=P(market<=b)
    """
    rng = np.random.default_rng(0)
    m = np.asarray(market_train, dtype=float)
    n = len(m)
    # event=1 (시장가 관측, 이긴 경매), event=0 (절단, 진 경매)
    event = (rng.random(n) < win_prob_at_train).astype(int)
    # 절단된 관측은 '그 가격 이상에서 절단'으로 취급
    order = np.argsort(m)
    m_sorted = m[order]
    e_sorted = event[order]

    # KM 생존함수 S(t)=P(market> t). win rate W(b)=1-S(b).
    at_risk = n
    S = 1.0
    times, surv = [], []
    i = 0
    while i < n:
        t = m_sorted[i]
        # 같은 시점 묶기
        j = i
        d = 0  # 이 시점의 event 수
        c = 0  # 이 시점의 censor 수
        while j < n and m_sorted[j] == t:
            if e_sorted[j] == 1:
                d += 1
            else:
                c += 1
            j += 1
        if at_risk > 0 and d > 0:
            S *= (1 - d / at_risk)
        times.append(t)
        surv.append(S)
        at_risk -= (d + c)
        i = j

    times = np.array(times)
    surv = np.array(surv)

    # ref_bids에서 W(b)=1-S(b) 계산 (step function)
    W = []
    for b in ref_bids:
        idx = np.searchsorted(times, b, side="right") - 1
        s = surv[idx] if idx >= 0 else 1.0
        W.append(1.0 - s)
    return np.array(W), (times, 1 - surv)


# ---------- ORTB lambda를 예산에 맞춰 결정 ----------
def ortb_bids(p, c, lam):
    return np.sqrt(c / lam * p + c * c) - c

def build_cond_mean_cost(km_times, km_W):
    """
    KM win rate 곡선 W(b)=P(market<=b)에서, 각 b에 대해
    E[market | market<=b] (이겼을 때 내는 2nd-price 평균 시장가)를 미리 계산.

    W는 b에 대한 누적분포(CDF)이므로, 인접 구간의 증가분 dW가 그 가격대의 확률질량.
    각 격자점 b_k의 대표가격을 b_k로 보면:
      E[market | market<=b] = Σ_{k: b_k<=b} b_k * dW_k / W(b)
    """
    t = np.asarray(km_times, dtype=float)
    W = np.asarray(km_W, dtype=float)
    dW = np.diff(W, prepend=0.0)          # 각 격자점의 확률질량
    dW = np.clip(dW, 0, None)
    cum_mass = np.cumsum(dW)              # = W(b_k) 근사
    cum_cost = np.cumsum(t * dW)          # Σ b_k * dW_k
    return t, cum_mass, cum_cost

def expected_cost_and_win(p, c, lam, km_times, km_W, cond=None):
    """KM win rate로 기대 낙찰 수와 기대 비용 추정 (시장가 모른다고 가정).
       비용은 2nd-price를 반영해 E[market|market<=bid] 사용."""
    bids = ortb_bids(p, c, lam)
    idx = np.searchsorted(km_times, bids, side="right") - 1
    idx = np.clip(idx, 0, len(km_W) - 1)
    wprob = km_W[idx]
    exp_win = wprob.sum()

    if cond is None:
        # 폴백: 옛 근사 (bid 사용 -> 과대추정)
        exp_cost = (bids * wprob).sum()
    else:
        t, cum_mass, cum_cost = cond
        # E[market|market<=bid] = cum_cost(bid)/cum_mass(bid)
        cidx = np.searchsorted(t, bids, side="right") - 1
        cidx = np.clip(cidx, 0, len(t) - 1)
        mass = cum_mass[cidx]
        cost = cum_cost[cidx]
        cond_mean = np.where(mass > 1e-12, cost / np.maximum(mass, 1e-12), bids)
        # 기대 비용 = Σ win_prob * E[market|win]
        exp_cost = (wprob * cond_mean).sum()
    return exp_cost, exp_win

def find_lambda_for_budget(p, c, budget, km_times, km_W, cond=None, lam_lo=1e-7, lam_hi=1e-3):
    """기대 비용이 예산과 같아지는 lambda를 이분탐색.
       lambda 클수록 입찰 작아짐 -> 비용 감소. 단조성 이용."""
    for _ in range(40):
        lam = np.sqrt(lam_lo * lam_hi)
        exp_cost, _ = expected_cost_and_win(p, c, lam, km_times, km_W, cond=cond)
        if exp_cost > budget:
            lam_lo = lam   # 비용 과다 -> lambda 키워 입찰 낮춤
        else:
            lam_hi = lam
    return np.sqrt(lam_lo * lam_hi)


# ---------- 오라클 시뮬레이터 (실제 시장가로 평가) ----------
def simulate(bids, market, click, budget):
    spend = 0.0; won = clicks = 0
    for i in range(len(bids)):
        b, mp = bids[i], market[i]
        if b >= mp and spend + mp <= budget:
            spend += mp; won += 1; clicks += click[i]
    return {"clicks": int(clicks), "won": won, "spend": round(spend,1),
            "win_rate": won/len(bids), "eCPC": (spend/clicks) if clicks else None}


def main():
    df = pd.read_parquet(SIM_PATH)
    market = df["PayingPrice"].values.astype(float)
    click = df["click"].values.astype(int)
    p = df["pCTR"].values.astype(float)
    full_cost = market.sum()
    c_ortb = 50

    # train 시장가가 따로 없으므로 test 시장가 분포를 landscape 추정에 사용(시연)
    ref = np.arange(1, 301)
    # censoring 50% 부여한 KM vs censoring 없는 경험적 CDF 비교
    W_km, (kt, kW) = km_winrate(market, ref, win_prob_at_train=0.5)
    W_emp = np.array([(market <= b).mean() for b in ref])

    print(f"full_cost={full_cost:,.0f}, avg market={market.mean():.1f}\n")

    # 2nd-price 반영: E[market|market<=b] 미리 계산
    km_times = ref.astype(float)
    cond = build_cond_mean_cost(km_times, W_km)

    budget_fracs = [1/64, 1/32, 1/16, 1/8, 1/4, 1/2]
    rows = []
    print("=== 정식 ORTB (KM 기반 lambda) vs 오라클 ORTB(그리드) ===")
    print(f"{'budget':>14} {'KM-ORTB clk':>12} {'lambda':>10} {'grid-ORTB clk':>14}")
    lam_grid = [1e-4,5e-5,2e-5,1e-5,5e-6,2e-6,1e-6,5e-7]
    for frac in budget_fracs:
        bud = full_cost*frac
        # KM 기반 lambda 선택 (2nd-price 비용 반영)
        lam_km = find_lambda_for_budget(p, c_ortb, bud, km_times, W_km, cond=cond)
        r_km = simulate(ortb_bids(p, c_ortb, lam_km), market, click, bud)
        # 오라클 그리드 (03 방식, 비교용)
        best = None
        for lam in lam_grid:
            r = simulate(ortb_bids(p, c_ortb, lam), market, click, bud)
            if best is None or r["clicks"]>best[1]["clicks"]:
                best=(lam,r)
        rows.append({"budget":round(bud),"frac":f"1/{int(1/frac)}",
                     "km_clicks":r_km["clicks"],"km_lambda":lam_km,
                     "grid_clicks":best[1]["clicks"],"grid_lambda":best[0]})
        print(f"1/{int(1/frac):<3} ({bud:>11,.0f}) {r_km['clicks']:>12} "
              f"{lam_km:>10.1e} {best[1]['clicks']:>14}")

    pd.DataFrame(rows).to_csv(PROC/"km_ortb_results.csv", index=False)

    # 시각화: KM vs 경험적 vs 03의 b/(b+c0)
    c0 = 55  # 03에서 적합된 값 근사
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    ax=axes[0]
    ax.plot(ref, W_emp, "-", lw=2, label="empirical CDF P(market<=b)")
    ax.plot(ref, W_km, "--", lw=2, label="KM (50% censored)")
    ax.plot(ref, ref/(ref+c0), ":", lw=2, label=f"b/(b+{c0}) [03]")
    ax.set_xlabel("bid"); ax.set_ylabel("win rate W(b)")
    ax.set_title("Win rate model 비교"); ax.legend(); ax.grid(alpha=0.3)

    ax=axes[1]
    x=[r["budget"] for r in rows]
    ax.plot(x,[r["km_clicks"] for r in rows],"^-",label="KM-based ORTB (정식)")
    ax.plot(x,[r["grid_clicks"] for r in rows],"s--",label="oracle grid ORTB")
    ax.set_xscale("log"); ax.set_xlabel("budget (log)"); ax.set_ylabel("clicks won")
    ax.set_title("정식 ORTB vs 오라클"); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("iPinYou 1458 — KM bid landscape & 정식 ORTB")
    fig.tight_layout()
    out=PROC/"km_ortb.png"; fig.savefig(out,dpi=120)
    print(f"\nSaved: {PROC/'km_ortb_results.csv'}")
    print(f"Saved plot: {out}")


if __name__=="__main__":
    main()