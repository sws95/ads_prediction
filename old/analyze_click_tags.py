"""
analyze_click_tags.py
클릭한 노출(click==1)의 UserTags 조합 분석.

- 캠페인별로, 클릭 노출에서 어떤 개별 태그가 자주 등장하는지
- 그 태그가 클릭 노출에서 차지하는 비율(lift) = (클릭 중 태그 보유율) / (전체 태그 보유율)
  lift가 높을수록 그 태그가 "클릭하는 사람"을 잘 가르는 신호
- 태그 ID -> 사람이 읽을 수 있는 이름 매핑(user.profile.tags.en.txt)도 함께 출력
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

BASE = Path(r"C:\Users\swson23\Desktop\test\ads_prediction\ipinyou.contest.dataset")
PROC = BASE / "processed"
IN_PATH = PROC / "imp_season2_all.parquet"
TAGS_FILE = BASE / "user.profile.tags.en.txt"

ADV_NAME = {
    "1458": "Chinese vertical e-commerce",
    "3358": "Software",
    "3386": "International e-commerce",
    "3427": "Oil",
    "3476": "Tire",
}


def load_tag_names():
    """user.profile.tags.en.txt -> {id: name}. 탭 또는 공백 구분."""
    m = {}
    if not TAGS_FILE.exists():
        return m
    with open(TAGS_FILE, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            # 탭 우선, 없으면 첫 공백 기준
            if "\t" in line:
                tid, name = line.split("\t", 1)
            else:
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                tid, name = parts
            m[tid.strip()] = name.strip()
    return m


def analyze_campaign(df_adv, tag_names, topn=15):
    n_all = len(df_adv)
    clk = df_adv[df_adv["click"] == 1]
    n_clk = len(clk)
    if n_clk == 0:
        print("  (클릭 없음)")
        return

    # 개별 태그별: 전체 보유 수, 클릭 보유 수
    all_cnt = defaultdict(int)
    clk_cnt = defaultdict(int)
    for tags in df_adv["UserTags"].astype(str):
        if tags in ("null", "nan", ""):
            continue
        for t in set(tags.split(",")):
            all_cnt[t] += 1
    for tags in clk["UserTags"].astype(str):
        if tags in ("null", "nan", ""):
            continue
        for t in set(tags.split(",")):
            clk_cnt[t] += 1

    rows = []
    for t, kc in clk_cnt.items():
        ac = all_cnt[t]
        share_clk = kc / n_clk            # 클릭 노출 중 이 태그 보유 비율
        share_all = ac / n_all            # 전체 노출 중 이 태그 보유 비율
        lift = share_clk / share_all if share_all > 0 else 0
        ctr_tag = kc / ac if ac > 0 else 0  # 이 태그 보유자의 클릭률
        rows.append((t, tag_names.get(t, "?"), ac, kc, ctr_tag, lift))

    res = pd.DataFrame(rows, columns=["tag", "name", "imp", "clk", "ctr", "lift"])
    # 클릭 보유 수 50건 이상 또는 클릭률 높은 것 위주
    res = res[res["clk"] >= 3].sort_values("lift", ascending=False)
    print(f"  클릭 노출 {n_clk}건 / 전체 {n_all:,}건. lift 높은 태그 top {topn}:")
    print(res.head(topn).to_string(index=False,
          formatters={"ctr": lambda x: f"{x*100:.2f}%", "lift": lambda x: f"{x:.1f}x"}))


def main():
    df = pd.read_parquet(IN_PATH)
    tag_names = load_tag_names()
    print(f"loaded {len(df):,} rows, tag names: {len(tag_names)}\n")

    for adv in ["1458", "3358", "3386", "3427", "3476"]:
        sub = df[df["AdvertiserID"] == adv]
        print(f"=== {adv} ({ADV_NAME.get(adv,'?')}) ===")
        analyze_campaign(sub, tag_names)
        print()


if __name__ == "__main__":
    main()
