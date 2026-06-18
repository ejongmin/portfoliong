# -*- coding: utf-8 -*-
"""Create a PNG visualization for the current live regime portfolio."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "alpha_lab"
PORTFOLIO = OUT / "live_regime_portfolio.csv"
PNG = OUT / "portfolio_allocation.png"

FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"

COLORS = {
    "us_equity_krx": "#2F6FED",
    "us_equity_etf": "#4FA3FF",
    "kr_equity_etf": "#7C3AED",
    "cash_krw": "#22A06B",
    "cash_usd": "#86C06A",
    "bond": "#F59E0B",
    "gold": "#D4A017",
}
CLASS_LABELS = {
    "us_equity_krx": "국내상장 미국주식",
    "us_equity_etf": "미국 ETF",
    "kr_equity_etf": "한국 주식 ETF",
    "cash_krw": "원화 현금성",
    "cash_usd": "달러 현금성",
    "bond": "채권",
    "gold": "금",
}


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size)


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def main() -> None:
    df = pd.read_csv(PORTFOLIO, encoding="utf-8-sig")
    by_class = df.groupby("asset_class")["weight"].sum().sort_values(ascending=False)
    df = df.sort_values("weight", ascending=True)

    W, H = 1600, 1450
    img = Image.new("RGB", (W, H), "#F7F8FA")
    d = ImageDraw.Draw(img)
    title_f = font(46)
    sub_f = font(24)
    label_f = font(24)
    small_f = font(20)

    d.text((70, 45), "Regime Overlay 포트폴리오 비율", font=title_f, fill="#111827")
    d.text((70, 105), "기본 실행 후보 | 최신 레짐: risk_on | 1천만원 기준 비중", font=sub_f, fill="#4B5563")

    # Donut chart
    cx, cy, r = 380, 410, 230
    box = (cx - r, cy - r, cx + r, cy + r)
    start = -90
    for ac, w in by_class.items():
        end = start + w * 360
        d.pieslice(box, start=start, end=end, fill=COLORS.get(ac, "#999999"))
        start = end
    d.ellipse((cx - 105, cy - 105, cx + 105, cy + 105), fill="#F7F8FA")
    d.text((cx - 75, cy - 30), "주식성", font=sub_f, fill="#111827")
    equity = by_class.reindex(["us_equity_krx", "us_equity_etf", "kr_equity_etf"], fill_value=0).sum()
    d.text((cx - 65, cy + 5), pct(equity), font=title_f, fill="#111827")

    legend_x, legend_y = 720, 210
    d.text((legend_x, 165), "자산군 구성", font=font(32), fill="#111827")
    y = legend_y
    for ac, w in by_class.items():
        color = COLORS.get(ac, "#999999")
        d.rounded_rectangle((legend_x, y + 4, legend_x + 28, y + 32), radius=6, fill=color)
        d.text((legend_x + 45, y), f"{CLASS_LABELS.get(ac, ac)}  {pct(w)}", font=label_f, fill="#111827")
        y += 48

    # Key risk stats
    d.rounded_rectangle((720, 590, 1480, 760), radius=18, fill="#FFFFFF", outline="#E5E7EB")
    stats = [
        ("주식성", pct(equity)),
        ("방어자산", pct(by_class.reindex(["cash_krw", "cash_usd", "bond"], fill_value=0).sum())),
        ("금", pct(by_class.get("gold", 0))),
        ("방어+금", pct(by_class.reindex(["cash_krw", "cash_usd", "bond", "gold"], fill_value=0).sum())),
    ]
    x = 755
    for name, value in stats:
        d.text((x, 620), name, font=small_f, fill="#6B7280")
        d.text((x, 655), value, font=font(34), fill="#111827")
        x += 180

    # Bar chart
    d.text((70, 790), "개별 자산 비중", font=font(32), fill="#111827")
    bar_x, bar_y = 430, 835
    max_w = df["weight"].max()
    for i, row in enumerate(df.itertuples(index=False)):
        y = bar_y + i * 30
        label = f"{row.ticker} {row.name}"
        color = COLORS.get(row.asset_class, "#999999")
        d.text((70, y - 4), label[:26], font=font(18), fill="#374151")
        width = int((row.weight / max_w) * 820)
        d.rounded_rectangle((bar_x, y, bar_x + width, y + 17), radius=5, fill=color)
        d.text((bar_x + width + 14, y - 4), pct(row.weight), font=small_f, fill="#111827")

    d.text((70, H - 55), f"source: {PORTFOLIO}", font=font(18), fill="#6B7280")
    img.save(PNG)
    print(PNG)


if __name__ == "__main__":
    main()
