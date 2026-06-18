# -*- coding: utf-8 -*-
"""Collect alpha_lab strategy summaries into one leaderboard."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "alpha_lab"


SOURCES = [
    ("ml_alpha_summary.csv", ["ML_RidgeAlpha"]),
    ("dl_alpha_summary.csv", ["DL_TinyMLP"]),
    ("reject_model_summary.csv", ["Core_InvVol_All", "ETF_Only_Core", "Reject_InvVol", "EqualWeight_All", "BM_KODEX200"]),
    ("regime_overlay_summary.csv", ["Regime_Overlay"]),
]


def main() -> None:
    rows = []
    seen = set()
    for file_name, names in SOURCES:
        df = pd.read_csv(OUT / file_name, index_col=0, encoding="utf-8-sig")
        for name in names:
            if name in df.index and name not in seen:
                row = df.loc[name].to_dict()
                row["strategy"] = name
                row["source"] = file_name
                rows.append(row)
                seen.add(name)

    board = pd.DataFrame(rows).set_index("strategy")
    order = ["CAGR", "Vol", "Sharpe", "MDD", "Calmar", "AvgIC", "HitIC", "RejectHit", "source"]
    board = board[[c for c in order if c in board.columns]]
    board = board.sort_values(["Sharpe", "Calmar"], ascending=False)
    board.to_csv(OUT / "strategy_leaderboard.csv", encoding="utf-8-sig")

    print("=== Strategy leaderboard ===")
    print(board.to_string(float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
