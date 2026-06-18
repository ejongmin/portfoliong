# -*- coding: utf-8 -*-
"""Run the full alpha_lab research and live-portfolio pipeline."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str], env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    print("\n$ " + " ".join(args))
    subprocess.run(args, cwd=ROOT, env=merged_env, check=True)


def main() -> None:
    py = sys.executable
    run([py, "alpha_lab/ml_alpha_backtest.py"])
    run([py, "alpha_lab/reject_model_backtest.py"])
    run([py, "alpha_lab/regime_overlay_backtest.py"])
    run([py, "alpha_lab/dl_alpha_backtest.py"])
    run([py, "alpha_lab/compare_strategies.py"])

    run([py, "alpha_lab/build_live_portfolio.py"])
    run([py, "alpha_lab/validate_alpha.py"])
    run([py, "alpha_lab/monthly_dca_plan.py"])

    run(
        [py, "alpha_lab/build_live_portfolio.py"],
        {
            "LIVE_MODEL": "Regime_Overlay",
            "LIVE_SOURCE": "regime_overlay_latest_weights.csv",
            "LIVE_OUTPUT": "live_regime_portfolio.csv",
        },
    )
    run(
        [py, "alpha_lab/validate_alpha.py"],
        {
            "VALIDATE_MODEL": "Regime_Overlay",
            "VALIDATE_PORTFOLIO": "live_regime_portfolio.csv",
            "VALIDATE_RETURNS": "regime_overlay_returns.csv",
            "VALIDATE_LATEST": "regime_overlay_latest_weights.csv",
        },
    )
    run(
        [py, "alpha_lab/monthly_dca_plan.py"],
        {
            "DCA_PORTFOLIO": "live_regime_portfolio.csv",
            "DCA_OUTPUT": "monthly_regime_dca_plan.csv",
        },
    )


if __name__ == "__main__":
    main()
