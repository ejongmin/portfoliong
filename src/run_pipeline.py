# -*- coding: utf-8 -*-
"""src/ 파이프라인 전체 실행 + SHARED.md 자동 업데이트.

실행: python src/run_pipeline.py  (투자/ 루트에서)
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "SHARED.md"


def run(args: list[str], env: dict[str, str] | None = None) -> None:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    print("\n$ " + " ".join(str(a) for a in args))
    subprocess.run(args, cwd=ROOT, env=merged, check=True)


def read_summary() -> dict:
    """실행 후 핵심 지표를 data/ 파일에서 읽어 반환."""
    import pandas as pd

    result: dict = {}
    try:
        bt = pd.read_csv(ROOT / "data/bt_summary.csv", index_col=0)
        hrp = bt.loc["HRP"]
        ew  = bt.loc["EqualWeight"]
        result["hrp_cagr"]   = f"{hrp['CAGR']:.1%}"
        result["hrp_sharpe"] = f"{hrp['Sharpe']:.2f}"
        result["hrp_mdd"]    = f"{hrp['MDD']:.1%}"
        result["ew_cagr"]    = f"{ew['CAGR']:.1%}"
    except Exception:
        pass

    try:
        fp = pd.read_csv(ROOT / "data/final_portfolio.csv", index_col=0)
        # 비중 컬럼 이름이 '비중'인 경우
        w_col = "비중" if "비중" in fp.columns else "weight"
        top = fp[w_col].sort_values(ascending=False)
        result["portfolio_top"] = ", ".join(
            f"{tk} {v:.1%}" for tk, v in top.head(4).items()
        )
    except Exception:
        pass

    try:
        import pandas as pd as pd2
    except Exception:
        pass

    try:
        raw = pd.read_csv(ROOT / "data/prices_raw.csv", index_col=0, parse_dates=True)
        krw_rate = raw["KRW=X"].ffill().iloc[-1]
        result["usdkrw"] = f"{krw_rate:,.0f}"
        result["data_end"] = str(raw.index.max().date())
    except Exception:
        pass

    return result


def append_shared(stats: dict) -> None:
    """SHARED.md 2026-MM-DD (Claude) 섹션을 월간 파이프라인 결과로 추가."""
    today = date.today().isoformat()
    ew_cagr    = stats.get("ew_cagr",    "—")
    hrp_sharpe = stats.get("hrp_sharpe", "—")
    hrp_mdd    = stats.get("hrp_mdd",    "—")
    hrp_cagr   = stats.get("hrp_cagr",   "—")
    top        = stats.get("portfolio_top", "—")
    usdkrw     = stats.get("usdkrw",     "—")
    data_end   = stats.get("data_end",   "—")

    block = textwrap.dedent(f"""
## {today} (Claude) — 월간 파이프라인 실행

**데이터 기준일**: {data_end} | **환율**: {usdkrw}원

### 백테스트 스냅샷

| 모델 | CAGR | Sharpe | MDD |
|---|---:|---:|---:|
| HRP | {hrp_cagr} | {hrp_sharpe} | {hrp_mdd} |
| EqualWeight | {ew_cagr} | — | — |

### 최종 BL 포트폴리오 상위 4개
{top}

### 산출물
- `data/prices_krw.csv` 갱신
- `data/candidates.csv`, `data/bt_summary.csv`, `data/bt_returns.csv` 갱신
- `data/bl_weights.csv`, `data/final_portfolio.csv` 갱신

### Claude → Codex
- `data/prices_krw.csv` 업데이트됨 → `alpha_lab/run_pipeline.py` 재실행 권장

---
""")

    if SHARED.exists():
        content = SHARED.read_text(encoding="utf-8")
        # 역할 분담 테이블 바로 뒤에 삽입 (---\n\n## 역할 분담 ... 이후 첫 번째 --- 앞에)
        marker = "\n---\n\n## 2026"
        idx = content.find(marker)
        if idx == -1:
            # fallback: 파일 끝에 추가
            SHARED.write_text(content.rstrip() + "\n" + block, encoding="utf-8")
        else:
            SHARED.write_text(content[:idx + 5] + block + content[idx + 5:], encoding="utf-8")
    else:
        SHARED.write_text(block, encoding="utf-8")

    print(f"\nSHARED.md 업데이트 완료 ({today})")


def main() -> None:
    py = sys.executable

    print("=" * 60)
    print("src/ 파이프라인 시작")
    print("=" * 60)

    run([py, "src/download_data.py"])
    run([py, "src/screen_factors.py"])
    run([py, "src/backtest.py"])
    run([py, "src/black_litterman.py"])
    run([py, "src/final_weights.py"])
    run([py, "src/final_portfolio.py"])

    print("\n" + "=" * 60)
    print("파이프라인 완료 — SHARED.md 업데이트 중")
    print("=" * 60)
    stats = read_summary()
    append_shared(stats)

    # 리스크 트리거도 자동 체크
    monitor = ROOT / "src/monitor_triggers.py"
    if monitor.exists():
        run([py, "src/monitor_triggers.py"])


if __name__ == "__main__":
    main()
