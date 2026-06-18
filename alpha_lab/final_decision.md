# Alpha Lab Final Decision

## 결론

현재 실행 기본값은 `Regime_Overlay`다. 더 보수적으로 운용하고 싶으면 `ETF_Only_Core`를 사용한다.

## 왜 Regime_Overlay인가

`Regime_Overlay`는 ETF-only 변동성 역가중 코어에 VIX, S&P500 추세, KOSPI 추세, 원/달러, 미국 단기금리 기반 레짐 필터를 얹은 전략이다.

검증 결과:

| 전략 | CAGR | Sharpe | MDD | 판단 |
|---|---:|---:|---:|---|
| Regime_Overlay | +14.2% | 1.56 | -8.5% | 기본 실행 |
| ETF_Only_Core | +14.0% | 1.54 | -8.5% | 보수 실행 |
| Core_InvVol_All | +20.1% | 1.68 | -10.0% | 연구 후보, 자산 수 많음 |
| ML_RidgeAlpha | +18.7% | 1.11 | -16.1% | 보조 신호 |
| DL_TinyMLP | +13.2% | 0.72 | -29.3% | 탈락 |

## 최신 Regime_Overlay 상태

- 최신 레짐: `risk_on`
- 주식성: 약 63.0%
- 방어자산: 약 30.8%
- 금: 약 6.3%
- 방어+금: 약 37.1%
- 전체 기간 MDD: -8.5%
- 시작일 민감도 MDD: -8.5% 이내

## 월 20만원 적립 실행안

신규 계좌 기준 현재 매수안:

| 티커 | 종목 | 금액 | 수량 |
|---|---|---:|---:|
| 360750.KS | TIGER 미국S&P500 | 195,055원 | 7주 |

잔여 현금은 4,945원이다.

## 운용 규칙

1. 매월 말 데이터 업데이트 후 `run_pipeline.py`를 실행한다.
2. 기본 실행 파일은 `live_regime_portfolio.csv`와 `monthly_regime_dca_plan.csv`다.
3. `regime_validation_checks.csv`가 모두 `True`인지 확인한다.
4. 보유수량이 생기면 `alpha_lab/current_holdings.csv`에 `ticker,shares`를 넣고 월 적립 플랜을 다시 생성한다.
5. `DL_TinyMLP`는 현재 탈락이므로 매수 판단에 쓰지 않는다.

## 정직한 한계

- 2019~2026 표본은 AI 랠리, 한국 멜트업, 환율 효과가 포함된 특수 구간이다.
- 백테스트 수익률은 미래 수익 보장이 아니다.
- 현재 ML/DL 신호는 독립 알파로 충분하지 않다.
- 실전에서는 매월 실제 체결가, 세금, 슬리피지, 추적오차를 기록해야 한다.

