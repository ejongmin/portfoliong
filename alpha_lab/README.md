# Alpha Lab

기존 클로드 산출물(`src/`, `reports/`, `data/`)은 수정하지 않고, 별도 실험실로 운용하는 퀀트/ML 포트폴리오 연구 공간이다. 목표는 "좋아 보이는 백테스트"가 아니라, 실거래에 넣기 전에 실패할 전략을 빠르게 걸러내는 것이다.

## 현재 진단

기존 포트폴리오는 데이터 수집, 팩터 스크리닝, 모델별 백테스트, Black-Litterman, 최종 비중 산출까지 갖춘 자산배분 엔진이다. 다만 알파 예측 엔진은 아직 약하다.

- 강점: 원화 기준 수익률, 거래비용, 월간 리밸런싱, MVO/RP/HRP 비교, BL 뷰 결합.
- 약점: 2019~2026 특수 구간 의존, 모멘텀 팩터 중심, ML/DL 없음, 리밸런싱 후 제약 재검사 부족.
- 즉시 보완점: 최종 비중에서 작은 포지션 제거 후 `방어+금 <= 50%` 같은 정책 제약을 다시 확인해야 한다.

## 알파 연구 원칙

1. 예측과 배분을 분리한다.
   - 모델은 다음 1개월 상대수익률을 예측한다.
   - 포트폴리오는 예측 점수, 변동성, 상한 제약으로 따로 만든다.

2. 시계열 검증만 사용한다.
   - 랜덤 k-fold 금지.
   - 매월 과거 데이터로만 학습하고 다음 달을 예측하는 walk-forward를 기본으로 한다.

3. 절대수익률보다 순위 예측을 중시한다.
   - 금융 데이터에서 수익률의 크기는 노이즈가 크다.
   - 상위/하위 랭킹, 제외해야 할 자산 식별, 리스크 온/오프 판단이 더 견고하다.

4. 거래 가능성을 항상 반영한다.
   - 월간 회전율, 거래비용, 포지션 상한, 자산군 상한을 성과보다 먼저 본다.

## 현재 구현

`ml_alpha_backtest.py`는 외부 ML 패키지 없이 돌아가는 1차 베이스라인이다.

- 데이터: `data/prices_krw.csv`, `data/meta.csv`
- 피처:
  - 1/3/6개월 모멘텀
  - 12-1개월 모멘텀
  - 1/3개월 변동성
  - 12개월 MDD
  - 1개월 반전
  - 자산군 더미
- 라벨: 다음 1개월 수익률에서 해당 월 전체 유니버스 평균을 뺀 상대수익률
- 모델: 표준화 + Ridge 회귀 앙상블
- 배분: 예측 점수 / 변동성 기반, 개별/자산군 상한 적용
- 검증: 월간 walk-forward, 거래비용 반영

`reject_model_backtest.py`는 ML을 직접 비중 산정에 쓰지 않고, 하위 예측 자산을 제외하거나 변동성 역가중 코어와 비교한다.

- `Core_InvVol_All`: 전체 유니버스 변동성 역가중
- `ETF_Only_Core`: ETF/금/채권/현금만 쓰는 소액 계좌용 변동성 역가중
- `Reject_EW`: 하위 예측 자산 제외 후 동일가중
- `Reject_InvVol`: 하위 예측 자산 제외 후 변동성 역가중
- `IC_Gated_RejectInvVol`: 최근 IC가 양수일 때만 Reject 필터 사용

`build_live_portfolio.py`는 `ETF_Only_Core` 최신 비중을 1천만원 기준 실행 금액표로 변환한다.

`validate_alpha.py`는 최신 후보의 제약과 안정성을 확인한다.

- 비중 합계
- 개별 자산 상한
- 방어자산 35% 이하
- 방어+금 50% 이하
- MDD -20% 한도
- 시작일 민감도

`monthly_dca_plan.py`는 월 20만원 적립 매수안을 만든다. `alpha_lab/current_holdings.csv`가 있으면 현재 보유수량을 반영하고, 없으면 신규 계좌 기준으로 목표 비중이 큰 자산부터 정수 주수로 매수한다.

`regime_overlay_backtest.py`는 ETF-only 코어에 매크로 레짐 필터를 얹는다.

- VIX 스트레스
- S&P500 6개월 하락 추세
- KOSPI 6개월 하락 추세
- 원/달러 3개월 급등
- 미국 단기금리 상승 압력

리스크 오프에서는 주식성 비중을 낮추고 금/방어 비중을 높인다. 리스크 온에서는 주식성 비중을 소폭 높인다.

`dl_alpha_backtest.py`는 외부 프레임워크 없이 `numpy`로 구현한 작은 MLP 실험이다. 현재 데이터 크기에서 DL 신호가 기존 ML/코어 포트폴리오보다 나은지 검증하기 위한 실패 가능성 높은 실험으로 둔다.

`compare_strategies.py`는 모든 전략 요약을 `strategy_leaderboard.csv`로 합친다.

## 다음 확장

1. Tree 모델
   - `lightgbm` 또는 `catboost` 설치 후 cross-sectional ranking model 추가.
   - 목표는 회귀보다 pairwise/ranking loss가 더 적합하다.

2. DL 모델
   - 데이터가 충분해지기 전까지는 코어 모델로 쓰지 않는다.
   - 추가한다면 `torch` 기반 Temporal CNN 또는 작은 Transformer를 보조 점수로 사용한다.
   - 입력은 최근 60~252거래일 수익률 시퀀스 + 매크로 피처.

3. 매크로 레짐
   - VIX, USD/KRW, 금리, SOX, KOSPI/S&P500 추세를 사용해 위험예산을 조절한다.
   - 리스크 오프에서는 주식 상한을 낮추고 단기채/금 비중을 올린다.

4. 실거래 루프
   - 월말 데이터 업데이트
   - 알파 점수 생성
   - 목표 비중 산출
   - 기존 보유 비중 대비 매수 후보만 제안
   - 다음 달 실제 성과 기록

## 실행

```bash
/Users/john9/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 alpha_lab/ml_alpha_backtest.py
/Users/john9/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 alpha_lab/reject_model_backtest.py
/Users/john9/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 alpha_lab/build_live_portfolio.py
/Users/john9/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 alpha_lab/validate_alpha.py
/Users/john9/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 alpha_lab/monthly_dca_plan.py
/Users/john9/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 alpha_lab/regime_overlay_backtest.py
/Users/john9/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 alpha_lab/dl_alpha_backtest.py
/Users/john9/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 alpha_lab/compare_strategies.py
LIVE_MODEL=Regime_Overlay LIVE_SOURCE=regime_overlay_latest_weights.csv LIVE_OUTPUT=live_regime_portfolio.csv /Users/john9/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 alpha_lab/build_live_portfolio.py
VALIDATE_MODEL=Regime_Overlay VALIDATE_PORTFOLIO=live_regime_portfolio.csv VALIDATE_RETURNS=regime_overlay_returns.csv VALIDATE_LATEST=regime_overlay_latest_weights.csv /Users/john9/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 alpha_lab/validate_alpha.py
DCA_PORTFOLIO=live_regime_portfolio.csv DCA_OUTPUT=monthly_regime_dca_plan.csv /Users/john9/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 alpha_lab/monthly_dca_plan.py
```

시스템 파이썬에 `numpy/pandas`가 설치되어 있다면 `python3 alpha_lab/ml_alpha_backtest.py`도 가능하다.

결과 파일:

- `alpha_lab/ml_alpha_summary.csv`
- `alpha_lab/ml_alpha_returns.csv`
- `alpha_lab/ml_alpha_latest_weights.csv`
- `alpha_lab/reject_model_summary.csv`
- `alpha_lab/reject_model_returns.csv`
- `alpha_lab/reject_model_latest_weights.csv`
- `alpha_lab/live_alpha_portfolio.csv`
- `alpha_lab/alpha_validation_checks.csv`
- `alpha_lab/alpha_start_sensitivity.csv`
- `alpha_lab/monthly_dca_plan.csv`
- `alpha_lab/regime_overlay_summary.csv`
- `alpha_lab/regime_overlay_history.csv`
- `alpha_lab/regime_overlay_latest_weights.csv`
- `alpha_lab/live_regime_portfolio.csv`
- `alpha_lab/regime_validation_checks.csv`
- `alpha_lab/regime_start_sensitivity.csv`
- `alpha_lab/monthly_regime_dca_plan.csv`
- `alpha_lab/dl_alpha_summary.csv`
- `alpha_lab/dl_alpha_returns.csv`
- `alpha_lab/dl_alpha_latest_weights.csv`
- `alpha_lab/strategy_leaderboard.csv`

## 1차 결과 해석

1차 실행 기준 `ML_RidgeAlpha`는 2019-02~2026-05 워크포워드에서 CAGR +18.7%, MDD -16.1%, 평균 Rank IC +0.029를 기록했다. 수익은 났지만 동일가중 벤치마크보다 낮으므로, 아직 독립 운용할 알파는 아니다.

현재 결론은 다음과 같다.

- ML 모델은 단독 포트폴리오보다 기존 HRP/EW/BL 포트폴리오의 위험 필터나 비중 틸트로 쓰는 편이 낫다.
- 현재 실전 후보는 ML 단독이 아니라 `Core_InvVol_All` 또는 소액 계좌용 `ETF_Only_Core`다.
- 다음 단계는 Ridge 회귀가 아니라 ranking/tree 모델과 매크로 레짐 필터다.
- 실전 투입 전에는 최소 3개 이상 다른 검증 구간과 파라미터 안정성 검사가 필요하다.

## 2차 결과 해석

`reject_model_backtest.py` 실행 기준:

- `Core_InvVol_All`: CAGR +20.1%, Sharpe 1.68, MDD -10.0%
- `ETF_Only_Core`: CAGR +14.0%, Sharpe 1.54, MDD -8.5%
- `Reject_InvVol`: CAGR +18.6%, Sharpe 1.37, MDD -11.4%
- `EqualWeight_All`: CAGR +29.4%, Sharpe 1.59, MDD -14.6%

현재 데이터에서는 ML Reject 필터가 코어 변동성 역가중을 이기지 못했다. 따라서 실전 후보는 `ETF_Only_Core`이고, ML은 월별 경고/관찰 신호로 유지한다.

`live_alpha_portfolio.csv`의 최신 ETF-only 구성은 미국 주식 ETF, 국내상장 미국 ETF, 단기채, 국고채, 금, 달러 현금성으로 구성된다. 1천만원 기준 주식성 약 57.5%, 방어 약 35.0%, 금 약 7.5%다.

검증 결과:

- 비중 합계 100% 통과
- 방어자산 35.0% 통과
- 방어+금 42.5% 통과
- 개별 자산 상한 통과
- 전체 기간 MDD -8.5%로 -20% 한도 통과
- 2020/2021/2022/2023 시작일 민감도에서도 MDD 한도 통과

월 20만원 신규 적립 예시:

- TIGER 미국S&P500(360750.KS) 7주
- 예상 집행액 195,055원
- 잔여 현금 4,945원

## 3차 결과 해석

매크로 레짐 오버레이는 ETF-only 코어를 소폭 개선했다.

- `ETF_Only_Core`: CAGR +14.0%, Sharpe 1.54, MDD -8.5%
- `Regime_Overlay`: CAGR +14.2%, Sharpe 1.56, MDD -8.5%

개선 폭은 크지 않지만 MDD를 키우지 않고 성과와 Sharpe를 약간 높였다. 최신 레짐은 2026-04-30 기준 `risk_on`이다. 따라서 최신 `live_regime_portfolio.csv`는 ETF-only 코어보다 주식성 자산을 더 높게 잡는다.

최신 Regime Overlay 구성:

- 주식성: 약 63.0%
- 방어자산: 약 30.8%
- 금: 약 6.3%
- 방어+금: 약 37.1%

검증 결과 방어자산 35% 이하, 방어+금 50% 이하, 개별 상한, MDD -20% 한도를 모두 통과했다.

## 4차 결과 해석

작은 `numpy` MLP로 DL 알파를 테스트했다.

- `DL_TinyMLP`: CAGR +13.2%, Sharpe 0.72, MDD -29.3%, AvgIC +0.002
- `ML_RidgeAlpha`: CAGR +18.7%, Sharpe 1.11, MDD -16.1%, AvgIC +0.029
- `Regime_Overlay`: CAGR +14.2%, Sharpe 1.56, MDD -8.5%

현재 데이터에서는 DL이 과최적화되어 실전 후보에서 탈락한다. 결론은 명확하다.

- 기본 실행: `Regime_Overlay`
- 보수 실행: `ETF_Only_Core`
- 공격형 연구 후보: `Core_InvVol_All`
- ML: 보조 신호와 모니터링
- DL: 데이터/피처/프레임워크 확장 전까지 실전 사용 금지
