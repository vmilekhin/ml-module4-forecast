# Модуль 4 — Прогнозирование временных рядов

Сравнение **Baseline vs MLP vs LSTM** на данных AAPL (Yahoo Finance).
Инженерный вычислительный эксперимент по слайдам 18-20.

## Что внутри

| Файл | Что делает |
|---|---|
| `forecast_aapl.py` | Прогноз **цен** AAPL |
| `forecast_returns.py` | Прогноз **log-доходностей** AAPL |
| `figures/` | Графики по ценам |
| `figures_returns/` | Графики по доходностям |
| `report/REPORT.md` | Отчёт в Markdown |
| `report/REPORT.pdf` | Отчёт в PDF |

## Ключевые моменты методики

- **Time split** (не random): train = прошлое, test = будущее
- **Scaling без data leakage**: `mean`/`std` только на train
- **Seed = 42** для воспроизводимости
- **Метрики**: MSE + MAE
- **Гипотеза**: LSTM снижает ошибку относительно MLP и baseline

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python forecast_aapl.py
python forecast_returns.py
```

## Результат

**MLP и LSTM не бьют baseline** по MSE/MAE. Negative result — ожидаемо
для дневных цен акций: они близки к случайному блужданию, а baseline
«завтра = сегодня» — сильный конкурент.

## Стек

Python 3.12, PyTorch 2.x (CPU), yfinance, pandas, numpy, matplotlib.
