"""Модуль 4 — второй эксперимент: прогноз доходностей AAPL.

Прогнозируем log_return = ln(P_t / P_{t-1}) вместо цен.
"""

import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# 1. Воспроизводимость
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 2. Загрузка и расчёт доходностей
# ---------------------------------------------------------------------------

def load_returns(start: str = "2018-01-01",
                 end: str = "2026-01-01") -> np.ndarray:
    """Загружает AAPL, возвращает лог-доходности."""
    print(f"Загружаю AAPL с {start} по {end}...")
    df = yf.download("AAPL", start=start, end=end,
                     auto_adjust=True, progress=False)
    prices = df["Close"].dropna().to_numpy().reshape(-1)
    print(f"  Цен: {len(prices)}")

    # r_t = ln(P_t / P_{t-1})
    log_returns = np.log(prices[1:] / prices[:-1])
    print(f"  Доходностей: {len(log_returns)}")
    print(f"  Среднее: {log_returns.mean():.6f}")
    print(f"  Std:     {log_returns.std():.6f}")
    return log_returns


# ---------------------------------------------------------------------------
# 3. Окна
# ---------------------------------------------------------------------------

def make_windows(series: torch.Tensor, window: int):
    x, y = [], []
    for i in range(len(series) - window):
        x.append(series[i:i + window])
        y.append(series[i + window])
    return torch.stack(x), torch.stack(y)


# ---------------------------------------------------------------------------
# 4. Модели
# ---------------------------------------------------------------------------

class PriceMLP(nn.Module):
    def __init__(self, window: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class PriceLSTM(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden,
                            batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)


# ---------------------------------------------------------------------------
# 5. Обучение
# ---------------------------------------------------------------------------

def train_model(model, loader, epochs: int, lr: float,
                log_every: int = 10):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        avg = epoch_loss / n
        losses.append(avg)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"    epoch {epoch+1:3d}/{epochs} — loss: {avg:.8f}")

    return losses, time.time() - t0


# ---------------------------------------------------------------------------
# 6. Метрики
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test, is_lstm: bool = False):
    model.eval()
    with torch.no_grad():
        x = X_test.unsqueeze(-1) if is_lstm else X_test
        pred = model(x)
    mse = nn.functional.mse_loss(pred, y_test).item()
    mae = nn.functional.l1_loss(pred, y_test).item()
    return pred, mse, mae


def baseline_metrics(X_test, y_test):
    # «Завтра = сегодня» для доходностей: r_{t+1} = r_t
    baseline_pred = X_test[:, -1]
    mse = nn.functional.mse_loss(baseline_pred, y_test).item()
    mae = nn.functional.l1_loss(baseline_pred, y_test).item()
    return baseline_pred, mse, mae


# ---------------------------------------------------------------------------
# 7. Один эксперимент
# ---------------------------------------------------------------------------

def run_experiment(returns: np.ndarray,
                   window: int = 20,
                   epochs: int = 30,
                   lr: float = 0.001,
                   batch_size: int = 64,
                   mlp_hidden: int = 64,
                   lstm_hidden: int = 32,
                   figures_dir: Path = Path("figures_returns"),
                   label: str = ""):
    print(f"\n{'=' * 60}")
    print(f"ЭКСПЕРИМЕНТ {label}: window={window}, lr={lr}, epochs={epochs}")
    print(f"{'=' * 60}")

    # --- Нормализация (без data leakage) ---
    split_index = int(len(returns) * 0.8)
    train_raw = returns[:split_index]
    mean = train_raw.mean()
    std = train_raw.std()
    scaled = (returns - mean) / std
    series = torch.tensor(scaled, dtype=torch.float32)

    # --- Окна ---
    X, y = make_windows(series, window)
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # --- DataLoader ---
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True,
    )
    X_train_lstm = X_train.unsqueeze(-1)
    X_test_lstm = X_test.unsqueeze(-1)
    lstm_loader = DataLoader(
        TensorDataset(X_train_lstm, y_train),
        batch_size=batch_size, shuffle=True,
    )

    # --- Baseline ---
    baseline_pred, baseline_mse, baseline_mae = baseline_metrics(X_test, y_test)
    print(f"Baseline   MSE={baseline_mse:.8f}  MAE={baseline_mae:.8f}")

    # --- MLP ---
    print("\nОбучаю MLP...")
    mlp = PriceMLP(window=window, hidden=mlp_hidden)
    mlp_losses, mlp_time = train_model(mlp, train_loader, epochs, lr)
    mlp_pred, mlp_mse, mlp_mae = evaluate(mlp, X_test, y_test)
    print(f"MLP        MSE={mlp_mse:.8f}  MAE={mlp_mae:.8f}  ({mlp_time:.1f} сек)")

    # --- LSTM ---
    print("\nОбучаю LSTM...")
    lstm = PriceLSTM(hidden=lstm_hidden)
    lstm_losses, lstm_time = train_model(lstm, lstm_loader, epochs, lr)
    lstm_pred, lstm_mse, lstm_mae = evaluate(
        lstm, X_test, y_test, is_lstm=True)
    print(f"LSTM       MSE={lstm_mse:.8f}  MAE={lstm_mae:.8f}  ({lstm_time:.1f} сек)")

    # --- Графики ---
    figures_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{label}" if label else ""

    # Предсказания
    plt.figure(figsize=(14, 5))
    plt.plot(y_test.numpy()[:200], label="Real", color="black",
             linewidth=1.2)
    plt.plot(baseline_pred.numpy()[:200], label="Baseline",
             color="gray", alpha=0.7)
    plt.plot(mlp_pred.numpy()[:200], label="MLP",
             color="steelblue", alpha=0.8)
    plt.plot(lstm_pred.numpy()[:200], label="LSTM",
             color="darkred", alpha=0.8)
    plt.title(f"Прогноз доходностей — первые 200 точек теста ({label})")
    plt.xlabel("День")
    plt.ylabel("Нормализованная доходность")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"returns_predictions{suffix}.png", dpi=120)
    plt.close()

    # Кривые обучения
    plt.figure(figsize=(10, 5))
    plt.plot(mlp_losses, label="MLP", color="steelblue")
    plt.plot(lstm_losses, label="LSTM", color="darkred")
    plt.title(f"Кривые обучения ({label})")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"returns_losses{suffix}.png", dpi=120)
    plt.close()

    return {
        "label": label,
        "lr": lr,
        "baseline_mse": baseline_mse,
        "baseline_mae": baseline_mae,
        "mlp_mse": mlp_mse,
        "mlp_mae": mlp_mae,
        "mlp_time": mlp_time,
        "lstm_mse": lstm_mse,
        "lstm_mae": lstm_mae,
        "lstm_time": lstm_time,
    }


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed(42)
    Path("figures_returns").mkdir(exist_ok=True)

    # Загрузка доходностей
    returns = load_returns()

    # График доходностей
    plt.figure(figsize=(14, 4))
    plt.plot(returns, color="darkred", linewidth=0.6)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.title("Лог-доходности AAPL")
    plt.xlabel("День")
    plt.ylabel("log(P_t / P_{t-1})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures_returns/returns_series.png", dpi=120)
    plt.close()

    # Базовый эксперимент
    results = [run_experiment(returns, lr=0.001, label="base")]

    # DOE по learning rate
    print(f"\n\n{'#' * 60}")
    print("DOE: learning rate")
    print(f"{'#' * 60}")

    for lr in [0.0001, 0.01]:
        results.append(run_experiment(returns, lr=lr, label=f"lr_{lr}"))

    # Итоги
    print(f"\n\n{'=' * 80}")
    print("ИТОГОВАЯ ТАБЛИЦА — ПРОГНОЗ ДОХОДНОСТЕЙ")
    print(f"{'=' * 80}")
    print(f"{'Label':12s} {'lr':>8s} {'Base MSE':>12s} "
          f"{'MLP MSE':>12s} {'LSTM MSE':>12s}")
    for r in results:
        print(f"{r['label']:12s} {r['lr']:>8.4f} "
              f"{r['baseline_mse']:>12.8f} "
              f"{r['mlp_mse']:>12.8f} "
              f"{r['lstm_mse']:>12.8f}")

    print(f"\n{'=' * 80}")
    print("ВЫВОДЫ")
    print(f"{'=' * 80}")

    best_mlp = min(results, key=lambda r: r["mlp_mse"])
    best_lstm = min(results, key=lambda r: r["lstm_mse"])
    base = results[0]

    print(f"Baseline MSE: {base['baseline_mse']:.8f}")
    print(f"Best MLP:     {best_mlp['mlp_mse']:.8f} (lr={best_mlp['lr']})")
    print(f"Best LSTM:    {best_lstm['lstm_mse']:.8f} (lr={best_lstm['lr']})")

    for name, best in [("MLP", best_mlp), ("LSTM", best_lstm)]:
        imp = (base["baseline_mse"] - best[f"{name.lower()}_mse"]) \
            / base["baseline_mse"] * 100
        sign = "улучшил" if imp > 0 else "ухудшил"
        print(f"{name} {sign} baseline на {abs(imp):.1f}%")

    print(f"\n[✓] Графики в figures_returns/")


if __name__ == "__main__":
    main()
