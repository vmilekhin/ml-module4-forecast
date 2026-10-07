"""Модуль 4 — прогнозирование временных рядов.

Сравнение Baseline vs MLP vs LSTM на данных AAPL (Yahoo Finance).
"""

import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # без GUI, сохраняем в файлы
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# 1. Воспроизводимость (слайд 30)
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 2. Загрузка данных (слайд 18)
# ---------------------------------------------------------------------------

def load_aapl(start: str = "2018-01-01", end: str = "2026-01-01") -> np.ndarray:
    print(f"Загружаю AAPL с {start} по {end}...")
    df = yf.download("AAPL", start=start, end=end,
                     auto_adjust=True, progress=False)
    prices = df["Close"].dropna()
    print(f"  Строк: {len(prices)}")
    print(f"  Период: {prices.index[0].date()} — {prices.index[-1].date()}")
    return prices.to_numpy().reshape(-1)


# ---------------------------------------------------------------------------
# 3. Окна (слайд 7)
# ---------------------------------------------------------------------------

def make_windows(series: torch.Tensor, window: int):
    x, y = [], []
    for i in range(len(series) - window):
        x.append(series[i:i + window])
        y.append(series[i + window])
    return torch.stack(x), torch.stack(y)


# ---------------------------------------------------------------------------
# 4. MLP (слайд 8)
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


# ---------------------------------------------------------------------------
# 5. LSTM (слайд 13, исправлена опечатка lst → lstm)
# ---------------------------------------------------------------------------

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
# 6. Обучение (слайды 9, 15)
# ---------------------------------------------------------------------------

def train_model(model, loader, epochs: int, lr: float, log_every: int = 5):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg = epoch_loss / n_batches
        losses.append(avg)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"    epoch {epoch+1:2d}/{epochs} — loss: {avg:.6f}")

    elapsed = time.time() - t0
    return losses, elapsed


# ---------------------------------------------------------------------------
# 7. Метрики (слайды 10, 16, 28)
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
    # Baseline: «завтра = сегодня» (слайд 5)
    baseline_pred = X_test[:, -1]
    mse = nn.functional.mse_loss(baseline_pred, y_test).item()
    mae = nn.functional.l1_loss(baseline_pred, y_test).item()
    return baseline_pred, mse, mae


# ---------------------------------------------------------------------------
# 8. Recursive forecasting (слайды 12, 17)
# ---------------------------------------------------------------------------

def recursive_forecast(model, last_window: torch.Tensor,
                       steps: int = 50, is_lstm: bool = False):
    model.eval()
    window = last_window.clone()
    future = []
    with torch.no_grad():
        for _ in range(steps):
            if is_lstm:
                x = window.reshape(1, -1, 1)
            else:
                x = window.unsqueeze(0)
            nv = model(x)[0]
            future.append(nv.item())
            window = torch.cat([window[1:], nv.unsqueeze(0)])
    return future


# ---------------------------------------------------------------------------
# 9. Графики
# ---------------------------------------------------------------------------

def plot_prices(prices: np.ndarray, out: Path) -> None:
    plt.figure(figsize=(14, 5))
    plt.plot(prices, color="steelblue", linewidth=1)
    plt.title("AAPL — цена закрытия (нормализованная)")
    plt.xlabel("День")
    plt.ylabel("Нормализованная цена")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_predictions(y_true: np.ndarray, predictions: dict, out: Path,
                     n_show: int = 200) -> None:
    plt.figure(figsize=(14, 5))
    plt.plot(y_true[:n_show], label="Real", color="black", linewidth=1.2)
    colors = {"Baseline": "gray", "MLP": "steelblue", "LSTM": "darkred"}
    for name, pred in predictions.items():
        plt.plot(pred[:n_show], label=name,
                 color=colors.get(name, "green"),
                 linewidth=1.0, alpha=0.8)
    plt.title(f"Прогноз на тесте (первые {n_show} точек)")
    plt.xlabel("День")
    plt.ylabel("Нормализованная цена")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_recursive(real_tail: np.ndarray, forecasts: dict, out: Path) -> None:
    plt.figure(figsize=(14, 5))
    plt.plot(range(len(real_tail)), real_tail,
             label="Последние наблюдения", color="black")
    offset = len(real_tail)
    colors = {"MLP": "steelblue", "LSTM": "darkred"}
    for name, future in forecasts.items():
        plt.plot(range(offset, offset + len(future)), future,
                 label=f"{name} (прогноз)", color=colors.get(name, "green"))
    plt.axvline(offset, color="gray", linestyle="--", alpha=0.6)
    plt.title("Recursive forecasting: 50 шагов вперёд")
    plt.xlabel("День")
    plt.ylabel("Нормализованная цена")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_losses(mlp_losses: list, lstm_losses: list, out: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(mlp_losses, label="MLP", color="steelblue")
    plt.plot(lstm_losses, label="LSTM", color="darkred")
    plt.title("Кривые обучения")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 10. Один эксперимент
# ---------------------------------------------------------------------------

def run_experiment(
    prices: np.ndarray,
    window: int = 20,
    epochs: int = 30,
    lr: float = 0.001,
    batch_size: int = 64,
    mlp_hidden: int = 64,
    lstm_hidden: int = 32,
    figures_dir: Path = Path("figures"),
    label: str = "",
):
    print(f"\n{'=' * 60}")
    print(f"ЭКСПЕРИМЕНТ {label}: window={window}, lr={lr}, epochs={epochs}")
    print(f"{'=' * 60}")

    # --- Слайд 19: preprocessing без data leakage ---
    split_index = int(len(prices) * 0.8)
    train_raw = prices[:split_index]
    mean = train_raw.mean()
    std = train_raw.std()
    scaled = (prices - mean) / std
    series = torch.tensor(scaled, dtype=torch.float32)

    # --- Слайды 7, 20: окна и split ---
    X, y = make_windows(series, window)
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # --- DataLoader MLP ---
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True,
    )

    # --- LSTM: добавляем признаковую ось ---
    X_train_lstm = X_train.unsqueeze(-1)
    X_test_lstm = X_test.unsqueeze(-1)
    lstm_loader = DataLoader(
        TensorDataset(X_train_lstm, y_train),
        batch_size=batch_size, shuffle=True,
    )

    # --- Baseline ---
    baseline_pred, baseline_mse, baseline_mae = baseline_metrics(X_test, y_test)
    print(f"Baseline   MSE={baseline_mse:.6f}  MAE={baseline_mae:.6f}")

    # --- MLP ---
    print("\nОбучаю MLP...")
    mlp = PriceMLP(window=window, hidden=mlp_hidden)
    mlp_losses, mlp_time = train_model(mlp, train_loader, epochs, lr)
    mlp_pred, mlp_mse, mlp_mae = evaluate(mlp, X_test, y_test)
    print(f"MLP        MSE={mlp_mse:.6f}  MAE={mlp_mae:.6f}  ({mlp_time:.1f} сек)")

    # --- LSTM ---
    print("\nОбучаю LSTM...")
    lstm = PriceLSTM(hidden=lstm_hidden)
    lstm_losses, lstm_time = train_model(lstm, lstm_loader, epochs, lr)
    lstm_pred, lstm_mse, lstm_mae = evaluate(lstm, X_test, y_test, is_lstm=True)
    print(f"LSTM       MSE={lstm_mse:.6f}  MAE={lstm_mae:.6f}  ({lstm_time:.1f} сек)")

    # --- Recursive forecasting ---
    last_window = X_test[-1]
    mlp_future = recursive_forecast(mlp, last_window, steps=50, is_lstm=False)
    lstm_future = recursive_forecast(lstm, last_window, steps=50, is_lstm=True)

    # --- Графики ---
    figures_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{label}" if label else ""

    plot_predictions(
        y_test.numpy(),
        {
            "Baseline": baseline_pred.numpy(),
            "MLP": mlp_pred.numpy(),
            "LSTM": lstm_pred.numpy(),
        },
        figures_dir / f"predictions{suffix}.png",
    )

    plot_recursive(
        y_test[-100:].numpy(),
        {"MLP": mlp_future, "LSTM": lstm_future},
        figures_dir / f"recursive{suffix}.png",
    )

    plot_losses(mlp_losses, lstm_losses, figures_dir / f"losses{suffix}.png")

    return {
        "label": label,
        "lr": lr,
        "window": window,
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
# 11. Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed(42)
    Path("figures").mkdir(exist_ok=True)

    # Загрузка AAPL
    prices = load_aapl()
    plot_prices(prices, Path("figures/aapl_price.png"))

    # Базовый эксперимент
    result = run_experiment(prices, window=20, epochs=30, lr=0.001,
                            label="base")

    # Слайд 32: DOE по learning rate
    print(f"\n\n{'#' * 60}")
    print("DOE: варьируем learning rate (слайд 32)")
    print(f"{'#' * 60}")

    learning_rates = [0.0001, 0.001, 0.01]
    doe_results = [result]

    for lr in learning_rates:
        if lr == 0.001:
            continue  # уже сделан в базовом
        r = run_experiment(prices, window=20, epochs=30, lr=lr,
                           label=f"lr_{lr}")
        doe_results.append(r)

    # Итоговая таблица
    print(f"\n\n{'=' * 80}")
    print("ИТОГОВАЯ ТАБЛИЦА (слайд 36)")
    print(f"{'=' * 80}")
    print(f"{'Label':12s} {'lr':>8s} {'MLP MSE':>10s} {'MLP MAE':>10s} "
          f"{'LSTM MSE':>10s} {'LSTM MAE':>10s}")
    for r in doe_results:
        print(f"{r['label']:12s} {r['lr']:>8.4f} "
              f"{r['mlp_mse']:>10.6f} {r['mlp_mae']:>10.6f} "
              f"{r['lstm_mse']:>10.6f} {r['lstm_mae']:>10.6f}")

    # Выводы
    print(f"\n{'=' * 80}")
    print("ВЫВОДЫ")
    print(f"{'=' * 80}")

    best_mlp = min(doe_results, key=lambda r: r["mlp_mse"])
    best_lstm = min(doe_results, key=lambda r: r["lstm_mse"])

    print(f"Baseline MSE: {result['baseline_mse']:.6f}")
    print(f"Best MLP:     {best_mlp['mlp_mse']:.6f} "
          f"(lr={best_mlp['lr']})")
    print(f"Best LSTM:    {best_lstm['lstm_mse']:.6f} "
          f"(lr={best_lstm['lr']})")

    improvement_mlp = (result["baseline_mse"] - best_mlp["mlp_mse"]) \
        / result["baseline_mse"] * 100
    improvement_lstm = (result["baseline_mse"] - best_lstm["lstm_mse"]) \
        / result["baseline_mse"] * 100

    print(f"\nMLP улучшил baseline на {improvement_mlp:.1f}%")
    print(f"LSTM улучшил baseline на {improvement_lstm:.1f}%")

    print(f"\n[✓] Графики сохранены в figures/")


if __name__ == "__main__":
    main()
