# Ceunia
Integración simbiótica 
Python
# ============================================
# CEUNIA 3-6-9 INTRADAY ENGINE — SINGLE BLOCK
# Listo para Google Colab (prototipo optimizable)
# ============================================

import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass

# =========================
# CONFIG
# =========================
SCALES = [3, 6, 9, 18, 36]

@dataclass
class Config:
    alpha: float = 4.0
    fee_bps: float = 1.0
    risk_per_trade: float = 0.01
    max_position: float = 1.0
    min_coherence: float = 0.6
    session_minutes: int = 390

CFG = Config()

# =========================
# FEATURE ENGINE
# =========================
class FeatureEngine:
    def seasonal(self, minute_idx):
        x = 2*np.pi*(minute_idx / CFG.session_minutes)
        return np.array([np.sin(x), np.cos(x), minute_idx, CFG.session_minutes - minute_idx])

    def compute(self, df, k, minute_idx):
        r = np.log(df["close"]).diff()

        feat = {
            "ret_1m": r.iloc[-1],
            f"ret_{k}m": r.rolling(k).sum().iloc[-1],
            f"vol_{k}m": r.rolling(k).std().iloc[-1],
            f"volu_{k}m": df["volume"].rolling(k).mean().iloc[-1],
        }

        if "bid" in df.columns and "ask" in df.columns:
            feat["spread"] = (df["ask"] - df["bid"]).iloc[-1]
        else:
            feat["spread"] = 0.0

        season = self.seasonal(minute_idx)
        for i, v in enumerate(season):
            feat[f"season_{i}"] = v

        return np.array(list(feat.values()), dtype=np.float32).reshape(1, -1)

# =========================
# MODELO ONLINE SIMPLE
# =========================
class OnlineLinear:
    def __init__(self, dim, lr=0.001):
        self.w = np.zeros(dim)
        self.lr = lr

    def predict(self, x):
        return float(x @ self.w)

    def update(self, x, y):
        pred = self.predict(x)
        grad = (pred - y) * x.flatten()
        self.w -= self.lr * grad

# =========================
# ENSEMBLE
# =========================
class Ensemble369:
    def __init__(self, models):
        self.models = models
        self.scores = {k: 0.0 for k in SCALES}
        self.weights = {k: 1.0 for k in SCALES}

    def softmax_weights(self):
        s = np.array([self.scores[k] for k in SCALES])
        w = np.exp(CFG.alpha * s)
        w /= w.sum()
        return dict(zip(SCALES, w))

    def predict(self, features_by_scale):
        preds = {}
        for k in SCALES:
            preds[k] = self.models[k].predict(features_by_scale[k])

        self.weights = self.softmax_weights()
        y_hat = sum(self.weights[k] * preds[k] for k in SCALES)

        coherence = np.mean(np.sign(list(preds.values())))
        return y_hat, preds, coherence

    def update(self, real_ret, preds):
        for k in SCALES:
            e = real_ret - preds[k]
            hit = 1.0 if np.sign(preds[k]) == np.sign(real_ret) else -1.0
            score = hit - abs(e) - CFG.fee_bps/1e4
            self.scores[k] = 0.9*self.scores[k] + 0.1*score

# =========================
# ERROR ENGINE
# =========================
class ErrorEngine:
    def __init__(self, maxlen=200):
        self.buffer = deque(maxlen=maxlen)

    def update(self, real, pred):
        e = real - pred
        self.buffer.append(e)
        return e

    def regime_shift(self):
        if len(self.buffer) < 50:
            return False
        arr = np.array(self.buffer)
        return np.std(arr[-50:]) > 1.5 * np.std(arr[:-50])

# =========================
# RISK
# =========================
class RiskManager:
    def position_size(self, signal_strength):
        return np.clip(signal_strength, -CFG.max_position, CFG.max_position)

# =========================
# EXECUTOR
# =========================
class Executor:
    def __init__(self):
        self.position = 0.0

    def execute(self, signal):
        self.position = signal
        return self.position

# =========================
# MAIN ENGINE
# =========================
class TradingEngine:
    def __init__(self, df):
        self.df = df.copy().reset_index(drop=True)
        self.fe = FeatureEngine()

        dim = 8
        self.models = {k: OnlineLinear(dim) for k in SCALES}
        self.ensemble = Ensemble369(self.models)
        self.error_engine = ErrorEngine()
        self.risk = RiskManager()
        self.exec = Executor()

    def step(self, i):
        sub_df = self.df.iloc[:i+1]

        features = {}
        for k in SCALES:
            features[k] = self.fe.compute(sub_df, k, i)

        y_hat, preds, coherence = self.ensemble.predict(features)

        signal = 0.0
        if coherence > CFG.min_coherence:
            signal = self.risk.position_size(y_hat)

        pos = self.exec.execute(signal)

        if i > 0:
            real_ret = np.log(self.df["close"]).diff().iloc[i]
        else:
            real_ret = 0.0

        # entrenamiento online
        for k in SCALES:
            self.models[k].update(features[k], real_ret)

        self.ensemble.update(real_ret, preds)
        self.error_engine.update(real_ret, y_hat)

        return y_hat, coherence, pos

    def run(self):
        results = []

        for i in range(max(SCALES), len(self.df)):
            y_hat, coherence, pos = self.step(i)
            results.append([y_hat, coherence, pos])

        return pd.DataFrame(results, columns=["prediction", "coherence", "position"])

# =========================
# DATA MOCK (para probar)
# =========================
np.random.seed(42)
n = 2000

price = np.cumsum(np.random.normal(0, 0.01, n)) + 100
volume = np.random.randint(100, 1000, n)

df = pd.DataFrame({
    "close": price,
    "volume": volume
})

# =========================
# RUN
# =========================
engine = TradingEngine(df)
result = engine.run()

print(result.tail())
