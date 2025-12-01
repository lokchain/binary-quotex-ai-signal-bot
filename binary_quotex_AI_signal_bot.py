# binary_quotex_AI_signal_bot.py
# Full-featured Quotex-style chart analyzer (single-file)
# Requirements:
#  pip install streamlit numpy pandas Pillow opencv-python matplotlib mplfinance scipy requests

import streamlit as st
st.set_page_config(page_title="Quotex Analyzer — Direction + Analysis", layout="wide")

import os, io, json, time, base64, math
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import mplfinance as mpf
from scipy.signal import medfilt, find_peaks
import requests

# -----------------------------
# SETTINGS
# -----------------------------
HISTORY_FILE = "history.json"
USED_KEYS_FILE = "used_keys.json"
MIN_CANDLES = 8
SYNTH_BUCKETS = 48
MIN_CONF_PERCENT = 15.0  # allow trade threshold
DEFAULT_SCALE_BASE = 100.0

# Ensure files exist and are valid JSON lists/dicts
def ensure_file(path, default):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf8") as f:
            f.write(json.dumps(default, indent=2))
    else:
        try:
            with open(path, "r", encoding="utf8") as f:
                json.load(f)
        except Exception:
            with open(path, "w", encoding="utf8") as f:
                f.write(json.dumps(default, indent=2))

ensure_file(HISTORY_FILE, {})
ensure_file(USED_KEYS_FILE, {})

def load_json(path):
    try:
        with open(path, "r", encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return {}

def _convert_recursive(o):
    # converts numpy/pandas objects recursively into plain python types
    if isinstance(o, dict):
        return {k: _convert_recursive(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_convert_recursive(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    # pandas types
    try:
        if hasattr(o, "tolist") and not isinstance(o, (str, bytes, bytearray)):
            return o.tolist()
    except Exception:
        pass
    # fallback
    return o

def save_json_safe(path, obj):
    try:
        converted = _convert_recursive(obj)
        txt = json.dumps(converted, indent=2, ensure_ascii=False)
        with open(path, "w", encoding="utf8") as f:
            f.write(txt)
    except Exception:
        # last-resort fallback: stringify
        with open(path, "w", encoding="utf8") as f:
            f.write(json.dumps(str(obj), indent=2))

# -----------------------------
# Setup-key: read from secrets -> [setup] keys = ["A","B","C"]
# -----------------------------
def load_valid_keys_from_secrets():
    try:
        keys = st.secrets["setup"]["keys"]
        if isinstance(keys, list):
            return set(keys)
        if isinstance(keys, str):
            return set([k.strip() for k in keys.split(",") if k.strip()])
    except Exception:
        pass
    return set()

VALID_KEYS = load_valid_keys_from_secrets()

# UI: Sidebar key gate
st.sidebar.header("Access / Setup Key")
user_key = st.sidebar.text_input("Enter your setup key (persistent on this browser)", type="password")
if not user_key:
    st.sidebar.info("Enter setup key to unlock the analyzer.")
    st.stop()
if user_key not in VALID_KEYS:
    st.sidebar.error("Invalid setup key.")
    st.stop()

# bind user to key for history separation
if "user_id" not in st.session_state:
    st.session_state.user_id = f"{user_key}"

USER_ID = st.session_state.user_id

# -----------------------------
# Helper: validate image is chart-like
# -----------------------------
def is_chart_image_pil(pil_img: Image.Image, min_candles=MIN_CANDLES):
    try:
        w, h = pil_img.size
        if w < 200 or h < 120:
            return False
        gray = np.array(pil_img.convert("L"))
        edges = cv2.Canny(gray, 80, 160)
        col_sum = edges.sum(axis=0) / 255.0
        peaks, _ = find_peaks(col_sum, height=(col_sum.mean()*1.2), distance=max(2, w//200))
        horiz = edges.sum(axis=1) / 255.0
        horiz_peaks, _ = find_peaks(horiz, height=(horiz.mean()*1.2), distance=max(2, h//200))
        if len(peaks) >= max(2, min_candles//2) and len(horiz_peaks) >= 1:
            return True
        # fallback edge ratio
        edge_ratio = edges.sum() / (255.0 * w * h)
        return edge_ratio > 0.0004
    except Exception:
        return False

# -----------------------------
# Image -> series extraction (two methods: color-aware and brightness sampling)
# -----------------------------
def pil_to_cv2(img_pil):
    arr = np.array(img_pil)
    if arr.ndim == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr

def extract_series_cv(cv_img, buckets=SYNTH_BUCKETS):
    h, w = cv_img.shape[:2]
    try:
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        cols = np.linspace(0, w - 1, buckets).astype(int)
        vals = []
        for c in cols:
            col = v[:, c]
            vals.append(np.median(col))
        arr = np.array(vals, dtype=float)
        if np.ptp(arr) == 0:
            arr = arr + np.linspace(-1, 1, len(arr))
        norm = (arr - arr.min()) / max(1e-9, arr.max() - arr.min())
        df = _build_synthetic_ohlc_from_norm(norm)
        return df
    except Exception:
        return pd.DataFrame([])

def extract_series_brightness(cv_img, buckets=SYNTH_BUCKETS):
    try:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        cols = np.linspace(0, w - 1, buckets).astype(int)
        vals = [np.mean(gray[:, c]) for c in cols]
        arr = np.array(vals, dtype=float)
        if np.ptp(arr) == 0:
            arr = arr + np.linspace(-1, 1, len(arr))
        norm = (arr - arr.min()) / max(1e-9, arr.max() - arr.min())
        df = _build_synthetic_ohlc_from_norm(norm)
        return df
    except Exception:
        return pd.DataFrame([])

def _build_synthetic_ohlc_from_norm(norm):
    series = []
    sd = np.clip(np.std(norm), 0.001, 0.25)
    for v in norm:
        o = np.clip(v + (np.random.rand() - 0.5) * sd, 0.0, 1.0)
        c = np.clip(v + (np.random.rand() - 0.5) * sd, 0.0, 1.0)
        hi = min(1.0, max(o, c) + abs(np.random.rand() * sd))
        lo = max(0.0, min(o, c) - abs(np.random.rand() * sd))
        series.append({"open": o, "high": hi, "low": lo, "close": c})
    return pd.DataFrame(series)

# -----------------------------
# Indicators
# -----------------------------
def sma(arr, n):
    return pd.Series(arr).rolling(n, min_periods=1).mean().values

def ema(arr, n):
    return pd.Series(arr).ewm(span=n, adjust=False).mean().values

def compute_rsi(arr, n=14):
    s = pd.Series(arr).astype(float)
    delta = s.diff()
    up = delta.clip(lower=0).rolling(window=n, min_periods=1).mean()
    down = -delta.clip(upper=0).rolling(window=n, min_periods=1).mean()
    rs = up / (down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).values

def compute_atr(df, n=14):
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=1).mean().fillna(tr.mean())
    return atr.values

# -----------------------------
# Pattern detectors
# -----------------------------
def detect_pinbar(df):
    if len(df) < 1: return False
    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    hi_tail = last["high"] - max(last["open"], last["close"])
    lo_tail = min(last["open"], last["close"]) - last["low"]
    if body < 1e-6: return False
    return (hi_tail > 2.5 * body) or (lo_tail > 2.5 * body)

def detect_engulfing(df):
    if len(df) < 2: return False
    last = df.iloc[-1]; prev = df.iloc[-2]
    prev_body = prev["close"] - prev["open"]
    last_body = last["close"] - last["open"]
    if prev_body * last_body < 0:
        if last["close"] >= prev["open"] and last["open"] <= prev["close"]:
            return True
    return False

def detect_doji(df):
    if len(df) < 1: return False
    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    rng = last["high"] - last["low"]
    return rng > 0 and (body / rng) < 0.12

# -----------------------------
# Anomaly score
# -----------------------------
def compute_anomaly_score(df):
    try:
        closes = df["close"].astype(float).values
        if len(closes) < 5: return 0.0
        rounded = np.round(closes, 4)
        unique_count = len(np.unique(rounded))
        ident_ratio = 1 - (unique_count / max(1, len(rounded)))
        score = 0.4 * np.clip(ident_ratio, 0, 1)
        returns = np.diff(closes)
        if len(returns) >= 3:
            cmat = np.corrcoef(returns[:-1], returns[1:])
            ac = 0.0
            if cmat.shape == (2,2) and not np.isnan(cmat[0,1]):
                ac = cmat[0,1]
            score += 0.3 * np.clip(abs(ac), 0, 1)
        kurt = pd.Series(returns).kurtosis(); skew = pd.Series(returns).skew()
        score += 0.3 * np.clip(np.tanh(abs(kurt)/5.0 + abs(skew)/3.0)/2.0, 0, 1)
        return float(np.clip(score, 0.0, 1.0))
    except Exception:
        return 0.0

# -----------------------------
# Direction & confidence (weighted)
# -----------------------------
def compute_direction_and_confidence(df, weights=None):
    if weights is None:
        weights = {"sma": 0.35, "rsi": 0.25, "momentum": 0.2, "pattern": 0.2}
    res = {"direction":"NO-TRADE", "confidence_pct":0.0, "reasons": []}
    if df.empty or len(df) < 3:
        return res
    closes = df["close"].astype(float).values
    sma5 = sma(closes, 5)[-1]
    sma20 = sma(closes, 20 if len(closes)>=20 else len(closes))[-1]
    rsi = compute_rsi(closes, n=min(14, len(closes)))[-1]
    mom = float(closes[-1] - closes[-2]) if len(closes)>=2 else 0.0
    pat_score = 0.0
    reasons = []
    # SMA reason
    if sma5 > sma20:
        reasons.append("Short SMA > Long SMA")
        sma_vote = 1
    elif sma5 < sma20:
        reasons.append("Short SMA < Long SMA")
        sma_vote = -1
    else:
        sma_vote = 0
    # RSI reason
    if rsi < 40:
        reasons.append("RSI low (bullish bias)")
        rsi_vote = 1
    elif rsi > 60:
        reasons.append("RSI high (bearish bias)")
        rsi_vote = -1
    else:
        rsi_vote = 0
    # momentum
    if mom > 0:
        mom_vote = 1
    elif mom < 0:
        mom_vote = -1
    else:
        mom_vote = 0
    # pattern
    pat_bull = detect_pinbar(df) and (df.iloc[-1]["close"] > df.iloc[-1]["open"])
    pat_engulf = detect_engulfing(df)
    if pat_bull:
        pat_score += 1; reasons.append("Bull pinbar")
    if pat_engulf:
        if df.iloc[-1]["close"] > df.iloc[-1]["open"]:
            pat_score += 1; reasons.append("Bullish engulfing")
        else:
            pat_score -= 1; reasons.append("Bearish engulfing")
    # aggregate votes
    vote_score = (weights["sma"] * sma_vote +
                  weights["rsi"] * rsi_vote +
                  weights["momentum"] * mom_vote +
                  weights["pattern"] * (np.sign(pat_score) if pat_score!=0 else 0))
    # Confidence scaling
    total_w = sum(weights.values())
    conf = (abs(vote_score) / (total_w if total_w>0 else 1)) * 100
    # boost if multiple patterns or strong ATR
    atr = compute_atr(df, n=min(14, len(df)))[-1]
    conf = conf * (1 + min(0.35, float(atr)*10))
    conf = float(np.clip(conf, 0.0, 99.9))
    if vote_score > 0.15:
        direction = "BUY"
    elif vote_score < -0.15:
        direction = "SELL"
    else:
        direction = "NO-TRADE"
    res["direction"] = direction
    res["confidence_pct"] = round(conf, 1)
    res["reasons"] = reasons
    return res

# -----------------------------
# Two-scenario text
# -----------------------------
def two_scenario_text(df):
    if len(df) < 1:
        return {"if_current_closes_up":"", "if_current_closes_down":""}
    last = df.iloc[-1]
    # compute threshold levels in normalized value space, then map to plotted scale
    up_close = last["open"] + (last["high"] - last["open"]) * 0.7
    down_close = last["open"] - (last["open"] - last["low"]) * 0.7
    return {
        "if_current_closes_up": f"If current closes ABOVE {up_close:.5f} → next likely BUY.",
        "if_current_closes_down": f"If current closes BELOW {down_close:.5f} → next likely SELL."
    }

# -----------------------------
# Telegram helper (optional)
# -----------------------------
def send_to_telegram(bot_token, chat_id, caption, image_bytes=None):
    if not bot_token or not chat_id:
        return {"ok": False, "error": "Missing token/id"}
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        files = {"photo": ("chart.png", image_bytes)} if image_bytes else None
        data = {"chat_id": chat_id, "caption": caption}
        r = requests.post(url, files=files, data=data, timeout=20)
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

# -----------------------------
# UI Layout
# -----------------------------
st.title("Quotex Chart Analyzer — Descriptive + Direction")
st.write("Upload candlestick screenshot. The tool reconstructs a candlestick series, computes indicators, gives a compact direction cue (BUY/SELL/NO-TRADE) and a two-scenario conditional cue based on the running candle close.")

colL, colR = st.columns([2,1])

with colR:
    st.header("Session")
    st.write(f"User key (masked): `{user_key[:3]}...`")
    st.write(f"Session ID: `{USER_ID}`")
    # minimal controls
    bot_token = st.text_input("Telegram Bot token (optional)", type="password")
    chat_id = st.text_input("Telegram Chat ID (optional)")
    auto_send = st.checkbox("Auto-send allowed signals to Telegram", value=False)
    st.markdown("---")
    st.write("Recent history (this user):")
    all_hist = load_json(HISTORY_FILE)
    my_hist = all_hist.get(USER_ID, [])
    for h in my_hist[:6]:
        ts = datetime.fromtimestamp(h.get("timestamp", time.time())).strftime("%Y-%m-%d %H:%M:%S")
        st.write(f"- {ts} — {h.get('direction','')} {h.get('confidence_pct','')}%")

with colL:
    st.header("Upload Chart (PNG/JPG)")
    uploaded = st.file_uploader("Drop chart screenshot", type=["png","jpg","jpeg"])
    st.caption("Prefer clear chart screenshots; UI overlay may affect extraction.")
    manual_srs_input = st.text_input("Optional manual S/R (comma separated normalized 0..1)", value="")

    if uploaded is None:
        st.info("Upload a chart image to analyze.")
        st.stop()

    # read image
    try:
        bytes_data = uploaded.read()
        pil = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    except Exception:
        st.error("Failed to read uploaded file. Make sure it's an image.")
        st.stop()

    st.subheader("Preview")
    st.image(pil, use_column_width=True)

    # validate chart
    if not is_chart_image_pil(pil, min_candles=MIN_CANDLES):
        st.error("Uploaded image does not look like a candlestick chart. Upload a proper chart screenshot.")
        st.stop()

    # prepare cv image
    cv_img = pil_to_cv2(pil)

    # try color-aware extraction then brightness fallback
    df_series = extract_series_cv(cv_img, buckets=SYNTH_BUCKETS)
    if df_series.empty or len(df_series) < MIN_CANDLES:
        df_series = extract_series_brightness(cv_img, buckets=SYNTH_BUCKETS)
    if df_series.empty or len(df_series) < MIN_CANDLES:
        st.error("Failed to extract candlesticks from image reliably.")
        st.stop()

    # attach synthetic datetime index
    times = pd.date_range(end=pd.Timestamp.now(), periods=len(df_series), freq="T")
    df_plot = df_series.copy()
    df_plot.index = times
    df_plot.index.name = "Datetime"
    # denormalize for plotting
    df_plot_num = df_plot.copy()
    df_plot_num[['open','high','low','close']] = df_plot_num[['open','high','low','close']] * 10 + DEFAULT_SCALE_BASE

    # Plot with mplfinance (candles + volume synthetic)
    st.subheader("Reconstructed Candlestick Series")
    try:
        fig, ax = plt.subplots(figsize=(10,3))
        mpf.plot(df_plot_num, type="candle", style="charles", ax=ax, volume=False, show_nontrading=True)
        st.pyplot(fig)
    except Exception:
        st.line_chart(df_plot_num['close'])

    # indicators & detection
    analysis = {}
    ind = {}
    closes = df_plot['close'].astype(float).values
    ind['sma5'] = float(sma(closes, 5)[-1])
    ind['sma20'] = float(sma(closes, 20 if len(closes)>=20 else len(closes))[-1])
    ind['rsi'] = float(compute_rsi(closes, n=min(14, len(closes)))[-1])
    ind['atr'] = float(compute_atr(df_plot, n=min(14, len(df_plot)))[-1])
    ind['momentum'] = float(closes[-1] - closes[-2]) if len(closes)>=2 else 0.0
    analysis['indicators'] = ind
    patterns = {
        "pinbar": detect_pinbar(df_plot),
        "engulfing": detect_engulfing(df_plot),
        "doji": detect_doji(df_plot)
    }
    analysis['patterns'] = patterns
    analysis['anomaly_score'] = compute_anomaly_score(df_plot)

    # compute direction & confidence
    dc = compute_direction_and_confidence(df_plot)
    analysis.update(dc)

    # two-scenario cues
    scenarios = two_scenario_text(df_plot)
    analysis['scenarios'] = scenarios

    # reasons / grouped reasoning
    analysis['reasons'] = dc.get("reasons", [])

    # Save per-user history (safe serializable)
    timestamp = time.time()
    hist_obj = {
        "timestamp": timestamp,
        "direction": analysis.get("direction"),
        "confidence_pct": analysis.get("confidence_pct"),
        "reasons": analysis.get("reasons"),
        "indicators": analysis.get("indicators"),
        "patterns": analysis.get("patterns"),
        "anomaly_score": analysis.get("anomaly_score")
    }
    all_hist = load_json(HISTORY_FILE)
    user_list = all_hist.get(USER_ID, [])
    user_list.insert(0, hist_obj)
    all_hist[USER_ID] = user_list[:500]
    try:
        save_json_safe(HISTORY_FILE, all_hist)
    except Exception:
        # non-fatal: continue but warn
        st.warning("Warning: failed to persist history file (permissions/serialization).")

    # show concise result
    st.subheader("Result")
    dir_disp = analysis.get("direction", "NO-TRADE")
    conf_disp = analysis.get("confidence_pct", 0.0)
    if dir_disp == "BUY":
        st.success(f"TRADE SUGGESTION: {dir_disp} — Confidence {conf_disp}%")
    elif dir_disp == "SELL":
        st.error(f"TRADE SUGGESTION: {dir_disp} — Confidence {conf_disp}%")
    else:
        st.warning(f"NO-TRADE (Low conviction) — Confidence {conf_disp}%")

    st.markdown("**Reasons:**")
    for r in analysis.get("reasons", []):
        st.write("- " + r)
    st.markdown("---")
    st.markdown("**Indicators snapshot**")
    c1,c2,c3 = st.columns(3)
    c1.metric("SMA5", f"{ind['sma5']:.4f}")
    c2.metric("SMA20", f"{ind['sma20']:.4f}")
    c3.metric("RSI", f"{ind['rsi']:.1f}")
    c4,c5,c6 = st.columns(3)
    c4.metric("ATR (norm)", f"{ind['atr']:.4f}")
    c5.metric("Momentum", f"{ind['momentum']:.4f}")
    c6.metric("Anomaly", f"{analysis['anomaly_score']:.2f}")

    st.markdown("**Patterns:**")
    for k,v in patterns.items():
        st.write(f"- {k}: {'Yes' if v else 'No'}")

    st.markdown("**Conditional scenarios (useful for binary next-candle logic)**")
    st.info(analysis['scenarios']['if_current_closes_up'])
    st.info(analysis['scenarios']['if_current_closes_down'])

    # Export options
    st.markdown("---")
    st.subheader("Export / Telegram")
    csv = df_plot_num.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown(f"[Download extracted series CSV](data:file/csv;base64,{b64})")

    json_analysis = json.dumps(_convert_recursive(analysis), indent=2)
    b64j = base64.b64encode(json_analysis.encode()).decode()
    st.markdown(f"[Download analysis JSON](data:application/json;base64,{b64j})")

    # Telegram send (optional)
    if auto_send and analysis.get("direction") in ("BUY","SELL") and analysis.get("confidence_pct",0) >= MIN_CONF_PERCENT:
        # render image bytes
        buf = io.BytesIO()
        try:
            fig2, ax2 = plt.subplots(figsize=(6,3))
            mpf.plot(df_plot_num, type="candle", style="charles", ax=ax2, volume=False, show_nontrading=True)
            fig2.savefig(buf, bbox_inches="tight")
            buf.seek(0)
            caption = f"Analyzer result: {analysis['direction']} ({analysis['confidence_pct']}%)\nReasons: {', '.join(analysis.get('reasons',[]))}"
            r = send_to_telegram(bot_token, chat_id, caption, buf.getvalue())
            if r.get("ok"):
                st.success("Sent to Telegram.")
            else:
                st.error("Telegram send failed.")
        except Exception as e:
            st.error("Telegram send exception: " + str(e))

# -----------------------------
# Footer / disclaimers
# -----------------------------
st.markdown("---")
st.caption("This tool provides descriptive analysis and conditional directional cues for research/testing only. Not financial advice.")
