import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import pytz
import uuid
import csv
import os

# === Configuration initiale ===
API_KEY = "K6meo5BSJuuduWI8"
EMAIL = "5rycnytkzh@privaterelay.appleid.com"
PASSWORD = "Lucas1234@"
BASE_URL = "https://demo-api-capital.backend-capital.com"

# Headers de base pour l'API
headers = {
    "X-CAP-API-KEY": API_KEY,
    "Content-Type": "application/json",
    "Accept": "application/json"
}

# Paramètres de trading
RISK_PER_TRADE = 0.01  # 1% du capital par trade
MAX_SLIPPAGE_PERCENT = 0.5  # Slippage maximum toléré
PRICE_TOLERANCE_PERCENT = 0.3  # Tolérance pour la vérification du prix
CHECK_INTERVAL = 60  # Intervalle entre les analyses (secondes)
CONFIRMATION_PERIOD = 2  # Nombre de bougies pour confirmation
ATR_THRESHOLD = 4.0  # Seuil de volatilité
MARKET = "GOLD"
SPREAD_COST = 0.6  # Spread estimé pour l'or (en points)
TRADE_LOG_FILE = "trade_log.csv"

# Dictionnaire pour stocker les positions ouvertes
open_positions = {}

# === Fonction utilitaire pour les requêtes API ===
def safe_request(method, url, headers, json=None, params=None, retries=3):
    for attempt in range(retries):
        try:
            response = requests.request(method, url, headers=headers, json=json, params=params)
            if response.ok:
                cst = response.headers.get("CST")
                security_token = response.headers.get("X-SECURITY-TOKEN")
                if cst and security_token:
                    headers["CST"] = cst
                    headers["X-SECURITY-TOKEN"] = security_token
                    print(f"🔑 Tokens mis à jour : CST={cst}, X-SECURITY-TOKEN={security_token}")
                return response
            else:
                print(f"❌ Erreur API (tentative {attempt + 1}) : {response.status_code} - {response.text}")
                if response.status_code == 401:
                    print("⚠️ Problème d'authentification, arrêt de la requête")
                    return None
                if response.status_code == 400:
                    error_message = response.json().get("errorCode", "Unknown error")
                    print(f"⚠️ Erreur spécifique : {error_message}")
                    if "INSUFFICIENT_FUNDS" in error_message:
                        return {"error": "INSUFFICIENT_FUNDS"}
                    elif "MARKET_CLOSED" in error_message:
                        return {"error": "MARKET_CLOSED"}
                    elif "error.invalid.stoploss.maxvalue" in error_message:
                        try:
                            max_stop_value = float(error_message.split(": ")[1])
                            return {"error": "INVALID_STOP_LOSS", "max_stop_value": max_stop_value}
                        except (IndexError, ValueError):
                            print("❌ Impossible d'extraire max_stop_value de l'erreur")
                            return {"error": "INVALID_STOP_LOSS"}
                    elif "error.invalid.dealId" in error_message:
                        return {"error": "INVALID_DEAL_ID"}
                time.sleep(2 ** attempt)
        except requests.RequestException as e:
            print(f"❌ Erreur réseau (tentative {attempt + 1}) : {e}")
        time.sleep(2 ** attempt)
    print(f"❌ Échec après {retries} tentatives")
    return None

# === Authentification à l'API ===
def authenticate():
    print("🔐 Tentative de connexion au compte démo...")
    url = f"{BASE_URL}/api/v1/session"
    payload = {"identifier": EMAIL, "password": PASSWORD}
    auth_headers = headers.copy()
    response = safe_request("POST", url, headers=auth_headers, json=payload)
    if response and response.status_code == 200:
        print("✅ Connexion réussie au compte démo")
        return auth_headers
    print(f"❌ Échec de la connexion : {response.status_code if response else 'Aucune réponse'}")
    return None

# === Recherche du marché Gold ===
def search_market(headers, search_term="GOLD"):
    print(f"🔍 Recherche du marché : {search_term}")
    url = f"{BASE_URL}/api/v1/markets"
    params = {"searchTerm": search_term}
    response = safe_request("GET", url, headers=headers, params=params)
    if response:
        markets = response.json().get("markets", [])
        if markets:
            print(f"📜 Marché trouvé : {markets[0]['instrumentName']} (EPIC: {markets[0]['epic']})")
            return markets[0]["epic"]
        print("⚠️ Aucun marché trouvé pour Gold")
    print("❌ Erreur lors de la recherche du marché")
    return None

# === Vérification des heures de trading ===
def is_market_open():
    now = datetime.now(pytz.timezone("Europe/Paris"))
    weekday = now.weekday()
    hour = now.hour
    if weekday < 5 or (weekday == 5 and hour < 23):
        print("🕒 Marché ouvert")
        return True
    print("⛔ Marché fermé")
    return False

# === Récupération du solde disponible ===
def get_available_balance(headers):
    print("💸 Vérification du solde disponible...")
# Récupération du solde disponible
    url = f"{BASE_URL}/api/v1/accounts"
    response = safe_request("GET", url, headers=headers)
    if response:
        accounts = response.json().get("accounts", [])
        if accounts:
            balance = accounts[0].get("balance", {}).get("available", 0.0)
            print(f"💰 Solde disponible : {balance} EUR")
            return balance
    print("❌ Erreur lors de la récupération du solde")
    return 0.0

# === Récupération des exigences de marge ===
def get_margin_requirement(headers, epic):
    print(f"📋 Récupération des règles pour {epic}...")
    url = f"{BASE_URL}/api/v1/markets/{epic}"
    response = safe_request("GET", url, headers=headers)
    if response:
        data = response.json()
        margin_factor = data.get("marginFactor", 0.2)
        min_size = data.get("minimumDealSize", 0.01)
        max_size = data.get("maximumDealSize", 100.0)
        print(f"📋 Règles : Marge={margin_factor*100}%, Taille min={min_size}, Taille max={max_size}")
        return margin_factor, min_size, max_size
    print(f"❌ Erreur lors de la récupération des règles pour {epic}")
    return 0.2, 0.01, 100.0

# === Récupération du prix actuel ===
def get_current_price(headers, epic):
    print(f"📈 Récupération du prix actuel pour {epic}...")
    url = f"{BASE_URL}/api/v1/prices/{epic}"
    params = {"resolution": "MINUTE", "max": 1}
    response = safe_request("GET", url, headers=headers, params=params)
    if response:
        prices = response.json().get("prices", [])
        if prices and "closePrice" in prices[0]:
            bid = prices[0]["closePrice"].get("bid", None)
            if bid is not None:
                print(f"📈 Prix actuel (bid) : {bid:.2f}")
                return bid
        print("⚠️ Données de prix invalides")
    print("❌ Erreur lors de la récupération du prix actuel")
    return None

# === Vérification du prix avant ordre ===
def verify_price_stability(headers, epic, expected_price):
    print(f"🔍 Vérification de la stabilité du prix pour {epic}...")
    current_price = get_current_price(headers, epic)
    if current_price is None:
        print("❌ Impossible de vérifier le prix actuel")
        return False
    price_diff_percent = abs(current_price - expected_price) / expected_price * 100
    if price_diff_percent > PRICE_TOLERANCE_PERCENT:
        print(f"⚠️ Écart de prix trop important : {price_diff_percent:.2f}% (tolérance max : {PRICE_TOLERANCE_PERCENT}%)")
        return False
    print(f"✅ Prix stable : écart de {price_diff_percent:.2f}%")
    return True

# === Vérification de l'exécution de l'ordre ===
def verify_order_execution(headers, deal_id, expected_price, direction, size):
    print(f"🔍 Vérification de l'exécution de l'ordre (ID: {deal_id})...")
    if not deal_id or deal_id == "unknown":
        print("❌ Deal ID invalide, impossible de vérifier l'exécution")
        return False
    url = f"{BASE_URL}/api/v1/confirms/{deal_id}"
    time.sleep(1)  # Short delay to allow API to process the order
    response = safe_request("GET", url, headers=headers)
    if response and isinstance(response, dict) and response.get("error") == "INVALID_DEAL_ID":
        print(f"❌ Deal ID {deal_id} invalide selon l'API")
        return False
    if response and response.ok:
        confirmation = response.json()
        status = confirmation.get("status", "UNKNOWN")
        executed_price = confirmation.get("level", None)
        if status == "OPEN" and executed_price is not None:
            slippage_percent = abs(executed_price - expected_price) / expected_price * 100
            if slippage_percent > MAX_SLIPPAGE_PERCENT:
                print(f"⚠️ Slippage excessif : {slippage_percent:.2f}% (max : {MAX_SLIPPAGE_PERCENT}%)")
                close_position(headers, deal_id, direction, size, executed_price)
                return False
            print(f"✅ Ordre exécuté correctement à {executed_price:.2f} (slippage : {slippage_percent:.2f}%)")
            return True
        print(f"⚠️ Statut de l'ordre non ouvert : {status}")
        return False
    print(f"❌ Échec de la vérification de l'ordre (ID: {deal_id})")
    return False

# === Journalisation des trades ===
def log_trade(deal_id, direction, size, entry_price, stop, limit, status, exit_price=None, profit_loss=None):
    timestamp = datetime.now(pytz.timezone("Europe/Paris")).strftime("%Y-%m-%d %H:%M:%S")
    trade_data = {
        "timestamp": timestamp,
        "deal_id": deal_id,
        "direction": direction,
        "size": size,
        "entry_price": entry_price,
        "stop_level": stop,
        "limit_level": limit,
        "status": status,
        "exit_price": exit_price if exit_price is not None else "",
        "profit_loss": profit_loss if profit_loss is not None else ""
    }
    file_exists = os.path.isfile(TRADE_LOG_FILE)
    with open(TRADE_LOG_FILE, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trade_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(trade_data)
    print(f"📝 Trade journalisé : {status} (ID: {deal_id})")

# === Récupération des prix historiques ===
def get_candles(headers, epic, resolution="MINUTE_15", limit=200):
    print(f"📈 Récupération des prix pour {epic} (résolution : {resolution}, limite : {limit})...")
    url = f"{BASE_URL}/api/v1/prices/{epic}"
    params = {"resolution": resolution, "max": limit}
    response = safe_request("GET", url, headers=headers, params=params)
    if response:
        candles = response.json().get("prices", [])
        if not candles:
            print("⚠️ Aucune donnée de prix reçue")
            return None
        df = pd.DataFrame(candles)
        df["timestamp"] = pd.to_datetime(df["snapshotTime"], utc=True).dt.tz_convert("Europe/Paris")
        df.set_index("timestamp", inplace=True)
        try:
            df["close"] = df["closePrice"].apply(lambda x: x["bid"] if isinstance(x, dict) and "bid" in x else None)
            df["open"] = df["openPrice"].apply(lambda x: x["bid"] if isinstance(x, dict) and "bid" in x else None)
            df["high"] = df["highPrice"].apply(lambda x: x["bid"] if isinstance(x, dict) and "bid" in x else None)
            df["low"] = df["lowPrice"].apply(lambda x: x["bid"] if isinstance(x, dict) and "bid" in x else None)
            df["volume"] = df["lastTradedVolume"]
        except (KeyError, TypeError) as e:
            print(f"❌ Erreur dans le format des données de prix : {e}")
            return None
        if df[["close", "open", "high", "low"]].isnull().any().any() or (df["close"] == 0).any():
            print("⚠️ Données de prix invalides")
            return None
        print(f"📈 Dernier prix Gold : {df['close'].iloc[-1]:.2f}")
        return df[["open", "high", "low", "close", "volume"]]
    print("❌ Erreur lors de la récupération des prix")
    return None

# === Récupération des prix pour la tendance (4 heures) ===
def get_trend_candles(headers, epic, resolution="HOUR_4", limit=50):
    print(f"📈 Récupération des prix pour analyse de tendance ({epic}, résolution : {resolution})...")
    url = f"{BASE_URL}/api/v1/prices/{epic}"
    params = {"resolution": resolution, "max": limit}
    response = safe_request("GET", url, headers=headers, params=params)
    if response:
        candles = response.json().get("prices", [])
        if not candles:
            print("⚠️ Aucune donnée de prix reçue pour la tendance")
            return None
        df = pd.DataFrame(candles)
        df["timestamp"] = pd.to_datetime(df["snapshotTime"], utc=True).dt.tz_convert("Europe/Paris")
        df.set_index("timestamp", inplace=True)
        try:
            df["close"] = df["closePrice"].apply(lambda x: x["bid"] if isinstance(x, dict) and "bid" in x else None)
        except (KeyError, TypeError) as e:
            print(f"❌ Erreur dans le format des données de tendance : {e}")
            return None
        if df["close"].isnull().any() or (df["close"] == 0).any():
            print("⚠️ Données de tendance invalides")
            return None
        return df[["close"]]
    print("❌ Erreur lors de la récupération des prix de tendance")
    return None

# === Calcul du RSI ===
def calculate_rsi(df, period=14):
    print(f"📊 Calcul du RSI sur {period} bougies...")
    if len(df) < period:
        print(f"⚠️ Pas assez de données pour calculer le RSI ({len(df)} lignes, requis {period})")
        df["rsi"] = 50
        return df
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)
    return df

# === Calcul de l'ADX ===
def calculate_adx(df, period=14):
    print(f"📊 Calcul de l'ADX sur {period} bougies...")
    if len(df) < period * 2:
        print(f"⚠️ Pas assez de données pour calculer l'ADX ({len(df)} lignes, requis {period * 2})")
        df["adx"] = 25
        return df
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
    df["plus_dm"] = df["high"].diff().where((df["high"].diff() > df["low"].diff()) & (df["high"].diff() > 0), 0)
    df["minus_dm"] = (-df["low"].diff()).where((df["low"].diff() < df["high"].diff()) & (df["low"].diff() < 0), 0)
    df["tr_smooth"] = df["tr"].rolling(window=period).mean()
    df["plus_di"] = 100 * (df["plus_dm"].rolling(window=period).mean() / df["tr_smooth"])
    df["minus_di"] = 100 * (df["minus_dm"].rolling(window=period).mean() / df["tr_smooth"])
    df["dx"] = 100 * abs(df["plus_di"] - df["minus_di"]) / (df["plus_di"] + df["minus_di"] + 1e-10)
    df["adx"] = df["dx"].rolling(window=period).mean()
    df["adx"] = df["adx"].fillna(25)
    return df

# === Calcul de l'ATR ===
def calculate_atr(df, period=14):
    print(f"📊 Calcul de l'ATR sur {period} bougies...")
    if len(df) < period:
        print(f"⚠️ Pas assez de données pour calculer l'ATR ({len(df)} lignes, requis {period})")
        df["atr"] = df["close"].std()
        return df
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
    df["atr"] = df["tr"].rolling(window=period).mean()
    df["atr"] = df["atr"].fillna(df["close"].std())
    return df

# === Calcul du MACD ===
def calculate_macd(df, fast=12, slow=26, signal=9):
    print(f"📊 Calcul du MACD (fast={fast}, slow={slow}, signal={signal})...")
    if len(df) < slow:
        print(f"⚠️ Pas assez de données pour calculer le MACD ({len(df)} lignes, requis {slow})")
        df["macd"] = 0
        df["macd_signal"] = 0
        return df
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd"] = df["macd"].fillna(0)
    df["macd_signal"] = df["macd_signal"].fillna(0)
    return df

# === Détection de la tendance (EMA 50/200) ===
def detect_trend(df_trend):
    print("🧠 Analyse de la tendance sur 4 heures...")
    if df_trend is None or len(df_trend) < 20:
        print("⚠️ Pas assez de données pour analyser la tendance, tendance neutre")
        return "NEUTRAL"
    df_trend["ema_50"] = df_trend["close"].ewm(span=50, adjust=False).mean()
    df_trend["ema_200"] = df_trend["close"].ewm(span=200, adjust=False).mean()
    last_ema_50 = df_trend["ema_50"].iloc[-1]
    last_ema_200 = df_trend["ema_200"].iloc[-1]
    if last_ema_50 > last_ema_200:
        print("📈 Tendance haussière détectée (EMA 50 > EMA 200)")
        return "BULLISH"
    elif last_ema_50 < last_ema_200:
        print("📉 Tendance baissière détectée (EMA 50 < EMA 200)")
        return "BEARISH"
    print("⚖️ Tendance neutre")
    return "NEUTRAL"

# === Calcul des seuils dynamiques ===
def calculate_dynamic_thresholds(df, market="GOLD"):
    print("🧠 Calcul des seuils dynamiques...")
    avg_atr = df["atr"].mean()
    atr_quantile = df["atr"].quantile(0.75)

    # Seuils RSI ajustés
    if avg_atr > atr_quantile:
        rsi_buy_threshold, rsi_sell_threshold = 60, 60  # Forte volatilité
    else:
        rsi_buy_threshold, rsi_sell_threshold = 60, 55  # Faible volatilité

    # Seuil ADX
    adx_threshold = 15 if market == "GOLD" else 10

    # Seuils fixes pour la variation de prix
    drop_threshold = 1.3  # Baisse de 1,3% pour achat
    rise_threshold = 1.3  # Hausse de 1,3% pour vente

    print(f"📊 Seuils dynamiques : RSI Buy={rsi_buy_threshold}, RSI Sell={rsi_sell_threshold}, "
          f"ADX={adx_threshold}, Drop={drop_threshold:.2f}%, Rise={rise_threshold:.2f}%")
    return rsi_buy_threshold, rsi_sell_threshold, adx_threshold, drop_threshold, rise_threshold

# === Détection des signaux d'achat/vente ===
def detect_signals(df, df_trend, periods_back=60, confirmation_period=2, atr_threshold=4.0, market="GOLD"):
    print(f"🧠 Analyse des prix pour détecter des signaux (confirmation : {confirmation_period} bougies)...")
    if len(df) < periods_back + confirmation_period - 1:
        print(f"⚠️ Pas assez de données ({len(df)} lignes, requis {periods_back + confirmation_period - 1})")
        df["buy_signal"] = False
        df["sell_signal"] = False
        return df

    # Calcul des indicateurs
    df = calculate_rsi(df, period=14)
    df = calculate_adx(df, period=14)
    df = calculate_atr(df, period=14)
    df = calculate_macd(df)

    if df[["rsi", "adx", "atr", "macd", "macd_signal"]].isnull().any().any():
        print("⚠️ Données des indicateurs invalides, aucun signal généré")
        df["buy_signal"] = False
        df["sell_signal"] = False
        return df

    # Calcul des seuils dynamiques
    rsi_buy_threshold, rsi_sell_threshold, adx_threshold, drop_threshold, rise_threshold = calculate_dynamic_thresholds(df, market)

    df["price_60min_ago"] = df["close"].shift(periods_back)
    df["pct_change"] = (df["close"] - df["price_60min_ago"]) / df["price_60min_ago"] * 100
    df["avg_volume_50"] = df["volume"].rolling(window=50).mean()

    df["buy_signal"] = False
    df["sell_signal"] = False
    trend = detect_trend(df_trend)

    # Vérification de la volatilité
    avg_atr = df["atr"].mean()
    current_atr = df["atr"].iloc[-1]
    if current_atr > atr_threshold * avg_atr:
        print(f"⚠️ Volatilité trop élevée (ATR={current_atr:.2f}, Moyenne={avg_atr:.2f})")
        return df

    # Vérification des signaux
    i = len(df) - confirmation_period
    macd_diff = abs(df["macd"].iloc[i + confirmation_period - 1] - df["macd_signal"].iloc[i + confirmation_period - 1])
    macd_threshold = 0.5 * df["atr"].iloc[i + confirmation_period - 1]
    buy_conditions = (
        all(df["pct_change"].iloc[i:i + confirmation_period] <= -drop_threshold) and  # Baisse de 1,3%
        df["rsi"].iloc[i + confirmation_period - 1] <= rsi_buy_threshold and
        df["adx"].iloc[i + confirmation_period - 1] >= adx_threshold and
        macd_diff <= macd_threshold and
        trend in ["BULLISH", "NEUTRAL"]
    )
    sell_conditions = (
        all(df["pct_change"].iloc[i:i + confirmation_period] >= rise_threshold) and  # Hausse de 1,3%
        df["rsi"].iloc[i + confirmation_period - 1] >= rsi_sell_threshold and
        df["adx"].iloc[i + confirmation_period - 1] >= adx_threshold and
        macd_diff <= macd_threshold and
        (trend in ["BEARISH", "NEUTRAL"] or (trend == "BULLISH" and df["rsi"].iloc[i + confirmation_period - 1] > 65))
    )

    if buy_conditions:
        df.iloc[i + confirmation_period - 1, df.columns.get_loc("buy_signal")] = True
        print(f"🔔 Signal d'achat confirmé : RSI={df['rsi'].iloc[i + confirmation_period - 1]:.2f}, "
              f"ADX={df['adx'].iloc[i + confirmation_period - 1]:.2f}, "
              f"MACD Diff={macd_diff:.2f}")
    if sell_conditions:
        df.iloc[i + confirmation_period - 1, df.columns.get_loc("sell_signal")] = True
        print(f"🔔 Signal de vente confirmé : RSI={df['rsi'].iloc[i + confirmation_period - 1]:.2f}, "
              f"ADX={df['adx'].iloc[i + confirmation_period - 1]:.2f}, "
              f"MACD Diff={macd_diff:.2f}")

    last_row = df.iloc[-1]
    buy_reasons = []
    sell_reasons = []

    # Raisons pour l'absence de signal d'achat
    if not last_row["buy_signal"]:
        if df["pct_change"].iloc[-1] > -drop_threshold:
            if df["pct_change"].iloc[-1] > 0:
                buy_reasons.append(f"Le prix a augmenté de {df['pct_change'].iloc[-1]:.2f}%, besoin d'une baisse d'au moins {drop_threshold:.2f}%")
            elif abs(df['pct_change'].iloc[-1]) < 0.01:
                buy_reasons.append(f"Le prix est stable, besoin d'une baisse d'au moins {drop_threshold:.2f}%")
            else:
                buy_reasons.append(f"Le prix n'a baissé que de {abs(df['pct_change'].iloc[-1]):.2f}%, besoin d'une baisse d'au moins {drop_threshold:.2f}%")
        if df["rsi"].iloc[-1] > rsi_buy_threshold:
            buy_reasons.append(f"RSI à {df['rsi'].iloc[-1]:.2f}, trop haut (besoin de ≤ {rsi_buy_threshold})")
        if df["adx"].iloc[-1] < adx_threshold:
            buy_reasons.append(f"ADX à {df['adx'].iloc[-1]:.2f}, insuffisant (besoin de ≥ {adx_threshold})")
        if macd_diff > macd_threshold:
            buy_reasons.append(f"MACD diff ({macd_diff:.2f}) trop éloigné du signal (besoin de ≤ {macd_threshold:.2f})")
        if trend not in ["BULLISH", "NEUTRAL"]:
            buy_reasons.append(f"Tendance baissière ({trend}), pas favorable pour acheter")
        if current_atr > atr_threshold * avg_atr:
            buy_reasons.append(f"ATR à {current_atr:.2f}, volatilité trop élevée (besoin de ≤ {atr_threshold * avg_atr:.2f})")

    # Raisons pour l'absence de signal de vente
    if not last_row["sell_signal"]:
        if df["pct_change"].iloc[-1] < rise_threshold:
            if df["pct_change"].iloc[-1] < 0:
                sell_reasons.append(f"Le prix a baissé de {abs(df['pct_change'].iloc[-1]):.2f}%, besoin d'une hausse d'au moins {rise_threshold:.2f}%")
            elif abs(df['pct_change'].iloc[-1]) < 0.01:
                sell_reasons.append(f"Le prix est stable, besoin d'une hausse d'au moins {rise_threshold:.2f}%")
            else:
                sell_reasons.append(f"Le prix n'a augmenté que de {df['pct_change'].iloc[-1]:.2f}%, besoin d'une hausse d'au moins {rise_threshold:.2f}%")
        if df["rsi"].iloc[-1] < rsi_sell_threshold:
            sell_reasons.append(f"RSI à {df['rsi'].iloc[-1]:.2f}, trop bas (besoin de ≥ {rsi_sell_threshold})")
        if df["adx"].iloc[-1] < adx_threshold:
            sell_reasons.append(f"ADX à {df['adx'].iloc[-1]:.2f}, insuffisant (besoin de ≥ {adx_threshold})")
        if macd_diff > macd_threshold:
            sell_reasons.append(f"MACD diff ({macd_diff:.2f}) trop éloigné du signal (besoin de ≤ {macd_threshold:.2f})")
        if trend not in ["BEARISH", "NEUTRAL"] and not (trend == "BULLISH" and df["rsi"].iloc[-1] > 65):
            sell_reasons.append(f"Tendance haussière ({trend}) avec RSI {df['rsi'].iloc[-1]:.2f}, pas favorable pour vendre (besoin de RSI > 65)")
        if current_atr > atr_threshold * avg_atr:
            sell_reasons.append(f"ATR à {current_atr:.2f}, volatilité trop élevée (besoin de ≤ {atr_threshold * avg_atr:.2f})")

    print(f"📊 Résumé : Prix={last_row['close']:.2f}, RSI={last_row['rsi']:.2f}, ADX={last_row['adx']:.2f}, "
          f"MACD Diff={macd_diff:.2f}, ATR={last_row['atr']:.2f}, Tendance={trend}, Achat={last_row['buy_signal']}, Vente={last_row['sell_signal']}")
    if buy_reasons or sell_reasons:
        print("ℹ️ Pourquoi aucun signal n'a été généré :")
        if buy_reasons:
            print("  - Pas de signal d'achat :")
            for reason in buy_reasons:
                print(f"    * {reason}")
        if sell_reasons:
            print("  - Pas de signal de vente :")
            for reason in sell_reasons:
                print(f"    * {reason}")
    return df

# === Vérification des positions ouvertes ===
def is_position_open(headers, epic):
    print(f"🔍 Vérification des positions ouvertes pour {epic}...")
    url = f"{BASE_URL}/api/v1/positions"
    response = safe_request("GET", url, headers=headers)
    if response:
        try:
            positions = response.json().get("positions", [])
            for pos in positions:
                if pos["market"]["epic"] == epic:
                    position = pos["position"]
                    if "dealId" not in position:
                        print(f"❌ Clé 'dealId' manquante pour la position {epic}")
                        continue
                    deal_id = position["dealId"]
                    size = position.get("size", position.get("dealSize", position.get("contractSize", 0.0)))
                    if size == 0.0:
                        print(f"⚠️ Taille de position non trouvée pour {epic}")
                        continue
                    open_level = position.get("openLevel", position.get("level", position.get("entryPrice", 0.0)))
                    stop_level = position.get("stopLevel", None)
                    limit_level = position.get("profitLevel", None)

                    stored_position = open_positions.get(deal_id, {})
                    if stop_level is None and "stop_level" in stored_position:
                        stop_level = stored_position["stop_level"]
                    if limit_level is None and "limit_level" in stored_position:
                        limit_level = stored_position["limit_level"]

                    stop_loss_percentage = abs(stop_level - open_level) / open_level * 100 if stop_level is not None else 0.0
                    take_profit_percentage = abs(limit_level - open_level) / open_level * 100 if limit_level is not None else 0.0

                    stop_display = f"{stop_level:.2f} ({stop_loss_percentage:.2f}%)" if stop_level is not None else "Non défini (0.00%)"
                    limit_display = f"{limit_level:.2f} ({take_profit_percentage:.2f}%)" if limit_level is not None else "Non défini (0.00%)"
                    print(f"📍 Position ouverte : {position['direction']} (Taille: {size:.4f}, Prix d'entrée: {open_level:.2f}, Stop-loss: {stop_display}, Take-profit: {limit_display})")

                    return {
                        "direction": position["direction"],
                        "size": size,
                        "deal_id": deal_id,
                        "open_level": open_level,
                        "stop_level": stop_level,
                        "limit_level": limit_level,
                        "stop_loss_percentage": stop_loss_percentage,
                        "take_profit_percentage": take_profit_percentage
                    }
        except KeyError as e:
            print(f"❌ Erreur dans la réponse de l'API : {e}")
            return None
    print("ℹ️ Aucune position ouverte")
    return None

# === Fermeture d'une position ===
def close_position(headers, deal_id, direction, size, current_price):
    print(f"🔐 Tentative de fermeture de la position (ID: {deal_id})...")
    if not deal_id or deal_id == "unknown":
        print("❌ Deal ID invalide, impossible de fermer la position")
        return False
    url = f"{BASE_URL}/api/v1/positions/{deal_id}"
    payload = {
        "direction": "SELL" if direction == "BUY" else "BUY",
        "size": size,
        "orderType": "MARKET",
        "level": current_price
    }
    print(f"📤 Envoi de la requête pour fermer : direction={payload['direction']}, taille={size:.4f}")
    response = safe_request("DELETE", url, headers=headers, json=payload)
    if response and isinstance(response, dict) and response.get("error") == "INVALID_DEAL_ID":
        print(f"❌ Deal ID {deal_id} invalide pour la fermeture")
        return False
    if response and response.ok:
        print(f"✅ Position fermée avec succès (ID: {deal_id})")
        if deal_id in open_positions:
            position = open_positions[deal_id]
            entry_price = position["open_level"]
            size = position["size"]
            profit_loss = (current_price - entry_price) * size if direction == "BUY" else (entry_price - current_price) * size
            profit_loss -= SPREAD_COST * size
            log_trade(deal_id, direction, size, entry_price, position["stop_level"], position["limit_level"], "CLOSED", current_price, profit_loss)
            del open_positions[deal_id]
            print(f"🗑️ Position {deal_id} supprimée de open_positions")
        return True
    print(f"❌ Échec de la fermeture de la position (ID: {deal_id})")
    return False

# === Passage d'un ordre ===
def place_order(headers, epic, direction, entry_price, df):
    print(f"📝 Préparation d'un ordre {direction} pour {epic}...")
    if not verify_price_stability(headers, epic, entry_price):
        print(f"❌ Ordre {direction} annulé : prix instable")
        return None, None, None, None, None

    available_balance = get_available_balance(headers)
    print(f"💸 Solde disponible pour l'ordre : {available_balance} EUR")
    margin_factor, min_size, max_size = get_margin_requirement(headers, epic)

    atr = df["atr"].iloc[-1]
    stop_loss_distance = atr * 2.0  # Base stop-loss distance
    take_profit_distance = atr * 2.0

    # Fetch market details to get stop-loss constraints
    url = f"{BASE_URL}/api/v1/markets/{epic}"
    response = safe_request("GET", url, headers=headers)
    max_stop_loss_distance = None
    if response and response.ok:
        data = response.json()
        max_stop_loss_distance = data.get("maxStopLossDistance", None)
        if max_stop_loss_distance is None:
            max_stop_loss_distance = entry_price * 0.02  # Default to 2%
        print(f"📋 Contrainte stop-loss max : {max_stop_loss_distance:.2f}")
    else:
        max_stop_loss_distance = entry_price * 0.02  # Fallback
        print(f"⚠️ Impossible de récupérer maxStopLossDistance, utilisation de la valeur par défaut : {max_stop_loss_distance:.2f}")

    # Cap stop-loss distance to respect platform constraints
    stop_loss_distance = min(stop_loss_distance, max_stop_loss_distance)

    if direction == "BUY":
        stop = entry_price - stop_loss_distance
        limit = entry_price + take_profit_distance
    else:
        stop = entry_price + stop_loss_distance
        limit = entry_price - take_profit_distance

    risk_amount = available_balance * RISK_PER_TRADE
    stop_loss_points = abs(entry_price - stop)
    size = risk_amount / stop_loss_points
    size = min(size, max_size)
    size = max(size, min_size)
    size = round(size, 4)

    stop_loss_percentage = abs(stop - entry_price) / entry_price * 100
    take_profit_percentage = abs(limit - entry_price) / entry_price * 100

    url = f"{BASE_URL}/api/v1/positions"
    payload = {
        "epic": epic,
        "direction": direction,
        "size": size,
        "stopLevel": round(stop, 2),
        "limitLevel": round(limit, 2),
        "orderType": "MARKET",
        "guaranteedStop": True
    }
    print(f"📤 Envoi de l'ordre {direction} : Prix={entry_price:.2f}, Stop={stop:.2f}, Limit={limit:.2f}, Taille={size:.4f}")
    print(f"📊 Perte potentielle (stop-loss) : {stop_loss_percentage:.2f}%")
    print(f"📊 Gain potentiel (take-profit) : {take_profit_percentage:.2f}%")

    response = safe_request("POST", url, headers=headers, json=payload)

    if response and isinstance(response, dict) and response.get("error") == "INVALID_STOP_LOSS":
        max_stop_value = response.get("max_stop_value")
        print(f"⚠️ Stop-loss trop éloigné, max autorisé : {max_stop_value:.2f}")
        # Adjust stop-loss to the maximum allowed
        if direction == "BUY":
            stop = max_stop_value
            stop_loss_distance = entry_price - stop
        else:
            stop = max_stop_value
            stop_loss_distance = stop - entry_price
        # Recalculate size based on new stop-loss
        stop_loss_points = abs(entry_price - stop)
        size = risk_amount / stop_loss_points
        size = min(size, max_size)
        size = max(size, min_size)
        size = round(size, 4)
        # Update payload with new stop-loss and size
        payload["stopLevel"] = round(stop, 2)
        payload["size"] = size
        stop_loss_percentage = abs(stop - entry_price) / entry_price * 100
        print(f"🔄 Réessai avec stop-loss ajusté : Stop={stop:.2f}, Taille={size:.4f}")
        print(f"📊 Perte potentielle ajustée (stop-loss) : {stop_loss_percentage:.2f}%")
        response = safe_request("POST", url, headers=headers, json=payload)

    if isinstance(response, dict) and "error" in response:
        error = response["error"]
        if error == "INSUFFICIENT_FUNDS":
            print("❌ Échec : Fonds insuffisants pour passer l'ordre")
        elif error == "MARKET_CLOSED":
            print("❌ Échec : Marché fermé")
        elif error == "INVALID_STOP_LOSS":
            print(f"❌ Échec : Stop-loss invalide même après ajustement")
        else:
            print(f"❌ Échec de l'ordre {direction} : {error}")
        return None, None, None, None, None

    if response and response.ok:
        try:
            response_data = response.json()
            print(f"📝 Réponse de l'API : {response_data}")
            deal_id = response_data.get("dealId")
            if not deal_id:
                print("❌ Échec : Aucun dealId retourné par l'API")
                return None, None, None, None, None
            print(f"✅ Ordre {direction} passé avec succès (Taille: {size:.4f}, Deal ID: {deal_id})")
            if not verify_order_execution(headers, deal_id, entry_price, direction, size):
                print(f"❌ Ordre {deal_id} annulé : problème d'exécution")
                close_position(headers, deal_id, direction, size, entry_price)
                return None, None, None, None, None

            open_positions[deal_id] = {
                "direction": direction,
                "size": size,
                "open_level": entry_price,
                "stop_level": stop,
                "limit_level": limit,
                "stop_loss_percentage": stop_loss_percentage,
                "take_profit_percentage": take_profit_percentage
            }
            log_trade(deal_id, direction, size, entry_price, stop, limit, "OPEN")
            print(f"💾 Position {deal_id} enregistrée dans open_positions")
            return response, stop, limit, stop_loss_percentage, take_profit_percentage
        except ValueError as e:
            print(f"❌ Erreur lors de l'analyse de la réponse JSON : {e}")
            return None, None, None, None, None

    print(f"❌ Échec de l'ordre {direction}. Réponse : {response.text if response else 'Aucune réponse'}")
    return None, None, None, None, None

# === Boucle principale du bot ===
def trading_bot(
    check_interval=60,
    confirmation_period=2,
    atr_threshold=4.0,
    market="GOLD"
):
    print("🚀 Démarrage du bot de trading...")
    auth_headers = authenticate()
    if not auth_headers:
        print("❌ Impossible de démarrer : échec de la connexion")
        return
    epic = search_market(auth_headers, market)
    if not epic:
        print(f"❌ Impossible de démarrer : aucun marché {market} trouvé")
        return
    print(f"📌 Bot configuré pour trader sur : {epic}")

    while True:
        try:
            print("🔄 Nouvelle itération du bot...")
            if not is_market_open():
                print(f"⏳ Marché fermé, pause de {check_interval * 2} secondes")
                time.sleep(check_interval * 2)
                continue

            df = get_candles(auth_headers, epic, resolution="MINUTE_15")
            df_trend = get_trend_candles(auth_headers, epic, resolution="HOUR_4")
            if df is not None and df_trend is not None:
                df = detect_signals(
                    df,
                    df_trend,
                    periods_back=60,
                    confirmation_period=confirmation_period,
                    atr_threshold=atr_threshold,
                    market=market
                )
                current_price = df["close"].iloc[-1]
                signal = df.iloc[-1]
                print(f"🔔 Résultat : Prix={current_price:.2f}, Achat={signal['buy_signal']}, Vente={signal['sell_signal']}")

                position = is_position_open(auth_headers, epic)
                if signal["buy_signal"]:
                    print("🔍 Signal d'achat détecté pour la dernière ligne")
                    if position and position["direction"] == "BUY":
                        print("ℹ️ Position d'achat déjà ouverte, rien à faire")
                        continue
                    if position:
                        print("🔄 Fermeture de la position de vente pour ouvrir un achat")
                        close_position(auth_headers, position["deal_id"], position["direction"], position["size"], current_price)
                    print(f"📈 Passage d'un ordre d'achat à {current_price:.2f}")
                    response, stop, limit, stop_loss_percentage, take_profit_percentage = place_order(auth_headers, epic, "BUY", current_price, df)
                    if response:
                        print(f"✅ Position d'achat ouverte pour {market} à {current_price:.2f}")
                        print(f"📊 Take-profit : {limit:.2f} ({take_profit_percentage:.2f}%)")
                        print(f"📊 Stop-loss : {stop:.2f} ({stop_loss_percentage:.2f}%)")
                elif signal["sell_signal"]:
                    print("🔍 Signal de vente détecté pour la dernière ligne")
                    if position and position["direction"] == "SELL":
                        print("ℹ️ Position de vente déjà ouverte, rien à faire")
                        continue
                    if position:
                        print("🔄 Fermeture de la position d'achat pour ouvrir une vente")
                        close_position(auth_headers, position["deal_id"], position["direction"], position["size"], current_price)
                    print(f"📉 Passage d'un ordre de vente à {current_price:.2f}")
                    response, stop, limit, stop_loss_percentage, take_profit_percentage = place_order(auth_headers, epic, "SELL", current_price, df)
                    if response:
                        print(f"✅ Position de vente ouverte pour {market} à {current_price:.2f}")
                        print(f"📊 Take-profit : {limit:.2f} ({take_profit_percentage:.2f}%)")
                        print(f"📊 Stop-loss : {stop:.2f} ({stop_loss_percentage:.2f}%)")
                else:
                    print("ℹ️ Aucun signal confirmé, pas d'action")
            print(f"⏳ Pause de {check_interval} secondes avant la prochaine analyse")
            time.sleep(check_interval)
        except KeyboardInterrupt:
            print("🛑 Bot arrêté par l'utilisateur")
            break
        except Exception as e:
            print(f"❌ Erreur inattendue : {e}")
            print(f"⏳ Pause de {check_interval * 2} secondes avant de réessayer")
            time.sleep(check_interval * 2)

# === Lancement du bot ===
if __name__ == "__main__":
    print("🎉 Lancement du bot de trading pour Gold")
    trading_bot(
        check_interval=60,
        confirmation_period=2,
        atr_threshold=4.0,
        market="GOLD"
    )