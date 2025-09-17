import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import pytz
import uuid
import os
import itertools  # Added for grid search in optimization

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
MAX_SLIPPAGE_PERCENT = 0.5  # Slippage maximum toléré
PRICE_TOLERANCE_PERCENT = 0.3  # Tolérance pour la vérification du prix
CHECK_INTERVAL = 60  # Intervalle entre les analyses (secondes)
CONFIRMATION_PERIOD = 2  # Nombre de bougies pour confirmation
ATR_THRESHOLD = 4.0  # Seuil de volatilité
MARKET = "GOLD"
SL_ATR_MULTIPLIER = 2.0  # Multiplicateur pour stop-loss dynamique basé sur ATR
TP_ATR_MULTIPLIER = 1.5  # Multiplicateur pour take-profit dynamique basé sur ATR
MINIMUM_BALANCE_BUFFER = 1.0  # Buffer réduit à 1 EUR
MAX_ORDER_RETRIES = 3  # Nombre de tentatives pour passer un ordre
LEVERAGE = 20  # Levier de 20:1
MARGIN_FACTOR = 1 / LEVERAGE  # Marge = 5% pour un levier de 20:1
RISK_PER_TRADE = 0.8  # Limite l'exposition à 80% du solde par position
MAX_SPREAD_COST = 2.0  # Limite maximale pour spread_cost
STOP_LOSS_PERCENT = 1.10  # Fallback pour stop-loss (en pourcentage)
TAKE_PROFIT_PERCENT = 0.5  # Fallback pour take-profit (en pourcentage)

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
                            return {"error": "INVALID_STOP_LOSS_MAX", "max_stop_value": max_stop_value}
                        except (IndexError, ValueError):
                            print("❌ Impossible d'extraire max_stop_value de l'erreur")
                            return {"error": "INVALID_STOP_LOSS_MAX"}
                    elif "error.invalid.stoploss.minvalue" in error_message:
                        try:
                            min_stop_value = float(error_message.split(": ")[1])
                            return {"error": "INVALID_STOP_LOSS_MIN", "min_stop_value": min_stop_value}
                        except (IndexError, ValueError):
                            print("❌ Impossible d'extraire min_stop_value de l'erreur")
                            return {"error": "INVALID_STOP_LOSS_MIN"}
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
    url = f"{BASE_URL}/api/v1/accounts"
    response = safe_request("GET", url, headers=headers)
    if response:
        accounts = response.json().get("accounts", [])
        if accounts:
            balance = accounts[0].get("balance", {}).get("available", 0.0)
            print(f"💰 Solde disponible : {balance:.2f} EUR")
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
        min_size = round(data.get("minimumDealSize", 0.01), 2)
        max_size = round(data.get("maximumDealSize", 100.0), 2)
        print(f"📋 Règles : Marge={MARGIN_FACTOR*100}%, Taille min={min_size:.2f}, Taille max={max_size:.2f}")
        return MARGIN_FACTOR, min_size, max_size
    print(f"❌ Erreur lors de la récupération des règles pour {epic}")
    return MARGIN_FACTOR, 0.01, 100.0

# === Récupération du prix actuel et du spread ===
def get_current_price(headers, epic):
    print(f"📈 Récupération du prix actuel pour {epic}...")
    url = f"{BASE_URL}/api/v1/prices/{epic}"
    params = {"resolution": "MINUTE", "max": 1}
    response = safe_request("GET", url, headers=headers, params=params)
    if response:
        prices = response.json().get("prices", [])
        if prices and "closePrice" in prices[0]:
            bid = prices[0]["closePrice"].get("bid", None)
            ask = prices[0]["closePrice"].get("ask", None)
            if bid is not None and ask is not None:
                spread = min(ask - bid, MAX_SPREAD_COST)  # Limiter le spread
                print(f"📈 Prix actuel (bid) : {bid:.2f}, Spread : {spread:.2f}")
                return bid, spread
        print("⚠️ Données de prix invalides")
    print("❌ Erreur lors de la récupération du prix actuel")
    return None, None

# === Vérification du prix avant ordre ===
def verify_price_stability(headers, epic, expected_price):
    print(f"🔍 Vérification de la stabilité du prix pour {epic}...")
    current_price, _ = get_current_price(headers, epic)
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
def verify_order_execution(headers, deal_ref, expected_price, expected_size):
    print(f"🔍 Vérification de l'exécution de l'ordre (Référence: {deal_ref})...")
    if not deal_ref:
        print("❌ Référence de deal invalide, impossible de vérifier l'exécution")
        return False, None, None
    url = f"{BASE_URL}/api/v1/confirms/{deal_ref}"
    time.sleep(2)  # Attendre que l'API traite l'ordre
    response = safe_request("GET", url, headers=headers)
    if response and isinstance(response, dict) and response.get("error") == "INVALID_DEAL_ID":
        print(f"❌ Référence de deal {deal_ref} invalide selon l'API")
        return False, None, None
    if response and response.ok:
        confirmation = response.json()
        print(f"📝 Réponse complète de confirmation : {confirmation}")
        status = confirmation.get("status", "UNKNOWN")
        executed_price = confirmation.get("level", None)
        deal_id = confirmation.get("dealId", None)
        affected_deals = confirmation.get("affectedDeals", [])
        reason = confirmation.get("reason", "No reason provided")
        executed_size = confirmation.get("size", None)  # Utiliser size au niveau racine
        if affected_deals and "size" in affected_deals[0]:
            executed_size = affected_deals[0].get("size", None)
            if executed_size is not None and abs(executed_size - expected_size) > 0.01:
                print(f"⚠️ Ajustement de taille par l'API : Taille demandée={expected_size:.2f}, Taille exécutée={executed_size:.2f}")
        if executed_size is None:
            print(f"⚠️ Taille exécutée non trouvée dans affectedDeals, utilisation de la taille racine : {confirmation.get('size', 'inconnue')}")
            executed_size = confirmation.get("size", expected_size)  # Fallback sur expected_size
        if affected_deals:
            deal_id = affected_deals[0].get("dealId", deal_id)
        if status == "OPEN" and executed_price is not None and deal_id:
            slippage_percent = abs(executed_price - expected_price) / expected_price * 100
            if slippage_percent > MAX_SLIPPAGE_PERCENT:
                print(f"⚠️ Slippage excessif : {slippage_percent:.2f}% (max : {MAX_SLIPPAGE_PERCENT}%)")
                return False, None, None
            print(f"✅ Ordre exécuté correctement à {executed_price:.2f} (slippage : {slippage_percent:.2f}%, taille exécutée : {executed_size:.2f})")
            return True, deal_id, executed_price
        print(f"⚠️ Statut de l'ordre non ouvert : {status}, Raison : {reason}")
        return False, None, None
    print(f"❌ Échec de la vérification de l'ordre (Référence: {deal_ref})")
    return False, None, None

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
        print(f"📈 Tendance haussière détectée (EMA 50={last_ema_50:.2f} > EMA 200={last_ema_200:.2f})")
        return "BULLISH"
    elif last_ema_50 < last_ema_200:
        print(f"📉 Tendance baissière détectée (EMA 50={last_ema_50:.2f} < EMA 200={last_ema_200:.2f})")
        return "BEARISH"
    print(f"⚖️ Tendance neutre (EMA 50={last_ema_50:.2f} ≈ EMA 200={last_ema_200:.2f})")
    return "NEUTRAL"

# === Calcul des seuils dynamiques ===
def calculate_dynamic_thresholds(df, market="GOLD"):
    print("🧠 Calcul des seuils dynamiques pour la stratégie...")
    avg_atr = df["atr"].mean()
    atr_quantile = df["atr"].quantile(0.75)
    print(f"📊 ATR moyen={avg_atr:.2f}, ATR quantile 75%={atr_quantile:.2f}")
    if avg_atr > atr_quantile:
        rsi_buy_threshold, rsi_sell_threshold = 60, 60
    else:
        rsi_buy_threshold, rsi_sell_threshold = 60, 55
    adx_threshold = 15 if market == "GOLD" else 10
    drop_threshold = max(0.1, 0.03 * avg_atr / df["close"].iloc[-1] * 100)
    rise_threshold = max(0.1, 0.03 * avg_atr / df["close"].iloc[-1] * 100)
    print(f"📊 Seuils dynamiques calculés : RSI Buy={rsi_buy_threshold}, RSI Sell={rsi_sell_threshold}, "
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
    df = calculate_rsi(df, period=14)
    df = calculate_adx(df, period=14)
    df = calculate_atr(df, period=14)
    df = calculate_macd(df)
    if df[["rsi", "adx", "atr", "macd", "macd_signal"]].isnull().any().any():
        print("⚠️ Données des indicateurs invalides, aucun signal généré")
        df["buy_signal"] = False
        df["sell_signal"] = False
        return df
    rsi_buy_threshold, rsi_sell_threshold, adx_threshold, drop_threshold, rise_threshold = calculate_dynamic_thresholds(df, market)
    df["price_60min_ago"] = df["close"].shift(periods_back)
    df["pct_change"] = (df["close"] - df["price_60min_ago"]) / df["price_60min_ago"] * 100
    df["avg_volume_50"] = df["volume"].rolling(window=50).mean()
    df["buy_signal"] = False
    df["sell_signal"] = False
    trend = detect_trend(df_trend)
    avg_atr = df["atr"].mean()
    current_atr = df["atr"].iloc[-1]
    print(f"📊 Vérification volatilité : ATR actuel={current_atr:.2f}, seuil max={atr_threshold * avg_atr:.2f}")
    if current_atr > atr_threshold * avg_atr:
        print(f"⚠️ Volatilité trop élevée (ATR={current_atr:.2f}, Moyenne={avg_atr:.2f})")
        return df
    i = len(df) - confirmation_period
    macd_diff = abs(df["macd"].iloc[i + confirmation_period - 1] - df["macd_signal"].iloc[i + confirmation_period - 1])
    macd_threshold = 0.5 * df["atr"].iloc[i + confirmation_period - 1]
    print(f"📊 MACD diff={macd_diff:.2f}, seuil MACD={macd_threshold:.2f}")
    buy_conditions = (
        all(df["pct_change"].iloc[i:i + confirmation_period] <= -drop_threshold) and
        df["rsi"].iloc[i + confirmation_period - 1] <= rsi_buy_threshold and
        df["adx"].iloc[i + confirmation_period - 1] >= adx_threshold and
        macd_diff <= macd_threshold and
        trend in ["BULLISH", "NEUTRAL"]
    )
    sell_conditions = (
        all(df["pct_change"].iloc[i:i + confirmation_period] >= rise_threshold) and
        df["rsi"].iloc[i + confirmation_period - 1] >= rsi_sell_threshold and
        df["adx"].iloc[i + confirmation_period - 1] >= adx_threshold and
        macd_diff <= macd_threshold and
        (
            trend in ["BEARISH", "NEUTRAL"] or
            (trend == "BULLISH" and df["rsi"].iloc[i + confirmation_period - 1] > 65)
        )
    )
    if buy_conditions:
        df.iloc[i + confirmation_period - 1, df.columns.get_loc("buy_signal")] = True
        print(f"🔔 Signal d'achat confirmé : RSI={df['rsi'].iloc[i + confirmation_period - 1]:.2f}, "
              f"ADX={df['adx'].iloc[i + confirmation_period - 1]:.2f}, "
              f"MACD Diff={macd_diff:.2f}, %Change={df['pct_change'].iloc[i + confirmation_period - 1]:.2f}%")
    if sell_conditions:
        df.iloc[i + confirmation_period - 1, df.columns.get_loc("sell_signal")] = True
        print(f"🔔 Signal de vente confirmé : RSI={df['rsi'].iloc[i + confirmation_period - 1]:.2f}, "
              f"ADX={df['adx'].iloc[i + confirmation_period - 1]:.2f}, "
              f"MACD Diff={macd_diff:.2f}, %Change={df['pct_change'].iloc[i + confirmation_period - 1]:.2f}%")
    last_row = df.iloc[-1]
    buy_reasons = []
    sell_reasons = []
    if not last_row["buy_signal"]:
        if not all(df["pct_change"].iloc[i:i + confirmation_period] <= -drop_threshold):
            buy_reasons.append(f"La baisse de prix n'a pas été confirmée sur {confirmation_period} bougies consécutives. "
                               f"Requis : baisse d'au moins {drop_threshold:.2f}% sur chaque bougie. "
                               f"Valeurs observées : {df['pct_change'].iloc[i:i + confirmation_period].values}")
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
    if not last_row["sell_signal"]:
        if not all(df["pct_change"].iloc[i:i + confirmation_period] >= rise_threshold):
            sell_reasons.append(f"La hausse de prix n'a pas été confirmée sur {confirmation_period} bougies consécutives. "
                                f"Requis : hausse d'au moins {rise_threshold:.2f}% sur chaque bougie. "
                                f"Valeurs observées : {df['pct_change'].iloc[i:i + confirmation_period].values}")
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
                    size = round(position.get("size", position.get("dealSize", position.get("contractSize", 0.0))), 2)
                    if size == 0.0:
                        print(f"⚠️ Taille de position non trouvée pour {epic}")
                        continue
                    open_level = position.get("openLevel", position.get("level", position.get("entryPrice", 0.0)))
                    stop_level = position.get("stopLevel")
                    limit_level = position.get("limitLevel")
                    stored_position = open_positions.get(deal_id, {})
                    if stop_level is None and "stop_level" in stored_position:
                        stop_level = stored_position["stop_level"]
                        print(f"ℹ️ Stop-level récupéré depuis open_positions : {stop_level:.2f}")
                    if limit_level is None and "limit_level" in stored_position:
                        limit_level = stored_position["limit_level"]
                        print(f"ℹ️ Limit-level récupéré depuis open_positions : {limit_level:.2f}")
                    if limit_level is None and "profit_distance" in stored_position and "direction" in stored_position:
                        profit_distance = stored_position["profit_distance"]
                        direction = stored_position["direction"]
                        if profit_distance is not None and open_level is not None:
                            if direction == "BUY":
                                limit_level = open_level + profit_distance
                            else:
                                limit_level = open_level - profit_distance
                            print(f"ℹ️ Limit-level recalculé à partir de profit_distance : {limit_level:.2f} (profit_distance={profit_distance:.2f})")
                    if limit_level is None:
                        print(f"⚠️ Aucun limit-level trouvé pour {deal_id}. Utilisation d'un fallback basé sur TAKE_PROFIT_PERCENT")
                        profit_distance = open_level * (TAKE_PROFIT_PERCENT / 100)
                        limit_level = open_level + profit_distance if position["direction"] == "BUY" else open_level - profit_distance
                        print(f"ℹ️ Fallback : Limit-level calculé à {limit_level:.2f} avec profit_distance={profit_distance:.2f}")
                    stop_loss_percentage = abs(stop_level - open_level) / open_level * 100 if stop_level is not None else 0.0
                    take_profit_percentage = abs(limit_level - open_level) / open_level * 100 if limit_level is not None else 0.0
                    stop_display = f"{stop_level:.2f} ({stop_loss_percentage:.2f}%)" if stop_level is not None else "Non défini (0.00%)"
                    limit_display = f"{limit_level:.2f} ({take_profit_percentage:.2f}%)" if limit_level is not None else "Non défini (0.00%)"
                    print(f"📍 Position ouverte : {position['direction']} (Taille: {size:.2f}, Prix d'entrée: {open_level:.2f}, Stop-loss: {stop_display}, Take-profit: {limit_display})")
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
def close_position(headers, deal_id, direction, size):
    print(f"🔐 Tentative de fermeture de la position (ID: {deal_id})...")
    if not deal_id:
        print("❌ Deal ID invalide, impossible de fermer la position")
        return False
    size = round(size, 2)
    url = f"{BASE_URL}/api/v1/positions/{deal_id}"
    payload = {
        "direction": "SELL" if direction == "BUY" else "BUY",
        "size": size,
        "orderType": "MARKET"
    }
    print(f"📤 Envoi de la requête pour fermer : direction={payload['direction']}, taille={size:.2f}")
    response = safe_request("DELETE", url, headers=headers, json=payload)
    if response and isinstance(response, dict) and response.get("error") == "INVALID_DEAL_ID":
        print(f"❌ Deal ID {deal_id} invalide pour la fermeture")
        return False
    if response and response.ok:
        print(f"✅ Position fermée avec succès (ID: {deal_id})")
        if deal_id in open_positions:
            del open_positions[deal_id]
            print(f"🗑️ Position {deal_id} supprimée de open_positions")
        return True
    print(f"❌ Échec de la fermeture de la position (ID: {deal_id})")
    return False

# === Passage d'un ordre ===
def place_order(headers, epic, direction, entry_price, df):
    print(f"📝 Préparation d'un ordre {direction} pour {epic} au prix d'entrée {entry_price:.2f}...")
    print(f"🔍 Étape 1 : Vérification de la stabilité du prix...")
    if not verify_price_stability(headers, epic, entry_price):
        print(f"❌ Ordre {direction} annulé : prix instable")
        return None, None, None, None, None
    print(f"🔍 Étape 2 : Récupération du solde disponible...")
    available_balance = get_available_balance(headers)
    print(f"💸 Solde disponible pour l'ordre : {available_balance:.2f} EUR")
    print(f"📈 Prix d'entrée utilisé : {entry_price:.2f}")
    print(f"🔍 Étape 3 : Récupération des exigences de marge...")
    _, min_size, max_size = get_margin_requirement(headers, epic)  # Ignorer margin_factor de l'API
    print(f"📋 Contraintes de taille : min_size={min_size:.2f}, max_size={max_size:.2f}")
    print(f"🔍 Étape 4 : Récupération des limites de stop/profit depuis l'API...")
    url = f"{BASE_URL}/api/v1/markets/{epic}"
    response = safe_request("GET", url, headers=headers)
    min_stop_distance = 0.0
    max_stop_distance = entry_price * 0.02  # 2% par défaut
    min_profit_distance = 0.0
    max_profit_distance = float('inf')
    if response and response.ok:
        data = response.json()
        min_stop_distance = float(data.get("minimumStopDistance", 0.0))
        max_stop_distance = float(data.get("maximumStopDistance", entry_price * 0.02))
        min_profit_distance = float(data.get("minimumProfitDistance", 0.0))
        max_profit_distance = float(data.get("maximumProfitDistance", float('inf')))
        print(f"📋 Limites API : Stop min={min_stop_distance:.1f}, Stop max={max_stop_distance:.1f}, "
              f"Profit min={min_profit_distance:.1f}, Profit max={max_profit_distance:.1f}")
    else:
        print(f"⚠️ Impossible de récupérer les limites, utilisation des valeurs par défaut")
    print(f"🔍 Étape 5 : Calcul des stop-loss et take-profit dynamiques basés sur l'ATR...")
    atr = float(df["atr"].iloc[-1])  # Convertir en float Python
    print(f"📊 ATR actuel = {atr:.1f}")
    stop_distance = atr * SL_ATR_MULTIPLIER
    profit_distance = atr * TP_ATR_MULTIPLIER
    print(f"📊 Calcul initial : Stop Distance={stop_distance:.1f} (ATR * {SL_ATR_MULTIPLIER}), Profit Distance={profit_distance:.1f} (ATR * {TP_ATR_MULTIPLIER})")
    # Ajuster stop_distance et profit_distance selon les limites de l'API, arrondi à 1 décimale
    stop_distance = round(max(min(stop_distance, max_stop_distance), min_stop_distance), 1)
    profit_distance = round(max(min(profit_distance, max_profit_distance), min_profit_distance), 1)
    print(f"📊 Après ajustement API : Stop Distance={stop_distance:.1f}, Profit Distance={profit_distance:.1f}")
    if profit_distance <= 0:
        print(f"❌ Erreur : profit_distance ({profit_distance:.1f}) est invalide ou nul")
        return None, None, None, None, None
    if direction == "BUY":
        stop_level = entry_price - stop_distance
        limit_level = entry_price + profit_distance
    else:
        stop_level = entry_price + stop_distance
        limit_level = entry_price - profit_distance
    stop_loss_percentage = (stop_distance / entry_price) * 100
    take_profit_percentage = (profit_distance / entry_price) * 100
    print(f"📊 Niveaux finaux : Stop-level={stop_level:.1f} ({stop_loss_percentage:.2f}%), Limit-level={limit_level:.1f} ({take_profit_percentage:.2f}%)")
    print(f"🔍 Étape 6 : Calcul de la taille de la position...")
    _, spread = get_current_price(headers, epic)
    spread_cost = min(spread if spread is not None else 0.1, MAX_SPREAD_COST)  # Limiter spread_cost
    print(f"📊 Spread utilisé : {spread_cost:.2f} (limité à {MAX_SPREAD_COST:.2f})")
    print(f"📊 Calcul de la taille pour utiliser {RISK_PER_TRADE*100}% du solde avec levier {LEVERAGE}:1...")
    size = (available_balance * RISK_PER_TRADE - MINIMUM_BALANCE_BUFFER) / (entry_price * MARGIN_FACTOR + spread_cost)
    print(f"📊 Taille calculée avant contraintes : {size:.4f}")
    size_before_constraints = size
    size = max(size, min_size)
    if size != size_before_constraints:
        print(f"⚠️ Taille ajustée de {size_before_constraints:.4f} à {size:.2f} en raison de min_size={min_size:.2f}")
    size = min(size, max_size)
    size = round(size, 2)
    required_margin = entry_price * size * MARGIN_FACTOR
    total_cost = required_margin + spread_cost * size + MINIMUM_BALANCE_BUFFER
    print(f"📊 Détail des coûts : Marge={required_margin:.2f}, Spread={spread_cost * size:.2f}, Buffer={MINIMUM_BALANCE_BUFFER:.2f}, Total={total_cost:.2f}")
    print(f"📊 Exposition totale : {entry_price * size:.2f} EUR")
    print(f"📊 Perte potentielle si stop-loss atteint : {stop_distance * size:.1f} EUR")
    if total_cost > available_balance:
        print(f"⚠️ Coût total ({total_cost:.2f} EUR) dépasse le solde disponible ({available_balance:.2f} EUR), réduction de la taille")
        size = (available_balance * RISK_PER_TRADE - MINIMUM_BALANCE_BUFFER) / (entry_price * MARGIN_FACTOR + spread_cost)
        size = max(size, min_size)
        size = min(size, max_size)
        size = round(size, 2)
        required_margin = entry_price * size * MARGIN_FACTOR
        total_cost = required_margin + spread_cost * size + MINIMUM_BALANCE_BUFFER
        print(f"📊 Détail des coûts ajustés : Marge={required_margin:.2f}, Spread={spread_cost * size:.2f}, Buffer={MINIMUM_BALANCE_BUFFER:.2f}, Total={total_cost:.2f}")
        print(f"📊 Exposition totale ajustée : {entry_price * size:.2f} EUR")
        print(f"📊 Perte potentielle ajustée : {stop_distance * size:.1f} EUR")
        if total_cost > available_balance or size < min_size:
            print(f"❌ Impossible d'ajuster la taille : Taille={size:.2f}, Coût total={total_cost:.2f} EUR")
            return None, None, None, None, None
    adjusted_stop_distance = stop_distance  # Stocker le stop_distance ajusté
    for attempt in range(MAX_ORDER_RETRIES):
        # Re-fetch current price and API limits before each attempt
        current_price, spread = get_current_price(headers, epic)
        if current_price is None:
            print(f"❌ Impossible de récupérer le prix actuel à la tentative {attempt + 1}")
            return None, None, None, None, None
        price_diff_percent = abs(current_price - entry_price) / entry_price * 100
        if price_diff_percent > PRICE_TOLERANCE_PERCENT:
            print(f"⚠️ Écart de prix trop important à la tentative {attempt + 1}: {price_diff_percent:.2f}% (tolérance max : {PRICE_TOLERANCE_PERCENT}%)")
            return None, None, None, None, None
        entry_price = float(current_price)  # Update entry_price and ensure float
        response = safe_request("GET", f"{BASE_URL}/api/v1/markets/{epic}", headers=headers)
        if response and response.ok:
            data = response.json()
            min_stop_distance = float(data.get("minimumStopDistance", 0.0))
            max_stop_distance = float(data.get("maximumStopDistance", entry_price * 0.02))
            print(f"📋 Limites API mises à jour : Stop min={min_stop_distance:.1f}, Stop max={max_stop_distance:.1f}")
        else:
            print(f"⚠️ Impossible de récupérer les limites à la tentative {attempt + 1}, utilisation de min_stop_distance={min_stop_distance:.1f}, max_stop_distance={max_stop_distance:.1f}")
        # Utiliser le stop_distance ajusté des tentatives précédentes
        stop_distance = round(max(min(adjusted_stop_distance, max_stop_distance), min_stop_distance), 1)
        profit_distance = round(max(min(atr * TP_ATR_MULTIPLIER, max_profit_distance), min_profit_distance), 1)
        if direction == "BUY":
            stop_level = entry_price - stop_distance
            limit_level = entry_price + profit_distance
        else:
            stop_level = entry_price + stop_distance
            limit_level = entry_price - profit_distance
        stop_loss_percentage = (stop_distance / entry_price) * 100
        take_profit_percentage = (profit_distance / entry_price) * 100
        payload = {
            "epic": epic,
            "direction": direction,
            "size": float(size),  # Convertir en float Python
            "orderType": "MARKET",
            "guaranteedStop": True,
            "trailingStop": False,
            "stopDistance": float(stop_distance),  # Convertir en float Python
            "profitDistance": float(profit_distance)  # Convertir en float Python
        }
        print(f"📤 Tentative {attempt + 1} - Envoi de l'ordre {direction} : Prix={entry_price:.2f}, Stop Distance={stop_distance:.1f}, Profit Distance={profit_distance:.1f}, Taille={size:.2f}")
        print(f"📊 Perte potentielle (stop-loss) : {stop_loss_percentage:.2f}%")
        print(f"📊 Gain potentiel (take-profit) : {take_profit_percentage:.2f}%")
        print(f"📝 Payload de l'ordre : {payload}")
        url = f"{BASE_URL}/api/v1/positions"
        response = safe_request("POST", url, headers=headers, json=payload)
        if response and isinstance(response, dict) and response.get("error") in ["INVALID_STOP_LOSS_MAX", "INVALID_STOP_LOSS_MIN"]:
            if response["error"] == "INVALID_STOP_LOSS_MAX":
                max_stop_value = response.get("max_stop_value")
                print(f"⚠️ Stop-loss trop éloigné, max autorisé : {max_stop_value:.2f}")
                if direction == "BUY":
                    adjusted_stop_distance = float(entry_price - max_stop_value)  # Calculer la distance
                    stop_level = float(max_stop_value)
                else:
                    adjusted_stop_distance = float(max_stop_value - entry_price)
                    stop_level = float(max_stop_value)
                # Appliquer une marge de sécurité (0.95) et arrondir à 1 décimale
                adjusted_stop_distance = round(adjusted_stop_distance * 0.95, 1)
            elif response["error"] == "INVALID_STOP_LOSS_MIN":
                min_stop_value = response.get("min_stop_value")
                print(f"⚠️ Stop-loss trop proche, min autorisé : {min_stop_value:.2f}")
                if direction == "BUY":
                    adjusted_stop_distance = float(min_stop_value - entry_price)  # Calculer la distance
                    stop_level = float(min_stop_value)
                else:
                    adjusted_stop_distance = float(min_stop_value - entry_price)
                    stop_level = float(min_stop_value)
                # Appliquer une marge de sécurité (1.05) et arrondir à 1 décimale
                adjusted_stop_distance = round(adjusted_stop_distance * 1.05, 1)
            if direction == "BUY":
                stop_level = entry_price - adjusted_stop_distance
            else:
                stop_level = entry_price + adjusted_stop_distance
            stop_loss_percentage = (adjusted_stop_distance / entry_price) * 100
            print(f"📊 Ajustement automatique : Stop Distance={adjusted_stop_distance:.1f}, Stop-level={stop_level:.1f} ({stop_loss_percentage:.2f}%)")
            # Vérifier si stop_distance est dans les limites
            if adjusted_stop_distance < min_stop_distance or adjusted_stop_distance > max_stop_distance:
                print(f"❌ Stop Distance {adjusted_stop_distance:.1f} hors limites (min={min_stop_distance:.1f}, max={max_stop_distance:.1f})")
                return None, None, None, None, None
            # Vérifier si l'ajustement est déjà optimisé
            if abs(adjusted_stop_distance - stop_distance) < 0.1:
                print(f"❌ Stop Distance {adjusted_stop_distance:.1f} déjà optimisé, abandon après {attempt + 1} tentatives")
                return None, None, None, None, None
            # Recalculer la taille pour respecter le risque
            size = (available_balance * RISK_PER_TRADE - MINIMUM_BALANCE_BUFFER) / (entry_price * MARGIN_FACTOR + spread_cost)
            print(f"📊 Taille recalculée après ajustement stop-loss : {size:.4f}")
            size = max(size, min_size)
            size = min(size, max_size)
            size = round(size, 2)
            required_margin = entry_price * size * MARGIN_FACTOR
            total_cost = required_margin + spread_cost * size + MINIMUM_BALANCE_BUFFER
            print(f"📊 Détail des coûts ajustés : Marge={required_margin:.2f}, Spread={spread_cost * size:.2f}, Buffer={MINIMUM_BALANCE_BUFFER:.2f}, Total={total_cost:.2f}")
            print(f"📊 Exposition totale ajustée : {entry_price * size:.2f} EUR")
            print(f"📊 Perte potentielle ajustée : {adjusted_stop_distance * size:.1f} EUR")
            if total_cost > available_balance or size < min_size:
                print(f"❌ Impossible d'ajuster la taille après correction du stop-loss : Taille={size:.2f}, Coût total={total_cost:.2f} EUR")
                return None, None, None, None, None
            if direction == "BUY":
                limit_level = entry_price + profit_distance
            else:
                limit_level = entry_price - profit_distance
            print(f"🔄 Réessai avec stop-loss ajusté : Stop Distance={adjusted_stop_distance:.1f}, Taille={size:.2f}")
            print(f"📊 Perte potentielle ajustée (stop-loss) : {stop_loss_percentage:.2f}%")
            continue
        if isinstance(response, dict) and "error" in response:
            error = response["error"]
            print(f"❌ Erreur API pour l'ordre : {response}")
            if error == "INSUFFICIENT_FUNDS":
                print(f"❌ Échec (tentative {attempt + 1}) : Fonds insuffisants")
                size *= 0.95  # Réduction progressive de 5%
                size = max(size, min_size)
                size = min(size, max_size)
                size = round(size, 2)
                required_margin = entry_price * size * MARGIN_FACTOR
                total_cost = required_margin + spread_cost * size + MINIMUM_BALANCE_BUFFER
                print(f"📊 Détail des coûts ajustés : Marge={required_margin:.2f}, Spread={spread_cost * size:.2f}, Buffer={MINIMUM_BALANCE_BUFFER:.2f}, Total={total_cost:.2f}")
                print(f"📊 Exposition totale ajustée : {entry_price * size:.2f} EUR")
                print(f"📊 Perte potentielle ajustée : {adjusted_stop_distance * size:.1f} EUR")
                if total_cost > available_balance or size < min_size:
                    print(f"❌ Impossible d'ajuster la taille : Taille={size:.2f}, Coût total={total_cost:.2f} EUR")
                    return None, None, None, None, None
                payload["size"] = float(size)
                print(f"🔄 Réessai avec taille réduite : Taille={size:.2f}")
                continue
            elif error == "MARKET_CLOSED":
                print("❌ Échec : Marché fermé")
                return None, None, None, None, None
            elif error == "RISK_CHECK":
                print(f"❌ Échec (tentative {attempt + 1}) : Rejet pour RISK_CHECK")
                size *= 0.95  # Réduction progressive de 5%
                size = max(size, min_size)
                size = min(size, max_size)
                size = round(size, 2)
                required_margin = entry_price * size * MARGIN_FACTOR
                total_cost = required_margin + spread_cost * size + MINIMUM_BALANCE_BUFFER
                print(f"📊 Détail des coûts ajustés : Marge={required_margin:.2f}, Spread={spread_cost * size:.2f}, Buffer={MINIMUM_BALANCE_BUFFER:.2f}, Total={total_cost:.2f}")
                print(f"📊 Exposition totale ajustée : {entry_price * size:.2f} EUR")
                print(f"📊 Perte potentielle ajustée : {adjusted_stop_distance * size:.1f} EUR")
                if total_cost > available_balance or size < min_size:
                    print(f"❌ Impossible d'ajuster la taille : Taille={size:.2f}, Coût total={total_cost:.2f} EUR")
                    return None, None, None, None, None
                payload["size"] = float(size)
                print(f"🔄 Réessai avec taille réduite pour RISK_CHECK : Taille={size:.2f}")
                continue
            else:
                print(f"❌ Échec de l'ordre {direction} : {error}")
                return None, None, None, None, None
        if response and response.ok:
            try:
                response_data = response.json()
                print(f"📝 Réponse de l'API pour l'ordre : {response_data}")
                deal_ref = response_data.get("dealReference")
                if not deal_ref:
                    print("❌ Échec : Aucun dealReference retourné par l'API")
                    return None, None, None, None, None
                print(f"🔍 Étape 7 : Vérification de l'exécution de l'ordre (dealRef={deal_ref})...")
                success, deal_id, executed_price = verify_order_execution(headers, deal_ref, entry_price, size)
                if not success:
                    print(f"❌ Ordre {deal_ref} annulé : problème d'exécution")
                    if deal_id:
                        print(f"🔄 Tentative de fermeture de l'ordre partiellement exécuté (deal_id={deal_id})...")
                        close_position(headers, deal_id, direction, size)
                    return None, None, None, None, None
                if limit_level is None or profit_distance is None:
                    print(f"⚠️ Avertissement : limit_level ou profit_distance non défini avant stockage")
                    profit_distance = round(float(atr * TP_ATR_MULTIPLIER), 1)  # Convertir en float et arrondir à 1 décimale
                    limit_level = entry_price + profit_distance if direction == "BUY" else entry_price - profit_distance
                    print(f"ℹ️ Recalcul de fallback : Limit-level={limit_level:.1f}, Profit Distance={profit_distance:.1f}")
                print(f"✅ Ordre {direction} passé avec succès (Taille: {size:.2f}, Deal ID: {deal_id})")
                print(f"💾 Enregistrement de la position {deal_id} : Stop={stop_level:.1f}, Limit={limit_level:.1f}, Profit Distance={profit_distance:.1f}")
                open_positions[deal_id] = {
                    "direction": direction,
                    "size": size,
                    "open_level": executed_price,
                    "stop_level": stop_level,
                    "limit_level": limit_level,
                    "stop_loss_percentage": stop_loss_percentage,
                    "take_profit_percentage": take_profit_percentage,
                    "profit_distance": profit_distance
                }
                return response, stop_level, limit_level, stop_loss_percentage, take_profit_percentage
            except ValueError as e:
                print(f"❌ Erreur lors de l'analyse de la réponse JSON : {e}")
                return None, None, None, None, None
        print(f"❌ Échec de l'ordre {direction}. Réponse complète : {response.text if response else 'Aucune réponse'}")
        return None, None, None, None, None
    print(f"❌ Échec après {MAX_ORDER_RETRIES} tentatives")
    return None, None, None, None, None

# === Fonction pour récupérer des données historiques étendues ===
def get_historical_candles(headers, epic, resolution="MINUTE_15", start_date=None, end_date=None, max_per_request=200):
    """
    Récupère des données historiques en plusieurs requêtes si nécessaire.
    Utilise des dates pour fetcher (l'API peut supporter fromDate/toDate si disponible, sinon simule avec limit).
    Note: Adapter si l'API supporte fromDate/toDate; ici on assume limit max, et on fetch en boucle si dates spécifiées.
    """
    print(f"📈 Récupération des données historiques pour {epic} de {start_date} à {end_date}")
    url = f"{BASE_URL}/api/v1/prices/{epic}"
    all_candles = []
    params = {"resolution": resolution, "max": max_per_request}
    if start_date and end_date:
        params["from"] = start_date  # Assumer format YYYY-MM-DD_HH:MM:SS (adapter si besoin)
        params["to"] = end_date
    response = safe_request("GET", url, headers=headers, params=params)
    if response:
        candles = response.json().get("prices", [])
        all_candles.extend(candles)
    # Si plus de données nécessaires, implémenter pagination si l'API le supporte (e.g., pageNumber)
    # Pour simplicité, assume un seul fetch; étendre si besoin.
    if not all_candles:
        return None
    df = pd.DataFrame(all_candles)
    df["timestamp"] = pd.to_datetime(df["snapshotTime"], utc=True).dt.tz_convert("Europe/Paris")
    df.set_index("timestamp", inplace=True)
    df["close"] = df["closePrice"].apply(lambda x: x["bid"])
    df["open"] = df["openPrice"].apply(lambda x: x["bid"])
    df["high"] = df["highPrice"].apply(lambda x: x["bid"])
    df["low"] = df["lowPrice"].apply(lambda x: x["bid"])
    df["volume"] = df["lastTradedVolume"]
    return df[["open", "high", "low", "close", "volume"]]

# === Fonction de Backtesting ===
def backtest_strategy(headers, epic, initial_balance=10000.0, start_date=None, end_date=None, resolution="MINUTE_15",
                      periods_back=60, confirmation_period=2, atr_threshold=4.0, market="GOLD",
                      sl_atr_multiplier=SL_ATR_MULTIPLIER, tp_atr_multiplier=TP_ATR_MULTIPLIER,
                      risk_per_trade=RISK_PER_TRADE):
    """
    Simule la stratégie sur des données historiques.
    - Fetch données historiques.
    - Applique detect_signals sur des fenêtres roulantes.
    - Simule trades: ouvre sur signal, ferme sur TP/SL ou fin.
    - Calcule métriques: PnL total, win rate, max drawdown, Sharpe ratio.
    """
    print("🔙 Démarrage du backtest...")
    df = get_historical_candles(headers, epic, resolution, start_date, end_date)
    if df is None or len(df) < 200:
        print("❌ Pas assez de données historiques pour backtest")
        return None

    df_trend = get_trend_candles(headers, epic, "HOUR_4", limit=50)  # Utilise trend récent; adapter pour historique si besoin

    balance = initial_balance
    equity = [initial_balance]
    trades = []
    position = None  # {'direction': 'BUY/SELL', 'entry_price': float, 'size': float, 'stop_level': float, 'limit_level': float}

    for i in range(max(periods_back + confirmation_period, 200), len(df)):  # Commencer après assez de données
        window_df = df.iloc[i - 200:i]  # Fenêtre roulante pour indicateurs
        window_df = detect_signals(window_df, df_trend, periods_back, confirmation_period, atr_threshold, market)
        signal = window_df.iloc[-1]

        current_price = signal['close']
        atr = window_df['atr'].iloc[-1]
        stop_distance = atr * sl_atr_multiplier
        profit_distance = atr * tp_atr_multiplier

        # Calcul taille simulée (simplifié, assume spread_cost fixe)
        spread_cost = 0.5  # Assumption
        size = (balance * risk_per_trade) / (current_price * MARGIN_FACTOR + spread_cost)

        if position:
            # Vérifier si hit SL ou TP
            if position['direction'] == 'BUY':
                if current_price <= position['stop_level']:
                    pnl = (position['stop_level'] - position['entry_price']) * position['size'] - spread_cost
                    balance += pnl
                    trades.append({'type': 'BUY_CLOSE_SL', 'pnl': pnl})
                    position = None
                elif current_price >= position['limit_level']:
                    pnl = (position['limit_level'] - position['entry_price']) * position['size'] - spread_cost
                    balance += pnl
                    trades.append({'type': 'BUY_CLOSE_TP', 'pnl': pnl})
                    position = None
            else:  # SELL
                if current_price >= position['stop_level']:
                    pnl = (position['entry_price'] - position['stop_level']) * position['size'] - spread_cost
                    balance += pnl
                    trades.append({'type': 'SELL_CLOSE_SL', 'pnl': pnl})
                    position = None
                elif current_price <= position['limit_level']:
                    pnl = (position['entry_price'] - position['limit_level']) * position['size'] - spread_cost
                    balance += pnl
                    trades.append({'type': 'SELL_CLOSE_TP', 'pnl': pnl})
                    position = None

        if not position:
            if signal['buy_signal']:
                stop_level = current_price - stop_distance
                limit_level = current_price + profit_distance
                position = {'direction': 'BUY', 'entry_price': current_price, 'size': size,
                            'stop_level': stop_level, 'limit_level': limit_level}
                trades.append({'type': 'BUY_OPEN', 'pnl': -spread_cost})
            elif signal['sell_signal']:
                stop_level = current_price + stop_distance
                limit_level = current_price - profit_distance
                position = {'direction': 'SELL', 'entry_price': current_price, 'size': size,
                            'stop_level': stop_level, 'limit_level': limit_level}
                trades.append({'type': 'SELL_OPEN', 'pnl': -spread_cost})

        equity.append(balance)

    # Fermer position ouverte à la fin
    if position:
        if position['direction'] == 'BUY':
            pnl = (df['close'].iloc[-1] - position['entry_price']) * position['size'] - spread_cost
        else:
            pnl = (position['entry_price'] - df['close'].iloc[-1]) * position['size'] - spread_cost
        balance += pnl
        trades.append({'type': 'CLOSE_END', 'pnl': pnl})

    # Calcul métriques
    pnls = [t['pnl'] for t in trades if 'CLOSE' in t['type']]
    total_pnl = balance - initial_balance
    win_rate = len([p for p in pnls if p > 0]) / len(pnls) if pnls else 0
    returns = pd.Series(equity).pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(len(returns)) if returns.std() != 0 else 0
    max_drawdown = (pd.Series(equity).cummax() - equity).max() / pd.Series(equity).cummax().max()

    results = {
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_trades': len(pnls),
        'final_balance': balance
    }
    print(f"📊 Résultats backtest: PnL={total_pnl:.2f}, Win Rate={win_rate:.2%}, Sharpe={sharpe_ratio:.2f}, Max DD={max_drawdown:.2%}")
    return results

# === Fonction d'Optimisation des Paramètres ===
def optimize_parameters(headers, epic, initial_balance=10000.0, start_date=None, end_date=None, resolution="MINUTE_15"):
    """
    Optimise les paramètres via grid search.
    Teste combinaisons de sl_atr_multiplier, tp_atr_multiplier, risk_per_trade.
    Choisit le meilleur basé sur Sharpe ratio.
    """
    print("🛠️ Démarrage de l'optimisation...")
    param_grid = {
        'sl_atr_multiplier': [1.5, 2.0, 2.5],
        'tp_atr_multiplier': [1.0, 1.5, 2.0],
        'risk_per_trade': [0.5, 0.8, 1.0]
    }
    best_sharpe = -np.inf
    best_params = None
    best_results = None

    for params in itertools.product(*param_grid.values()):
        sl, tp, risk = params
        results = backtest_strategy(headers, epic, initial_balance, start_date, end_date, resolution,
                                    sl_atr_multiplier=sl, tp_atr_multiplier=tp, risk_per_trade=risk)
        if results and results['sharpe_ratio'] > best_sharpe:
            best_sharpe = results['sharpe_ratio']
            best_params = {'sl_atr_multiplier': sl, 'tp_atr_multiplier': tp, 'risk_per_trade': risk}
            best_results = results

    print(f"✅ Meilleurs params: {best_params}, Sharpe={best_sharpe:.2f}")
    return best_params, best_results

# === Boucle principale du bot ===
def trading_bot(check_interval=60, confirmation_period=2, atr_threshold=4.0, market="GOLD"):
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
                        print(f"🔄 Fermeture de la position existante ({position['direction']}) avant nouvel ordre...")
                        close_position(auth_headers, position["deal_id"], position["direction"], position["size"])
                    print(f"📈 Passage d'un ordre d'achat à {current_price:.2f}")
                    response, stop_level, limit_level, stop_loss_percentage, take_profit_percentage = place_order(
                        auth_headers, epic, "BUY", current_price, df
                    )
                    if response:
                        print(f"✅ Ordre d'achat exécuté avec succès")
                    else:
                        print(f"❌ Échec de l'ordre d'achat")
                elif signal["sell_signal"]:
                    print("🔍 Signal de vente détecté pour la dernière ligne")
                    if position and position["direction"] == "SELL":
                        print("ℹ️ Position de vente déjà ouverte, rien à faire")
                        continue
                    if position:
                        print(f"🔄 Fermeture de la position existante ({position['direction']}) avant nouvel ordre...")
                        close_position(auth_headers, position["deal_id"], position["direction"], position["size"])
                    print(f"📈 Passage d'un ordre de vente à {current_price:.2f}")
                    response, stop_level, limit_level, stop_loss_percentage, take_profit_percentage = place_order(
                        auth_headers, epic, "SELL", current_price, df
                    )
                    if response:
                        print(f"✅ Ordre de vente exécuté avec succès")
                    else:
                        print(f"❌ Échec de l'ordre de vente")
                else:
                    print("ℹ️ Aucun signal d'achat ou de vente détecté")
            else:
                print("❌ Données de prix ou de tendance manquantes, saut de cette itération")
            print(f"⏳ Pause de {check_interval} secondes avant la prochaine analyse")
            time.sleep(check_interval)
        except Exception as e:
            print(f"❌ Erreur inattendue dans la boucle principale : {e}")
            print(f"⏳ Pause de {check_interval} secondes avant de réessayer")
            time.sleep(check_interval)

# === Lancement du bot ===
if __name__ == "__main__":
    # Exemple d'utilisation du backtest et optimisation
    # auth_headers = authenticate()
    # epic = search_market(auth_headers, MARKET)
    # backtest_strategy(auth_headers, epic, start_date="2024-01-01", end_date="2024-12-31")
    # optimize_parameters(auth_headers, epic, start_date="2024-01-01", end_date="2024-12-31")

    trading_bot(check_interval=CHECK_INTERVAL, confirmation_period=CONFIRMATION_PERIOD, atr_threshold=ATR_THRESHOLD, market=MARKET)