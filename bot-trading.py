import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
import pytz

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
        except requests.RequestException as e:
            print(f"❌ Erreur réseau (tentative {attempt + 1}) : {e}")
        print(f"⏳ Attente avant nouvelle tentative ({2 ** attempt} secondes)")
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
        cst = response.headers.get("CST")
        security_token = response.headers.get("X-SECURITY-TOKEN")
        if cst and security_token:
            auth_headers["CST"] = cst
            auth_headers["X-SECURITY-TOKEN"] = security_token
        return auth_headers
    print(f"❌ Échec de la connexion : {response.status_code if response else 'Aucune réponse'}")
    return None

# === Recherche du marché BTC/EUR ===
def search_market(headers, search_term="BTC/EUR"):
    print(f"🔍 Recherche du marché : {search_term}")
    url = f"{BASE_URL}/api/v1/markets"
    params = {"searchTerm": search_term}
    response = safe_request("GET", url, headers=headers, params=params)
    if response:
        markets = response.json().get("markets", [])
        if markets:
            print(f"📜 Marché trouvé : {markets[0]['instrumentName']} (EPIC: {markets[0]['epic']})")
            return markets[0]["epic"]
        print("⚠️ Aucun marché trouvé pour BTC/EUR")
    print("❌ Erreur lors de la recherche du marché")
    return None

# === Récupération du solde disponible ===
def get_available_balance(headers):
    print("💸 Vérification du solde disponible...")
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
    print(f"📋 Récupération des règles pour le marché {epic}...")
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
        print(f"📈 Dernier prix BTC/EUR : {df['close'].iloc[-1]:.2f}")
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

# === Détection des signaux d'achat/vente ===
def detect_signals(df, df_trend, periods_back=60, drop=1.5, rise=1.5, confirmation_period=5, adx_threshold=20, atr_threshold=2):
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

    if df[["rsi", "adx", "atr"]].isnull().any().any():
        print("⚠️ Données des indicateurs invalides, aucun signal généré")
        df["buy_signal"] = False
        df["sell_signal"] = False
        return df

    df["price_60min_ago"] = df["close"].shift(periods_back)
    df["pct_change"] = (df["close"] - df["price_60min_ago"]) / df["price_60min_ago"] * 100
    df["avg_volume_20"] = df["volume"].rolling(window=20).mean()

    df["buy_signal"] = False
    df["sell_signal"] = False
    trend = detect_trend(df_trend)

    # Filtre de volatilité basé sur l'ATR
    avg_atr = df["atr"].mean()
    current_atr = df["atr"].iloc[-1]
    if current_atr > atr_threshold * avg_atr:
        print(f"⚠️ Volatilité trop élevée (ATR={current_atr:.2f}, Moyenne={avg_atr:.2f}), aucun signal généré")
        return df

    # Vérification des signaux pour la dernière période
    i = len(df) - confirmation_period
    buy_conditions = (
        all(df["pct_change"].iloc[i:i + confirmation_period] <= -drop) and
        df["rsi"].iloc[i + confirmation_period - 1] <= 35 and
        df["volume"].iloc[i + confirmation_period - 1] > df["avg_volume_20"].iloc[i + confirmation_period - 1] and
        df["adx"].iloc[i + confirmation_period - 1] >= adx_threshold and
        trend in ["BULLISH", "NEUTRAL"]
    )
    sell_conditions = (
        all(df["pct_change"].iloc[i:i + confirmation_period] >= rise) and
        df["rsi"].iloc[i + confirmation_period - 1] >= 65 and
        df["volume"].iloc[i + confirmation_period - 1] > df["avg_volume_20"].iloc[i + confirmation_period - 1] and
        df["adx"].iloc[i + confirmation_period - 1] >= adx_threshold and
        trend in ["BEARISH", "NEUTRAL"]
    )

    if buy_conditions:
        df.iloc[i + confirmation_period - 1, df.columns.get_loc("buy_signal")] = True
        print(f"🔔 Signal d'achat confirmé : RSI={df['rsi'].iloc[i + confirmation_period - 1]:.2f}, "
              f"ADX={df['adx'].iloc[i + confirmation_period - 1]:.2f}")
    if sell_conditions:
        df.iloc[i + confirmation_period - 1, df.columns.get_loc("sell_signal")] = True
        print(f"🔔 Signal de vente confirmé : RSI={df['rsi'].iloc[i + confirmation_period - 1]:.2f}, "
              f"ADX={df['adx'].iloc[i + confirmation_period - 1]:.2f}")

    last_row = df.iloc[-1]
    buy_reasons = []
    sell_reasons = []

    # Raisons pour l'absence de signal d'achat
    if not last_row["buy_signal"]:
        if df["pct_change"].iloc[-1] > -drop:
            if df["pct_change"].iloc[-1] > 0:
                buy_reasons.append(f"Le prix a augmenté de {df['pct_change'].iloc[-1]:.2f}%, besoin d'une baisse d'au moins {drop}%")
            elif abs(df["pct_change"].iloc[-1]) < 0.01:  # Considéré comme stable
                buy_reasons.append(f"Le prix est stable, besoin d'une baisse d'au moins {drop}%")
            else:
                buy_reasons.append(f"Le prix n'a baissé que de {abs(df['pct_change'].iloc[-1]):.2f}%, besoin d'une baisse d'au moins {drop}%")
        if df["rsi"].iloc[-1] > 35:
            buy_reasons.append(f"RSI à {df['rsi'].iloc[-1]:.2f}, trop haut (besoin de ≤ 35 pour survente)")
        if df["volume"].iloc[-1] <= df["avg_volume_20"].iloc[-1]:
            buy_reasons.append("Volume trop faible par rapport à la moyenne")
        if df["adx"].iloc[-1] < adx_threshold:
            buy_reasons.append(f"ADX à {df['adx'].iloc[-1]:.2f}, trop faible (besoin de ≥ {adx_threshold} pour une tendance forte)")
        if trend not in ["BULLISH", "NEUTRAL"]:
            buy_reasons.append(f"Tendance baissière ({trend}), pas favorable pour acheter")

    # Raisons pour l'absence de signal de vente
    if not last_row["sell_signal"]:
        if df["pct_change"].iloc[-1] < rise:
            if df["pct_change"].iloc[-1] < 0:
                sell_reasons.append(f"Le prix a baissé de {abs(df['pct_change'].iloc[-1]):.2f}%, besoin d'une hausse d'au moins {rise}%")
            elif abs(df["pct_change"].iloc[-1]) < 0.01:  # Considéré comme stable
                sell_reasons.append(f"Le prix est stable, besoin d'une hausse d'au moins {rise}%")
            else:
                sell_reasons.append(f"Le prix n'a augmenté que de {df['pct_change'].iloc[-1]:.2f}%, besoin d'une hausse d'au moins {rise}%")
        if df["rsi"].iloc[-1] < 65:
            sell_reasons.append(f"RSI à {df['rsi'].iloc[-1]:.2f}, trop bas (besoin de ≥ 65 pour surachat)")
        if df["volume"].iloc[-1] <= df["avg_volume_20"].iloc[-1]:
            sell_reasons.append("Volume trop faible par rapport à la moyenne")
        if df["adx"].iloc[-1] < adx_threshold:
            sell_reasons.append(f"ADX à {df['adx'].iloc[-1]:.2f}, trop faible (besoin de ≥ {adx_threshold} pour une tendance forte)")
        if trend not in ["BEARISH", "NEUTRAL"]:
            sell_reasons.append(f"Tendance haussière ({trend}), pas favorable pour vendre")

    print(f"📊 Résumé : Prix={last_row['close']:.2f}, RSI={last_row['rsi']:.2f}, ADX={last_row['adx']:.2f}, "
          f"ATR={last_row['atr']:.2f}, Tendance={trend}, Achat={last_row['buy_signal']}, Vente={last_row['sell_signal']}")
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
                    size = position.get("size", position.get("dealSize", position.get("contractSize", 0.0)))
                    if size == 0.0:
                        print(f"⚠️ Taille de position non trouvée pour {epic}")
                        continue
                    open_level = position.get("openLevel", position.get("level", position.get("entryPrice", 0.0)))
                    print(f"📍 Position ouverte : {position['direction']} (Taille: {size:.4f}, Prix d'entrée: {open_level:.2f})")
                    return {
                        "direction": position["direction"],
                        "size": size,
                        "deal_id": position["dealId"]
                    }
        except KeyError as e:
            print(f"❌ Erreur dans la réponse de l'API : {e}")
            return None
    print("ℹ️ Aucune position ouverte")
    return None

# === Fermeture d'une position ===
def close_position(headers, deal_id, direction, size, current_price):
    print(f"🔐 Tentative de fermeture de la position (ID: {deal_id})...")
    url = f"{BASE_URL}/api/v1/positions/{deal_id}"
    payload = {
        "direction": "SELL" if direction == "BUY" else "BUY",
        "size": size,
        "orderType": "MARKET",
        "level": current_price
    }
    print(f"📤 Envoi de la requête pour fermer : direction={payload['direction']}, taille={size:.4f}")
    response = safe_request("DELETE", url, headers=headers, json=payload)
    if response and response.ok:
        print(f"✅ Position fermée avec succès (ID: {deal_id})")
        return True
    print(f"❌ Échec de la fermeture de la position (ID: {deal_id})")
    return False

# === Passage d'un ordre ===
def place_order(headers, epic, direction, entry_price, df):
    print(f"📝 Préparation d'un ordre {direction} pour {epic}...")
    available_balance = get_available_balance(headers)
    print(f"💸 Solde disponible pour l'ordre : {available_balance} EUR")
    margin_factor, min_size, max_size = get_margin_requirement(headers, epic)
    max_size_possible = available_balance / (entry_price * margin_factor)
    size = min(max_size_possible, max_size)
    size = max(size, min_size)
    size = round(size, 4)

    # Calcul du stop-loss et take-profit basé sur l'ATR
    atr = df["atr"].iloc[-1]
    stop_loss_distance = atr * 1.5
    take_profit_distance = atr * 2.0
    if direction == "BUY":
        stop = entry_price - stop_loss_distance
        limit = entry_price + take_profit_distance
    else:
        stop = entry_price + stop_loss_distance
        limit = entry_price - take_profit_distance

    # Calcul des pourcentages de perte/gain
    stop_loss_percentage = abs(stop - entry_price) / entry_price * 100
    take_profit_percentage = abs(limit - entry_price) / entry_price * 100

    url = f"{BASE_URL}/api/v1/positions"
    payload = {
        "epic": epic,
        "direction": direction,
        "size": size,
        "stopLevel": round(stop, 2),
        "limitLevel": round(limit, 2),
        "orderType": "MARKET"
    }
    print(f"📤 Envoi de l'ordre {direction} : Prix={entry_price:.2f}, Stop={stop:.2f}, Limit={limit:.2f}, Taille={size:.4f}")
    print(f"📊 Perte potentielle (stop-loss) : {stop_loss_percentage:.2f}%")
    print(f"📊 Gain potentiel (take-profit) : {take_profit_percentage:.2f}%")
    response = safe_request("POST", url, headers=headers, json=payload)
    if response and response.ok:
        print(f"✅ Ordre {direction} passé avec succès (Taille: {size:.4f})")
        return response
    print(f"❌ Échec de l'ordre {direction}. Réponse : {response.text if response else 'Aucune réponse'}")
    return None

# === Boucle principale du bot ===
def trading_bot(
    drop=1.5,
    rise=1.5,
    check_interval=60,
    confirmation_period=5,
    adx_threshold=20,
    atr_threshold=2
):
    print("🚀 Démarrage du bot de trading...")
    auth_headers = authenticate()
    if not auth_headers:
        print("❌ Impossible de démarrer : échec de la connexion")
        return
    epic = search_market(auth_headers, "BTC/EUR")
    if not epic:
        print("❌ Impossible de démarrer : aucun marché BTC/EUR trouvé")
        return
    print(f"📌 Bot configuré pour trader sur : {epic}")

    while True:
        try:
            print("🔄 Nouvelle itération du bot...")
            df = get_candles(auth_headers, epic, resolution="MINUTE_15")
            df_trend = get_trend_candles(auth_headers, epic, resolution="HOUR_4")
            if df is not None and df_trend is not None:
                df = detect_signals(
                    df,
                    df_trend,
                    drop=drop,
                    rise=rise,
                    confirmation_period=confirmation_period,
                    adx_threshold=adx_threshold,
                    atr_threshold=atr_threshold
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
                    place_order(auth_headers, epic, "BUY", current_price, df)
                elif signal["sell_signal"]:
                    print("🔍 Signal de vente détecté pour la dernière ligne")
                    if position and position["direction"] == "SELL":
                        print("ℹ️ Position de vente déjà ouverte, rien à faire")
                        continue
                    if position:
                        print("🔄 Fermeture de la position d'achat pour ouvrir une vente")
                        close_position(auth_headers, position["deal_id"], position["direction"], position["size"], current_price)
                    print(f"📉 Passage d'un ordre de vente à {current_price:.2f}")
                    place_order(auth_headers, epic, "SELL", current_price, df)
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
    print("🎉 Lancement du bot de trading pour BTC/EUR")
    trading_bot(
        drop=1.5,
        rise=1.5,
        check_interval=60,
        confirmation_period=5,
        adx_threshold=20,
        atr_threshold=2
    )