import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import pytz
import uuid
import os
import itertools
import logging
import smtplib  # Ajout pour l'envoi email-to-SMS
from email.mime.text import MIMEText  # Ajout pour le formatage du message email

# === Fonction pour logs et prints ===
def log_message(message, level="info"):
    print(message)
    if level == "info":
        logging.info(message)
    elif level == "error":
        logging.error(message)
    logging.getLogger().handlers[0].flush()  # Forcer l'écriture immédiate

# Supprimer tous les gestionnaires existants pour éviter les conflits avec Pyzo
logging.getLogger().handlers = []

# Configuration du logging avec FileHandler
log_file = '/Users/juanlucas/Documents/Bot-Trading/trading_log.txt'
handler = logging.FileHandler(log_file)
handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Vérification initiale des permissions et configuration du logging
print(f"Répertoire de travail courant : {os.getcwd()}")
if not os.access(os.getcwd(), os.W_OK):
    print("❌ Erreur : Pas de permission d'écriture dans le répertoire courant")
    log_message("❌ Erreur : Pas de permission d'écriture dans le répertoire courant", level="error")
else:
    log_message("✅ Permissions d'écriture OK")
    # Écrire un message de test pour vérifier
    log_message("Démarrage du bot de trading")
    handler.flush()  # Forcer l'écriture immédiate
    # Vérifier si le fichier a été créé
    if os.path.exists(log_file):
        print(f"✅ Fichier {log_file} créé avec succès")
        log_message(f"Fichier {log_file} créé avec succès")
    else:
        print(f"❌ Échec de la création du fichier {log_file}")
        log_message(f"Échec de la création du fichier {log_file}", level="error")

# === Configuration initiale ===
API_KEY = "K6meo5BSJuuduWI8"
EMAIL = "5rycnytkzh@privaterelay.appleid.com"
PASSWORD = "Lucas1234@"
BASE_URL = "https://demo-api-capital.backend-capital.com"

# Configuration pour SMS via Email-to-SMS (Gmail)
EMAIL_SENDER = "juan.lucas.3b@gmail.com"  # Remplace par ton email Gmail
EMAIL_PASSWORD = "iqcx gqpm zjcp xhlj"  # Remplace par le mot de passe d'application Gmail (active 2FA et génère-le)
PHONE_NUMBER = "0781473142"  # Remplace par ton numéro de téléphone (format français sans +33)
CARRIER_GATEWAY = "@orange.fr"  # Remplace par ton gateway (ex: "@free-mobile.fr" pour Free, "@sfr.fr" pour SFR)

print("🚀 Démarrage du bot - Chargement des configurations...")
print(f"📧 Email chargé : {EMAIL[:5]}... (caché pour sécurité)")
print(f"🔑 Clé API chargée : {API_KEY[:10]}... (cachée pour sécurité)")
print(f"📱 Configuration SMS : Numéro {PHONE_NUMBER[:4]}... via {CARRIER_GATEWAY}")

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
SL_ATR_MULTIPLIER = 2.0  # Multiplicateur pour stop-loss
TP_ATR_MULTIPLIER = 1.5  # Multiplicateur pour take-profit
MINIMUM_BALANCE_BUFFER = 1.0  # Buffer minimum (EUR)
MAX_ORDER_RETRIES = 3  # Nombre max de tentatives pour un ordre
LEVERAGE = 20  # Levier de 20:1
MARGIN_FACTOR = 1 / LEVERAGE  # Marge = 5%
RISK_PER_TRADE = 0.8  # Risque 80% du solde par position
MAX_SPREAD_COST = 2.0  # Limite spread
STOP_LOSS_PERCENT = 1.10  # Fallback stop-loss (%)
TAKE_PROFIT_PERCENT = 0.5  # Fallback take-profit (%)
MAX_DRAWDOWN_PERCENT = 20.0  # Arrêt si perte > 20%
TRAILING_STOP_ENABLED = False  # Désactivé pour éviter le conflit avec guaranteedStop
VOLUME_THRESHOLD_MULTIPLIER = 1.2  # Volume doit être 1.2x la moyenne
INITIAL_BALANCE = 1000.0  # Solde initial pour drawdown
NEWS_AVOID_TIMES = [(14, 30), (15, 0)]  # Éviter trades ±30min autour de ces heures

print(f"⚙️ Paramètres chargés : Risque par trade = {RISK_PER_TRADE*100}%, Levier = {LEVERAGE}:1")
print(f"📊 Indicateurs : ATR seuil = {ATR_THRESHOLD}, Confirmation = {CONFIRMATION_PERIOD} bougies")
print(f"🛡️ Risques : Drawdown max = {MAX_DRAWDOWN_PERCENT}%, Trailing stop = {'Activé' if TRAILING_STOP_ENABLED else 'Désactivé'}")

# Dictionnaire pour stocker les positions ouvertes
open_positions = {}

# Variable globale pour le pic d'équité
peak_equity = INITIAL_BALANCE  # Initialisé au solde de départ

# === Fonction pour envoi SMS via Email-to-SMS Gateway ===
def send_sms_via_email(equity, phone_number=PHONE_NUMBER, carrier_gateway=CARRIER_GATEWAY):
    log_message(f"\n📩 Envoi SMS quotidien via Email-to-SMS Gateway à 20h...")
    now = datetime.now(pytz.timezone("Europe/Paris"))
    message = f"Solde trading: {equity:.2f} EUR | {now.strftime('%d/%m/%Y %H:%M')}"
    if len(message) > 160:  # Limite SMS standard
        message = message[:157] + "..."

    to_email = f"{phone_number}{carrier_gateway}"
    msg = MIMEText(message)
    msg['Subject'] = ''  # Pas de sujet pour SMS
    msg['From'] = EMAIL_SENDER
    msg['To'] = to_email

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, to_email, msg.as_string())
        server.quit()
        log_message(f"   ✅ SMS envoyé à {phone_number} via {carrier_gateway} : {message}")
        return True
    except Exception as e:
        log_message(f"   ❌ Erreur envoi SMS : {str(e)}", "error")
        return False

# === Fonction utilitaire pour requêtes API ===
def safe_request(method, url, headers, json=None, params=None, retries=3):
    log_message(f"🌐 Envoi requête : {method} {url} (tentatives max: {retries})")
    for attempt in range(retries):
        try:
            log_message(f"   Tentative {attempt + 1}/{retries}...")
            response = requests.request(method, url, headers=headers, json=json, params=params)
            if response.ok:
                cst = response.headers.get("CST")
                security_token = response.headers.get("X-SECURITY-TOKEN")
                if cst and security_token:
                    headers["CST"] = cst
                    headers["X-SECURITY-TOKEN"] = security_token
                    log_message(f"   🔑 Tokens mis à jour : CST et Security-Token OK")
                log_message(f"   ✅ Réponse OK (code {response.status_code})")
                return response
            else:
                log_message(f"   ❌ Erreur API (code {response.status_code}) : {response.text[:100]}...")
                if response.status_code == 401:
                    log_message("   ⚠️ Problème d'authentification détecté")
                    return None
                if response.status_code == 400:
                    error_message = response.json().get("errorCode", "Unknown error")
                    log_message(f"   ⚠️ Erreur spécifique : {error_message}")
                    if "INSUFFICIENT_FUNDS" in error_message:
                        return {"error": "INSUFFICIENT_FUNDS"}
                    elif "MARKET_CLOSED" in error_message:
                        return {"error": "MARKET_CLOSED"}
                    elif "error.invalid.stoploss.maxvalue" in error_message:
                        try:
                            max_stop_value = float(error_message.split(": ")[1])
                            return {"error": "INVALID_STOP_LOSS_MAX", "max_stop_value": max_stop_value}
                        except (IndexError, ValueError):
                            log_message("   ❌ Impossible d'extraire max_stop_value")
                            return {"error": "INVALID_STOP_LOSS_MAX"}
                    elif "error.invalid.stoploss.minvalue" in error_message:
                        try:
                            min_stop_value = float(error_message.split(": ")[1])
                            return {"error": "INVALID_STOP_LOSS_MIN", "min_stop_value": min_stop_value}
                        except (IndexError, ValueError):
                            log_message("   ❌ Impossible d'extraire min_stop_value")
                            return {"error": "INVALID_STOP_LOSS_MIN"}
                    elif "error.invalid.dealId" in error_message:
                        return {"error": "INVALID_DEAL_ID"}
                log_message(f"   ⏳ Attente 2^{attempt} secondes avant retry...")
                time.sleep(2 ** attempt)
        except requests.RequestException as e:
            log_message(f"   ❌ Erreur réseau : {e}")
            log_message(f"   ⏳ Attente 2^{attempt} secondes avant retry...")
            time.sleep(2 ** attempt)
    log_message(f"❌ Échec total après {retries} tentatives")
    return None

# === Authentification ===
def authenticate():
    log_message("\n🔐 Étape 1 : Connexion au compte démo...")
    url = f"{BASE_URL}/api/v1/session"
    payload = {"identifier": EMAIL, "password": PASSWORD}
    log_message(f"   Envoi payload : identifier={EMAIL}, password=*** (caché)")
    auth_headers = headers.copy()
    response = safe_request("POST", url, headers=auth_headers, json=payload)
    if response and response.status_code == 200:
        log_message("   ✅ Connexion réussie ! Prêt à trader.")
        return auth_headers
    log_message("   ❌ Échec de la connexion. Vérifie tes identifiants.")
    return None

# === Recherche du marché Gold ===
def search_market(headers, search_term="GOLD"):
    log_message(f"\n🔍 Étape 2 : Recherche du marché '{search_term}'...")
    url = f"{BASE_URL}/api/v1/markets"
    params = {"searchTerm": search_term}
    response = safe_request("GET", url, headers=headers, params=params)
    if response:
        markets = response.json().get("markets", [])
        if markets:
            epic = markets[0]["epic"]
            instrument = markets[0]['instrumentName']
            log_message(f"   ✅ Marché trouvé : {instrument} (EPIC: {epic})")
            return epic
        log_message("   ⚠️ Aucun marché trouvé pour 'GOLD'. Vérifie le nom.")
    log_message("   ❌ Erreur lors de la recherche du marché.")
    return None

# === Vérification des heures de trading ===
def is_market_open():
    log_message("\n🕒 Étape 3 : Vérification si le marché est ouvert...")
    now = datetime.now(pytz.timezone("Europe/Paris"))
    weekday = now.weekday()
    hour, minute = now.hour, now.minute
    log_message(f"   Heure actuelle (Paris) : {now.strftime('%Y-%m-%d %H:%M:%S')} (jour {weekday+1}/7)")
    if weekday >= 5 or (weekday == 4 and hour >= 23):
        log_message("   ⛔ Marché fermé (weekend ou après 23h vendredi).")
        return False
    # Check news
    for h, m in NEWS_AVOID_TIMES:
        news_time = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if abs((now - news_time).total_seconds()) < 1800:  # ±30min
            log_message(f"   ⚠️ Éviter trading : Proche d'une annonce économique à {h:02d}:{m:02d}.")
            return False
    log_message("   ✅ Marché ouvert et pas d'annonces en cours.")
    return True

# === Récupération des soldes (available et equity) ===
def get_balances(headers):
    log_message("\n💸 Récupération des soldes (available et equity)...")
    url = f"{BASE_URL}/api/v1/accounts"
    response = safe_request("GET", url, headers=headers)
    if response:
        accounts = response.json().get("accounts", [])
        if accounts:
            balance_data = accounts[0].get("balance", {})
            available = balance_data.get("available", 0.0)
            # Calcul de l'equity : funds ("balance") + profitLoss
            funds = balance_data.get("balance", 0.0)  # Ou "deposit" si c'est funds
            profit_loss = balance_data.get("profitLoss", 0.0)
            equity = funds + profit_loss  # Equity réelle
            log_message(f"   ✅ Available : {available:.2f} EUR | Equity : {equity:.2f} EUR | P&L : {profit_loss:.2f} EUR")
            return available, equity
    log_message("   ❌ Erreur lors de la récupération des soldes.")
    return 0.0, 0.0  # Fallback

# === Récupération des exigences de marge ===
def get_margin_requirement(headers, epic):
    log_message(f"\n📋 Étape 5 : Récupération des règles pour {epic}...")
    url = f"{BASE_URL}/api/v1/markets/{epic}"
    response = safe_request("GET", url, headers=headers)
    if response:
        data = response.json()
        min_size = round(data.get("minimumDealSize", 0.01), 2)
        max_size = round(data.get("maximumDealSize", 100.0), 2)
        log_message(f"   ✅ Règles : Marge requise = {MARGIN_FACTOR*100}%, Taille min = {min_size:.2f}, Taille max = {max_size:.2f}")
        return MARGIN_FACTOR, min_size, max_size
    log_message("   ❌ Erreur lors de la récupération des règles.")
    return MARGIN_FACTOR, 0.01, 100.0

# === Récupération du prix actuel et spread ===
def get_current_price(headers, epic):
    log_message(f"\n📈 Étape 6 : Récupération du prix actuel pour {epic}...")
    url = f"{BASE_URL}/api/v1/prices/{epic}"
    params = {"resolution": "MINUTE", "max": 1}
    response = safe_request("GET", url, headers=headers, params=params)
    if response:
        prices = response.json().get("prices", [])
        if prices and "closePrice" in prices[0]:
            bid = prices[0]["closePrice"].get("bid", None)
            ask = prices[0]["closePrice"].get("ask", None)
            if bid is not None and ask is not None:
                spread = min(ask - bid, MAX_SPREAD_COST)
                log_message(f"   ✅ Prix actuel (bid) : {bid:.2f}, Prix ask : {ask:.2f}, Spread : {spread:.2f}")
                return bid, spread
        log_message("   ⚠️ Données de prix invalides dans la réponse.")
    log_message("   ❌ Erreur lors de la récupération du prix.")
    return None, None

# === Vérification stabilité prix ===
def verify_price_stability(headers, epic, expected_price):
    log_message(f"\n🔍 Étape 7 : Vérification de la stabilité du prix pour {epic} (attendu: {expected_price:.2f})...")
    current_price, _ = get_current_price(headers, epic)
    if current_price is None:
        log_message("   ❌ Impossible de vérifier le prix actuel.")
        return False
    price_diff_percent = abs(current_price - expected_price) / expected_price * 100
    log_message(f"   Prix actuel : {current_price:.2f}, Écart : {price_diff_percent:.2f}%")
    if price_diff_percent > PRICE_TOLERANCE_PERCENT:
        log_message(f"   ⚠️ Écart trop important : {price_diff_percent:.2f}% (tolérance max : {PRICE_TOLERANCE_PERCENT}%)")
        return False
    log_message(f"   ✅ Prix stable : écart de {price_diff_percent:.2f}% OK.")
    return True

# === Vérification exécution ordre ===
def verify_order_execution(headers, deal_ref, expected_price, expected_size):
    log_message(f"\n🔍 Étape 8 : Vérification de l'exécution de l'ordre (Référence: {deal_ref})...")
    if not deal_ref:
        log_message("   ❌ Référence de deal invalide, impossible de vérifier.")
        return False, None, None
    url = f"{BASE_URL}/api/v1/confirms/{deal_ref}"
    log_message("   ⏳ Attente 2 secondes pour que l'API traite l'ordre...")
    time.sleep(2)
    response = safe_request("GET", url, headers=headers)
    if response and isinstance(response, dict) and response.get("error") == "INVALID_DEAL_ID":
        log_message(f"   ❌ Référence de deal {deal_ref} invalide selon l'API.")
        return False, None, None
    if response and response.ok:
        confirmation = response.json()
        log_message(f"   📝 Réponse de confirmation complète : {confirmation}")
        status = confirmation.get("status", "UNKNOWN")
        executed_price = confirmation.get("level", None)
        deal_id = confirmation.get("dealId", None)
        affected_deals = confirmation.get("affectedDeals", [])
        reason = confirmation.get("reason", "No reason provided")
        executed_size = confirmation.get("size", None)
        if affected_deals and "size" in affected_deals[0]:
            executed_size = affected_deals[0].get("size", None)
            if executed_size is not None and abs(executed_size - expected_size) > 0.01:
                log_message(f"   ⚠️ Ajustement de taille par l'API : Demandée={expected_size:.2f}, Exécutée={executed_size:.2f}")
        if executed_size is None:
            executed_size = confirmation.get("size", expected_size)
            log_message(f"   ⚠️ Taille exécutée non trouvée, utilisation fallback : {executed_size:.2f}")
        if affected_deals:
            deal_id = affected_deals[0].get("dealId", deal_id)
        if status == "OPEN" and executed_price is not None and deal_id:
            slippage_percent = abs(executed_price - expected_price) / expected_price * 100
            log_message(f"   Prix exécuté : {executed_price:.2f}, Slippage : {slippage_percent:.2f}%")
            if slippage_percent > MAX_SLIPPAGE_PERCENT:
                log_message(f"   ⚠️ Slippage excessif : {slippage_percent:.2f}% (max toléré : {MAX_SLIPPAGE_PERCENT}%)")
                return False, None, None
            log_message(f"   ✅ Ordre exécuté correctement ! Taille : {executed_size:.2f}, Deal ID : {deal_id}")
            return True, deal_id, executed_price
        log_message(f"   ⚠️ Statut de l'ordre non ouvert : {status}, Raison : {reason}")
        return False, None, None
    log_message(f"   ❌ Échec de la vérification de l'ordre (Réf: {deal_ref}).")
    return False, None, None

# === Récupération prix historiques ===
def get_candles(headers, epic, resolution="MINUTE_15", limit=200):
    log_message(f"\n📈 Étape 9 : Récupération des prix historiques pour {epic}...")
    log_message(f"   Résolution : {resolution}, Limite : {limit} bougies")
    url = f"{BASE_URL}/api/v1/prices/{epic}"
    params = {"resolution": resolution, "max": limit}
    response = safe_request("GET", url, headers=headers, params=params)
    if response:
        candles = response.json().get("prices", [])
        if not candles:
            log_message("   ⚠️ Aucune donnée de prix reçue.")
            return None
        log_message(f"   ✅ {len(candles)} bougies récupérées.")
        df = pd.DataFrame(candles)
        df["timestamp"] = pd.to_datetime(df["snapshotTime"], utc=True).dt.tz_convert("Europe/Paris")
        df.set_index("timestamp", inplace=True)
        try:
            df["close"] = df["closePrice"].apply(lambda x: x["bid"] if isinstance(x, dict) and "bid" in x else None)
            df["open"] = df["openPrice"].apply(lambda x: x["bid"] if isinstance(x, dict) and "bid" in x else None)
            df["high"] = df["highPrice"].apply(lambda x: x["bid"] if isinstance(x, dict) and "bid" in x else None)
            df["low"] = df["lowPrice"].apply(lambda x: x["bid"] if isinstance(x, dict) and "bid" in x else None)
            df["volume"] = df["lastTradedVolume"]
            log_message(f"   Données formatées : Open/High/Low/Close/Volume OK.")
        except (KeyError, TypeError) as e:
            log_message(f"   ❌ Erreur dans le format des données de prix : {e}", "error")
            return None
        if df[["close", "open", "high", "low"]].isnull().any().any() or (df["close"] == 0).any():
            log_message("   ⚠️ Données de prix invalides (valeurs nulles ou zéro).")
            return None
        last_close = df['close'].iloc[-1]
        log_message(f"   ✅ Dernier prix Gold : {last_close:.2f} (du {df.index[-1].strftime('%H:%M:%S')})")
        return df[["open", "high", "low", "close", "volume"]]
    log_message("   ❌ Erreur lors de la récupération des prix.")
    return None

# === Récupération prix pour tendance ===
def get_trend_candles(headers, epic, resolution="HOUR_4", limit=50):
    log_message(f"\n📈 Étape 10 : Récupération des prix pour analyse de tendance ({epic})...")
    log_message(f"   Résolution : {resolution}, Limite : {limit} bougies")
    url = f"{BASE_URL}/api/v1/prices/{epic}"
    params = {"resolution": resolution, "max": limit}
    response = safe_request("GET", url, headers=headers, params=params)
    if response:
        candles = response.json().get("prices", [])
        if not candles:
            log_message("   ⚠️ Aucune données de prix pour la tendance.")
            return None
        log_message(f"   ✅ {len(candles)} bougies de tendance récupérées.")
        df = pd.DataFrame(candles)
        df["timestamp"] = pd.to_datetime(df["snapshotTime"], utc=True).dt.tz_convert("Europe/Paris")
        df.set_index("timestamp", inplace=True)
        try:
            df["close"] = df["closePrice"].apply(lambda x: x["bid"] if isinstance(x, dict) and "bid" in x else None)
            log_message("   Données de close formatées OK.")
        except (KeyError, TypeError) as e:
            log_message(f"   ❌ Erreur dans le format des données de tendance : {e}", "error")
            return None
        if df["close"].isnull().any() or (df["close"] == 0).any():
            log_message("   ⚠️ Données de tendance invalides.")
            return None
        last_trend_close = df["close"].iloc[-1]
        log_message(f"   ✅ Dernier prix tendance (4h) : {last_trend_close:.2f}")
        return df[["close"]]
    log_message("   ❌ Erreur lors de la récupération des prix de tendance.")
    return None

# === Calcul RSI ===
def calculate_rsi(df, period=14):
    log_message(f"\n📊 Étape 11 : Calcul du RSI sur {period} bougies...")
    if len(df) < period:
        log_message(f"   ⚠️ Pas assez de données pour RSI ({len(df)} < {period}). Valeur par défaut : 50.")
        df["rsi"] = 50
        return df
    log_message("   Calcul des gains/pertes et moyenne...")
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)
    last_rsi = df["rsi"].iloc[-1]
    log_message(f"   ✅ RSI calculé. Dernière valeur : {last_rsi:.2f} (sur {len(df)} bougies)")
    return df

# === Calcul ADX ===
def calculate_adx(df, period=14):
    log_message(f"\n📊 Étape 12 : Calcul de l'ADX sur {period} bougies...")
    if len(df) < period * 2:
        log_message(f"   ⚠️ Pas assez de données pour ADX ({len(df)} < {period * 2}). Valeur par défaut : 25.")
        df["adx"] = 25
        return df
    log_message("   Calcul des True Range et Directional Indicators...")
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
    last_adx = df["adx"].iloc[-1]
    log_message(f"   ✅ ADX calculé. Dernière valeur : {last_adx:.2f}")
    return df

# === Calcul ATR ===
def calculate_atr(df, period=14):
    log_message(f"\n📊 Étape 13 : Calcul de l'ATR sur {period} bougies...")
    if len(df) < period:
        log_message(f"   ⚠️ Pas assez de données pour ATR ({len(df)} < {period}). Utilisation écart-type : {df['close'].std():.2f}")
        df["atr"] = df["close"].std()
        return df
    log_message("   Calcul des True Range...")
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
    df["atr"] = df["tr"].rolling(window=period).mean()
    df["atr"] = df["atr"].fillna(df["close"].std())
    last_atr = df["atr"].iloc[-1]
    log_message(f"   ✅ ATR calculé. Dernière valeur : {last_atr:.2f}")
    return df

# === Calcul MACD ===
def calculate_macd(df, fast=12, slow=26, signal=9):
    log_message(f"\n📊 Étape 14 : Calcul du MACD (fast={fast}, slow={slow}, signal={signal})...")
    if len(df) < slow:
        log_message(f"   ⚠️ Pas assez de données pour MACD ({len(df)} < {slow}). Valeurs par défaut : 0.")
        df["macd"] = 0
        df["macd_signal"] = 0
        return df
    log_message("   Calcul des EMA rapide et lente...")
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd"] = df["macd"].fillna(0)
    df["macd_signal"] = df["macd_signal"].fillna(0)
    last_macd = df["macd"].iloc[-1]
    last_signal = df["macd_signal"].iloc[-1]
    log_message(f"   ✅ MACD calculé. Dernier MACD : {last_macd:.4f}, Signal : {last_signal:.4f}")
    return df

# === Calcul Bandes de Bollinger ===
def calculate_bollinger_bands(df, period=20, std_dev=2):
    log_message(f"\n📊 Étape 15 : Calcul des Bandes de Bollinger (période={period}, écart-type={std_dev})...")
    if len(df) < period:
        log_message(f"   ⚠️ Pas assez de données pour Bollinger ({len(df)} < {period}). Utilisation prix actuel.")
        df["bb_mid"] = df["close"]
        df["bb_lower"] = df["close"]
        df["bb_upper"] = df["close"]
        return df
    log_message("   Calcul de la moyenne mobile et écart-type...")
    df["bb_mid"] = df["close"].rolling(window=period).mean()
    df["bb_std"] = df["close"].rolling(window=period).std()
    df["bb_upper"] = df["bb_mid"] + (df["bb_std"] * std_dev)
    df["bb_lower"] = df["bb_mid"] - (df["bb_std"] * std_dev)
    df[["bb_mid", "bb_upper", "bb_lower"]] = df[["bb_mid", "bb_upper", "bb_lower"]].bfill()
    last_upper = df["bb_upper"].iloc[-1]
    last_lower = df["bb_lower"].iloc[-1]
    log_message(f"   ✅ Bandes calculées. Haut : {last_upper:.2f}, Bas : {last_lower:.2f}, Prix actuel : {df['close'].iloc[-1]:.2f}")
    return df

# === Détection tendance (EMA 50/200) ===
def detect_trend(df_trend):
    log_message("\n🧠 Étape 16 : Analyse de la tendance sur timeframe 4h...")
    if df_trend is None or len(df_trend) < 20:
        log_message("   ⚠️ Pas assez de données pour analyser la tendance. Tendance neutre par défaut.")
        return "NEUTRAL"
    log_message("   Calcul des EMA 50 et 200...")
    df_trend["ema_50"] = df_trend["close"].ewm(span=50, adjust=False).mean()
    df_trend["ema_200"] = df_trend["close"].ewm(span=200, adjust=False).mean()
    last_ema_50 = df_trend["ema_50"].iloc[-1]
    last_ema_200 = df_trend["ema_200"].iloc[-1]
    log_message(f"   EMA 50 : {last_ema_50:.2f}, EMA 200 : {last_ema_200:.2f}")
    if last_ema_50 > last_ema_200:
        log_message("   📈 Tendance haussière détectée (EMA 50 > EMA 200).")
        return "BULLISH"
    elif last_ema_50 < last_ema_200:
        log_message("   📉 Tendance baissière détectée (EMA 50 < EMA 200).")
        return "BEARISH"
    log_message("   ⚖️ Tendance neutre (EMA 50 ≈ EMA 200).")
    return "NEUTRAL"

# === Calcul seuils dynamiques ===
def calculate_dynamic_thresholds(df, market="GOLD"):
    log_message("\n🧠 Étape 17 : Calcul des seuils dynamiques pour la stratégie...")
    avg_atr = df["atr"].mean()
    atr_quantile = df["atr"].quantile(0.75)
    log_message(f"   ATR moyen : {avg_atr:.2f}, ATR quantile 75% : {atr_quantile:.2f}")
    if avg_atr > atr_quantile:
        rsi_buy_threshold, rsi_sell_threshold = 60, 60
        log_message("   Marché volatile : Seuils RSI ajustés à 60/60.")
    else:
        rsi_buy_threshold, rsi_sell_threshold = 60, 55
        log_message("   Marché calme : Seuils RSI à 60/55.")
    adx_threshold = 15 if market == "GOLD" else 10
    drop_threshold = max(0.1, 0.03 * avg_atr / df["close"].iloc[-1] * 100)
    rise_threshold = max(0.1, 0.03 * avg_atr / df["close"].iloc[-1] * 100)
    log_message(f"   Seuils finaux : RSI Achat={rsi_buy_threshold}, Vente={rsi_sell_threshold}, ADX≥{adx_threshold}")
    log_message(f"   Seuil baisse : {drop_threshold:.2f}%, Hausse : {rise_threshold:.2f}%")
    return rsi_buy_threshold, rsi_sell_threshold, adx_threshold, drop_threshold, rise_threshold

# === Détection signaux achat/vente ===
def detect_signals(df, df_trend, periods_back=60, confirmation_period=2, atr_threshold=4.0, market="GOLD"):
    log_message(f"\n🧠 Étape 18 : Détection des signaux (confirmation sur {confirmation_period} bougies, lookback {periods_back} min)...")
    if len(df) < periods_back + confirmation_period - 1:
        log_message(f"   ⚠️ Pas assez de données pour analyse ({len(df)} bougies disponibles). Signaux désactivés.")
        df["buy_signal"] = False
        df["sell_signal"] = False
        return df
    log_message("   Calcul des indicateurs techniques...")
    df = calculate_rsi(df, period=14)
    df = calculate_adx(df, period=14)
    df = calculate_atr(df, period=14)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    if df[["rsi", "adx", "atr", "macd", "macd_signal", "bb_upper", "bb_lower"]].isnull().any().any():
        log_message("   ⚠️ Données des indicateurs invalides. Aucun signal généré.")
        df["buy_signal"] = False
        df["sell_signal"] = False
        return df
    log_message("   Calcul des seuils dynamiques...")
    rsi_buy_threshold, rsi_sell_threshold, adx_threshold, drop_threshold, rise_threshold = calculate_dynamic_thresholds(df, market)
    log_message("   Calcul des changements de prix et volume moyen...")
    df["price_60min_ago"] = df["close"].shift(periods_back)
    df["pct_change"] = (df["close"] - df["price_60min_ago"]) / df["price_60min_ago"] * 100
    df["avg_volume_50"] = df["volume"].rolling(window=50).mean()
    df["buy_signal"] = False
    df["sell_signal"] = False
    trend = detect_trend(df_trend)
    avg_atr = df["atr"].mean()
    current_atr = df["atr"].iloc[-1]
    log_message(f"   Vérification volatilité : ATR actuel = {current_atr:.2f}, Seuil max = {atr_threshold * avg_atr:.2f}")
    if current_atr > atr_threshold * avg_atr:
        log_message(f"   ⚠️ Volatilité trop élevée (ATR = {current_atr:.2f}). Pas de signaux.")
        return df
    i = len(df) - confirmation_period
    macd_diff = abs(df["macd"].iloc[i + confirmation_period - 1] - df["macd_signal"].iloc[i + confirmation_period - 1])
    macd_threshold = 0.5 * df["atr"].iloc[i + confirmation_period - 1]
    log_message(f"   Différence MACD : {macd_diff:.4f}, Seuil MACD : {macd_threshold:.2f}")
    log_message("   Vérification conditions d'achat...")
    bb_tolerance = 0.002  # Tolérance de 0.2% pour les bandes de Bollinger
    buy_conditions = (
        df["pct_change"].iloc[i + confirmation_period - 1] <= -drop_threshold and  # Baisse sur la dernière bougie
        df["rsi"].iloc[i + confirmation_period - 1] <= rsi_buy_threshold and
        df["adx"].iloc[i + confirmation_period - 1] >= adx_threshold and
        macd_diff <= macd_threshold and
        trend in ["BULLISH", "NEUTRAL"] and
        df["close"].iloc[i + confirmation_period - 1] <= df["bb_lower"].iloc[i + confirmation_period - 1] * (1 + bb_tolerance) and
        df["volume"].iloc[i + confirmation_period - 1] > df["avg_volume_50"].iloc[i + confirmation_period - 1] * 1.0  # Réduit à 1.0
    )
    if buy_conditions:
        df.iloc[i + confirmation_period - 1, df.columns.get_loc("buy_signal")] = True
        rsi_val = df['rsi'].iloc[i + confirmation_period - 1]
        adx_val = df['adx'].iloc[i + confirmation_period - 1]
        pct_val = df['pct_change'].iloc[i + confirmation_period - 1]
        log_message(f"   🔔 SIGNAL D'ACHAT CONFIRMÉ ! RSI={rsi_val:.2f} ≤ {rsi_buy_threshold}, ADX={adx_val:.2f} ≥ {adx_threshold}")
        log_message(f"   Détails : %Chg={pct_val:.2f}%, MACD Diff={macd_diff:.4f} ≤ {macd_threshold:.2f}, Tendance={trend}")
        log_message(f"   Bollinger : Prix={df['close'].iloc[i + confirmation_period - 1]:.2f} ≤ Bas={df['bb_lower'].iloc[i + confirmation_period - 1]:.2f}")
        log_message(f"   Volume : {df['volume'].iloc[i + confirmation_period - 1]:.0f} > Moyenne x 1.0 = {df['avg_volume_50'].iloc[i + confirmation_period - 1]:.0f}")
    log_message("   Vérification conditions de vente...")
    sell_conditions = (
        df["pct_change"].iloc[i + confirmation_period - 1] >= rise_threshold and
        df["rsi"].iloc[i + confirmation_period - 1] >= rsi_sell_threshold and
        df["adx"].iloc[i + confirmation_period - 1] >= adx_threshold and
        macd_diff <= macd_threshold and
        (
            trend in ["BEARISH", "NEUTRAL"] or
            (trend == "BULLISH" and df["rsi"].iloc[i + confirmation_period - 1] > 65)
        ) and
        df["close"].iloc[i + confirmation_period - 1] >= df["bb_upper"].iloc[i + confirmation_period - 1] * (1 - bb_tolerance) and
        df["volume"].iloc[i + confirmation_period - 1] > df["avg_volume_50"].iloc[i + confirmation_period - 1] * 1.0
    )
    if sell_conditions:
        df.iloc[i + confirmation_period - 1, df.columns.get_loc("sell_signal")] = True
        rsi_val = df['rsi'].iloc[i + confirmation_period - 1]
        adx_val = df['adx'].iloc[i + confirmation_period - 1]
        pct_val = df['pct_change'].iloc[i + confirmation_period - 1]
        log_message(f"   🔔 SIGNAL DE VENTE CONFIRMÉ ! RSI={rsi_val:.2f} ≥ {rsi_sell_threshold}, ADX={adx_val:.2f} ≥ {adx_threshold}")
        log_message(f"   Détails : %Chg={pct_val:.2f}%, MACD Diff={macd_diff:.4f} ≤ {macd_threshold:.2f}, Tendance={trend}")
        log_message(f"   Bollinger : Prix={df['close'].iloc[i + confirmation_period - 1]:.2f} ≥ Haut={df['bb_upper'].iloc[i + confirmation_period - 1]:.2f}")
        log_message(f"   Volume : {df['volume'].iloc[i + confirmation_period - 1]:.0f} > Moyenne x 1.0 = {df['avg_volume_50'].iloc[i + confirmation_period - 1]:.0f}")
    last_row = df.iloc[-1]
    log_message(f"\n   📊 RÉSUMÉ ACTUEL : Prix={last_row['close']:.2f}, RSI={last_row['rsi']:.2f}, ADX={last_row['adx']:.2f}")
    log_message(f"   MACD Diff={macd_diff:.4f}, ATR={last_row['atr']:.2f}, Tendance={trend}")
    log_message(f"   Signal Achat : {last_row['buy_signal']}, Signal Vente : {last_row['sell_signal']}")
    if not last_row["buy_signal"] and not last_row["sell_signal"]:
        log_message("   ℹ️ Pourquoi pas de signal ? Vérification détaillée...")
        buy_reasons, sell_reasons = [], []
        # Raisons pour achat non validé
        if df["pct_change"].iloc[i + confirmation_period - 1] > -drop_threshold:
            buy_reasons.append("Baisse pas confirmée sur la dernière bougie")
        if last_row["rsi"] > rsi_buy_threshold:
            buy_reasons.append(f"RSI trop haut ({last_row['rsi']:.2f} > {rsi_buy_threshold})")
        if last_row["adx"] < adx_threshold:
            buy_reasons.append(f"ADX trop bas ({last_row['adx']:.2f} < {adx_threshold})")
        if macd_diff > macd_threshold:
            buy_reasons.append(f"MACD diff trop grande ({macd_diff:.4f} > {macd_threshold:.2f})")
        if trend not in ["BULLISH", "NEUTRAL"]:
            buy_reasons.append(f"Tendance défavorable ({trend})")
        if last_row["close"] > last_row["bb_lower"] * (1 + bb_tolerance):
            buy_reasons.append(f"Prix pas proche du Bollinger bas ({last_row['close']:.2f} > {last_row['bb_lower']*(1+bb_tolerance):.2f})")
        if last_row["volume"] <= last_row["avg_volume_50"] * 1.0:
            buy_reasons.append(f"Volume insuffisant ({last_row['volume']:.0f} <= {last_row['avg_volume_50']*1.0:.0f})")
        # Raisons pour vente non validée
        if df["pct_change"].iloc[i + confirmation_period - 1] < rise_threshold:
            sell_reasons.append("Hausse pas confirmée sur la dernière bougie")
        if last_row["rsi"] < rsi_sell_threshold:
            sell_reasons.append(f"RSI trop bas ({last_row['rsi']:.2f} < {rsi_sell_threshold})")
        if last_row["adx"] < adx_threshold:
            sell_reasons.append(f"ADX trop bas ({last_row['adx']:.2f} < {adx_threshold})")
        if macd_diff > macd_threshold:
            sell_reasons.append(f"MACD diff trop grande ({macd_diff:.4f} > {macd_threshold:.2f})")
        if trend not in ["BEARISH", "NEUTRAL"] and not (trend == "BULLISH" and last_row["rsi"] > 65):
            sell_reasons.append(f"Tendance défavorable pour vente ({trend}, RSI={last_row['rsi']:.2f})")
        if last_row["close"] < last_row["bb_upper"] * (1 - bb_tolerance):
            sell_reasons.append(f"Prix pas proche du Bollinger haut ({last_row['close']:.2f} < {last_row['bb_upper']*(1-bb_tolerance):.2f})")
        if last_row["volume"] <= last_row["avg_volume_50"] * 1.0:
            sell_reasons.append(f"Volume insuffisant ({last_row['volume']:.0f} <= {last_row['avg_volume_50']*1.0:.0f})")
        if buy_reasons:
            log_message("   - Raisons pas d'achat :")
            for reason in buy_reasons:
                log_message(f"     * {reason}")
        if sell_reasons:
            log_message("   - Raisons pas de vente :")
            for reason in sell_reasons:
                log_message(f"     * {reason}")
    log_message("   --- Fin analyse signaux ---")
    return df

# === Vérification positions ouvertes ===
def is_position_open(headers, epic):
    log_message(f"\n🔍 Étape 19 : Vérification des positions ouvertes pour {epic}...")
    url = f"{BASE_URL}/api/v1/positions"
    response = safe_request("GET", url, headers=headers)
    if response:
        try:
            positions = response.json().get("positions", [])
            log_message(f"   {len(positions)} positions totales trouvées.")
            for pos in positions:
                if pos["market"]["epic"] == epic:
                    position = pos["position"]
                    deal_id = position.get("dealId")
                    size = round(position.get("size", position.get("dealSize", position.get("contractSize", 0.0))), 2)
                    if size == 0.0:
                        log_message("   ⚠️ Taille de position non trouvée, skip.")
                        continue
                    open_level = position.get("openLevel", position.get("level", position.get("entryPrice", 0.0)))
                    stop_level = position.get("stopLevel")
                    limit_level = position.get("limitLevel")
                    stored_position = open_positions.get(deal_id, {})
                    if stop_level is None and "stop_level" in stored_position:
                        stop_level = stored_position["stop_level"]
                        log_message(f"   ℹ️ Stop-level récupéré du stock : {stop_level:.2f}")
                    if limit_level is None and "limit_level" in stored_position:
                        limit_level = stored_position["limit_level"]
                        log_message(f"   ℹ️ Limit-level récupéré du stock : {limit_level:.2f}")
                    if limit_level is None and "profit_distance" in stored_position:
                        profit_distance = stored_position["profit_distance"]
                        direction = stored_position["direction"]
                        if profit_distance is not None and open_level is not None:
                            limit_level = open_level + profit_distance if direction == "BUY" else open_level - profit_distance
                            log_message(f"   ℹ️ Limit-level recalculé à partir de profit_distance : {limit_level:.2f}")
                    if limit_level is None:
                        profit_distance = open_level * (TAKE_PROFIT_PERCENT / 100)
                        limit_level = open_level + profit_distance if position["direction"] == "BUY" else open_level - profit_distance
                        log_message(f"   ℹ️ Fallback : Limit-level calculé à {limit_level:.2f} (basé sur {TAKE_PROFIT_PERCENT}%)")
                    stop_loss_percentage = abs(stop_level - open_level) / open_level * 100 if stop_level else 0.0
                    take_profit_percentage = abs(limit_level - open_level) / open_level * 100 if limit_level else 0.0
                    direction = position["direction"]
                    log_message(f"   ✅ Position ouverte trouvée : {direction} | Taille : {size:.2f} | Entrée : {open_level:.2f}")
                    log_message(f"   Stop-loss : {stop_level:.2f} ({stop_loss_percentage:.2f}%) | Take-profit : {limit_level:.2f} ({take_profit_percentage:.2f}%)")
                    log_message(f"   Deal ID : {deal_id}")
                    return {
                        "direction": direction,
                        "size": size,
                        "deal_id": deal_id,
                        "open_level": open_level,
                        "stop_level": stop_level,
                        "limit_level": limit_level,
                        "stop_loss_percentage": stop_loss_percentage,
                        "take_profit_percentage": take_profit_percentage
                    }
            log_message("   ℹ️ Aucune position ouverte pour ce marché.")
            return None
        except KeyError as e:
            log_message(f"   ❌ Erreur dans la réponse API positions : {e}", "error")
            return None
    log_message("   ❌ Erreur lors de la récupération des positions.")
    return None

# === Fermeture position ===
def close_position(headers, deal_id, direction, size):
    log_message(f"\n🔐 Étape 20 : Fermeture de la position (ID: {deal_id}, Direction: {direction}, Taille: {size:.2f})...")
    if not deal_id:
        log_message("   ❌ Deal ID invalide, impossible de fermer.")
        return False
    size = round(size, 2)
    url = f"{BASE_URL}/api/v1/positions/{deal_id}"
    close_direction = "SELL" if direction == "BUY" else "BUY"
    payload = {
        "direction": close_direction,
        "size": size,
        "orderType": "MARKET"
    }
    log_message(f"   Payload fermeture : Direction={close_direction}, Taille={size:.2f}, Type=MARKET")
    response = safe_request("DELETE", url, headers=headers, json=payload)
    if response and isinstance(response, dict) and response.get("error") == "INVALID_DEAL_ID":
        log_message(f"   ❌ Deal ID {deal_id} invalide pour fermeture.")
        return False
    if response and response.ok:
        log_message(f"   ✅ Position fermée avec succès ! (ID: {deal_id})")
        if deal_id in open_positions:
            del open_positions[deal_id]
            log_message(f"   🗑️ Position supprimée du dictionnaire local.")
        return True
    log_message(f"   ❌ Échec de la fermeture de la position (ID: {deal_id}). Réponse : {response.text if response else 'Aucune'}")
    return False

# === Passage d'un ordre ===
def place_order(headers, epic, direction, entry_price, df):
    log_message(f"\n📝 Étape 21 : Préparation d'un ordre {direction} pour {epic} au prix d'entrée {entry_price:.2f}...")
    log_message("   Sous-étape 1 : Vérification stabilité prix...")
    if not verify_price_stability(headers, epic, entry_price):
        log_message(f"   ❌ Ordre {direction} annulé : Prix trop instable.")
        return None, None, None, None, None
    log_message("   Sous-étape 2 : Récupération solde disponible...")
    available, _ = get_balances(headers)  # Utilise available pour les trades
    log_message(f"   Solde disponible : {available:.2f} EUR")
    log_message("   Sous-étape 3 : Récupération exigences de marge...")
    _, min_size, max_size = get_margin_requirement(headers, epic)
    log_message(f"   Contraintes taille : Min={min_size:.2f}, Max={max_size:.2f}")
    log_message("   Sous-étape 4 : Récupération limites stop/profit de l'API...")
    url = f"{BASE_URL}/api/v1/markets/{epic}"
    response = safe_request("GET", url, headers=headers)
    min_stop_distance = 0.0
    max_stop_distance = entry_price * 0.02
    min_profit_distance = 0.0
    max_profit_distance = float('inf')
    if response and response.ok:
        data = response.json()
        min_stop_distance = float(data.get("minimumStopDistance", 0.0))
        max_stop_distance = float(data.get("maximumStopDistance", entry_price * 0.02))
        min_profit_distance = float(data.get("minimumProfitDistance", 0.0))
        max_profit_distance = float(data.get("maximumProfitDistance", float('inf')))
        log_message(f"   Limites API : Stop min={min_stop_distance:.1f}, max={max_stop_distance:.1f}")
        log_message(f"   Profit min={min_profit_distance:.1f}, max={max_profit_distance:.1f}")
    else:
        log_message("   ⚠️ Limites non récupérées, utilisation valeurs par défaut.")
    log_message("   Sous-étape 5 : Calcul stop-loss et take-profit dynamiques (basés sur ATR)...")
    atr = float(df["atr"].iloc[-1])
    log_message(f"   ATR actuel = {atr:.1f}")
    stop_distance = atr * SL_ATR_MULTIPLIER
    profit_distance = atr * TP_ATR_MULTIPLIER
    log_message(f"   Calcul initial : Stop = ATR x {SL_ATR_MULTIPLIER} = {stop_distance:.1f}")
    log_message(f"   Profit = ATR x {TP_ATR_MULTIPLIER} = {profit_distance:.1f}")
    # Ajustements API
    stop_distance = round(max(min(stop_distance, max_stop_distance), min_stop_distance), 1)
    profit_distance = round(max(min(profit_distance, max_profit_distance), min_profit_distance), 1)
    log_message(f"   Après ajustements API : Stop = {stop_distance:.1f}, Profit = {profit_distance:.1f}")
    if profit_distance <= 0:
        log_message(f"   ❌ Profit_distance invalide ({profit_distance:.1f}). Ordre annulé.")
        return None, None, None, None, None
    if direction == "BUY":
        stop_level = entry_price - stop_distance
        limit_level = entry_price + profit_distance
    else:
        stop_level = entry_price + stop_distance
        limit_level = entry_price - profit_distance
    stop_loss_percentage = (stop_distance / entry_price) * 100
    take_profit_percentage = (profit_distance / entry_price) * 100
    log_message(f"   Niveaux finaux : Stop-loss = {stop_level:.1f} ({stop_loss_percentage:.2f}%)")
    log_message(f"   Take-profit = {limit_level:.1f} ({take_profit_percentage:.2f}%)")
    log_message("   Sous-étape 6 : Calcul de la taille de la position...")
    current_price, spread = get_current_price(headers, epic)
    spread_cost = min(spread if spread is not None else 0.1, MAX_SPREAD_COST)
    log_message(f"   Spread actuel : {spread_cost:.2f}")
    log_message(f"   Calcul taille pour risquer {RISK_PER_TRADE*100}% du solde avec levier {LEVERAGE}:1...")
    size = (available * RISK_PER_TRADE - MINIMUM_BALANCE_BUFFER) / (entry_price * MARGIN_FACTOR + spread_cost)
    log_message(f"   Taille brute calculée : {size:.4f}")
    size = max(size, min_size)
    if size > min_size:
        log_message(f"   Ajusté à min_size : {size:.2f}")
    size = min(size, max_size)
    size = round(size, 2)
    log_message(f"   Taille finale : {size:.2f} (dans limites {min_size}-{max_size})")
    required_margin = entry_price * size * MARGIN_FACTOR
    total_cost = required_margin + spread_cost * size + MINIMUM_BALANCE_BUFFER
    log_message(f"   Détail coûts : Marge = {required_margin:.2f} EUR, Spread = {spread_cost * size:.2f} EUR")
    log_message(f"   Buffer = {MINIMUM_BALANCE_BUFFER:.2f} EUR, Total coût = {total_cost:.2f} EUR")
    log_message(f"   Exposition totale : {entry_price * size:.2f} EUR")
    log_message(f"   Perte potentielle (stop-loss) : {stop_distance * size:.1f} EUR")
    if total_cost > available:
        log_message(f"   ⚠️ Coût total ({total_cost:.2f}) > solde ({available:.2f}). Réduction taille...")
        size = (available * RISK_PER_TRADE - MINIMUM_BALANCE_BUFFER) / (entry_price * MARGIN_FACTOR + spread_cost)
        size = max(size, min_size)
        size = min(size, max_size)
        size = round(size, 2)
        required_margin = entry_price * size * MARGIN_FACTOR
        total_cost = required_margin + spread_cost * size + MINIMUM_BALANCE_BUFFER
        log_message(f"   Nouvelle taille : {size:.2f}, Nouveau total : {total_cost:.2f} EUR")
        if total_cost > available or size < min_size:
            log_message(f"   ❌ Impossible d'ajuster. Ordre annulé.")
            return None, None, None, None, None
    adjusted_stop_distance = stop_distance
    log_message(f"   Début envoi ordre (max retries : {MAX_ORDER_RETRIES})...")
    for attempt in range(MAX_ORDER_RETRIES):
        log_message(f"   Tentative {attempt + 1}/{MAX_ORDER_RETRIES}...")
        current_price, spread = get_current_price(headers, epic)
        if current_price is None:
            log_message(f"   ❌ Prix actuel non récupéré. Retry impossible.")
            return None, None, None, None, None
        price_diff_percent = abs(current_price - entry_price) / entry_price * 100
        log_message(f"   Prix mis à jour : {current_price:.2f}, Écart : {price_diff_percent:.2f}%")
        if price_diff_percent > PRICE_TOLERANCE_PERCENT:
            log_message(f"   ⚠️ Écart trop important. Ordre annulé.")
            return None, None, None, None, None
        entry_price = float(current_price)
        response = safe_request("GET", f"{BASE_URL}/api/v1/markets/{epic}", headers=headers)
        if response and response.ok:
            data = response.json()
            min_stop_distance = float(data.get("minimumStopDistance", 0.0))
            max_stop_distance = float(data.get("maximumStopDistance", entry_price * 0.02))
            log_message(f"   Limites API actualisées : Stop min={min_stop_distance:.1f}, max={max_stop_distance:.1f}")
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
            "size": float(size),
            "orderType": "MARKET",
            "guaranteedStop": True,
            "trailingStop": False,
            "stopDistance": float(stop_distance),
            "profitDistance": float(profit_distance)
        }
        log_message(f"   Payload ordre : Direction={direction}, Taille={size:.2f}, StopDist={stop_distance:.1f}, ProfitDist={profit_distance:.1f}")
        url = f"{BASE_URL}/api/v1/positions"
        response = safe_request("POST", url, headers=headers, json=payload)
        if response and isinstance(response, dict) and response.get("error") in ["INVALID_STOP_LOSS_MAX", "INVALID_STOP_LOSS_MIN"]:
            if response["error"] == "INVALID_STOP_LOSS_MAX":
                max_stop_value = response.get("max_stop_value")
                log_message(f"   ⚠️ Stop-loss trop éloigné, max autorisé : {max_stop_value:.2f}")
                if direction == "BUY":
                    adjusted_stop_distance = float(entry_price - max_stop_value)
                    stop_level = float(max_stop_value)
                else:
                    adjusted_stop_distance = float(max_stop_value - entry_price)
                    stop_level = float(max_stop_value)
                adjusted_stop_distance = round(adjusted_stop_distance * 0.95, 1)
            elif response["error"] == "INVALID_STOP_LOSS_MIN":
                min_stop_value = response.get("min_stop_value")
                log_message(f"   ⚠️ Stop-loss trop proche, min autorisé : {min_stop_value:.2f}")
                if direction == "BUY":
                    adjusted_stop_distance = float(entry_price - min_stop_value)
                    stop_level = float(min_stop_value)
                else:
                    adjusted_stop_distance = float(min_stop_value - entry_price)
                    stop_level = float(min_stop_value)
                adjusted_stop_distance = round(adjusted_stop_distance * 1.05, 1)
            if direction == "BUY":
                stop_level = entry_price - adjusted_stop_distance
            else:
                stop_level = entry_price + adjusted_stop_distance
            stop_loss_percentage = (adjusted_stop_distance / entry_price) * 100
            log_message(f"   📊 Ajustement automatique : Stop Distance={adjusted_stop_distance:.1f}, Stop-level={stop_level:.1f} ({stop_loss_percentage:.2f}%)")
            # Vérifier si stop_distance est dans les limites
            if adjusted_stop_distance < min_stop_distance or adjusted_stop_distance > max_stop_distance:
                log_message(f"   ❌ Stop Distance {adjusted_stop_distance:.1f} hors limites (min={min_stop_distance:.1f}, max={max_stop_distance:.1f})")
                return None, None, None, None, None
            # Vérifier si l'ajustement est déjà optimisé
            if abs(adjusted_stop_distance - stop_distance) < 0.1:
                log_message(f"   ❌ Stop Distance {adjusted_stop_distance:.1f} déjà optimisé, abandon après {attempt + 1} tentatives")
                return None, None, None, None, None
            size = (available * RISK_PER_TRADE - MINIMUM_BALANCE_BUFFER) / (entry_price * MARGIN_FACTOR + spread_cost)
            log_message(f"   📊 Taille recalculée après ajustement stop-loss : {size:.4f}")
            size = max(size, min_size)
            size = min(size, max_size)
            size = round(size, 2)
            required_margin = entry_price * size * MARGIN_FACTOR
            total_cost = required_margin + spread_cost * size + MINIMUM_BALANCE_BUFFER
            log_message(f"   📊 Détail des coûts ajustés : Marge={required_margin:.2f}, Spread={spread_cost * size:.2f}, Buffer={MINIMUM_BALANCE_BUFFER:.2f}, Total={total_cost:.2f}")
            log_message(f"   📊 Exposition totale ajustée : {entry_price * size:.2f} EUR")
            log_message(f"   📊 Perte potentielle ajustée : {adjusted_stop_distance * size:.1f} EUR")
            if total_cost > available or size < min_size:
                log_message(f"   ❌ Impossible d'ajuster la taille après correction du stop-loss : Taille={size:.2f}, Coût total={total_cost:.2f} EUR")
                return None, None, None, None, None
            if direction == "BUY":
                limit_level = entry_price + profit_distance
            else:
                limit_level = entry_price - profit_distance
            log_message(f"   🔄 Réessai avec stop-loss ajusté : Stop Distance={adjusted_stop_distance:.1f}, Taille={size:.2f}")
            log_message(f"   📊 Perte potentielle ajustée (stop-loss) : {stop_loss_percentage:.2f}%")
            continue
        if isinstance(response, dict) and "error" in response:
            error = response["error"]
            log_message(f"   ❌ Erreur API pour ordre : {response}")
            if error == "INSUFFICIENT_FUNDS":
                log_message(f"   ❌ Échec (tentative {attempt + 1}) : Fonds insuffisants")
                size *= 0.95
                size = max(size, min_size)
                size = min(size, max_size)
                size = round(size, 2)
                required_margin = entry_price * size * MARGIN_FACTOR
                total_cost = required_margin + spread_cost * size + MINIMUM_BALANCE_BUFFER
                log_message(f"   📊 Détail des coûts ajustés : Marge={required_margin:.2f}, Spread={spread_cost * size:.2f}, Buffer={MINIMUM_BALANCE_BUFFER:.2f}, Total={total_cost:.2f}")
                log_message(f"   📊 Exposition totale ajustée : {entry_price * size:.2f} EUR")
                log_message(f"   📊 Perte potentielle ajustée : {adjusted_stop_distance * size:.1f} EUR")
                if total_cost > available or size < min_size:
                    log_message(f"   ❌ Impossible d'ajuster la taille : Taille={size:.2f}, Coût total={total_cost:.2f} EUR")
                    return None, None, None, None, None
                payload["size"] = float(size)
                log_message(f"   🔄 Réessai avec taille réduite : Taille={size:.2f}")
                continue
            elif error == "MARKET_CLOSED":
                log_message("   ❌ Marché fermé. Ordre impossible.")
                return None, None, None, None, None
            elif error == "RISK_CHECK":
                log_message(f"   ❌ Échec (tentative {attempt + 1}) : Rejet pour RISK_CHECK")
                size *= 0.95
                size = max(size, min_size)
                size = min(size, max_size)
                size = round(size, 2)
                required_margin = entry_price * size * MARGIN_FACTOR
                total_cost = required_margin + spread_cost * size + MINIMUM_BALANCE_BUFFER
                log_message(f"   📊 Détail des coûts ajustés : Marge={required_margin:.2f}, Spread={spread_cost * size:.2f}, Buffer={MINIMUM_BALANCE_BUFFER:.2f}, Total={total_cost:.2f}")
                log_message(f"   📊 Exposition totale ajustée : {entry_price * size:.2f} EUR")
                log_message(f"   📊 Perte potentielle ajustée : {adjusted_stop_distance * size:.1f} EUR")
                if total_cost > available or size < min_size:
                    log_message(f"   ❌ Impossible d'ajuster la taille : Taille={size:.2f}, Coût total={total_cost:.2f} EUR")
                    return None, None, None, None, None
                payload["size"] = float(size)
                log_message(f"   🔄 Réessai avec taille réduite pour RISK_CHECK : Taille={size:.2f}")
                continue
            else:
                log_message(f"   ❌ Échec de l'ordre {direction} : {error}")
                return None, None, None, None, None
        if response and response.ok:
            try:
                response_data = response.json()
                log_message(f"   📝 Réponse de l'API pour l'ordre : {response_data}")
                deal_ref = response_data.get("dealReference")
                if not deal_ref:
                    log_message("   ❌ Échec : Aucun dealReference retourné par l'API")
                    return None, None, None, None, None
                log_message(f"   🔍 Étape 7 : Vérification de l'exécution de l'ordre (dealRef={deal_ref})...")
                success, deal_id, executed_price = verify_order_execution(headers, deal_ref, entry_price, size)
                if not success:
                    log_message(f"   ❌ Ordre {deal_ref} annulé : problème d'exécution")
                    if deal_id:
                        log_message(f"   🔄 Tentative de fermeture de l'ordre partiellement exécuté (deal_id={deal_id})...")
                        close_position(headers, deal_id, direction, size)
                    return None, None, None, None, None
                if limit_level is None or profit_distance is None:
                    log_message(f"   ⚠️ Avertissement : limit_level ou profit_distance non défini avant stockage")
                    profit_distance = round(float(atr * TP_ATR_MULTIPLIER), 1)
                    limit_level = entry_price + profit_distance if direction == "BUY" else entry_price - profit_distance
                    log_message(f"   ℹ️ Recalcul de fallback : Limit-level={limit_level:.1f}, Profit Distance={profit_distance:.1f}")
                log_message(f"   ✅ Ordre {direction} passé avec succès (Taille: {size:.2f}, Deal ID: {deal_id})")
                log_message(f"   💾 Enregistrement de la position {deal_id} : Stop={stop_level:.1f}, Limit={limit_level:.1f}, Profit Distance={profit_distance:.1f}")
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
                log_message(f"   ❌ Erreur lors de l'analyse de la réponse JSON : {e}")
                return None, None, None, None, None
        log_message(f"   ❌ Échec de l'ordre {direction}. Réponse complète : {response.text if response else 'Aucune réponse'}")
        return None, None, None, None, None
    log_message(f"   ❌ Échec après {MAX_ORDER_RETRIES} tentatives")
    return None, None, None, None, None

# === Récupération données historiques (pour backtest) ===
def get_historical_candles(headers, epic, resolution="MINUTE_15", start_date=None, end_date=None, max_per_request=200):
    log_message(f"\n🔙 Récupération données historiques pour {epic} (résolution: {resolution})...")
    if start_date and end_date:
        log_message(f"   Période : {start_date} à {end_date}")
    url = f"{BASE_URL}/api/v1/prices/{epic}"
    all_candles = []
    params = {"resolution": resolution, "max": max_per_request}
    if start_date and end_date:
        params["from"] = start_date
        params["to"] = end_date
    response = safe_request("GET", url, headers=headers, params=params)
    if response:
        candles = response.json().get("prices", [])
        all_candles.extend(candles)
        log_message(f"   ✅ {len(all_candles)} bougies historiques récupérées.")
    if not all_candles:
        log_message("   ❌ Aucune donnée historique.")
        return None
    df = pd.DataFrame(all_candles)
    df["timestamp"] = pd.to_datetime(df["snapshotTime"], utc=True).dt.tz_convert("Europe/Paris")
    df.set_index("timestamp", inplace=True)
    df["close"] = df["closePrice"].apply(lambda x: x["bid"])
    df["open"] = df["openPrice"].apply(lambda x: x["bid"])
    df["high"] = df["highPrice"].apply(lambda x: x["bid"])
    df["low"] = df["lowPrice"].apply(lambda x: x["bid"])
    df["volume"] = df["lastTradedVolume"]
    log_message("   Données formatées pour backtest OK.")
    return df[["open", "high", "low", "close", "volume"]]

# === Backtesting ===
def backtest_strategy(headers, epic, initial_balance=10000.0, start_date=None, end_date=None, resolution="MINUTE_15",
                     periods_back=60, confirmation_period=2, atr_threshold=4.0, market="GOLD",
                     sl_atr_multiplier=SL_ATR_MULTIPLIER, tp_atr_multiplier=TP_ATR_MULTIPLIER, risk_per_trade=RISK_PER_TRADE):
    log_message("\n🔙 === DÉMARRAGE BACKTEST ===")
    log_message(f"   Période : {start_date} à {end_date} | Résolution : {resolution} | Solde initial : {initial_balance} EUR")
    df = get_historical_candles(headers, epic, resolution, start_date, end_date)
    if df is None or len(df) < 200:
        log_message("   ❌ Pas assez de données pour backtest (min 200 bougies).")
        return None
    df_trend = get_trend_candles(headers, epic, "HOUR_4", limit=50)
    balance = initial_balance
    equity = [initial_balance]
    trades = []
    position = None
    trade_start_time = None
    num_signals_checked = 0
    for i in range(max(periods_back + confirmation_period, 200), len(df)):
        num_signals_checked += 1
        if num_signals_checked % 100 == 0:
            log_message(f"   Backtest progression : {num_signals_checked}/{len(df)-200} itérations, Solde actuel : {balance:.2f}")
        window_df = df.iloc[i - 200:i]
        window_df = detect_signals(window_df, df_trend, periods_back, confirmation_period, atr_threshold, market)
        signal = window_df.iloc[-1]
        current_price = signal['close']
        atr = window_df['atr'].iloc[-1]
        stop_distance = atr * sl_atr_multiplier
        profit_distance = atr * tp_atr_multiplier
        spread_cost = 0.5  # Estimation
        size = (balance * risk_per_trade) / (current_price * MARGIN_FACTOR + spread_cost)
        if position:
            # Vérifier SL/TP
            closed = False
            if position['direction'] == 'BUY':
                if current_price <= position['stop_level']:
                    pnl = (position['stop_level'] - position['entry_price']) * position['size'] - spread_cost
                    balance += pnl
                    trades.append({'type': 'BUY_CLOSE_SL', 'pnl': pnl, 'duration': (df.index[i] - trade_start_time).total_seconds() / 3600 if trade_start_time else 0})
                    log_message(f"   📉 Trade fermé SL (BUY) : PnL = {pnl:.2f} EUR, Solde = {balance:.2f}")
                    closed = True
                elif current_price >= position['limit_level']:
                    pnl = (position['limit_level'] - position['entry_price']) * position['size'] - spread_cost
                    balance += pnl
                    trades.append({'type': 'BUY_CLOSE_TP', 'pnl': pnl, 'duration': (df.index[i] - trade_start_time).total_seconds() / 3600 if trade_start_time else 0})
                    log_message(f"   📈 Trade fermé TP (BUY) : PnL = {pnl:.2f} EUR, Solde = {balance:.2f}")
                    closed = True
            else:  # SELL
                if current_price >= position['stop_level']:
                    pnl = (position['entry_price'] - position['stop_level']) * position['size'] - spread_cost
                    balance += pnl
                    trades.append({'type': 'SELL_CLOSE_SL', 'pnl': pnl, 'duration': (df.index[i] - trade_start_time).total_seconds() / 3600 if trade_start_time else 0})
                    log_message(f"   📉 Trade fermé SL (SELL) : PnL = {pnl:.2f} EUR, Solde = {balance:.2f}")
                    closed = True
                elif current_price <= position['limit_level']:
                    pnl = (position['entry_price'] - position['limit_level']) * position['size'] - spread_cost
                    balance += pnl
                    trades.append({'type': 'SELL_CLOSE_TP', 'pnl': pnl, 'duration': (df.index[i] - trade_start_time).total_seconds() / 3600 if trade_start_time else 0})
                    log_message(f"   📈 Trade fermé TP (SELL) : PnL = {pnl:.2f} EUR, Solde = {balance:.2f}")
                    closed = True
            if closed:
                position = None
        if not position:
            if signal['buy_signal']:
                stop_level = current_price - stop_distance
                limit_level = current_price + profit_distance
                position = {'direction': 'BUY', 'entry_price': current_price, 'size': size,
                            'stop_level': stop_level, 'limit_level': limit_level}
                trades.append({'type': 'BUY_OPEN', 'pnl': -spread_cost})
                trade_start_time = df.index[i]
                log_message(f"   🛒 Trade ouvert BUY à {current_price:.2f}, Taille {size:.2f}, SL {stop_level:.2f}, TP {limit_level:.2f}")
            elif signal['sell_signal']:
                stop_level = current_price + stop_distance
                limit_level = current_price - profit_distance
                position = {'direction': 'SELL', 'entry_price': current_price, 'size': size,
                            'stop_level': stop_level, 'limit_level': limit_level}
                trades.append({'type': 'SELL_OPEN', 'pnl': -spread_cost})
                trade_start_time = df.index[i]
                log_message(f"   🛒 Trade ouvert SELL à {current_price:.2f}, Taille {size:.2f}, SL {stop_level:.2f}, TP {limit_level:.2f}")
        equity.append(balance)
    # Fermer position finale si ouverte
    if position:
        if position['direction'] == 'BUY':
            pnl = (df['close'].iloc[-1] - position['entry_price']) * position['size'] - spread_cost
        else:
            pnl = (position['entry_price'] - df['close'].iloc[-1]) * position['size'] - spread_cost
        balance += pnl
        trades.append({'type': 'CLOSE_END', 'pnl': pnl, 'duration': (df.index[-1] - trade_start_time).total_seconds() / 3600 if trade_start_time else 0})
        log_message(f"   📊 Fin backtest : Position fermée à fin période, PnL = {pnl:.2f} EUR")
    # Métriques
    pnls = [t['pnl'] for t in trades if 'CLOSE' in t['type']]
    total_pnl = balance - initial_balance
    win_rate = len([p for p in pnls if p > 0]) / len(pnls) * 100 if pnls else 0
    returns = pd.Series(equity).pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(len(returns)) if returns.std() != 0 else 0
    max_drawdown = ((pd.Series(equity).cummax() - equity).max() / pd.Series(equity).cummax().max()) * 100
    avg_trade_duration = np.mean([t.get('duration', 0) for t in trades if 'duration' in t]) if any('duration' in t for t in trades) else 0
    log_message("\n📊 === RÉSULTATS BACKTEST ===")
    log_message(f"   PnL total : {total_pnl:.2f} EUR ({(total_pnl / initial_balance * 100):.2f}%)")
    log_message(f"   Win Rate : {win_rate:.1f}% ({len([p for p in pnls if p > 0])} gains / {len(pnls)} trades)")
    log_message(f"   Ratio Sharpe : {sharpe_ratio:.2f}")
    log_message(f"   Max Drawdown : {max_drawdown:.2f}%")
    log_message(f"   Nombre de trades : {len(pnls)}")
    log_message(f"   Solde final : {balance:.2f} EUR")
    log_message(f"   Durée moyenne trade : {avg_trade_duration:.2f} heures")
    results = {
        'total_pnl': total_pnl,
        'win_rate': win_rate / 100,  # En décimal pour compatibilité
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown / 100,
        'num_trades': len(pnls),
        'final_balance': balance,
        'avg_trade_duration_hours': avg_trade_duration
    }
    return results

# === Optimisation paramètres ===
def optimize_parameters(headers, epic, initial_balance=10000.0, start_date=None, end_date=None, resolution="MINUTE_15"):
    log_message("\n🛠️ === OPTIMISATION PARAMÈTRES VIA GRID SEARCH ===")
    param_grid = {
        'sl_atr_multiplier': [1.5, 2.0, 2.5],
        'tp_atr_multiplier': [1.0, 1.5, 2.0],
        'risk_per_trade': [0.5, 0.8, 1.0]
    }
    log_message(f"   Grille testée : {len(list(itertools.product(*param_grid.values())))} combinaisons")
    best_sharpe = -np.inf
    best_params = None
    best_results = None
    for idx, params in enumerate(itertools.product(*param_grid.values()), 1):
        sl, tp, risk = params
        log_message(f"   Test {idx} : SL_mult={sl}, TP_mult={tp}, Risk={risk}")
        results = backtest_strategy(headers, epic, initial_balance, start_date, end_date, resolution,
                                   sl_atr_multiplier=sl, tp_atr_multiplier=tp, risk_per_trade=risk)
        if results and results['sharpe_ratio'] > best_sharpe:
            best_sharpe = results['sharpe_ratio']
            best_params = {'sl_atr_multiplier': sl, 'tp_atr_multiplier': tp, 'risk_per_trade': risk}
            best_results = results
            log_message(f"   🏆 Nouveau meilleur ! Sharpe = {best_sharpe:.2f}")
    log_message(f"\n✅ OPTIMISATION TERMINÉE : Meilleurs params = {best_params}")
    log_message(f"   Sharpe optimal : {best_sharpe:.2f}")
    return best_params, best_results

# === Boucle principale bot ===
def trading_bot(check_interval=60, confirmation_period=2, atr_threshold=4.0, market="GOLD"):
    global peak_equity
    log_message("\n🚀 === DÉMARRAGE DU BOT EN LIVE ===")
    log_message(f"   Intervalle check : {check_interval}s | Marché : {market} | Compteur itérations : 0")
    auth_headers = authenticate()
    if not auth_headers:
        log_message("❌ Impossible de démarrer : Échec connexion. Arrêt bot.", "error")
        return
    epic = search_market(auth_headers, market)
    if not epic:
        log_message(f"❌ Impossible de démarrer : Marché {market} non trouvé. Arrêt bot.", "error")
        return
    log_message(f"📌 Bot configuré pour trader sur EPIC : {epic}")
    iteration = 0
    last_sms_date = None  # Pour éviter d'envoyer plusieurs SMS le même jour
    while True:
        iteration += 1
        log_message(f"\n{'='*50}")
        log_message(f"🔄 ITÉRATION {iteration} - {datetime.now(pytz.timezone('Europe/Paris')).strftime('%Y-%m-%d %H:%M:%S')}")
        log_message(f"{'='*50}")
        try:
            log_message("   Check drawdown...")
            available, equity = get_balances(auth_headers)
            # Mise à jour du pic sur equity
            if equity > peak_equity:
                peak_equity = equity
                log_message(f"   📈 Nouveau pic d'équité : {peak_equity:.2f} EUR")
            # Calcul drawdown sur equity
            if peak_equity > 0:
                drawdown = ((peak_equity - equity) / peak_equity) * 100
            else:
                drawdown = 0.0
            log_message(f"   Equity actuelle : {equity:.2f} EUR | Drawdown (du pic) : {drawdown:.2f}% | Pic equity : {peak_equity:.2f} EUR")
            log_message(f"   Available (pour trades) : {available:.2f} EUR")
            # Envoi SMS quotidien à 20h00 (une fois par jour)
            now = datetime.now(pytz.timezone("Europe/Paris"))
            current_date = now.date()
            if now.hour == 23 and now.minute == 55 and last_sms_date != current_date:
                send_sms_via_email(equity)
                last_sms_date = current_date
            # Vérification sur drawdown d'equity
            if drawdown > MAX_DRAWDOWN_PERCENT:
                log_message(f"   ❌ Drawdown max atteint ({drawdown:.2f}% > {MAX_DRAWDOWN_PERCENT}%). ARRÊT BOT.", "error")
                break
            if not is_market_open():
                log_message(f"   ⏳ Marché fermé ou news en cours. Pause {check_interval * 2}s.")
                time.sleep(check_interval * 2)
                continue
            log_message("   Récupération données prix...")
            df = get_candles(auth_headers, epic, resolution="MINUTE_15")
            df_trend = get_trend_candles(auth_headers, epic, resolution="HOUR_4")
            if df is not None and df_trend is not None:
                log_message("   Analyse signaux...")
                df = detect_signals(df, df_trend, periods_back=60, confirmation_period=confirmation_period,
                                    atr_threshold=atr_threshold, market=market)
                current_price = df["close"].iloc[-1]
                signal = df.iloc[-1]
                log_message(f"   🔔 Résultat analyse : Prix actuel = {current_price:.2f}")
                log_message(f"   Signal Achat : {signal['buy_signal']} | Signal Vente : {signal['sell_signal']}")
                position = is_position_open(auth_headers, epic)
                if signal["buy_signal"]:
                    log_message("   🔍 Signal d'ACHAT détecté !")
                    if position and position["direction"] == "BUY":
                        log_message("   ℹ️ Position BUY déjà ouverte. Rien à faire.")
                        continue
                    if position:
                        log_message(f"   🔄 Fermeture position existante ({position['direction']}) avant nouvel ordre...")
                        if close_position(auth_headers, position["deal_id"], position["direction"], position["size"]):
                            log_message("   Fermeture réussie.")
                        else:
                            log_message("   Échec fermeture. Skip ordre.")
                            continue
                    log_message(f"   📈 Passage ordre BUY à {current_price:.2f}...")
                    response, stop_level, limit_level, stop_loss_percentage, take_profit_percentage = place_order(
                        auth_headers, epic, "BUY", current_price, df
                    )
                    if response:
                        log_message("   ✅ Ordre BUY exécuté avec succès ! Position ouverte.")
                    else:
                        log_message("   ❌ Échec ordre BUY.")
                elif signal["sell_signal"]:
                    log_message("   🔍 Signal de VENTE détecté !")
                    if position and position["direction"] == "SELL":
                        log_message("   ℹ️ Position SELL déjà ouverte. Rien à faire.")
                        continue
                    if position:
                        log_message(f"   🔄 Fermeture position existante ({position['direction']}) avant nouvel ordre...")
                        if close_position(auth_headers, position["deal_id"], position["direction"], position["size"]):
                            log_message("   Fermeture réussie.")
                        else:
                            log_message("   Échec fermeture. Skip ordre.")
                            continue
                    log_message(f"   📉 Passage ordre SELL à {current_price:.2f}...")
                    response, stop_level, limit_level, stop_loss_percentage, take_profit_percentage = place_order(
                        auth_headers, epic, "SELL", current_price, df
                    )
                    if response:
                        log_message("   ✅ Ordre SELL exécuté avec succès ! Position ouverte.")
                    else:
                        log_message("   ❌ Échec ordre SELL.")
                else:
                    log_message("   ℹ️ Aucun signal détecté. Attente...")
            else:
                log_message("   ❌ Données de prix ou tendance manquantes. Skip itération.")
            log_message(f"   ⏳ Pause de {check_interval} secondes avant itération {iteration+1}...")
            time.sleep(check_interval)
        except Exception as e:
            log_message(f"   ❌ Erreur inattendue dans la boucle : {e}", "error")
            log_message(f"   Type erreur : {type(e).__name__}", "error")
            log_message("   ⏳ Pause et retry...")
            time.sleep(check_interval)

# === Lancement bot ===
if __name__ == "__main__":
    log_message("🎯 Lancement du script principal...")
    log_message("Pour tester backtest, décommente les lignes ci-dessous.")
    # auth_headers = authenticate()
    # epic = search_market(auth_headers, MARKET)
    # backtest_strategy(auth_headers, epic, start_date="2024-01-01", end_date="2024-12-31")
    # optimize_parameters(auth_headers, epic, start_date="2024-01-01", end_date="2024-12-31")
    trading_bot(check_interval=CHECK_INTERVAL, confirmation_period=CONFIRMATION_PERIOD,
                atr_threshold=ATR_THRESHOLD, market=MARKET)