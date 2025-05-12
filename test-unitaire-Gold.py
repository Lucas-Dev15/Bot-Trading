import requests
import time

# Configuration API Capital.com
API_URL = "https://demo-api-capital.backend-capital.com/api/v1"
API_KEY = "K6meo5BSJuuduWI8"
EMAIL = "5rycnytkzh@privaterelay.appleid.com"
PASSWORD = "Lucas1234@"
EPIC_GOLD = "GOLD"

def get_auth_token():
    headers = {"X-CAP-API-KEY": API_KEY}
    payload = {"identifier": EMAIL, "password": PASSWORD}
    response = requests.post(f"{API_URL}/session", headers=headers, json=payload)
    response.raise_for_status()
    return response.headers["CST"], response.headers["X-SECURITY-TOKEN"]

def get_market_details(epic, cst, security_token):
    headers = {"X-CAP-API-KEY": API_KEY, "CST": cst, "X-SECURITY-TOKEN": security_token}
    response = requests.get(f"{API_URL}/markets/{epic}", headers=headers)
    response.raise_for_status()
    return response.json()

def get_current_price(epic, cst, security_token):
    headers = {"X-CAP-API-KEY": API_KEY, "CST": cst, "X-SECURITY-TOKEN": security_token}
    response = requests.get(f"{API_URL}/prices/{epic}?resolution=MINUTE", headers=headers)
    response.raise_for_status()
    data = response.json()
    return data["prices"][-1]["closePrice"]["bid"], data["prices"][-1]["closePrice"]["ask"]

def place_order(epic, direction, size, cst, security_token, tp_distance=None, sl_distance=None):
    headers = {"X-CAP-API-KEY": API_KEY, "CST": cst, "X-SECURITY-TOKEN": security_token}
    payload = {
        "epic": epic,
        "direction": direction,
        "size": size,
        "orderType": "MARKET",
        "guaranteedStop": sl_distance is not None,
        "trailingStop": False
    }
    if tp_distance:
        payload["profitDistance"] = tp_distance
    if sl_distance:
        payload["stopDistance"] = sl_distance

    response = requests.post(f"{API_URL}/positions", headers=headers, json=payload)
    response.raise_for_status()
    deal_ref = response.json().get("dealReference")
    time.sleep(2)

    confirm_resp = requests.get(f"{API_URL}/confirms/{deal_ref}", headers=headers)
    confirm_resp.raise_for_status()
    confirm_data = confirm_resp.json()

    affected_deals = confirm_data.get("affectedDeals", [])
    if not affected_deals:
        raise Exception("Aucun dealId trouvé.")
    return deal_ref, affected_deals[0]["dealId"], confirm_data["level"]

def get_position_details(deal_id, cst, security_token):
    headers = {"X-CAP-API-KEY": API_KEY, "CST": cst, "X-SECURITY-TOKEN": security_token}
    response = requests.get(f"{API_URL}/positions/{deal_id}", headers=headers)
    response.raise_for_status()
    return response.json()

def list_open_positions(cst, security_token):
    headers = {"X-CAP-API-KEY": API_KEY, "CST": cst, "X-SECURITY-TOKEN": security_token}
    response = requests.get(f"{API_URL}/positions", headers=headers)
    response.raise_for_status()
    return response.json().get("positions", [])

def close_position(deal_id, cst, security_token, size, direction):
    headers = {"X-CAP-API-KEY": API_KEY, "CST": cst, "X-SECURITY-TOKEN": security_token}
    payload = {"direction": direction, "size": size, "orderType": "MARKET"}
    response = requests.delete(f"{API_URL}/positions/{deal_id}", headers=headers, json=payload)
    response.raise_for_status()

def main():
    try:
        cst, token = get_auth_token()
        print("✅ Authentification réussie.")

        market = get_market_details(EPIC_GOLD, cst, token)
        min_size = market.get("dealingRules", {}).get("minDealSize", {}).get("value", 0.01)

        bid, ask = get_current_price(EPIC_GOLD, cst, token)
        print(f"🎯 Prix actuel - Bid: {bid}, Ask: {ask}")

        tp_dist = 200
        sl_dist = 100

        # Achat
        print("🟢 Envoi ordre d'achat...")
        buy_ref, buy_deal_id, level = place_order(EPIC_GOLD, "BUY", min_size, cst, token, tp_distance=tp_dist, sl_distance=sl_dist)
        print(f"Achat exécuté à {level} (Deal ID: {buy_deal_id})")

        time.sleep(15)
        for pos in list_open_positions(cst, token):
            if pos["market"]["epic"] == EPIC_GOLD and pos["position"]["direction"] == "BUY":
                close_position(pos["position"]["dealId"], cst, token, min_size, direction="SELL")
                print("❌ Achat fermé.")
                break

        # Vente
        print("🔴 Envoi ordre de vente...")
        sell_ref, sell_deal_id, level = place_order(EPIC_GOLD, "SELL", min_size, cst, token, tp_distance=tp_dist, sl_distance=sl_dist)
        print(f"Vente exécutée à {level} (Deal ID: {sell_deal_id})")

        time.sleep(15)
        for pos in list_open_positions(cst, token):
            if pos["market"]["epic"] == EPIC_GOLD and pos["position"]["direction"] == "SELL":
                close_position(pos["position"]["dealId"], cst, token, min_size, direction="BUY")
                print("❌ Vente fermée.")
                break

    except Exception as e:
        print(f"❗ Erreur : {e}")

if __name__ == "__main__":
    main()
