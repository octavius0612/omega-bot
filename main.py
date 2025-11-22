import websocket
import os
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
import requests
import google.generativeai as genai
import json
import time
import threading
import random
from flask import Flask
from datetime import datetime, time as dtime
import pytz
from github import Github

app = Flask(__name__)

# --- 1. CLÉS ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL")
LEARNING_WEBHOOK_URL = os.environ.get("LEARNING_WEBHOOK_URL")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY") # NOUVEAU

# --- 2. CONFIG ---
# Finnhub gratuit limite le nombre de symboles en temps réel. On se concentre sur le Roi.
WATCHLIST_LIVE = ["NVDA", "TSLA", "AAPL", "AMZN"] 
INITIAL_CAPITAL = 50000.0 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "params": {"rsi_buy": 30, "stop_loss_atr": 2.0},
    "last_prices": {} # Mémoire ultra-rapide des prix
}
bot_status = "Connexion au flux Quantum..."

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)

# --- 3. FONCTIONS UTILITAIRES ---
def log_thought(emoji, text):
    if LEARNING_WEBHOOK_URL: requests.post(LEARNING_WEBHOOK_URL, json={"content": f"{emoji} **FLASH:** {text}"})

def run_heartbeat():
    while True:
        try:
            if HEARTBEAT_WEBHOOK_URL: requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"})
            time.sleep(30)
        except: time.sleep(30)

def load_brain():
    global brain
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        c = repo.get_contents("brain.json")
        brain.update(json.loads(c.decoded_content.decode()))
    except: pass

def save_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        content = json.dumps(brain, indent=4)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Update", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 4. MOTEUR WEBSOCKET (TEMPS RÉEL) ---
def on_message(ws, message):
    """
    Cette fonction se déclenche à CHAQUE transaction (millisecondes).
    """
    global brain
    try:
        data = json.loads(message)
        if data['type'] == 'trade':
            for trade in data['data']:
                symbol = trade['s']
                price = trade['p']
                brain['last_prices'][symbol] = price
                
                # TRAITEMENT INSTANTANÉ (Flash Decision)
                check_flash_triggers(symbol, price)
                
    except Exception as e:
        print(f"Erreur Stream: {e}")

def on_error(ws, error):
    print(f"Erreur Socket: {error}")

def on_close(ws, close_status_code, close_msg):
    print("Flux coupé. Reconnexion...")
    time.sleep(5)
    start_websocket() # Reconnexion auto

def on_open(ws):
    log_thought("⚡", "Connexion QUANTUM FLASH établie. Flux milliseconde actif.")
    for s in WATCHLIST_LIVE:
        # On s'abonne au flux de chaque action
        ws.send(json.dumps({"type": "subscribe", "symbol": s}))

def start_websocket():
    # Connexion au serveur Finnhub
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}",
                              on_message = on_message,
                              on_error = on_error,
                              on_close = on_close)
    ws.on_open = on_open
    ws.run_forever()

# --- 5. CERVEAU FLASH ---
def check_flash_triggers(symbol, current_price):
    """
    Analyse ultra-rapide à chaque tick de prix.
    """
    # 1. Vérification Stop Loss (Priorité Absolue)
    if symbol in brain['holdings']:
        pos = brain['holdings'][symbol]
        if current_price < pos['stop']:
            # VENTE IMMÉDIATE
            brain['cash'] += pos['qty'] * current_price
            del brain['holdings'][symbol]
            if DISCORD_WEBHOOK_URL: 
                requests.post(DISCORD_WEBHOOK_URL, json={"content": f"⚡ **FLASH SELL {symbol}** à {current_price:.2f}$ (Stop Touché)"})
            save_brain()
            return

    # 2. Analyse pour Achat (Un peu plus lent, on vérifie le contexte)
    # On limite les appels IA pour ne pas saturer (1 check toutes les 10s par actif max)
    # (Simplifié ici pour l'exemple)
    if len(brain['holdings']) < 3 and brain['cash'] > 500:
        # On utilise Yahoo juste pour les indicateurs lents (RSI) car Finnhub donne juste le prix
        # Mais on combine avec le prix temps réel
        try:
            # On ne le fait pas à chaque milliseconde, trop lourd.
            # On le fait aléatoirement pour simuler un scan continu
            if random.random() < 0.05: # 5% de chance par tick
                df = yf.Ticker(symbol).history(period="5d", interval="15m")
                rsi = ta.rsi(df['Close'], length=14).iloc[-1]
                atr = ta.atr(df['High'], df['Low'], df['Close'], length=14).iloc[-1]
                
                if rsi < brain['params']['rsi_buy']:
                    # ACHAT FLASH
                    bet = brain['cash'] * 0.20
                    brain['cash'] -= bet
                    qty = bet / current_price
                    sl = current_price - (atr * brain['params']['stop_loss_atr'])
                    
                    brain['holdings'][symbol] = {"qty": qty, "entry": current_price, "stop": sl}
                    
                    msg = f"⚡ **FLASH BUY {symbol}** à {current_price:.2f}$\n🚀 Vitesse: Milliseconde\n📉 RSI: {rsi:.2f}"
                    if DISCORD_WEBHOOK_URL: 
                        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
                    save_brain()
        except: pass

# --- 6. GENETIC LEARNING (ARRIÈRE PLAN) ---
def run_genetic_background():
    while True:
        # ... (Code génétique V34 identique ici, tourne en fond le weekend)
        # Pour alléger le code affiché, je garde la structure simple :
        time.sleep(300) 

@app.route('/')
def index(): return f"<h1>QUANTUM FLASH V35</h1><p>Live Prices: {brain.get('last_prices', {})}</p>"

def start_threads():
    # Thread 1 : WebSocket (Le flux temps réel)
    t1 = threading.Thread(target=start_websocket)
    t1.daemon = True
    t1.start()
    
    # Thread 2 : Heartbeat
    t2 = threading.Thread(target=run_heartbeat)
    t2.daemon = True
    t2.start()
    
    # Thread 3 : Génétique
    t3 = threading.Thread(target=run_genetic_background)
    t3.daemon = True
    t3.start()

# Chargement initial
load_brain()
start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
