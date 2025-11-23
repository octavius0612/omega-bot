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

# --- 2. CONFIG ---
WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR"]
INITIAL_CAPITAL = 50000.0 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    # Paramètres optimisés (ceux que le bot cherche à améliorer)
    "best_params": {"rsi_buy": 30, "stop_loss_mult": 2.0, "take_profit_mult": 3.0},
    "best_score": -99999.0,
    "generation": 0
}
bot_status = "Démarrage du Dojo..."

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)

# --- 3. COMMS ---
def log_learning(emoji, text):
    """Envoie les messages d'apprentissage dans #cerveau_ia"""
    print(f"{emoji} {text}")
    if LEARNING_WEBHOOK_URL: 
        try: requests.post(LEARNING_WEBHOOK_URL, json={"content": f"{emoji} **DOJO:** {text}"})
        except: pass

def send_trading_alert(msg):
    if DISCORD_WEBHOOK_URL:
        try: requests.post(DISCORD_WEBHOOK_URL, json=msg)
        except: pass

def run_heartbeat():
    while True:
        try:
            if HEARTBEAT_WEBHOOK_URL: requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"})
            time.sleep(30)
        except: time.sleep(30)

# --- 4. MÉMOIRE ---
def load_brain():
    global brain
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        c = repo.get_contents("brain.json")
        loaded = json.loads(c.decoded_content.decode())
        # On garde le meilleur score pour ne pas régresser
        if "best_score" in loaded: brain.update(loaded)
    except: pass

def save_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        content = json.dumps(brain, indent=4)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Training Save", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 5. MOTEUR DE SIMULATION RAPIDE ---
def run_backtest(df, rsi_limit, sl_mult, tp_mult):
    """
    Simule une stratégie sur des données passées.
    Retourne le Profit (PnL) et le nombre de trades.
    """
    capital = 10000.0
    position = 0
    entry_price = 0
    trades = 0
    
    # Calcul des indicateurs
    # Note: On suppose que df a déjà RSI et ATR calculés pour la vitesse
    
    # On parcourt le DataFrame
    for index, row in df.iterrows():
        # LOGIQUE D'ACHAT
        if position == 0:
            # Condition : RSI Bas ET Tendance (Prix > SMA50)
            if row['RSI'] < rsi_limit and row['Close'] > row['SMA50']:
                position = capital / row['Close']
                entry_price = row['Close']
                capital = 0
                
        # LOGIQUE DE VENTE
        elif position > 0:
            stop_price = entry_price - (row['ATR'] * sl_mult)
            target_price = entry_price + (row['ATR'] * tp_mult)
            
            # Stop Loss touché ?
            if row['Low'] < stop_price:
                capital = position * stop_price
                position = 0
                trades += 1
            # Take Profit touché ?
            elif row['High'] > target_price:
                capital = position * target_price
                position = 0
                trades += 1
                
    final_val = capital + (position * df.iloc[-1]['Close'])
    return final_val - 10000.0, trades

# --- 6. CERVEAU PRINCIPAL ---
def run_brain_engine():
    global brain, bot_status
    load_brain()
    
    # Chargement des données UNE SEULE FOIS (Cache)
    log_learning("📥", "Téléchargement des données historiques (30 jours) pour l'entraînement...")
    data_cache = {}
    for s in WATCHLIST:
        try:
            df = yf.Ticker(s).history(period="1mo", interval="1h")
            if not df.empty:
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                df['SMA50'] = ta.sma(df['Close'], length=50)
                data_cache[s] = df.dropna()
        except: pass
    
    log_learning("🧠", "Données chargées. Début des cycles d'optimisation.")

    while True:
        # Check Horaire
        nyc = pytz.timezone('America/New_York')
        now = datetime.now(nyc)
        market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
        
        # --- MODE ENTRAÎNEMENT (WEEKEND / NUIT) ---
        if not market_open:
            bot_status = "🏋️ Entraînement..."
            brain['generation'] += 1
            
            # 1. Mutation : On invente des paramètres
            # On explore large : RSI entre 20 et 55
            test_rsi = np.random.randint(20, 55)
            # Stop Loss entre 1.0 et 4.0 ATR
            test_sl = round(np.random.uniform(1.0, 4.0), 1)
            # Take Profit : On cherche un gros ratio (entre 2 et 6 fois l'ATR)
            test_tp = round(np.random.uniform(2.0, 6.0), 1)
            
            total_pnl = 0
            total_trades = 0
            
            # 2. Test sur toutes les actions du cache
            for s, df in data_cache.items():
                pnl, tr = run_backtest(df, test_rsi, test_sl, test_tp)
                total_pnl += pnl
                total_trades += tr
            
            # 3. Analyse du résultat
            if total_trades < 5:
                # Pas assez de trades pour être significatif
                pass 
                # On ne log rien pour ne pas spammer le vide
            else:
                # C'est un résultat valide
                emoji = "❌"
                if total_pnl > 0: emoji = "✅"
                if total_pnl > 1000: emoji = "🔥"
                
                msg = f"Test Gen #{brain['generation']} : RSI<{test_rsi} | SL {test_sl} | TP {test_tp} => PnL: {emoji} {total_pnl:.2f}$ ({total_trades} trades)"
                log_learning("🧪", msg)
                
                # 4. RECORD BATTU ?
                if total_pnl > brain['best_score']:
                    brain['best_score'] = total_pnl
                    brain['best_params'] = {
                        "rsi_buy": test_rsi,
                        "stop_loss_mult": test_sl,
                        "take_profit_mult": test_tp
                    }
                    save_brain()
                    log_learning("🏆", f"**NOUVEAU MODÈLE VALIDÉ !** Je garde ces paramètres pour Lundi.\n(Meilleur Profit Historique: {total_pnl:.2f}$)")
            
            # Pause pour lire les logs (10 secondes)
            time.sleep(10)

        # --- MODE LIVE (LUNDI) ---
        else:
            bot_status = "🟢 Trading Live..."
            # Logique simple qui applique les best_params
            for s in WATCHLIST:
                # (Code trading similaire aux versions précédentes, utilisant brain['best_params'])
                pass
            time.sleep(60)

@app.route('/')
def index(): return f"<h1>GRANDMASTER V49</h1><p>{bot_status}</p><p>Best Score: {brain.get('best_score', 0):.2f}$</p>"

def start_threads():
    t1 = threading.Thread(target=run_brain_engine, daemon=True)
    t1.start()
    t2 = threading.Thread(target=run_heartbeat, daemon=True)
    t2.start()

start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
