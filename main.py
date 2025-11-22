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
WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "MSFT", "AMD", "PLTR", "COIN", "MSTR"]
INITIAL_CAPITAL = 25000.0 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "params": {"rsi_buy": 30, "stop_loss_atr": 2.0, "monte_carlo_threshold": 60},
    "best_pnl_found": 0
}
bot_status = "Démarrage..."

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)

# --- 3. COMMUNICATION ---
def log_thought(emoji, text):
    """Parle dans le salon cerveau"""
    print(f"🧠 {emoji} {text}")
    if LEARNING_WEBHOOK_URL:
        try:
            requests.post(LEARNING_WEBHOOK_URL, json={"content": f"{emoji} **IA:** {text}"})
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

# --- 5. TRAINING INFINI (BOUCLE CONTINUE) ---
def run_continuous_learning():
    """
    Cette fonction tourne en boucle quand le marché est fermé.
    Elle ne s'arrête JAMAIS.
    """
    global brain, bot_status
    bot_status = "🧠 Learning Loop..."
    
    log_thought("🧬", "Démarrage du protocole d'apprentissage profond (Deep Learning Loop).")
    
    # On charge les données UNE FOIS pour ne pas spammer Yahoo
    cache_data = {}
    for s in ["NVDA", "TSLA", "BTC-USD"]:
        try:
            df = yf.Ticker(s).history(period="1mo", interval="1h")
            if not df.empty:
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                cache_data[s] = df.dropna()
        except: pass
    
    iteration = 0
    
    # BOUCLE INFINIE D'APPRENTISSAGE
    while True:
        # Vérification si le marché ouvre (pour sortir de la boucle)
        nyc = pytz.timezone('America/New_York')
        now = datetime.now(nyc)
        if now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0):
            log_thought("🔔", "Marché ouvert ! Fin de l'entraînement. Passage en mode Trading.")
            break
            
        iteration += 1
        
        # 1. Mutation : On invente de nouveaux paramètres au hasard
        test_rsi = np.random.randint(20, 50)
        test_sl = np.random.uniform(1.5, 4.0)
        test_mc = np.random.randint(50, 80)
        
        # 2. Simulation Rapide (Backtest) sur les données en cache
        total_sim_pnl = 0
        trades = 0
        
        # On choisit un actif au hasard pour tester
        s = random.choice(list(cache_data.keys()))
        df = cache_data[s]
        
        entry = 0
        in_trade = False
        
        for index, row in df.iterrows():
            if not in_trade and row['RSI'] < test_rsi:
                entry = row['Close']
                in_trade = True
                stop = entry - (row['ATR'] * test_sl)
                tp = entry + (row['ATR'] * test_sl * 1.5)
            
            elif in_trade:
                if row['Low'] < stop:
                    total_sim_pnl -= (entry - stop)
                    trades += 1
                    in_trade = False
                elif row['High'] > tp:
                    total_sim_pnl += (tp - entry)
                    trades += 1
                    in_trade = False
        
        # 3. Analyse du Résultat
        result_emoji = "❌"
        if total_sim_pnl > 0: result_emoji = "✅"
        
        log_msg = f"Sim #{iteration} sur {s}: RSI<{test_rsi}, SL={test_sl:.1f}. Résultat: {result_emoji} {total_sim_pnl:.2f}$ ({trades} trades)."
        
        # On envoie le message TOUT DE SUITE (Feedback Constant)
        log_thought("🧪", log_msg)
        
        # 4. Évolution (Si c'est mieux que le record actuel)
        if total_sim_pnl > brain.get('best_pnl_found', 0):
            brain['best_pnl_found'] = total_sim_pnl
            brain['params'] = {"rsi_buy": test_rsi, "stop_loss_atr": test_sl, "monte_carlo_threshold": test_mc}
            save_brain()
            log_thought("🚀", f"**NOUVEAU RECORD !** Je mets à jour ma stratégie pour Lundi.\nNouveau PnL record: {total_sim_pnl:.2f}$")
        
        # Petite pause pour ne pas flood Discord (10 secondes)
        time.sleep(10)

# --- 6. RESTE DU CODE (TRADING) ---
def get_data(s):
    try:
        df = yf.Ticker(s).history(period="1mo", interval="15m")
        if df.empty: return None
        last = df['Close'].iloc[-1]
        return {
            "s": s, "p": last, 
            "rsi": ta.rsi(df['Close'], length=14).iloc[-1],
            "atr": ta.atr(df['High'], df['Low'], df['Close'], length=14).iloc[-1],
            "prob": 65 # Simulation simple pour l'exemple
        }
    except: return None

def run_trading():
    global brain, bot_status
    load_brain()
    
    # Message de démarrage immédiat pour tester le webhook
    log_thought("👋", "Système V30 (Eternal Student) en ligne. Test des connexions...")
    
    count = 0
    while True:
        try:
            count += 1
            if count >= 10: save_brain(); count = 0
            
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if not market_open:
                # C'EST ICI QUE ÇA SE PASSE : Boucle d'apprentissage
                run_continuous_learning()
                # Quand on sort de la fonction (lundi matin), on reprend la boucle trading
                continue

            bot_status = "🟢 Trading Live..."
            # ... (Logique de trading inchangée pour la clarté) ...
            
            time.sleep(60)
        except Exception as e:
            print(f"Erreur: {e}")
            time.sleep(10)

@app.route('/')
def index(): return f"<h1>V30 ETERNAL</h1><p>{bot_status}</p>"

def start_threads():
    t1 = threading.Thread(target=run_trading)
    t1.daemon = True
    t1.start()
    t2 = threading.Thread(target=run_heartbeat)
    t2.daemon = True
    t2.start()

start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
