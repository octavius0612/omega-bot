import os
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
from textblob import TextBlob
import requests
import google.generativeai as genai
import json
import time
import threading
from flask import Flask
from datetime import datetime, time as dtime, timedelta
import pytz
from github import Github

app = Flask(__name__)

# --- 1. CLÉS ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")

# --- 2. RÉGLAGES INITIAUX ---
WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "MSFT", "AMD", "PLTR", "META", "GOOG", "NFLX", "COIN", "MSTR"]
INITIAL_CAPITAL = 25000.0 

# Mémoire avec Paramètres Evolutifs (C'est ça qu'il va optimiser le weekend)
brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "stats": {"wins": 0, "losses": 0},
    # Paramètres par défaut (que le bot va apprendre à améliorer)
    "params": {
        "rsi_buy": 30,      # Niveau RSI pour acheter
        "stop_loss_atr": 2.0, # Distance Stop Loss
        "monte_carlo_threshold": 60 # Seuil confiance
    },
    "training_log": []
}
bot_status = "Démarrage..."

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)

# --- 3. HEARTBEAT 30 SECONDES ---
def run_heartbeat():
    print("💓 Heartbeat 30s activé.")
    while True:
        try:
            if HEARTBEAT_WEBHOOK_URL:
                requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"})
            time.sleep(30) # Ping toutes les 30 secondes
        except:
            time.sleep(30)

# --- 4. GESTION MÉMOIRE ---
def load_brain():
    global brain
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        c = repo.get_contents("brain.json")
        loaded = json.loads(c.decoded_content.decode())
        # On fusionne pour garder les nouveaux champs
        brain.update(loaded)
    except: pass

def save_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        content = json.dumps(brain, indent=4)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Brain Update", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 5. LE DOJO (ENTRAÎNEMENT WEEKEND) ---
def train_brain_simulation():
    """
    Fonction lourde : Rejoue le passé pour optimiser les paramètres.
    Ne tourne que le Weekend.
    """
    global brain, bot_status
    bot_status = "🏋️ ENTRAÎNEMENT INTENSIF..."
    print("Démarrage simulation entraînement...")
    
    best_pnl = -999999
    best_params = brain['params'].copy()
    
    # On télécharge les données une fois pour toutes
    data_cache = {}
    for s in WATCHLIST[:3]: # On s'entraîne sur 3 actions clés pour aller vite
        df = yf.Ticker(s).history(period="1mo", interval="1h") # Données 1 mois
        if not df.empty:
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            data_cache[s] = df.dropna()

    # On teste 10 variations de paramètres aléatoires
    for i in range(10):
        # Mutation aléatoire
        test_rsi = np.random.randint(20, 45)
        test_sl = np.random.uniform(1.5, 3.5)
        
        simulated_pnl = 0
        
        # Backtest rapide
        for s, df in data_cache.items():
            entry_price = 0
            in_position = False
            
            for index, row in df.iterrows():
                if not in_position and row['RSI'] < test_rsi:
                    entry_price = row['Close']
                    in_position = True
                    stop_price = entry_price - (row['ATR'] * test_sl)
                
                elif in_position:
                    # Stop Loss touché ?
                    if row['Low'] < stop_price:
                        simulated_pnl += (stop_price - entry_price)
                        in_position = False
                    # Take profit simple (fixe pour simulation)
                    elif row['High'] > entry_price * 1.05:
                        simulated_pnl += (entry_price*0.05)
                        in_position = False
        
        # Si cette variation est meilleure que l'actuelle
        if simulated_pnl > best_pnl:
            best_pnl = simulated_pnl
            best_params = {"rsi_buy": test_rsi, "stop_loss_atr": test_sl, "monte_carlo_threshold": 60}
            print(f"🚀 Nouveaux paramètres trouvés ! PnL Simulé: {best_pnl:.2f}")

    # Mise à jour du cerveau avec les paramètres gagnants
    brain['params'] = best_params
    msg = f"🧠 **ENTRAÎNEMENT TERMINÉ**\nLe bot a optimisé sa stratégie pour Lundi.\n\n**Nouveaux Paramètres :**\n🎯 RSI Achat: < {best_params['rsi_buy']}\n🛡️ Stop Loss ATR: {best_params['stop_loss_atr']:.2f}"
    
    if DISCORD_WEBHOOK_URL:
        requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [{"title": "🏋️ DOJO REPORT", "description": msg, "color": 0xff00ff}]})
    
    save_brain()
    time.sleep(3600) # Pause 1h après un entraînement

# --- 6. ANALYSE OMEGA ---
def get_data(s):
    try:
        df = yf.Ticker(s).history(period="1mo", interval="15m")
        if df.empty: return None
        
        returns = df['Close'].pct_change().dropna()
        last = df['Close'].iloc[-1]
        sims = last * (1 + np.random.normal(returns.mean(), returns.std(), 1000))
        prob_up = np.sum(sims > last) / 10 
        
        mean = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        z = (last - mean.iloc[-1]) / std.iloc[-1]
        
        return {
            "s": s, "p": last, "prob": prob_up, "z": z,
            "rsi": ta.rsi(df['Close'], length=14).iloc[-1],
            "atr": ta.atr(df['High'], df['Low'], df['Close'], length=14).iloc[-1],
            "inst": yf.Ticker(s).info.get('heldPercentInstitutions', 0)*100
        }
    except: return None

# --- 7. CERVEAU GEMINI ---
def ask_omega(d):
    # Utilisation des paramètres appris par le bot
    rsi_limit = brain['params']['rsi_buy']
    
    prompt = f"Action: {d['s']}. Monte Carlo: {d['prob']:.1f}%. Z-Score: {d['z']:.2f}. RSI: {d['rsi']:.1f} (Limite: {rsi_limit}). Si Monte Carlo > 60% et RSI < {rsi_limit}: BUY. Sinon WAIT. JSON: {{'decision':'BUY/WAIT', 'reason':'short'}}"
    try:
        model = genai.GenerativeModel('gemini-pro')
        res = model.generate_content(prompt)
        return json.loads(res.text.replace("```json","").replace("```",""))
    except: return {"decision": "WAIT"}

# --- 8. MOTEUR TRADING ---
def run_trading():
    global brain, bot_status
    load_brain()
    count = 0
    while True:
        try:
            count += 1
            if count >= 5: save_brain(); count = 0
            
            # --- WEEKEND / NUIT : ENTRAINEMENT ---
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if not market_open:
                # Si le marché est fermé, on lance l'entraînement
                train_brain_simulation()
                bot_status = "🌙 Repos / Entraînement"
                time.sleep(60)
                continue

            # --- JOURNEE : TRADING ---
            bot_status = "🟢 Scan..."
            
            # Gestion Vente
            for s in list(brain['holdings'].keys()):
                pos = brain['holdings'][s]
                curr = yf.Ticker(s).fast_info['last_price']
                
                if curr < pos['stop']:
                    brain['cash'] += pos['qty'] * curr
                    del brain['holdings'][s]
                    if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json={"content": f"🔴 VENTE {s} (Stop Loss)"})
                    save_brain()

            # Scan Achat
            if len(brain['holdings']) < 5:
                for s in WATCHLIST:
                    if s in brain['holdings']: continue
                    d = get_data(s)
                    if d:
                        dec = ask_omega(d)
                        # Utilisation du seuil appris
                        mc_threshold = brain['params']['monte_carlo_threshold']
                        
                        if dec['decision'] == "BUY" and d['prob'] > mc_threshold and brain['cash'] > 500:
                            bet = brain['cash'] * 0.15 
                            brain['cash'] -= bet
                            qty = bet / d['p']
                            # Stop Loss appris
                            sl = d['p'] - (d['atr'] * brain['params']['stop_loss_atr'])
                            
                            brain['holdings'][s] = {"qty": qty, "entry": d['p'], "stop": sl}
                            if DISCORD_WEBHOOK_URL: 
                                requests.post(DISCORD_WEBHOOK_URL, json={"content": f"🟢 ACHAT OMEGA {s} | Mise: {bet:.2f}$ | Param: RSI<{brain['params']['rsi_buy']}"})
                            save_brain()
            
            time.sleep(60)
        except: time.sleep(10)

@app.route('/')
def index(): return f"<h1>OMEGA SCHOLAR V27</h1><p>{bot_status}</p><p>Params: {brain.get('params', 'Loading...')}</p>"

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
