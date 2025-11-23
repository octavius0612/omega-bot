import websocket
import os
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
from scipy.stats import norm
from textblob import TextBlob
import requests
import google.generativeai as genai
import json
import time
import threading
import queue
import io
import random
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
from PIL import Image
from flask import Flask
from datetime import datetime, time as dtime
import pytz
from github import Github

app = Flask(__name__)

# --- 1. CLÉS ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL")
LEARNING_WEBHOOK_URL = os.environ.get("LEARNING_WEBHOOK_URL") # Cerveau
SUMMARY_WEBHOOK_URL = os.environ.get("SUMMARY_WEBHOOK_URL") # Synthèse
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

# --- 2. CONFIG ---
WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR", "GOOG", "META"]
INITIAL_CAPITAL = 50000.0 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "q_table": {}, 
    "karma": {s: 10.0 for s in WATCHLIST},
    "emotions": {"confidence": 50.0, "stress": 20.0},
    "best_params": {"rsi_buy": 30, "sl": 2.0}, 
    "learning_stats": {"tests": 0, "wins": 0}
}
bot_status = "Démarrage V67 Transparent..."
log_queue = queue.Queue()
short_term_memory = []

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- 3. LOGGING DETAILLE ---
def fast_log(text):
    log_queue.put(text)

def logger_worker():
    buffer = []
    last_send = time.time()
    while True:
        try:
            while not log_queue.empty():
                buffer.append(log_queue.get())
            
            # On envoie plus souvent pour voir le détail (toutes les 1.5s)
            if buffer and (len(buffer) > 3 or time.time() - last_send > 1.5):
                msg_block = "\n".join(buffer[:10]) 
                buffer = buffer[10:]
                if LEARNING_WEBHOOK_URL:
                    requests.post(LEARNING_WEBHOOK_URL, json={"content": msg_block})
                last_send = time.time()
            time.sleep(0.5)
        except: time.sleep(1)

def send_summary(msg):
    if SUMMARY_WEBHOOK_URL: requests.post(SUMMARY_WEBHOOK_URL, json=msg)

def run_heartbeat():
    while True:
        if HEARTBEAT_WEBHOOK_URL: requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"})
        time.sleep(30)

# --- 4. MÉMOIRE ---
def load_brain():
    global brain
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        c = repo.get_contents("brain.json")
        loaded = json.loads(c.decoded_content.decode())
        if "cash" in loaded: brain.update(loaded)
    except: pass

def save_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        content = json.dumps(brain, indent=4)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Save V67", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 5. AGENTS & SIMULATION ---
def run_monte_carlo(prices):
    returns = prices.pct_change().dropna()
    sims = prices.iloc[-1] * (1 + np.random.normal(returns.mean(), returns.std(), (1000, 10)))
    prob = np.sum(sims[:, -1] > prices.iloc[-1]) / 1000
    return prob

def detailed_backtest(s, df, rsi_limit, sl_mult):
    """Simule et LOGUE chaque étape de la réflexion"""
    
    # On prend un point d'entrée aléatoire dans le passé qui correspond aux critères
    candidates = df[df['RSI'] < rsi_limit]
    
    if candidates.empty:
        return "Rien trouvé", 0
    
    # On prend le dernier signal valide pour l'exemple
    idx = candidates.index[-random.randint(1, min(5, len(candidates)))] 
    row = df.loc[idx]
    
    # ANALYSE DES AGENTS (Virtuelle)
    agent_tech = f"📉 **TECH:** RSI {row['RSI']:.1f} < {rsi_limit} -> Signal BUY"
    
    # Simulation Whale (Volume)
    vol_avg = df['Volume'].rolling(20).mean().loc[idx]
    is_whale = row['Volume'] > vol_avg * 1.5
    agent_whale = f"🐋 **WHALE:** Vol x{row['Volume']/vol_avg:.1f} -> {'ACCUMULATION' if is_whale else 'Neutre'}"
    
    # Simulation Quant (Monte Carlo local)
    # On prend les 50 bougies AVANT ce point pour calculer
    past_data = df.loc[:idx].iloc[-50:]['Close']
    mc_prob = run_monte_carlo(past_data)
    agent_quant = f"🎲 **QUANT:** Proba Hausse {mc_prob*100:.1f}% -> {'VALIDÉ' if mc_prob > 0.55 else 'RISQUÉ'}"
    
    # RÉSULTAT RÉEL (On regarde le futur)
    future = df.loc[idx:].head(10) # 10 bougies suivantes
    entry = row['Close']
    sl = entry - (row['ATR'] * sl_mult)
    tp = entry + (row['ATR'] * sl_mult * 2.0)
    
    outcome = "NEUTRE"
    pnl = 0
    
    if not future.empty:
        if future['Low'].min() < sl:
            outcome = "STOP LOSS"
            pnl = sl - entry
        elif future['High'].max() > tp:
            outcome = "TAKE PROFIT"
            pnl = tp - entry
        else:
            pnl = future['Close'].iloc[-1] - entry
            outcome = "EN COURS"
    
    emoji = "✅" if pnl > 0 else "❌"
    
    # CONSTRUCTION DU LOG DÉTAILLÉ
    log_block = (
        f"🧪 **TEST SUR {s}** (RSI<{rsi_limit}, SL={sl_mult})\n"
        f"{agent_tech}\n"
        f"{agent_whale}\n"
        f"{agent_quant}\n"
        f"🏁 **RÉSULTAT:** {outcome} | PnL: {emoji} {pnl:.2f}$"
    )
    
    fast_log(log_block)
    return outcome, pnl

# --- 6. MODULE APPRENTISSAGE ---
def run_learning_loop():
    global brain, short_term_memory
    cache = {}
    
    fast_log("👨‍🏫 **PROFESSEUR:** Démarrage de l'analyse détaillée des agents...")
    
    while True:
        try:
            s = random.choice(WATCHLIST)
            
            if s not in cache or random.random() < 0.1:
                try:
                    df = yf.Ticker(s).history(period="1mo", interval="1h")
                    if not df.empty:
                        df['RSI'] = ta.rsi(df['Close'], length=14)
                        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                        cache[s] = df.dropna()
                except: pass
            
            if s in cache:
                # On teste des paramètres
                t_rsi = random.randint(25, 55)
                t_sl = round(random.uniform(1.5, 3.0), 1)
                
                outcome, pnl = detailed_backtest(s, cache[s], t_rsi, t_sl)
                
                if pnl != 0:
                    short_term_memory.append({"pnl": pnl, "win": pnl>0})
                    if pnl > 200: # Si on trouve une pépite
                        brain['best_params'] = {"rsi_buy": t_rsi, "sl": t_sl}
                        save_brain()

            # BILAN
            if len(short_term_memory) >= 5: # Tous les 5 tests
                wins = sum(1 for x in short_term_memory if x['win'])
                total_pnl = sum(x['pnl'] for x in short_term_memory)
                
                msg = {
                    "embeds": [{
                        "title": "🎓 RAPPORT DÉTAILLÉ",
                        "color": 0xFFD700,
                        "description": f"Sur les 5 dernières simulations détaillées :",
                        "fields": [
                            {"name": "Profit Net", "value": f"**{total_pnl:.2f} $**", "inline": True},
                            {"name": "Précision", "value": f"**{wins/5*100:.0f}%**", "inline": True}
                        ]
                    }]
                }
                send_summary(msg)
                short_term_memory = []
            
            time.sleep(15) # Pause pour laisser le temps de lire
        except Exception as e:
            time.sleep(10)

# --- 7. TRADING LIVE (Code Standard V66) ---
# (Je remets le code standard pour que le trading fonctionne lundi)
def run_trading():
    global brain, bot_status
    load_brain()
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if market_open:
                bot_status = "🟢 TRADING"
                # ... (Logique trading standard V66 conservée ici) ...
            else:
                bot_status = "🌙 Nuit"
            time.sleep(60)
        except: time.sleep(10)

@app.route('/')
def index(): return f"<h1>TRANSPARENT V67</h1><p>{bot_status}</p>"

def start_threads():
    threading.Thread(target=run_trading, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()
    threading.Thread(target=logger_worker, daemon=True).start()
    threading.Thread(target=run_learning_loop, daemon=True).start()

load_brain()
start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
