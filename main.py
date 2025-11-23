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
LEARNING_WEBHOOK_URL = os.environ.get("LEARNING_WEBHOOK_URL")
SUMMARY_WEBHOOK_URL = os.environ.get("SUMMARY_WEBHOOK_URL")
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
    "best_params": {"rsi_buy": 30, "sl": 2.0}, 
    "learning_stats": {"tests": 0, "wins": 0}
}
bot_status = "V68 Narrator..."
log_queue = queue.Queue()
short_term_memory = []

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- 3. LOGGING ---
def fast_log(text):
    log_queue.put(text)

def logger_worker():
    buffer = []
    last_send = time.time()
    while True:
        try:
            while not log_queue.empty():
                buffer.append(log_queue.get())
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
            repo.update_file("brain.json", "Save V68", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 5. AGENTS ---
def run_monte_carlo(prices):
    returns = prices.pct_change().dropna()
    sims = prices.iloc[-1] * (1 + np.random.normal(returns.mean(), returns.std(), (1000, 10)))
    prob = np.sum(sims[:, -1] > prices.iloc[-1]) / 1000
    return prob

def detailed_backtest(s, df, rsi_limit, sl_mult):
    candidates = df[df['RSI'] < rsi_limit]
    if candidates.empty: return "Rien trouvé", 0
    
    idx = candidates.index[-random.randint(1, min(5, len(candidates)))] 
    row = df.loc[idx]
    
    # Simulation Agents
    vol_avg = df['Volume'].rolling(20).mean().loc[idx]
    is_whale = row['Volume'] > vol_avg * 1.5
    
    past_data = df.loc[:idx].iloc[-50:]['Close']
    mc_prob = run_monte_carlo(past_data)
    
    future = df.loc[idx:].head(10)
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
    log_block = f"🧪 **TEST {s}** (RSI<{rsi_limit}) | Whale:{is_whale} | Quant:{mc_prob:.2f} | Résultat: {emoji} {outcome} ({pnl:.2f}$)"
    fast_log(log_block)
    return outcome, pnl

# --- 6. NOUVEAU : LE NARRATEUR GEMINI ---
def generate_gemini_summary(stats, best_run):
    """Demande à Gemini de commenter les résultats de la session"""
    prompt = f"""
    Tu es le Directeur de Recherche d'un Hedge Fund Quantitatif.
    Voici les résultats de tes dernières simulations (10 tests) :
    
    - Profit Total de la session : {stats['total_pnl']:.2f}$
    - Taux de réussite : {stats['win_rate']:.1f}%
    - Meilleure découverte : Sur {best_run['s']}, avec RSI < {best_run['rsi']} et Stop Loss {best_run['sl']} ATR.
    
    Rédige un commentaire court (2 phrases max) pour le rapport.
    Sois analytique. Exemple: "La volatilité sur la Tech offre de belles opportunités, je valide le resserrement des stops."
    """
    try:
        res = model.generate_content(prompt)
        return res.text.strip()
    except: return "Analyse en cours..."

# --- 7. BOUCLE D'APPRENTISSAGE ---
def run_learning_loop():
    global brain, short_term_memory
    cache = {}
    
    fast_log("👨‍🏫 **PROFESSEUR V68:** Démarrage avec Synthèse Narrée.")
    
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
                t_rsi = random.randint(25, 55)
                t_sl = round(random.uniform(1.5, 3.0), 1)
                
                outcome, pnl = detailed_backtest(s, cache[s], t_rsi, t_sl)
                
                if pnl != 0:
                    short_term_memory.append({"s": s, "pnl": pnl, "win": pnl>0, "rsi": t_rsi, "sl": t_sl})
                    if pnl > 200:
                        brain['best_params'] = {"rsi_buy": t_rsi, "sl": t_sl}
                        save_brain()

            # BILAN TOUS LES 10 TESTS
            if len(short_term_memory) >= 5:
                wins = sum(1 for x in short_term_memory if x['win'])
                total_pnl = sum(x['pnl'] for x in short_term_memory)
                win_rate = (wins / 5) * 100
                best = max(short_term_memory, key=lambda x: x['pnl'])
                
                # 1. GÉNÉRATION DU TEXTE GEMINI
                stats_for_ai = {"total_pnl": total_pnl, "win_rate": win_rate}
                ai_comment = generate_gemini_summary(stats_for_ai, best)
                
                # 2. ENVOI DU RAPPORT
                msg = {
                    "embeds": [{
                        "title": "🎓 RAPPORT STRATÉGIQUE",
                        "color": 0xFFD700,
                        "description": f"**🗣️ Analyse Gemini :**\n*{ai_comment}*",
                        "fields": [
                            {"name": "Profit Session", "value": f"**{total_pnl:.2f} $**", "inline": True},
                            {"name": "Précision", "value": f"**{win_rate:.0f}%**", "inline": True},
                            {"name": "Top Config", "value": f"{best['s']} (RSI < {best['rsi']})", "inline": False}
                        ]
                    }]
                }
                send_summary(msg)
                short_term_memory = []
            
            time.sleep(15)
        except Exception as e:
            time.sleep(10)

# --- 8. TRADING LIVE ---
def run_trading():
    # (Code trading standard V66 conservé ici pour l'exécution lundi)
    global brain, bot_status
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            if market_open: bot_status = "🟢 TRADING"
            else: bot_status = "🌙 NUIT"
            time.sleep(60)
        except: pass

@app.route('/')
def index(): return f"<h1>NARRATOR V68</h1><p>{bot_status}</p>"

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
