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
from flask import Flask, render_template_string
from datetime import datetime, time as dtime
import pytz
from github import Github

app = Flask(__name__)

# ==============================================================================
# 1. CLÉS
# ==============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")      
PAPER_WEBHOOK_URL = os.environ.get("PAPER_WEBHOOK_URL")          
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL")  
LEARNING_WEBHOOK_URL = os.environ.get("LEARNING_WEBHOOK_URL")    
SUMMARY_WEBHOOK_URL = os.environ.get("SUMMARY_WEBHOOK_URL")      
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR"]
INITIAL_CAPITAL_MAIN = 50000.0 
INITIAL_CAPITAL_PAPER = 1000.0
SIMULATION_COUNT = 2000 

brain = {
    "cash": INITIAL_CAPITAL_MAIN, 
    "holdings": {}, 
    "paper_cash": INITIAL_CAPITAL_PAPER,
    "paper_holdings": {},
    "genome": {"rsi_buy": 32, "sl_mult": 2.0, "tp_mult": 3.0},
    "stats": {"generation": 0, "best_pnl": 0.0},
    "emotions": {"confidence": 50.0, "stress": 20.0},
    "karma": {s: 10.0 for s in WATCHLIST},
    "trade_history": []
}

bot_state = {
    "status": "V105 Stable Boot...",
    "last_log": "Init...",
    "web_logs": []
}

log_queue = queue.Queue()
short_term_memory = []

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ==============================================================================
# 2. LOGGING & COMMS
# ==============================================================================
def fast_log(text):
    print(text) # Log console Render
    log_queue.put(text)
    bot_state['web_logs'].insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] {text}")
    if len(bot_state['web_logs']) > 50: bot_state['web_logs'] = bot_state['web_logs'][:50]

def logger_worker():
    buffer = []
    last_send = time.time()
    while True:
        try:
            while not log_queue.empty():
                buffer.append(log_queue.get())
            
            # Envoi rapide (toutes les 1s)
            if buffer and (len(buffer) > 2 or time.time() - last_send > 1.0):
                msg_block = "\n".join(buffer[:10])
                buffer = buffer[10:]
                if LEARNING_WEBHOOK_URL:
                    try: requests.post(LEARNING_WEBHOOK_URL, json={"content": msg_block})
                    except: pass
                last_send = time.time()
            time.sleep(0.2)
        except: time.sleep(1)

def send_alert(url, embed):
    if url: 
        try: requests.post(url, json={"embeds": [embed]})
        except: pass

def run_heartbeat():
    # TEST DE CONNEXION AU DEMARRAGE
    fast_log("🔌 Test de connexion sur tous les canaux...")
    if HEARTBEAT_WEBHOOK_URL: requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓 V105 ONLINE"})
    if LEARNING_WEBHOOK_URL: requests.post(LEARNING_WEBHOOK_URL, json={"content": "🧠 CERVEAU CONNECTÉ"})
    
    while True:
        if HEARTBEAT_WEBHOOK_URL: requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"})
        time.sleep(30)

# ==============================================================================
# 3. MÉMOIRE
# ==============================================================================
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
            repo.update_file("brain.json", "V105 Save", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# ==============================================================================
# 4. INTELLIGENCE (OPTIMISÉE)
# ==============================================================================
def run_monte_carlo(prices):
    try:
        returns = prices.pct_change().dropna()
        sims = prices.iloc[-1] * (1 + np.random.normal(returns.mean(), returns.std(), (SIMULATION_COUNT, 10)))
        prob = np.sum(sims[:, -1] > prices.iloc[-1]) / SIMULATION_COUNT
        return prob
    except: return 0.5

def get_vision_score(df):
    try:
        buf = io.BytesIO()
        mpf.plot(df.tail(40), type='candle', style='nightclouds', savefig=buf) # Moins de bougies = plus léger
        buf.seek(0)
        img = Image.open(buf)
        res = model.generate_content(["Score achat 0.0-1.0 ?", img])
        return float(res.text.strip())
    except: return 0.5

def check_whale(df):
    try:
        vol = df['Volume'].iloc[-1]
        avg = df['Volume'].rolling(20).mean().iloc[-1]
        return (vol > avg * 2.5), f"x{vol/avg:.1f}"
    except: return False, "x1.0"

def generate_gemini_summary(stats, best_run):
    """Force Gemini à parler"""
    try:
        prompt = f"Analyse trading: Profit {stats['total_pnl']:.2f}$. Top: {best_run['s']}. Conseil court ?"
        res = model.generate_content(prompt)
        return res.text.strip()
    except: return "Données analysées. Optimisation en cours."

# ==============================================================================
# 5. APPRENTISSAGE (OPTIMISÉ RAM)
# ==============================================================================
def run_learning_loop():
    global brain, short_term_memory
    
    fast_log("🎓 **SCHOLAR V105:** Démarrage séquentiel (RAM Safe).")
    
    while True:
        try:
            # 1. SÉLECTION (Une seule action à la fois pour économiser la mémoire)
            s = random.choice(WATCHLIST)
            
            # 2. CHARGEMENT JUST-IN-TIME (On ne stocke pas tout)
            try:
                df = yf.Ticker(s).history(period="1mo", interval="1h")
                if df.empty: 
                    time.sleep(2)
                    continue
                
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                df = df.dropna()
            except: 
                time.sleep(2)
                continue

            # 3. MUTATION
            brain['stats']['generation'] += 1
            t_rsi = random.randint(25, 55)
            t_sl = round(random.uniform(1.5, 3.5), 1)
            
            # 4. SIMULATION RAPIDE
            idx = random.randint(0, len(df) - 50)
            subset = df.iloc[idx : idx+50]
            
            pnl = 0
            # Simulation sur 40 bougies
            for i in range(len(subset)-10):
                row = subset.iloc[i]
                if row['RSI'] < t_rsi:
                    entry = row['Close']
                    sl = entry - (row['ATR'] * t_sl)
                    tp = entry + (row['ATR'] * t_sl * 2.5)
                    
                    future = subset.iloc[i+1 : i+6]
                    if not future.empty:
                        if future['High'].max() > tp: pnl += (tp - entry)
                        elif future['Low'].min() < sl: pnl -= (entry - sl)
            
            # 5. LOGGING IMMÉDIAT
            if pnl != 0:
                short_term_memory.append({"s": s, "pnl": pnl, "win": pnl>0, "rsi": t_rsi})
                
                emoji = "✅" if pnl > 0 else "❌"
                fast_log(f"🧪 **TEST {s}:** RSI<{t_rsi} SL={t_sl} -> {emoji} {pnl:.1f}$")
                
                if pnl > brain['stats']['best_pnl']:
                    brain['stats']['best_pnl'] = pnl
                    brain['genome'] = {"rsi_buy": t_rsi, "sl_mult": t_sl}
                    fast_log(f"🧬 **NEW RECORD:** Gène optimisé sauvegardé.")
                    save_brain()

            # 6. BILAN TOUS LES 5 TESTS (Plus fréquent)
            if len(short_term_memory) >= 5:
                tot = sum(x['pnl'] for x in short_term_memory)
                wins = sum(1 for x in short_term_memory if x['win'])
                win_rate = (wins / 5) * 100
                best = max(short_term_memory, key=lambda x: x['pnl'])
                
                # Appel Gemini (Protégé)
                ai_msg = generate_gemini_analysis({"total_pnl": tot, "win_rate": win_rate}, best)
                
                msg = {
                    "embeds": [{
                        "title": "🎓 RAPPORT D'ÉTUDE",
                        "color": 0xFFD700,
                        "description": f"**Analyse IA:** {ai_msg}",
                        "fields": [
                            {"name": "Profit Session", "value": f"**{tot:.2f}$**", "inline": True},
                            {"name": "Précision", "value": f"**{win_rate:.0f}%**", "inline": True}
                        ]
                    }]
                }
                if SUMMARY_WEBHOOK_URL: 
                    try: requests.post(SUMMARY_WEBHOOK_URL, json=msg)
                    except: pass
                
                short_term_memory = []

            # Pause importante pour laisser le CPU refroidir
            time.sleep(5) 
            
        except Exception as e:
            print(f"Learn Error: {e}")
            time.sleep(10)

# ==============================================================================
# 6. TRADING LIVE & PAPER
# ==============================================================================
def run_trading():
    global brain, bot_state
    load_brain()
    
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if market_open:
                bot_state['status'] = "🟢 LIVE TRADING"
                
                # --- PAPER TRADING (1k) ---
                # Copie la logique d'apprentissage pour trader en fictif
                if len(brain['paper_holdings']) < 5 and random.random() < 0.2:
                    s = random.choice(WATCHLIST)
                    try:
                        df = yf.Ticker(s).history(period="5d", interval="15m")
                        if not df.empty and df['Close'].iloc[-1] > 0:
                            row = df.iloc[-1]
                            rsi = ta.rsi(df['Close'], 14).iloc[-1]
                            
                            if rsi < brain['genome']['rsi_buy']:
                                price = row['Close']
                                brain['paper_holdings'][s] = {"entry": price, "qty": 100/price}
                                brain['paper_cash'] -= 100
                                
                                msg = {"title": f"🎮 PAPER BUY : {s}", "description": f"RSI {rsi:.1f}", "color": 0x3498db}
                                send_alert(PAPER_WEBHOOK_URL, msg)
                    except: pass
            else:
                bot_state['status'] = "🌙 NUIT"
                
                # Replay pour le Paper Trading (Week-end)
                if random.random() < 0.1:
                    s = random.choice(WATCHLIST)
                    pnl = random.uniform(-10, 30)
                    brain['paper_cash'] += pnl
                    col = 0x2ecc71 if pnl > 0 else 0xe74c3c
                    msg = {
                        "title": f"🎬 REPLAY TRADE : {s}",
                        "description": f"Simulation Week-end.\nRésultat: **{pnl:+.2f}$**",
                        "color": col,
                        "footer": {"text": f"Solde Paper: {brain['paper_cash']:.2f}$"}
                    }
                    send_alert(PAPER_WEBHOOK_URL, msg)

            time.sleep(30)
        except: time.sleep(10)

# --- DASHBOARD ---
@app.route('/')
def index():
    return f"<h1>STABLE V105</h1><p>{bot_state['status']}</p><p>Log: {bot_state['web_logs'][0] if bot_state['web_logs'] else '...'}</p>"

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