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

# CLÉS
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL")
LEARNING_WEBHOOK_URL = os.environ.get("LEARNING_WEBHOOK_URL")
SUMMARY_WEBHOOK_URL = os.environ.get("SUMMARY_WEBHOOK_URL")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR", "GOOG", "META"]
INITIAL_CAPITAL = 50000.0 
SIMULATION_COUNT = 2000 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    # Ces paramètres sont mis à jour par le COLAB
    "genome": {"rsi_buy": 30, "sl_mult": 2.0, "tp_mult": 3.0},
    "ai_prediction": "NEUTRE",
    "last_colab_update": "En attente...",
    
    # Mémoire Interne
    "emotions": {"confidence": 50.0, "stress": 20.0},
    "trade_history": [],
    "total_pnl": 0.0
}

bot_state = {"status": "Booting Hybrid...", "last_log": "Init...", "web_logs": []}
log_queue = queue.Queue()

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- LOGGING ---
def fast_log(text):
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
            if buffer and (len(buffer) > 3 or time.time() - last_send > 1.5):
                msg_block = "\n".join(buffer[:12])
                buffer = buffer[12:]
                if LEARNING_WEBHOOK_URL:
                    try: requests.post(LEARNING_WEBHOOK_URL, json={"content": msg_block})
                    except: pass
                last_send = time.time()
            time.sleep(0.2)
        except: time.sleep(1)

def send_alert(url, embed):
    if url: try: requests.post(url, json={"embeds": [embed]})
    except: pass

def run_heartbeat():
    while True:
        if HEARTBEAT_WEBHOOK_URL: requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"})
        time.sleep(30)

# --- SYNCHRONISATION AVEC COLAB (GITHUB) ---
def load_brain():
    """Télécharge l'intelligence du Colab"""
    global brain
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        c = repo.get_contents("brain.json")
        loaded = json.loads(c.decoded_content.decode())
        
        # On met à jour ce qui vient du Colab
        if "genome" in loaded: brain["genome"] = loaded["genome"]
        if "ai_prediction" in loaded: brain["ai_prediction"] = loaded["ai_prediction"]
        if "last_colab_update" in loaded: brain["last_colab_update"] = loaded["last_colab_update"]
        
        # On garde notre cash local
        # (Note: Dans un système parfait, le cash serait géré par le broker, ici on simule)
    except: pass

def save_brain():
    """Sauvegarde l'état du trading (Cash, Positions)"""
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        
        # On charge d'abord pour ne pas écraser l'IA du Colab
        try:
            c = repo.get_contents("brain.json")
            current_remote = json.loads(c.decoded_content.decode())
        except: current_remote = {}
        
        # On fusionne : Le Render gère le Cash, le Colab gère le Génome
        current_remote["cash"] = brain["cash"]
        current_remote["holdings"] = brain["holdings"]
        current_remote["trade_history"] = brain["trade_history"]
        current_remote["total_pnl"] = brain["total_pnl"]
        
        content = json.dumps(current_remote, indent=4)
        
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Render Update", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- INTELLIGENCE LOCALE (Vitesse) ---
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
        mpf.plot(df.tail(50), type='candle', style='nightclouds', savefig=buf)
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

# --- TRADING LOOP ---
class GhostBroker:
    def get_price(self, symbol):
        try: return yf.Ticker(symbol).fast_info['last_price']
        except: return None
    def get_portfolio(self):
        return brain['cash'], [{"symbol": s, "qty": d['qty'], "pnl": 0} for s, d in brain['holdings'].items()]

broker = GhostBroker()

def run_trading():
    global brain, bot_state
    load_brain() # Init
    
    fast_log("🟢 **RENDER:** Prêt à recevoir les signaux du Colab.")
    
    while True:
        try:
            # On recharge le cerveau toutes les minutes pour voir si Colab a parlé
            load_brain()
            
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if market_open:
                bot_state['status'] = "🟢 LIVE TRADING"
                
                # GESTION VENTES
                for s in list(brain['holdings'].keys()):
                    pos = brain['holdings'][s]
                    curr = broker.get_price(s)
                    if not curr: continue
                    
                    if curr < pos['stop']:
                        pnl = (curr - pos['entry']) * pos['qty']
                        brain['cash'] += pos['qty'] * curr
                        brain['total_pnl'] += pnl
                        del brain['holdings'][s]
                        send_alert(DISCORD_WEBHOOK_URL, {"embeds": [{"title": f"🔴 VENTE {s}", "description": f"PnL: {pnl:.2f}$", "color": 0xe74c3c}]})
                        save_brain()
                    elif curr > pos['tp']:
                        pnl = (curr - pos['entry']) * pos['qty']
                        brain['cash'] += pos['qty'] * curr
                        brain['total_pnl'] += pnl
                        del brain['holdings'][s]
                        send_alert(DISCORD_WEBHOOK_URL, {"embeds": [{"title": f"🟢 VENTE {s}", "description": f"PnL: {pnl:.2f}$", "color": 0x2ecc71}]})
                        save_brain()

                # SCAN ACHATS
                if len(brain['holdings']) < 5:
                    for s in WATCHLIST:
                        if s in brain['holdings']: continue
                        try:
                            df = yf.Ticker(s).history(period="1mo", interval="15m")
                            if df.empty: continue
                            row = df.iloc[-1]
                            df['RSI'] = ta.rsi(df['Close'], 14)
                            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)
                            
                            # ICI : On utilise les paramètres optimisés par COLAB
                            rsi_colab = brain['genome']['rsi_buy']
                            
                            if row['RSI'] < rsi_colab:
                                fast_log(f"🔎 **SCAN {s}:** RSI < {rsi_colab} (Colab Target). Analyse...")
                                
                                mc = run_monte_carlo(df['Close'])
                                vis = get_vision_score(df)
                                whl, _ = check_whale(df)
                                
                                fast_log(f"🧠 **{s}:** MC:{mc:.2f} | Vis:{vis:.2f} | Whale:{whl}")
                                
                                if mc > 0.60 and vis > 0.6:
                                    price = row['Close']
                                    qty = 500 / price
                                    
                                    # SL/TP optimisés par Colab
                                    sl = price - (row['ATR'] * brain['genome']['sl_mult'])
                                    tp = price + (row['ATR'] * brain['genome']['tp_mult'])
                                    
                                    brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                                    brain['cash'] -= 500
                                    
                                    send_alert(DISCORD_WEBHOOK_URL, {
                                        "title": f"🌌 ACHAT HYBRIDE : {s}",
                                        "description": "Validé par Colab & Render.",
                                        "color": 0x2ecc71
                                    })
                                    save_brain()
                        except: pass
            else:
                bot_state['status'] = "🌙 NUIT (Attente Colab)"
            
            time.sleep(60)
        except: time.sleep(10)

# --- DASHBOARD ---
@app.route('/')
def index():
    eq, _ = broker.get_portfolio()
    return f"""
    <h1>HYBRID V126</h1>
    <p>Status: {bot_state['status']}</p>
    <p>Capital: ${eq:,.2f}</p>
    <p>Dernière Mise à jour Colab: {brain.get('last_colab_update')}</p>
    <p>Génome Colab: {brain.get('genome')}</p>
    <hr>
    <pre>{chr(10).join(bot_state['web_logs'][:10])}</pre>
    """

def start_threads():
    threading.Thread(target=run_trading, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()
    threading.Thread(target=logger_worker, daemon=True).start()

load_brain()
start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
