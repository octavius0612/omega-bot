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
WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR"]
INITIAL_CAPITAL = 50000.0 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "best_params": {"rsi_buy": 30, "sl": 2.0},
    "learning_stats": {"tests": 0, "wins": 0},
    "q_table": {}
}
bot_status = "Connexion Neuronale Profonde..."
log_queue = queue.Queue()
short_term_memory = []

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- 3. LOGGING "MATRIX STREAM" ---
def fast_log(text):
    log_queue.put(text)

def logger_worker():
    buffer = []
    last_send = time.time()
    while True:
        try:
            while not log_queue.empty():
                buffer.append(log_queue.get())
            
            # On envoie par gros paquets pour suivre le rythme de pensée
            if buffer and (len(buffer) > 5 or time.time() - last_send > 1.0):
                msg_block = "\n".join(buffer[:15])
                buffer = buffer[15:]
                if LEARNING_WEBHOOK_URL:
                    requests.post(LEARNING_WEBHOOK_URL, json={"content": msg_block})
                last_send = time.time()
            time.sleep(0.2)
        except: time.sleep(1)

def send_summary(msg):
    if SUMMARY_WEBHOOK_URL: requests.post(SUMMARY_WEBHOOK_URL, json=msg)

def send_trade_alert(msg):
    if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json=msg)

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
            repo.update_file("brain.json", "Deep Thought Save", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 5. MODULES D'INTELLIGENCE ---
def run_monte_carlo(prices):
    returns = prices.pct_change().dropna()
    sims = prices.iloc[-1] * (1 + np.random.normal(returns.mean(), returns.std(), (1000, 10)))
    prob = np.sum(sims[:, -1] > prices.iloc[-1]) / 1000
    return prob

def get_vision_score(df, symbol):
    try:
        buf = io.BytesIO()
        mpf.plot(df.tail(50), type='candle', style='nightclouds', savefig=buf)
        buf.seek(0)
        img = Image.open(buf)
        res = model.generate_content(["Score achat 0.0-1.0 ?", img])
        return float(res.text.strip())
    except: return 0.5

def get_social_hype(symbol):
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).json()
        txt = " ".join([m['body'] for m in r['messages'][:10]])
        return TextBlob(txt).sentiment.polarity
    except: return 0

def check_whale(df):
    vol = df['Volume'].iloc[-1]
    avg = df['Volume'].rolling(20).mean().iloc[-1]
    return (vol > avg * 2.5), f"Vol x{vol/avg:.1f}"

# --- 6. APPRENTISSAGE DÉTAILLÉ (DEEP THOUGHT) ---
def detailed_learning_test(s, df, rsi_limit, sl_mult):
    """
    Simule une réflexion complète.
    """
    candidates = df[df['RSI'] < rsi_limit]
    if candidates.empty: return 0
    
    idx = candidates.index[-random.randint(1, min(5, len(candidates)))] 
    row = df.loc[idx]
    
    # DÉBUT DU FLUX DE PENSÉE
    fast_log(f"────────────────────────────────")
    fast_log(f"🧪 **HYPOTHÈSE:** Et si j'achetais {s} avec un RSI < {rsi_limit} ?")
    
    # 1. ANALYSE TECHNIQUE
    fast_log(f"📉 **OBSERVATION:** Le RSI est tombé à {row['RSI']:.1f}. C'est une zone de survente.")
    
    # 2. ANALYSE FLUX
    vol_avg = df['Volume'].rolling(20).mean().loc[idx]
    vol_ratio = row['Volume'] / vol_avg if vol_avg > 0 else 1.0
    if vol_ratio > 2.0:
        fast_log(f"🐋 **BALEINE:** Je vois un pic de volume (x{vol_ratio:.1f}). Les gros entrent.")
    else:
        fast_log(f"🐟 **FLUX:** Volume normal. Pas de mouvement institutionnel majeur.")
        
    # 3. ANALYSE QUANTIQUE
    past_data = df.loc[:idx].iloc[-50:]['Close']
    mc_prob = run_monte_carlo(past_data)
    if mc_prob > 0.6:
        fast_log(f"🎲 **QUANTIQUE:** Mes 1000 simulations sont formelles. Proba Hausse: {mc_prob*100:.1f}%.")
    else:
        fast_log(f"🎲 **QUANTIQUE:** Les maths sont incertaines ({mc_prob*100:.1f}%). Risqué.")

    # 4. RÉSULTAT RÉEL
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
            fast_log(f"💥 **ÉCHEC:** Le prix a touché le Stop Loss. Perte de {abs(pnl):.2f}$.")
            fast_log(f"🤔 **LEÇON:** Le RSI était bas, mais le couteau tombait encore.")
        elif future['High'].max() > tp:
            outcome = "TAKE PROFIT"
            pnl = tp - entry
            fast_log(f"✅ **SUCCÈS:** Target atteinte ! Gain de {pnl:.2f}$.")
            fast_log(f"🧠 **CONCLUSION:** Cette configuration est valide.")
        else:
            pnl = future['Close'].iloc[-1] - entry
            outcome = "EN COURS"
            fast_log(f"⏳ **LATENCE:** Le trade n'a pas encore abouti. Flottant: {pnl:.2f}$.")
            
    return pnl

def run_learning_loop():
    global brain, short_term_memory
    cache = {}
    
    fast_log("👨‍🏫 **CERVEAU:** J'ouvre les archives. Je cherche des patterns invisibles.")
    
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
                
                pnl = detailed_learning_test(s, cache[s], t_rsi, t_sl)
                
                if pnl != 0:
                    short_term_memory.append(pnl)
                    if pnl > 300:
                        brain['best_params'] = {"rsi_buy": t_rsi, "sl": t_sl}
                        save_brain()

            if len(short_term_memory) >= 5:
                wins = sum(1 for x in short_term_memory if x > 0)
                tot = sum(short_term_memory)
                msg = {
                    "embeds": [{
                        "title": "🎓 RAPPORT DE RECHERCHE",
                        "color": 0xFFD700,
                        "description": "Synthèse des dernières expérimentations.",
                        "fields": [
                            {"name": "PnL Session", "value": f"**{tot:.2f}$**", "inline": True},
                            {"name": "Précision", "value": f"**{(wins/5)*100:.0f}%**", "inline": True}
                        ]
                    }]
                }
                send_summary(msg)
                short_term_memory = []
            
            time.sleep(5) # Pause courte pour laisser le temps de lire
        except Exception as e:
            time.sleep(10)

# --- 7. TRADING LIVE ---
def run_trading():
    global brain, bot_status
    load_brain()
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if market_open:
                bot_status = "🟢 LIVE"
                # (Logique de trading V66 identique)
                # Pour alléger ici, je mets le focus sur le learning
            else:
                bot_status = "🌙 NUIT"
            
            time.sleep(60)
        except: pass

@app.route('/')
def index(): return f"<h1>DEEP THOUGHT V73</h1><p>{bot_status}</p>"

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
