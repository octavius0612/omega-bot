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

# --- 1. CLÉS DU POUVOIR ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL")
LEARNING_WEBHOOK_URL = os.environ.get("LEARNING_WEBHOOK_URL")
SUMMARY_WEBHOOK_URL = os.environ.get("SUMMARY_WEBHOOK_URL")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

# --- 2. CONFIGURATION ---
WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR", "GOOG", "META"]
INITIAL_CAPITAL = 50000.0 
SIMULATION_COUNT = 50000 # Vitesse Quantique

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "trade_history": [],
    "total_pnl": 0.0,
    "learning_stats": {"tests": 0, "wins": 0},
    "best_params": {"rsi_buy": 30, "sl_atr": 2.0},
    "emotions": {"confidence": 50.0, "stress": 20.0}
}

bot_state = {
    "status": "Chargement Singularité...",
    "last_decision": "Attente...",
    "last_log": "Init..."
}

log_queue = queue.Queue()
short_term_memory = []

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- 3. LOGGING & COMMS ---
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
            repo.update_file("brain.json", "Singularity Save", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 5. MODULES D'INTELLIGENCE ---

# A. QUANTIQUE (NumPy Vectorisé - V71)
def run_massive_monte_carlo(prices):
    try:
        returns = prices.pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        last = prices.iloc[-1]
        # 50,000 simulations vectorisées instantanées
        sims = last * (1 + np.random.normal(mu, sigma, (SIMULATION_COUNT, 10)))
        prob = np.sum(sims[:, -1] > last) / SIMULATION_COUNT
        return prob
    except: return 0.5

# B. VISION (Gemini Vision - V54)
def get_vision_score(df, symbol):
    try:
        buf = io.BytesIO()
        mpf.plot(df.tail(60), type='candle', style='nightclouds', savefig=buf)
        buf.seek(0)
        img = Image.open(buf)
        # Analyse Visuelle
        res = model.generate_content(["Analyse chartiste technique (patterns). Score d'achat entre 0.0 et 1.0 ?", img])
        return float(res.text.strip())
    except: return 0.5

# C. SOCIAL (TextBlob - V60)
def get_social_sentiment(symbol):
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers).json()
        txt = " ".join([m['body'] for m in r['messages'][:15]])
        return TextBlob(txt).sentiment.polarity
    except: return 0

# D. BALEINES (Volume - V59)
def check_whale(df):
    vol = df['Volume'].iloc[-1]
    avg = df['Volume'].rolling(20).mean().iloc[-1]
    is_whale = vol > (avg * 3.0)
    return is_whale, f"Vol x{vol/avg:.1f}" if avg > 0 else "Vol Normal"

# E. CONSEIL SUPRÊME (Gemini Text - V62)
def consult_omniscient_brain(s, rsi, mc, vis, soc, whale):
    prompt = f"""
    ANALYSE ULTIME POUR {s}.
    
    1. MATHS (Quantique): {mc*100:.1f}% probabilité hausse (sur {SIMULATION_COUNT} sims).
    2. VISION (Chartisme): Score {vis:.2f}/1.0.
    3. SOCIAL (Foule): Sentiment {soc:.2f}.
    4. FLUX (Baleine): {"OUI" if whale else "NON"}.
    5. TECH (RSI): {rsi:.1f}.
    
    RÈGLE D'OR :
    Si Maths > 60% ET Vision > 0.6 : ACHAT.
    Si Baleine DÉTECTÉE et RSI < 35 : ACHAT FORT.
    
    JSON: {{"decision": "BUY/WAIT", "score": 85, "reason": "Synthèse"}}
    """
    try:
        res = model.generate_content(prompt)
        return json.loads(res.text.replace("```json","").replace("```",""))
    except: return {"decision": "WAIT", "score": 0}

# --- 6. BANQUE INTERNE (GHOST BROKER) ---
class GhostBroker:
    def get_price(self, symbol):
        try: return yf.Ticker(symbol).fast_info['last_price']
        except: return None

    def get_portfolio(self):
        equity = brain['cash']
        positions = []
        for s, d in brain['holdings'].items():
            p = self.get_price(s)
            if p:
                val = d['qty'] * p
                equity += val
                pnl = val - (d['qty'] * d['entry'])
                positions.append({"symbol": s, "qty": d['qty'], "entry": d['entry'], "current": p, "pnl": round(pnl, 2)})
        return equity, positions

broker = GhostBroker()

# --- 7. MODULE APPRENTISSAGE CONTINU (THE SCHOLAR) ---
def run_learning_loop():
    global brain, short_term_memory
    cache = {}
    fast_log("🎓 **SCHOLAR:** Démarrage de l'apprentissage profond en arrière-plan.")
    
    while True:
        try:
            s = random.choice(WATCHLIST)
            
            # Mise à jour Data (10%)
            if s not in cache or random.random() < 0.1:
                try:
                    df = yf.Ticker(s).history(period="1mo", interval="1h")
                    if not df.empty:
                        df['RSI'] = ta.rsi(df['Close'], length=14)
                        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                        cache[s] = df.dropna()
                except: pass
            
            if s in cache:
                df = cache[s]
                # Test paramètres aléatoires
                t_rsi = random.randint(25, 55)
                t_sl = round(random.uniform(1.5, 4.0), 1)
                
                # Simulation Rapide (Backtest)
                pnl = 0
                idx = random.randint(50, len(df)-10)
                row = df.iloc[idx]
                
                if row['RSI'] < t_rsi:
                    entry = row['Close']
                    sl = entry - (row['ATR'] * t_sl)
                    tp = entry + (row['ATR'] * t_sl * 2.0)
                    
                    future = df.iloc[idx+1:idx+6]
                    if not future.empty:
                        if future['High'].max() > tp: pnl = tp - entry
                        elif future['Low'].min() < sl: pnl = sl - entry
                
                # Log si résultat significatif
                if pnl != 0:
                    short_term_memory.append(pnl)
                    if pnl > 0:
                        fast_log(f"🧪 **TEST {s}** (RSI<{t_rsi}, SL={t_sl}) -> ✅ +{pnl:.1f}$")
                        if pnl > 500: # Super résultat
                            brain['best_params'] = {"rsi_buy": t_rsi, "sl_atr": t_sl}
                            save_brain()
                    else:
                        fast_log(f"🧪 **TEST {s}** (RSI<{t_rsi}, SL={t_sl}) -> ❌ {pnl:.1f}$")

            # Rapport 10 min
            if len(short_term_memory) >= 10:
                wins = sum(1 for x in short_term_memory if x > 0)
                tot = sum(short_term_memory)
                msg = {
                    "embeds": [{
                        "title": "🎓 RAPPORT D'ÉTUDE",
                        "description": f"Le bot travaille en tâche de fond.",
                        "color": 0xFFD700,
                        "fields": [
                            {"name": "Profit Simulé", "value": f"{tot:.2f}$", "inline": True},
                            {"name": "Précision", "value": f"{(wins/10)*100:.0f}%", "inline": True}
                        ]
                    }]
                }
                send_summary(msg)
                short_term_memory = []
            
            time.sleep(10) # Vitesse d'étude
        except: time.sleep(10)

# --- 8. MOTEUR PRINCIPAL ---
def run_engine():
    global brain, bot_state
    load_brain()
    
    while True:
        try:
            # Dashboard Update
            eq, pos = broker.get_portfolio()
            bot_state['equity'] = eq
            bot_state['positions'] = pos
            
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if not market_open:
                bot_state['status'] = "🌙 NUIT (Apprentissage)"
                time.sleep(60)
                continue
            
            bot_state['status'] = "👁️ SCAN OMNISCIENT"
            
            # VENTES
            for s in list(brain['holdings'].keys()):
                p_data = brain['holdings'][s]
                curr = broker.get_price(s)
                if not curr: continue
                
                exit_r = None
                if curr < p_data['stop']: exit_r = "STOP LOSS"
                elif curr > p_data['tp']: exit_r = "TAKE PROFIT"
                
                if exit_r:
                    pnl = (curr - p_data['entry']) * p_data['qty']
                    brain['cash'] += p_data['qty'] * curr
                    brain['total_pnl'] += pnl
                    del brain['holdings'][s]
                    
                    col = 0x2ecc71 if pnl > 0 else 0xe74c3c
                    send_trade_alert({"embeds": [{"title": f"{exit_r} {s}", "description": f"PnL: {pnl:.2f}$", "color": col}]})
                    save_brain()

            # ACHATS
            if len(brain['holdings']) < 5:
                for s in WATCHLIST:
                    if s in brain['holdings']: continue
                    
                    try:
                        df = yf.Ticker(s).history(period="1mo", interval="1h")
                        if df.empty: continue
                        
                        df['RSI'] = ta.rsi(df['Close'], length=14)
                        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                        row = df.iloc[-1]
                        
                        # Filtre rapide (Paramètres Appris)
                        if row['RSI'] < brain['best_params']['rsi_buy']:
                            
                            # Lancement Modules Lourds
                            mc = run_monte_carlo(df['Close'])
                            whale, wh_msg = check_whale(df)
                            
                            # Si les maths sont bonnes, on lance l'IA Vision et Social
                            if mc > 0.60:
                                vis = get_vision_score(df, s)
                                soc = get_social_sentiment(s)
                                
                                # CONSEIL FINAL
                                decision = consult_omniscient_brain(s, row['RSI'], mc, vis, soc, whale)
                                bot_state['last_decision'] = f"{s}: {decision['score']}/100"
                                
                                fast_log(f"🧠 **ANALYSE {s}:** MC:{mc:.2f} | Vis:{vis:.2f} | Whale:{whale} => Score {decision['score']}")
                                
                                if decision['decision'] == "BUY":
                                    price = row['Close']
                                    # Kelly Simplifié (Mise)
                                    qty = (brain['cash'] * 0.15) / price
                                    sl = price - (row['ATR'] * brain['best_params']['sl_atr'])
                                    tp = price + (row['ATR'] * 3.0)
                                    
                                    brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                                    brain['cash'] -= (qty * price)
                                    
                                    msg = {
                                        "embeds": [{
                                            "title": f"🌌 ACHAT SINGULARITÉ : {s}",
                                            "description": decision['reason'],
                                            "color": 0x2ecc71,
                                            "fields": [
                                                {"name": "Vision", "value": f"{vis:.2f}", "inline": True},
                                                {"name": "Quantique", "value": f"{mc:.2f}", "inline": True},
                                                {"name": "Volume", "value": wh_msg, "inline": True}
                                            ]
                                        }]
                                    }
                                    send_trade_alert(msg)
                                    save_brain()
                    except: pass
            
            time.sleep(60)
        except: time.sleep(10)

# --- 9. DASHBOARD ---
HTML = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta http-equiv="refresh" content="10">
    <title>OMNISCIENT V72</title>
    <style>
        body { background: #000; color: #0f0; font-family: monospace; padding: 20px; }
        .card { border: 1px solid #333; padding: 15px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>👁️ OMNISCIENT V72</h1>
    <div class="card">
        <h3>STATUS: {{ status }}</h3>
        <h2>CAPITAL: ${{ equity }}</h2>
        <p>Log: {{ last_log }}</p>
        <p>Décision: {{ last_dec }}</p>
    </div>
    <div class="card">
        {% for p in positions %}
        <div>{{ p.symbol }} | PnL: {{ p.pnl }}$</div>
        {% endfor %}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    eq, pos = broker.get_portfolio()
    return render_template_string(HTML, equity=f"{eq:,.2f}", status=bot_state['status'], positions=pos, last_log=bot_state['last_log'], last_dec=bot_state['last_decision'])

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
