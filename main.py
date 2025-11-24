import os
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
from scipy.stats import norm
from textblob import TextBlob
# MACHINE LEARNING
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
# 1. CLÉS & CONFIGURATION ULTIME
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

WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR", "GOOG", "META"]
INITIAL_CAPITAL = 50000.0 
SIMULATION_COUNT = 2000

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "paper_cash": 1000.0,
    "paper_holdings": {},
    "genome": {"rsi_buy": 30, "sl_mult": 2.0, "tp_mult": 3.5},
    "stats": {"generation": 0, "best_pnl": 0.0},
    "ai_performance": {}, # Précision du ML
    "emotions": {"confidence": 50.0, "stress": 20.0},
    "karma": {s: 10.0 for s in WATCHLIST},
    "trade_history": [],
    "total_pnl": 0.0
}

bot_state = {
    "status": "Booting V111...",
    "mode": "INIT",
    "last_log": "Chargement...",
    "web_logs": []
}

log_queue = queue.Queue()
short_term_memory = []
active_models = {} # Modèles ML en RAM

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ==============================================================================
# 2. LOGGING & COMMS
# ==============================================================================
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
            if buffer and (len(buffer) > 5 or time.time() - last_send > 1.5):
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
        if "paper_cash" not in brain: brain['paper_cash'] = 1000.0
    except: pass

def save_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        content = json.dumps(brain, indent=4)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Omniscient Save", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# ==============================================================================
# 4. MOTEUR MACHINE LEARNING (PRÉCOGNITION)
# ==============================================================================
def prepare_ml_data(symbol):
    try:
        df = yf.Ticker(symbol).history(period="60d", interval="1h")
        if len(df) < 100: return None, None
        
        # Features Avancées
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['SMA50'] = ta.sma(df['Close'], length=50)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['Return'] = df['Close'].pct_change()
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int) # Cible : Hausse future
        
        df = df.dropna()
        features = ['RSI', 'SMA50', 'ATR', 'Return', 'Volume']
        return df, features
    except: return None, None

def train_model(symbol):
    fast_log(f"🧠 **TRAINING:** Entraînement neuronal sur {symbol}...")
    df, feats = prepare_ml_data(symbol)
    if df is None: return None
    
    X = df[feats]
    y = df['Target']
    split = int(len(df)*0.8)
    
    model_ml = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=100)
    model_ml.fit(X.iloc[:split], y.iloc[:split])
    
    score = model_ml.score(X.iloc[split:], y.iloc[split:])
    brain['ai_performance'][symbol] = score
    fast_log(f"✅ **{symbol} APPRIS:** Précision {score*100:.1f}%")
    save_brain()
    return model_ml

def get_ai_prediction(symbol, model_ml, feats):
    try:
        df, _ = prepare_ml_data(symbol)
        if df is None: return 0.5
        current = df.iloc[[-1]][feats]
        prob = model_ml.predict_proba(current)[0][1]
        return prob
    except: return 0.5

def run_ml_maintenance():
    """Maintient les cerveaux ML à jour"""
    while True:
        try:
            for s in WATCHLIST:
                if s not in active_models or random.random() < 0.05:
                    m = train_model(s)
                    if m: active_models[s] = m
                time.sleep(10)
            time.sleep(300)
        except: time.sleep(60)

# ==============================================================================
# 5. MOTEUR QUANTIQUE & SENSORIEL
# ==============================================================================
def run_monte_carlo(prices):
    try:
        ret = prices.pct_change().dropna()
        sims = prices.iloc[-1] * (1 + np.random.normal(ret.mean(), ret.std(), (SIMULATION_COUNT, 10)))
        return np.sum(sims[:, -1] > prices.iloc[-1]) / SIMULATION_COUNT
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

def get_social_hype(symbol):
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).json()
        txt = " ".join([m['body'] for m in r['messages'][:10]])
        return TextBlob(txt).sentiment.polarity
    except: return 0

def consult_council(s, rsi, mc, vis, soc, whale, ai_prob):
    prompt = f"""
    DÉCISION ULTIME {s}.
    1. PRECOGNITION (ML): {ai_prob*100:.1f}% prob.
    2. QUANTIQUE: {mc*100:.1f}% prob.
    3. VISION: {vis:.2f}.
    4. SOCIAL: {soc:.2f}.
    5. BALEINE: {whale}.
    6. RSI: {rsi:.1f}.
    
    RÈGLE:
    - Si (ML > 0.6 ET Quantique > 0.6) OU (Baleine ET RSI < 30): BUY.
    - Sinon WAIT.
    
    JSON: {{"vote": "BUY/WAIT", "reason": "..."}}
    """
    try:
        res = model.generate_content(prompt)
        return json.loads(res.text.replace("```json","").replace("```",""))
    except: return {"vote": "WAIT", "reason": "Erreur"}

# ==============================================================================
# 6. MOTEUR D'APPRENTISSAGE GÉNÉTIQUE (LE RÊVE)
# ==============================================================================
def generate_gemini_summary(stats, best_run):
    try:
        prompt = f"Analyse résultats trading: Profit {stats['total_pnl']:.2f}$. Top: {best_run['s']}. Conseil court ?"
        return model.generate_content(prompt).text.strip()
    except: return "Analyse effectuée."

def run_dream_learning():
    global brain
    cache = {}
    fast_log("🧬 **GENESIS:** Cycle d'évolution génétique activé.")
    
    for s in WATCHLIST:
        try:
            df = yf.Ticker(s).history(period="1mo", interval="1h")
            if not df.empty:
                df['RSI'] = ta.rsi(df['Close'], 14)
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)
                cache[s] = df.dropna()
        except: pass

    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if market_open:
                time.sleep(60)
                continue
            
            # Mutation
            parent = brain['genome']
            mutant = {
                "rsi_buy": max(15, min(60, parent['rsi_buy'] + random.randint(-4, 4))),
                "sl_mult": round(max(1.0, parent['sl_mult'] + random.uniform(-0.3, 0.3)), 1),
                "tp_mult": round(max(1.5, parent['tp_mult'] + random.uniform(-0.3, 0.3)), 1)
            }
            
            # Simulation
            s = random.choice(list(cache.keys()))
            df = cache[s]
            idx = random.randint(0, len(df)-50)
            subset = df.iloc[idx:idx+50]
            pnl = 0
            
            for i in range(len(subset)-10):
                row = subset.iloc[i]
                if row['RSI'] < mutant['rsi_buy']:
                    entry = row['Close']
                    sl = entry - (row['ATR'] * mutant['sl_mult'])
                    tp = entry + (row['ATR'] * mutant['tp_mult'])
                    fut = subset.iloc[i+1:i+6]
                    if not fut.empty:
                        if fut['High'].max() > tp: pnl += (tp - entry)
                        elif fut['Low'].min() < sl: pnl -= (entry - sl)
            
            if pnl != 0:
                short_term_memory.append({"s": s, "pnl": pnl, "win": pnl>0, "rsi": mutant['rsi_buy']})
                
                emoji = "✅" if pnl > 0 else "❌"
                fast_log(f"🧪 **TEST {s}:** RSI<{mutant['rsi_buy']} -> {emoji} {pnl:.1f}$")
                
                if pnl > brain['stats']['best_pnl']:
                    brain['stats']['best_pnl'] = pnl
                    brain['genome'] = mutant
                    save_brain()

            # Bilan 10 tests
            if len(short_term_memory) >= 10:
                wins = sum(1 for x in short_term_memory if x['win'])
                tot = sum(x['pnl'] for x in short_term_memory)
                best = max(short_term_memory, key=lambda x: x['pnl'])
                
                ai_text = generate_gemini_analysis({"total_pnl": tot, "win_rate": (wins/10)*100}, best)
                
                msg = {
                    "embeds": [{
                        "title": "🎓 RAPPORT D'ÉTUDE",
                        "color": 0xFFD700,
                        "description": f"**IA:** *{ai_text}*",
                        "fields": [
                            {"name": "Profit", "value": f"**{tot:.2f}$**", "inline": True},
                            {"name": "Top Config", "value": f"{best['s']} (Gain {best['pnl']:.0f}$)", "inline": False}
                        ]
                    }]
                }
                if SUMMARY_WEBHOOK_URL: requests.post(SUMMARY_WEBHOOK_URL, json=msg)
                short_term_memory = []

            time.sleep(5)
        except: time.sleep(10)

# ==============================================================================
# 7. MOTEUR TRADING LIVE
# ==============================================================================
class GhostBroker:
    def get_price(self, symbol):
        try: return yf.Ticker(symbol).fast_info['last_price']
        except: return None
    def get_portfolio(self):
        return brain['cash'], [{"symbol": s, "qty": d['qty'], "pnl": 0} for s, d in brain['holdings'].items()]
broker = GhostBroker()

def run_trading():
    global brain, bot_state
    load_brain()
    
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if market_open:
                bot_state['status'] = "🟢 LIVE OMNISCIENT"
                
                # VENTES
                for pf in [brain['holdings'], brain['paper_holdings']]:
                    for s in list(pf.keys()):
                        # (Logique vente standard simplifiée pour espace, identique V107)
                        # ...
                        pass

                # ACHATS
                if len(brain['holdings']) < 5:
                    for s in WATCHLIST:
                        if s in brain['holdings']: continue
                        try:
                            df = yf.Ticker(s).history(period="1mo", interval="1h")
                            if df.empty: continue
                            df['RSI'] = ta.rsi(df['Close'], 14); df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)
                            row = df.iloc[-1]
                            
                            if row['RSI'] < brain['genome']['rsi_buy']:
                                # Lancement Analyse Totale
                                mc = run_monte_carlo(df['Close'])
                                vis = get_vision_score(df)
                                soc = get_social_hype(s)
                                whl, _ = check_whale(df)
                                
                                # Prédiction ML
                                ai_prob = 0.5
                                if s in active_models:
                                    ai_prob = get_ai_prediction(s, active_models[s], ['RSI', 'SMA50', 'ATR', 'Return', 'Volume'])
                                
                                council = consult_council(s, row['RSI'], mc, vis, soc, whl)
                                
                                fast_log(f"🧠 **{s}:** ML:{ai_prob:.2f} | MC:{mc:.2f} | Vis:{vis:.2f} => {council['vote']}")
                                
                                if council['vote'] == "BUY":
                                    # ... (Logique achat standard V107) ...
                                    # + Alertes Discord
                                    pass
                        except: pass
            else:
                bot_state['status'] = "🌙 NUIT"
            
            time.sleep(30)
        except: time.sleep(10)

# ==============================================================================
# 8. DASHBOARD
# ==============================================================================
@app.route('/')
def index():
    eq, _ = broker.get_portfolio()
    return f"<h1>OMNISCIENT V111</h1><p>Status: {bot_state['status']}</p><p>Capital: ${eq:,.2f}</p>"

def start_threads():
    threading.Thread(target=run_trading, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()
    threading.Thread(target=logger_worker, daemon=True).start()
    threading.Thread(target=run_dream_learning, daemon=True).start()
    threading.Thread(target=run_ml_maintenance, daemon=True).start()

load_brain()
start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)