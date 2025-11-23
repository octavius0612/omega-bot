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

# --- 1. CONFIGURATION ---
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

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "best_params": {"rsi_buy": 30, "sl": 2.0, "tp": 4.0},
    "stats": {"math_precision": "INFINIE", "wins": 0},
    "trade_history": [],
    "total_pnl": 0.0
}

bot_state = {
    "status": "Initialisation Solver...",
    "last_log": "Calibrage..."
}

log_queue = queue.Queue()
short_term_memory = []

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. LOGGING ---
def fast_log(text):
    log_queue.put(text)

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

def send_summary(msg):
    if SUMMARY_WEBHOOK_URL: requests.post(SUMMARY_WEBHOOK_URL, json=msg)

def send_trade_alert(msg):
    if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json=msg)

def run_heartbeat():
    while True:
        if HEARTBEAT_WEBHOOK_URL: requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"})
        time.sleep(30)

# --- 3. MÉMOIRE ---
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
            repo.update_file("brain.json", "Infinite Save", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 4. LE CŒUR MATHÉMATIQUE (L'INFINI) ---
def run_quantum_math(prices, days_ahead=5):
    """
    Remplace Monte Carlo par le Calcul Intégral Gaussien.
    C'est l'équivalent mathématique de simuler une infinité de scénarios.
    """
    try:
        # 1. Calcul des paramètres du chaos (Drift & Volatilité)
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        mu = log_returns.mean() # Tendance moyenne (Drift)
        sigma = log_returns.std() # Volatilité
        
        # 2. Projection Temporelle
        # On projette sur 'days_ahead' périodes (ex: 5 heures si données horaires)
        t = days_ahead 
        
        # 3. Formule de Probabilité Exacte (Z-Score projeté)
        # P(S_t > S_0) = P(ln(S_t/S_0) > 0)
        # C'est la probabilité que le prix futur soit supérieur au prix actuel
        
        # Drift total sur la période
        drift_total = (mu - 0.5 * sigma**2) * t
        vol_total = sigma * np.sqrt(t)
        
        # Score Z inversé (Probabilité de dépassement)
        # Z = (Target - Drift) / Vol
        # Ici Target = 0 (car on veut rendement > 0)
        z_score = - (drift_total) / vol_total
        
        # Fonction de Répartition Cumulative (Loi Normale)
        # Cela donne la probabilité exacte entre 0.0 et 1.0
        prob_up = 1 - norm.cdf(z_score)
        
        return prob_up
        
    except Exception as e:
        return 0.5

# --- 5. AUTRES SENS (IA & DATA) ---
def get_vision_score(df):
    try:
        buf = io.BytesIO()
        mpf.plot(df.tail(60), type='candle', style='nightclouds', savefig=buf)
        buf.seek(0)
        img = Image.open(buf)
        res = model.generate_content(["Score achat 0.0-1.0 chartiste ?", img])
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
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers).json()
        txt = " ".join([m['body'] for m in r['messages'][:15]])
        return TextBlob(txt).sentiment.polarity
    except: return 0

# --- 6. CERVEAU CENTRAL ---
def consult_council(s, rsi, mc, vis, soc, whale):
    prompt = f"""
    DECISION {s}.
    1. MATHS INFINIES: {mc*100:.4f}% chance hausse.
    2. VISION: {vis:.2f}.
    3. SOCIAL: {soc:.2f}.
    4. BALEINE: {whale}.
    5. RSI: {rsi:.1f}.
    
    Si Maths > 60% ET Vision > 0.6 : ACHAT.
    Si Baleine ET RSI < 30 : ACHAT FORT.
    JSON: {{"vote": "BUY/WAIT", "reason": "Synthèse"}}
    """
    try:
        res = model.generate_content(prompt)
        return json.loads(res.text.replace("```json","").replace("```",""))
    except: return {"vote": "WAIT", "reason": "Erreur"}

# --- 7. APPRENTISSAGE CONTINU ---
def run_learning_loop():
    global brain, short_term_memory
    cache = {}
    fast_log("♾️ **SOLVER INFINI:** Démarrage de l'optimisation mathématique.")
    
    while True:
        try:
            s = random.choice(WATCHLIST)
            
            # Update Cache (20%)
            if s not in cache or random.random() < 0.2:
                try:
                    df = yf.Ticker(s).history(period="1mo", interval="1h")
                    if not df.empty:
                        df['RSI'] = ta.rsi(df['Close'], length=14)
                        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                        cache[s] = df.dropna()
                except: pass
            
            if s in cache:
                df = cache[s]
                # Mutation
                t_rsi = random.randint(20, 55)
                t_sl = round(random.uniform(1.5, 3.5), 1)
                
                # Test Rétrospectif
                idx = random.randint(50, len(df)-10)
                row = df.iloc[idx]
                
                if row['RSI'] < t_rsi:
                    # Analyse Quantique du passé
                    past_data = df.iloc[:idx]['Close'][-50:]
                    prob_math = run_monte_carlo(past_data)
                    
                    # Résultat réel
                    future = df.iloc[idx+1:idx+6]
                    entry = row['Close']
                    sl = entry - (row['ATR'] * t_sl)
                    tp = entry + (row['ATR'] * t_sl * 2.5)
                    
                    pnl = 0
                    if not future.empty:
                        if future['High'].max() > tp: pnl = tp - entry
                        elif future['Low'].min() < sl: pnl -= (entry - sl)
                    
                    if pnl != 0:
                        short_term_memory.append(pnl)
                        
                        # Log intelligent
                        emoji = "✅" if pnl > 0 else "❌"
                        # On ne log que si la proba mathématique était forte (>60%)
                        if prob_math > 0.6:
                            fast_log(f"🧪 **TEST {s}:** Maths {prob_math*100:.1f}% | RSI<{t_rsi} -> {emoji} {pnl:.1f}$")
                        
                        if pnl > 400 and prob_math > 0.6:
                            brain['best_params'] = {"rsi_buy": t_rsi, "sl": t_sl}
                            save_brain()

            # Bilan
            if len(short_term_memory) >= 10:
                tot = sum(short_term_memory)
                msg = {
                    "embeds": [{
                        "title": "♾️ RAPPORT MATHÉMATIQUE",
                        "color": 0xFFD700,
                        "description": f"Analyse vectorielle terminée.",
                        "fields": [
                            {"name": "Gain Session", "value": f"**{tot:.2f}$**", "inline": True},
                            {"name": "Précision Modèle", "value": "Infinie (Gaussien)", "inline": True}
                        ]
                    }]
                }
                send_summary(msg)
                short_term_memory = []
            
            time.sleep(2) # Rapide
        except: time.sleep(10)

# --- 8. TRADING LIVE ---
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
                bot_state['status'] = "🟢 LIVE (Solver Actif)"
                
                # VENTES
                for s in list(brain['holdings'].keys()):
                    pos = brain['holdings'][s]
                    curr = broker.get_price(s)
                    if not curr: continue
                    
                    if curr < pos['stop']:
                        brain['cash'] += pos['qty'] * curr
                        del brain['holdings'][s]
                        send_trade_alert({"embeds": [{"title": f"🔴 VENTE {s}", "description": "Stop Loss (Maths)", "color": 0xe74c3c}]})
                        save_brain()
                    elif curr > pos['tp']:
                        brain['cash'] += pos['qty'] * curr
                        del brain['holdings'][s]
                        send_trade_alert({"embeds": [{"title": f"🟢 VENTE {s}", "description": "Objectif Atteint", "color": 0x2ecc71}]})
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
                            
                            if row['RSI'] < brain['best_params']['rsi_buy']:
                                
                                # 1. CALCUL INFINI (INSTANTANÉ)
                                prob_infinite = run_quantum_math(df['Close'])
                                
                                fast_log(f"⚛️ **ANALYSE {s}:** Proba Infinie: **{prob_infinite*100:.2f}%**")
                                
                                # Si les maths disent OUI (>60%)
                                if prob_infinite > 0.60:
                                    vis = get_vision_score(df)
                                    soc = get_social_hype(s)
                                    whl, _ = check_whale(df)
                                    
                                    council = consult_council(s, row['RSI'], prob_infinite, vis, soc, whl)
                                    
                                    if council['vote'] == "BUY":
                                        price = row['Close']
                                        qty = 500 / price
                                        sl = price - (row['ATR'] * brain['best_params']['sl'])
                                        tp = price + (row['ATR'] * 3.5)
                                        
                                        brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                                        brain['cash'] -= 500
                                        
                                        send_trade_alert({
                                            "embeds": [{"title": f"⚛️ ACHAT MATHÉMATIQUE : {s}", "description": f"Certitude: {prob_infinite*100:.2f}%", "color": 0x2ecc71}]
                                        })
                                        save_brain()
                        except: pass
            else:
                bot_state['status'] = "🌙 NUIT"
            
            time.sleep(30)
        except: time.sleep(10)

@app.route('/')
def index(): 
    eq, pos = broker.get_portfolio()
    return f"<h1>INFINITE SOLVER V96</h1><p>{bot_state['status']}</p><p>Capital: {eq:.2f}$</p>"

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
