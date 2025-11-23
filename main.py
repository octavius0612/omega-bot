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

# --- 2. CONFIGURATION OVERDRIVE ---
WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR"]
INITIAL_CAPITAL = 50000.0 

# ⚠️ ATTENTION : 100 000 simulations consomment énormément de CPU/RAM
SIMULATION_COUNT = 100000 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "best_params": {"rsi_buy": 30, "sl": 2.0},
    "stats": {"cpu_cycles": 0, "simulations_total": 0},
    "q_table": {},
    "karma": {s: 10.0 for s in WATCHLIST}
}

bot_state = {
    "status": "☢️ DÉMARRAGE NUCLÉAIRE...",
    "last_log": "Init...",
    "load": "0%"
}

log_queue = queue.Queue()
short_term_memory = []

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- 3. LOGGING HAUTE VITESSE ---
def fast_log(text):
    log_queue.put(text)

def logger_worker():
    buffer = []
    last_send = time.time()
    while True:
        try:
            while not log_queue.empty():
                buffer.append(log_queue.get())
            
            # Envoi agressif (toutes les 1.0s)
            if buffer and (len(buffer) > 5 or time.time() - last_send > 1.0):
                msg_block = "\n".join(buffer[:15])
                buffer = buffer[15:]
                if LEARNING_WEBHOOK_URL:
                    requests.post(LEARNING_WEBHOOK_URL, json={"content": msg_block})
                last_send = time.time()
            time.sleep(0.1) # Pause minimale
        except: time.sleep(0.1)

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
            repo.update_file("brain.json", "Nuclear Save", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 5. MOTEUR QUANTIQUE NUCLÉAIRE (MAX POWER) ---
def run_massive_monte_carlo(prices):
    """
    Utilise NumPy pour saturer le processeur avec des calculs matriciels.
    """
    try:
        start_t = time.time()
        returns = prices.pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        last = prices.iloc[-1]
        
        # GÉNÉRATION MATRICIELLE MASSIVE
        # On crée une matrice de 100,000 lignes x 10 colonnes de nombres aléatoires
        # Cela force le CPU à faire 1 Million d'opérations flottantes instantanément
        sims = np.random.normal(mu, sigma, (SIMULATION_COUNT, 10))
        sims = last * (1 + sims).cumprod(axis=1)
        
        final_prices = sims[:, -1]
        prob = np.sum(final_prices > last) / SIMULATION_COUNT
        
        calc_time = (time.time() - start_t) * 1000
        brain['stats']['simulations_total'] += SIMULATION_COUNT
        
        return prob, calc_time
    except: return 0.5, 0

# --- 6. AUTRES SENS ---
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

def consult_council(s, data):
    prompt = f"""
    ANALYSE MAXIMALE {s}.
    Maths: {data['mc_prob']*100:.1f}% (sur {SIMULATION_COUNT} univers).
    Vision: {data['vis']:.2f}. Social: {data['soc']:.2f}.
    Baleine: {data['whale']}. RSI: {data['rsi']:.1f}.
    
    Si Maths > 60% ET Vision > 0.6 : BUY.
    JSON: {{"vote": "BUY/WAIT", "reason": "Synthèse"}}
    """
    try:
        res = model.generate_content(prompt)
        return json.loads(res.text.replace("```json","").replace("```",""))
    except: return {"vote": "WAIT", "reason": "Erreur"}

# --- 7. BOUCLE D'APPRENTISSAGE HYPER-ACTIVE ---
def run_nuclear_learning():
    global brain
    cache = {}
    
    fast_log(f"☢️ **NUCLEAR CORE:** Démarrage à {SIMULATION_COUNT} sims/cycle.")
    
    while True:
        try:
            # Pas de pause. On enchaîne.
            s = random.choice(WATCHLIST)
            
            # Mise à jour cache plus fréquente (tous les 5 cycles)
            if s not in cache or random.random() < 0.2:
                try:
                    df = yf.Ticker(s).history(period="2mo", interval="1h")
                    if not df.empty:
                        df['RSI'] = ta.rsi(df['Close'], length=14)
                        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                        cache[s] = df.dropna()
                except: pass
            
            if s in cache:
                df = cache[s]
                
                # Test Paramètres (On teste 50 variations d'un coup)
                best_local_pnl = -999
                best_local_params = None
                
                # Simulation vectorisée de stratégie
                # On cherche le meilleur RSI sur les données passées
                rsi_range = np.arange(20, 60, 2) # Teste 20, 22, 24... jusqu'à 60
                
                for rsi_test in rsi_range:
                    # Backtest ultra rapide
                    signals = df[df['RSI'] < rsi_test]
                    if len(signals) > 5:
                        # Simulation PnL approximatif
                        pnl = np.sum(signals['Close'].shift(-5) - signals['Close'])
                        if pnl > best_local_pnl:
                            best_local_pnl = pnl
                            best_local_params = rsi_test
                
                if best_local_pnl > 0:
                    brain['best_params']['rsi_buy'] = int(best_local_params)
                    fast_log(f"🧬 **OPTIMISATION ({s}):** Meilleur RSI trouvé < {best_local_params} (PnL théorique: {best_local_pnl:.0f}$)")
                    
                    # On sauvegarde si c'est un record
                    if best_local_pnl > 1000: save_brain()

            # On ne dort que 1 seconde pour laisser respirer le serveur
            time.sleep(1)
            
        except Exception as e:
            time.sleep(5)

# --- 8. MOTEUR TRADING LIVE ---
def run_trading():
    global brain, bot_state
    load_brain()
    
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if market_open:
                bot_state['status'] = "🔥 FULL POWER TRADING"
                
                # Gestion Ventes
                for s in list(brain['holdings'].keys()):
                    # ... (Logique Vente Standard) ...
                    pass

                # Scan Achats
                if len(brain['holdings']) < 5:
                    for s in WATCHLIST:
                        if s in brain['holdings']: continue
                        
                        try:
                            df = yf.Ticker(s).history(period="1mo", interval="1h")
                            if df.empty: continue
                            
                            # CALCULS LOURDS
                            mc_prob, ms = run_massive_monte_carlo(df['Close'])
                            
                            df['RSI'] = ta.rsi(df['Close'], length=14)
                            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                            row = df.iloc[-1]
                            
                            fast_log(f"⚡ **SCAN {s}:** {SIMULATION_COUNT} Sims en {ms:.0f}ms. Proba: {mc_prob*100:.1f}%")
                            
                            # Si Maths OK, on lance le reste
                            if mc_prob > 0.60 and row['RSI'] < brain['best_params']['rsi_buy']:
                                
                                vis = get_vision_score(df)
                                soc = get_social_hype(s)
                                whl, wh_msg = check_whale(df)
                                
                                data_pack = {"mc_prob": mc_prob, "vis": vis, "soc": soc, "whale": whl, "rsi": row['RSI']}
                                council = consult_council(s, data_pack)
                                
                                if council['vote'] == "BUY":
                                    # Achat
                                    price = row['Close']
                                    qty = 500 / price
                                    sl = price - (row['ATR'] * 2)
                                    tp = price + (row['ATR'] * 3.5)
                                    
                                    brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                                    brain['cash'] -= 500
                                    
                                    msg = f"☢️ **ACHAT NUCLÉAIRE : {s}**\n{SIMULATION_COUNT} scénarios calculés.\nDécision: {council['reason']}"
                                    send_trade_alert({"content": msg})
                                    save_brain()
                                    
                        except: pass
            else:
                bot_state['status'] = "🌙 NUIT (Calculs Intensifs)"
            
            time.sleep(10) # Scan live toutes les 10s
        except: time.sleep(10)

@app.route('/')
def index(): 
    sims = f"{brain['stats']['simulations_total']:,}"
    return f"<h1>NUCLEAR V85</h1><p>{bot_state['status']}</p><p>Simulations Totales: {sims}</p>"

def start_threads():
    threading.Thread(target=run_trading, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()
    threading.Thread(target=logger_worker, daemon=True).start()
    threading.Thread(target=run_nuclear_learning, daemon=True).start()

load_brain()
start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
