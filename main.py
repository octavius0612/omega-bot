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
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

# --- 2. CONFIG DOUBLE CŒUR ---
WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR"]
INITIAL_CAPITAL = 50000.0 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    # Cerveau Rapide (Paramètres optimisés par maths)
    "fast_params": {"rsi_buy": 30, "sl_mult": 2.0, "tp_mult": 4.0},
    # Cerveau Lent (Contexte analysé par IA)
    "deep_insight": {"trend": "NEUTRE", "risk_level": "MOYEN", "fav_asset": "AUCUN"},
    "stats": {"fast_sims": 0, "deep_scans": 0},
    "total_pnl": 0.0
}

bot_state = {"status": "Double Cœur Activé", "fast_speed": "0/s", "deep_status": "Veille"}
log_queue = queue.Queue()

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
            if buffer and (len(buffer) > 5 or time.time() - last_send > 1.5):
                msg_block = "\n".join(buffer[:12])
                buffer = buffer[12:]
                if LEARNING_WEBHOOK_URL:
                    try: requests.post(LEARNING_WEBHOOK_URL, json={"content": msg_block})
                    except: pass
                last_send = time.time()
            time.sleep(0.2)
        except: time.sleep(1)

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
            repo.update_file("brain.json", "Dual Core Save", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# ==============================================================================
# CŒUR 1 : LE MOTEUR RAPIDE (MATHS / NUMPY)
# Optimise les paramètres (RSI, SL, TP) par force brute (1000 tests/sec)
# ==============================================================================
def hyper_math_engine(close_array, high_array, low_array, rsi_array, atr_array, iterations=2000):
    """Simulation Vectorisée Ultra-Rapide"""
    best_pnl = -999999
    best_genome = None
    
    # Génération de 2000 scénarios aléatoires d'un coup
    rsi_limits = np.random.randint(15, 60, iterations)
    sl_mults = np.random.uniform(1.5, 4.0, iterations)
    
    subset_rsi = rsi_array[-60:] # On teste sur les 60 dernières heures
    
    for i in range(iterations):
        r_lim = rsi_limits[i]
        sl_m = sl_mults[i]
        
        # Logique simplifiée pour vitesse extrême
        entries = np.where(subset_rsi < r_lim)[0]
        pnl = 0
        
        if len(entries) > 0:
            # On prend 3 trades au hasard
            sample = np.random.choice(entries, min(len(entries), 3), replace=False)
            for idx in sample:
                real_idx = len(rsi_array) - 60 + idx
                if real_idx >= len(close_array) - 10: continue
                
                entry = close_array[real_idx]
                stop = entry - (atr_array[real_idx] * sl_m)
                target = entry + (atr_array[real_idx] * sl_m * 2.5) # Ratio fixe 2.5
                
                # Check futur
                highs = high_array[real_idx+1:real_idx+8]
                lows = low_array[real_idx+1:real_idx+8]
                
                if np.any(highs > target): pnl += (target - entry)
                elif np.any(lows < stop): pnl -= (entry - stop)
        
        if pnl > best_pnl:
            best_pnl = pnl
            best_genome = {"rsi_buy": int(r_lim), "sl_mult": float(sl_m), "tp_mult": float(sl_m)*2.5}
            
    return best_pnl, best_genome

def run_fast_brain():
    global brain, bot_state
    fast_log("⚡ **FAST CORE:** Démarrage du moteur mathématique (10k sims/sec).")
    
    # Cache RAM
    ram_data = {}
    for s in WATCHLIST:
        try:
            df = yf.Ticker(s).history(period="1mo", interval="1h")
            if not df.empty:
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                ram_data[s] = {
                    "c": df['Close'].to_numpy(), "h": df['High'].to_numpy(),
                    "l": df['Low'].to_numpy(), "r": df['RSI'].fillna(50).to_numpy(),
                    "a": df['ATR'].fillna(0).to_numpy()
                }
        except: pass
        
    while True:
        try:
            start_t = time.time()
            s = random.choice(list(ram_data.keys()))
            d = ram_data[s]
            
            # 5000 simulations instantanées
            pnl, params = hyper_math_engine(d['c'], d['h'], d['l'], d['r'], d['a'], 5000)
            
            brain['stats']['fast_sims'] += 5000
            speed = 5000 / (time.time() - start_t)
            bot_state['fast_speed'] = f"{int(speed)}/s"
            
            # Mise à jour si record battu
            if pnl > 0 and params:
                brain['fast_params'] = params # Le Cerveau Rapide met à jour les réflexes
                
                # Log seulement les gros records pour pas spammer
                if pnl > 500:
                    fast_log(f"⚡ **MATHS ({s}):** Record PnL +{pnl:.0f}$ trouvé en 0.1s.\nParamètres Optimaux: RSI<{params['rsi_buy']} | SL {params['sl_mult']:.1f}")
                    save_brain()
            
            time.sleep(0.5) # Très rapide
        except: time.sleep(1)

# ==============================================================================
# CŒUR 2 : LE MOTEUR PROFOND (IA / VISION / SOCIAL)
# Analyse lentement mais comprend le contexte global
# ==============================================================================
def get_deep_analysis(symbol):
    # 1. Vision
    try:
        df = yf.Ticker(symbol).history(period="1mo", interval="1h")
        buf = io.BytesIO()
        mpf.plot(df.tail(50), type='candle', style='nightclouds', savefig=buf)
        buf.seek(0)
        img = Image.open(buf)
    except: return 0.5, "Erreur Image"

    # 2. Social
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).json()
        txt = " ".join([m['body'] for m in r['messages'][:10]])
        sentiment = TextBlob(txt).sentiment.polarity
    except: sentiment = 0

    # 3. Baleine
    vol_spike = (df['Volume'].iloc[-1] > df['Volume'].mean() * 2.0)
    
    # 4. Synthèse Gemini
    prompt = f"""
    Analyse Profonde pour {symbol}.
    Social Sentiment: {sentiment:.2f}.
    Volume Spike: {vol_spike}.
    
    Regarde le graphique joint.
    Détermine :
    1. La tendance de fond (HAUSSIERE/BAISSIERE).
    2. Le niveau de risque (FAIBLE/ELEVE).
    
    Réponds JSON : {{"trend": "...", "risk": "...", "score": 0-100}}
    """
    try:
        res = model.generate_content([prompt, img])
        return json.loads(res.text.replace("```json","").replace("```",""))
    except: return {"trend": "NEUTRE", "risk": "MOYEN", "score": 50}

def run_deep_brain():
    global brain, bot_state
    fast_log("🧠 **DEEP MIND:** Démarrage de l'analyse cognitive (Vision + Social).")
    
    while True:
        try:
            s = random.choice(WATCHLIST)
            bot_state['deep_status'] = f"Analyse {s}..."
            
            # Analyse lourde (prend 3-4 secondes)
            insight = get_deep_analysis(s)
            
            # Mise à jour de la "Vision du Monde" du bot
            brain['deep_insight'] = {
                "fav_asset": s if insight['score'] > 80 else brain['deep_insight']['fav_asset'],
                "trend": insight['trend'],
                "risk_level": insight['risk']
            }
            brain['stats']['deep_scans'] += 1
            
            # Si découverte majeure
            if insight['score'] > 85:
                fast_log(f"👁️ **VISION IA ({s}):** Opportunité Majeure détectée !\nScore: {insight['score']}/100 | Tendance: {insight['trend']} | Risque: {insight['risk']}")
                save_brain()
            
            time.sleep(30) # Lent et profond
        except: time.sleep(30)

# ==============================================================================
# MOTEUR DE TRADING (LA FUSION)
# ==============================================================================
def run_trading():
    global brain
    load_brain()
    
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if market_open:
                # SCAN ACHAT
                if len(brain['holdings']) < 5:
                    for s in WATCHLIST:
                        if s in brain['holdings']: continue
                        
                        try:
                            df = yf.Ticker(s).history(period="5d", interval="15m")
                            if df.empty: continue
                            rsi = ta.rsi(df['Close'], length=14).iloc[-1]
                            
                            # 1. FILTRE RAPIDE (Issu des Maths)
                            # On utilise le paramètre optimisé à la milliseconde près par le Fast Core
                            fast_limit = brain['fast_params']['rsi_buy']
                            
                            if rsi < fast_limit:
                                # 2. VERIFICATION PROFONDE (Issue de l'IA)
                                # Si l'IA dit que le risque est ÉLEVÉ sur le marché, on n'achète pas
                                if brain['deep_insight']['risk_level'] == "ELEVE":
                                    fast_log(f"🛡️ **BLOCAGE IA:** Signal Math sur {s}, mais l'IA détecte un risque élevé.")
                                    continue
                                
                                # 3. EXÉCUTION
                                price = df['Close'].iloc[-1]
                                atr = ta.atr(df['High'], df['Low'], df['Close'], length=14).iloc[-1]
                                
                                qty = 500 / price
                                sl = price - (atr * brain['fast_params']['sl_mult'])
                                tp = price + (atr * brain['fast_params']['tp_mult'])
                                
                                brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                                brain['cash'] -= 500
                                
                                msg = {
                                    "embeds": [{
                                        "title": f"🧬 ACHAT DUAL-CORE : {s}",
                                        "description": "Fusion Maths + IA validée.",
                                        "color": 0x2ecc71,
                                        "fields": [
                                            {"name": "Maths (Fast)", "value": f"RSI {rsi:.1f} < {fast_limit}", "inline": True},
                                            {"name": "IA (Deep)", "value": f"Trend: {brain['deep_insight']['trend']}", "inline": True}
                                        ]
                                    }]
                                }
                                send_trade_alert(msg)
                                save_brain()
                        except: pass
                
                # GESTION VENTES (Classique)
                for s in list(brain['holdings'].keys()):
                    pos = brain['holdings'][s]
                    curr = yf.Ticker(s).fast_info['last_price']
                    if curr < pos['stop']:
                        brain['cash'] += pos['qty'] * curr
                        del brain['holdings'][s]
                        send_trade_alert({"content": f"🔴 VENTE {s} (Stop) | PnL: {(curr-pos['entry'])*pos['qty']:.2f}$"})
                        save_brain()
            
            time.sleep(10)
        except: time.sleep(10)

@app.route('/')
def index(): 
    return f"<h1>DUAL CORE V83</h1><p>Fast: {bot_state.get('fast_speed')} | Deep: {brain['deep_insight']['trend']}</p>"

def start_threads():
    threading.Thread(target=run_trading, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()
    threading.Thread(target=logger_worker, daemon=True).start()
    threading.Thread(target=run_fast_brain, daemon=True).start() # Cerveau Rapide
    threading.Thread(target=run_deep_brain, daemon=True).start() # Cerveau Lent

load_brain()
start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
