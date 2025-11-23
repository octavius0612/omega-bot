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
# 1. CLÉS & CONFIGURATION OMEGA
# ==============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL")
LEARNING_WEBHOOK_URL = os.environ.get("LEARNING_WEBHOOK_URL")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR"]
INITIAL_CAPITAL = 50000.0 
SIMULATION_COUNT = 2000

# CERVEAU COMPLET
brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "trade_history": [],
    "total_pnl": 0.0,
    # Réseau Neuronal (Poids Synaptiques)
    "synapses": {
        "w_rsi": -0.5, "w_trend": 0.5, "w_vol": 0.2, "w_social": 0.1, "w_quant": 0.6, "w_vision": 0.4, "w_bias": 0.0
    },
    # Mémoire Émotionnelle
    "emotions": {"confidence": 50.0, "stress": 20.0},
    # Mémoire des traumatismes (Black Box)
    "black_box": []
}

bot_state = {
    "status": "Booting Omega...",
    "last_decision": "Init...",
    "last_log": "Chargement modules..."
}

log_queue = queue.Queue()

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ==============================================================================
# 2. SYSTÈME NERVEUX (LOGGING & COMMS)
# ==============================================================================
def fast_log(text):
    log_queue.put(text)

def logger_worker():
    buffer = []
    last_send = time.time()
    while True:
        try:
            while not log_queue.empty():
                buffer.append(log_queue.get())
            if buffer and (len(buffer) > 5 or time.time() - last_send > 2.0):
                msg_block = "\n".join(buffer[:12])
                buffer = buffer[12:]
                if LEARNING_WEBHOOK_URL:
                    try: requests.post(LEARNING_WEBHOOK_URL, json={"content": msg_block})
                    except: pass
                last_send = time.time()
            time.sleep(0.2)
        except: time.sleep(1)

def send_trade_alert(embed):
    if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed]})

def run_heartbeat():
    while True:
        if HEARTBEAT_WEBHOOK_URL: requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"})
        time.sleep(30)

# ==============================================================================
# 3. HIPPOCAMPE (MÉMOIRE)
# ==============================================================================
def load_brain():
    global brain
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        c = repo.get_contents("brain.json")
        loaded = json.loads(c.decoded_content.decode())
        if "synapses" in loaded: brain.update(loaded)
    except: pass

def save_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        content = json.dumps(brain, indent=4)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Omega Save", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# ==============================================================================
# 4. MOTEUR NEURONAL (PERCEPTRON)
# ==============================================================================
class NeuralNet:
    def activate(self, inputs):
        w = brain['synapses']
        # Calcul du score d'activation (Somme pondérée des sens)
        z = (inputs['rsi_norm'] * w['w_rsi']) + \
            (inputs['trend_norm'] * w['w_trend']) + \
            (inputs['vol_norm'] * w['w_vol']) + \
            (inputs['social'] * w['w_social']) + \
            (inputs['quant'] * w['w_quant']) + \
            (inputs['vision'] * w['w_vision']) + \
            w['w_bias']
        return 1 / (1 + np.exp(-z)) # Sigmoid

    def train(self, inputs, outcome):
        # Apprentissage par renforcement (Backpropagation simple)
        lr = 0.01 # Learning rate
        pred = self.activate(inputs)
        error = outcome - pred
        w = brain['synapses']
        
        w['w_rsi'] += lr * error * inputs['rsi_norm']
        w['w_quant'] += lr * error * inputs['quant']
        w['w_vision'] += lr * error * inputs['vision']
        w['w_bias'] += lr * error
        
        brain['synapses'] = w # Sauvegarde des nouveaux poids

neural_engine = NeuralNet()

# ==============================================================================
# 5. LES SENS (MODULES D'ANALYSE)
# ==============================================================================

# A. QUANTIQUE
def run_monte_carlo(prices):
    try:
        returns = prices.pct_change().dropna()
        sims = prices.iloc[-1] * (1 + np.random.normal(returns.mean(), returns.std(), (SIMULATION_COUNT, 10)))
        return np.sum(sims[:, -1] > prices.iloc[-1]) / SIMULATION_COUNT
    except: return 0.5

# B. VISION
def get_vision_score(df):
    try:
        buf = io.BytesIO()
        mpf.plot(df.tail(50), type='candle', style='nightclouds', savefig=buf)
        buf.seek(0)
        img = Image.open(buf)
        res = model.generate_content(["Analyse technique visuelle. Score d'achat 0.0 à 1.0. Réponds chiffre uniquement.", img])
        return float(res.text.strip())
    except: return 0.5

# C. SOCIAL
def get_social_sentiment(symbol):
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).json()
        txt = " ".join([m['body'] for m in r['messages'][:10]])
        return TextBlob(txt).sentiment.polarity
    except: return 0

# D. BALEINE
def check_whale(df):
    vol = df['Volume'].iloc[-1]
    avg = df['Volume'].rolling(20).mean().iloc[-1]
    return (vol > avg * 2.5), f"x{vol/avg:.1f}" if avg > 0 else "N/A"

# ==============================================================================
# 6. GHOST BROKER & GESTION
# ==============================================================================
class GhostBroker:
    def get_price(self, symbol):
        try: return yf.Ticker(symbol).fast_info['last_price']
        except: return None
    def get_portfolio(self):
        eq = brain['cash']
        pos = []
        for s, d in brain['holdings'].items():
            p = self.get_price(s)
            if p:
                val = d['qty'] * p
                eq += val
                pnl = val - (d['qty'] * d['entry'])
                pos.append({"symbol": s, "qty": d['qty'], "entry": d['entry'], "current": p, "pnl": round(pnl, 2)})
        return eq, pos

broker = GhostBroker()

def update_emotions(pnl):
    e = brain['emotions']
    if pnl > 0: e['confidence'] = min(e['confidence']+5, 100); e['stress'] = max(e['stress']-5, 0)
    else: e['confidence'] = max(e['confidence']-10, 10); e['stress'] = min(e['stress']+10, 100)

# ==============================================================================
# 7. APPRENTISSAGE CONTINU (LE RÊVE)
# ==============================================================================
def run_dream_learning():
    global brain
    cache = {}
    fast_log("🧠 **RÊVE:** Démarrage de l'entraînement neuronal en fond.")
    
    while True:
        try:
            # On s'entraîne quand le marché est fermé
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if market_open:
                time.sleep(300)
                continue
                
            s = random.choice(WATCHLIST)
            if s not in cache or random.random() < 0.1:
                try:
                    df = yf.Ticker(s).history(period="1mo", interval="1h")
                    if not df.empty:
                        df['RSI'] = ta.rsi(df['Close'], length=14)
                        df['SMA50'] = ta.sma(df['Close'], length=50)
                        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                        cache[s] = df.dropna()
                except: pass

            if s in cache:
                df = cache[s]
                idx = random.randint(50, len(df)-10)
                row = df.iloc[idx]
                
                # Simulation Inputs
                inputs = {
                    "rsi_norm": row['RSI']/100,
                    "trend_norm": 1.0 if row['Close'] > row['SMA50'] else -1.0,
                    "vol_norm": min(row['ATR']/row['Close']*100, 1.0),
                    "social": 0.0, # Simulé
                    "quant": 0.5, # Simulé pour vitesse
                    "vision": 0.5 # Simulé
                }
                
                # Futur
                future = df.iloc[idx+5]['Close']
                target = 1.0 if future > row['Close'] * 1.01 else 0.0
                
                # Entraînement Neurone
                error = neural_engine.train(inputs, target)
                
                if abs(error) < 0.1: # Bonne prédiction
                    fast_log(f"🎓 **APPRENTISSAGE:** Neurone affiné sur {s} (Erreur: {error:.2f})")
                    save_brain()
            
            time.sleep(2)
        except: time.sleep(10)

# ==============================================================================
# 8. MOTEUR TRADING LIVE
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
                bot_state['status'] = "🟢 TRADING NEURONAL"
                
                # VENTES
                for s in list(brain['holdings'].keys()):
                    pos = brain['holdings'][s]
                    curr = broker.get_price(s)
                    if not curr: continue
                    
                    exit_r = None
                    if curr < pos['stop']: exit_r = "STOP LOSS"
                    elif curr > pos['tp']: exit_r = "TAKE PROFIT"
                    
                    if exit_r:
                        pnl = (curr - pos['entry']) * pos['qty']
                        brain['cash'] += pos['qty'] * curr
                        brain['total_pnl'] += pnl
                        del brain['holdings'][s]
                        update_emotions(pnl)
                        
                        col = 0x2ecc71 if pnl > 0 else 0xe74c3c
                        send_trade_alert({"embeds": [{"title": f"{exit_r} {s}", "description": f"PnL: **{pnl:.2f}$**", "color": col}]})
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
                            df['SMA50'] = ta.sma(df['Close'], length=50)
                            row = df.iloc[-1]
                            
                            # COLLECTE DES SENS (LOURD)
                            # On filtre d'abord par RSI pour économiser les ressources
                            if row['RSI'] < 45:
                                fast_log(f"🔎 **SCAN {s}:** Signal détecté. Activation des sens...")
                                
                                # 1. Quantique
                                mc_prob = run_monte_carlo(df['Close'])
                                # 2. Vision
                                vis_score = get_vision_score(df)
                                # 3. Social
                                soc_score = get_social_sentiment(s)
                                # 4. Baleine
                                is_whale, wh_msg = check_whale(df)
                                
                                # INPUTS NEURONAUX
                                inputs = {
                                    "rsi_norm": row['RSI']/100,
                                    "trend_norm": 1.0 if row['Close'] > row['SMA50'] else -1.0,
                                    "vol_norm": min(row['ATR']/row['Close']*100, 1.0),
                                    "social": soc_score,
                                    "quant": mc_prob,
                                    "vision": vis_score
                                }
                                
                                # ACTIVATION DU NEURONE
                                activation = neural_engine.activate(inputs)
                                bot_state['last_decision'] = f"{s}: {activation*100:.1f}%"
                                
                                fast_log(f"🧠 **{s}:** Neurone: {activation:.2f} | Vis:{vis_score:.2f} | MC:{mc_prob:.2f}")
                                
                                # DÉCISION FINALE
                                if activation > 0.85:
                                    price = row['Close']
                                    # Kelly + Psycho
                                    bet_factor = 1.2 if brain['emotions']['confidence'] > 80 else 0.8
                                    qty = (500 * bet_factor) / price
                                    
                                    sl = price - (row['ATR'] * 2.0)
                                    tp = price + (row['ATR'] * 3.5)
                                    
                                    brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                                    brain['cash'] -= (500 * bet_factor)
                                    
                                    msg = {
                                        "title": f"🌌 ACHAT OMEGA : {s}",
                                        "description": f"**Activation Neuronale: {activation:.4f}**",
                                        "color": 0x2ecc71,
                                        "fields": [
                                            {"name": "Vision", "value": f"{vis_score:.2f}", "inline": True},
                                            {"name": "Quantique", "value": f"{mc_prob:.2f}", "inline": True},
                                            {"name": "Confiance", "value": f"{brain['emotions']['confidence']}%", "inline": True}
                                        ]
                                    }
                                    send_trade_alert(msg)
                                    save_brain()
                        except: pass
            else:
                bot_state['status'] = "🌙 NUIT"
            
            time.sleep(30)
        except: time.sleep(10)

# --- 9. DASHBOARD WEB ---
HTML = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta http-equiv="refresh" content="10">
    <title>OMEGA V92</title>
    <style>
        body { background: #000; color: #0f0; font-family: monospace; padding: 20px; }
        .card { border: 1px solid #333; padding: 15px; margin-bottom: 20px; }
        h1 { margin: 0; color: #fff; }
        table { width: 100%; border-collapse: collapse; color: #aaa; }
        td { border-bottom: 1px solid #222; padding: 5px; }
    </style>
</head>
<body>
    <h1>👁️ OMEGA BRAIN V92</h1>
    <div class="card">
        <h3>STATUS: {{ status }}</h3>
        <h2>CAPITAL: ${{ equity }}</h2>
        <p>Log: {{ last_log }}</p>
    </div>
    <div class="card">
        <h3>POIDS SYNAPTIQUES (Cerveau)</h3>
        <ul>
        {% for k, v in synapses.items() %}
            <li>{{ k }}: {{ v|round(4) }}</li>
        {% endfor %}
        </ul>
    </div>
    <div class="card">
        <table>
            <tr><th>ACTIF</th><th>QTE</th><th>PRIX MOY</th><th>VALEUR</th><th>PNL</th></tr>
            {% for p in positions %}
            <tr>
                <td>{{ p.symbol }}</td>
                <td>{{ p.qty|round(2) }}</td>
                <td>{{ p.entry }}</td>
                <td>{{ (p.qty * p.current)|round(2) }}</td>
                <td style="color: {{ 'red' if p.pnl < 0 else '#0f0' }}">{{ p.pnl }}$</td>
            </tr>
            {% endfor %}
        </table>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    eq, pos = broker.get_portfolio()
    return render_template_string(HTML, 
        equity=f"{eq:,.2f}", 
        status=bot_state['status'], 
        positions=pos,
        last_log=bot_state['last_log'],
        synapses=brain['synapses']
    )

def start_threads():
    threading.Thread(target=run_trading, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()
    threading.Thread(target=logger_worker, daemon=True).start()
    threading.Thread(target=run_dream_learning, daemon=True).start()

load_brain()
start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
