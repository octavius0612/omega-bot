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
matplotlib.use('Agg') # Important pour le serveur
import mplfinance as mpf
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
from datetime import datetime, time as dtime
import pytz
from github import Github

app = Flask(__name__)

# ==============================================================================
# 1. CLÉS & CONFIGURATION
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
    "genome": {"rsi_buy": 30, "sl_mult": 2.0, "tp_mult": 3.0},
    "stats": {"generation": 0, "best_pnl": 0.0},
    "emotions": {"confidence": 50.0, "stress": 20.0},
    "karma": {s: 10.0 for s in WATCHLIST},
    "total_pnl": 0.0,
    "colab_signal": None
}

bot_state = {
    "status": "Démarrage V131...",
    "last_log": "Init...",
    "web_logs": []
}

log_queue = queue.Queue()

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ==============================================================================
# 2. LOGGING & COMMS (CORRIGÉ)
# ==============================================================================
def fast_log(text):
    log_queue.put(text)
    try:
        t = datetime.now().strftime('%H:%M:%S')
        bot_state['web_logs'].insert(0, f"[{t}] {text}")
        if len(bot_state['web_logs']) > 50:
            bot_state['web_logs'] = bot_state['web_logs'][:50]
    except: pass

def logger_worker():
    buffer = []
    last_send = time.time()
    while True:
        try:
            while not log_queue.empty():
                buffer.append(log_queue.get())
            
            if buffer and (len(buffer) > 3 or time.time() - last_send > 2.0):
                msg_block = "\n".join(buffer[:10])
                buffer = buffer[10:]
                if LEARNING_WEBHOOK_URL:
                    try:
                        requests.post(LEARNING_WEBHOOK_URL, json={"content": msg_block}, timeout=5)
                    except: pass
                last_send = time.time()
            time.sleep(0.5)
        except: time.sleep(1)

def send_alert(url, embed):
    # CORRECTION SYNTAXE ICI
    if url:
        try:
            requests.post(url, json={"embeds": [embed]}, timeout=5)
        except:
            pass

def run_heartbeat():
    while True:
        if HEARTBEAT_WEBHOOK_URL:
            try:
                requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"}, timeout=5)
            except: pass
        time.sleep(30)

# ==============================================================================
# 3. MÉMOIRE & COLAB RECEPTION
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
        content = json.dumps(brain, indent=4, default=str)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Save V131", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- ENDPOINT POUR LE COLAB ---
@app.route('/receive_signal', methods=['POST'])
def receive_signal():
    data = request.json
    s = data.get('symbol')
    score = data.get('score')
    reason = data.get('reason')
    
    brain['colab_signal'] = data
    fast_log(f"🧠 **SIGNAL COLAB REÇU:** {s} (Score {score:.2f})")
    
    # Si le signal est fort, on déclenche un achat immédiat sur le Paper
    if score > 0.8:
        execute_paper_buy(s, 120.0, reason) # Prix simulé pour l'exemple rapide
        
    return jsonify({"status": "OK"}), 200

# ==============================================================================
# 4. MODULES D'INTELLIGENCE (LOCAL)
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

def consult_council(s, rsi, mc, vis, whale):
    prompt = f"""
    DECISION {s}.
    1. Maths: {mc*100:.1f}%.
    2. Vision: {vis:.2f}.
    3. Baleine: {whale}.
    4. RSI: {rsi:.1f}.
    
    Si Maths > 0.6 ET Vision > 0.6 : BUY.
    JSON: {{"vote": "BUY/WAIT", "reason": "Synthèse"}}
    """
    try:
        res = model.generate_content(prompt)
        return json.loads(res.text.replace("```json","").replace("```",""))
    except: return {"vote": "WAIT", "reason": "Erreur"}

# ==============================================================================
# 5. EXÉCUTION TRADES
# ==============================================================================
def execute_paper_buy(s, price, reason):
    if len(brain['paper_holdings']) < 5:
        qty = 200 / price
        sl = price * 0.98
        tp = price * 1.04
        
        brain['paper_holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
        brain['paper_cash'] -= 200
        
        send_alert(PAPER_WEBHOOK_URL, {
            "title": f"🎮 PAPER BUY : {s}",
            "description": reason,
            "color": 0x3498db
        })
        save_brain()

class GhostBroker:
    def get_price(self, symbol):
        try: return yf.Ticker(symbol).fast_info['last_price']
        except: return None
    def get_portfolio(self):
        return brain['cash'], [{"symbol": s, "qty": d['qty'], "pnl": 0} for s, d in brain['holdings'].items()]
broker = GhostBroker()

# ==============================================================================
# 6. MOTEUR PRINCIPAL
# ==============================================================================
def run_trading_engine():
    global brain, bot_state
    load_brain()
    
    fast_log("🌌 **OMNIVERSE V131:** Systèmes nominaux. Prêt.")
    
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if market_open:
                bot_state['status'] = "🟢 LIVE"
                
                # 1. VENTES
                for s in list(brain['holdings'].keys()):
                    pos = brain['holdings'][s]
                    curr = broker.get_price(s)
                    if not curr: continue
                    
                    if curr < pos['stop']:
                        brain['cash'] += pos['qty'] * curr
                        del brain['holdings'][s]
                        send_alert(DISCORD_WEBHOOK_URL, {"title": f"🔴 VENTE {s} (Stop)", "color": 0xe74c3c})
                        save_brain()
                    elif curr > pos['tp']:
                        brain['cash'] += pos['qty'] * curr
                        del brain['holdings'][s]
                        send_alert(DISCORD_WEBHOOK_URL, {"title": f"🟢 VENTE {s} (Target)", "color": 0x2ecc71})
                        save_brain()

                # 2. ACHATS
                if len(brain['holdings']) < 5:
                    for s in WATCHLIST:
                        if s in brain['holdings']: continue
                        try:
                            df = yf.Ticker(s).history(period="1mo", interval="15m")
                            if df.empty: continue
                            rsi = ta.rsi(df['Close'], 14).iloc[-1]
                            
                            if rsi < 35:
                                fast_log(f"🔎 Scan {s} (RSI {rsi:.1f})...")
                                mc = run_monte_carlo(df['Close'])
                                vis = get_vision_score(df)
                                whl, _ = check_whale(df)
                                
                                council = consult_council(s, rsi, mc, vis, whl)
                                
                                if council['vote'] == "BUY":
                                    price = df['Close'].iloc[-1]
                                    qty = 1000 / price
                                    sl = price * 0.98
                                    tp = price * 1.04
                                    
                                    brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                                    brain['cash'] -= 1000
                                    
                                    send_alert(DISCORD_WEBHOOK_URL, {
                                        "title": f"🌌 ACHAT : {s}",
                                        "description": council['reason'],
                                        "color": 0x2ecc71
                                    })
                                    save_brain()
                        except: pass
            else:
                bot_state['status'] = "🌙 NUIT (Replay)"
                # Simulation Replay pour Paper
                if random.random() < 0.1:
                    s = random.choice(WATCHLIST)
                    brain['paper_cash'] += random.uniform(-10, 30)
                    send_alert(PAPER_WEBHOOK_URL, {"title": f"🎬 REPLAY : {s}", "color": 0x3498db})
            
            time.sleep(30)
        except Exception as e:
            print(f"Err Loop: {e}")
            time.sleep(10)

# --- DASHBOARD ---
@app.route('/')
def index():
    eq, pos = broker.get_portfolio()
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="fr">
    <head><meta http-equiv="refresh" content="5"><style>body{background:#000;color:#0f0;font-family:monospace;padding:20px}</style></head>
    <body>
        <h1>👁️ OMNIVERSE V131</h1>
        <p>Status: {{ status }}</p>
        <p>Main Capital: ${{ eq }}</p>
        <p>Paper Capital: ${{ paper }}</p>
        <hr>
        <div style="height:400px;overflow:auto;background:#111;padding:10px;font-size:12px">
            {% for l in logs %}<div>{{ l }}</div>{% endfor %}
        </div>
    </body>
    </html>
    """, status=bot_state['status'], eq=f"{eq:,.2f}", paper=f"{brain['paper_cash']:,.2f}", logs=bot_state['web_logs'])

def start_threads():
    threading.Thread(target=run_trading, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()
    threading.Thread(target=logger_worker, daemon=True).start()

load_brain()
start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
