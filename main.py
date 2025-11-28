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

# --- CLÉS ---
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
INITIAL_CAPITAL_MAIN = 50000.0 
INITIAL_CAPITAL_PAPER = 1000.0
SIMULATION_COUNT = 2000 

brain = {
    "cash": INITIAL_CAPITAL_MAIN, 
    "holdings": {}, 
    "paper_cash": INITIAL_CAPITAL_PAPER,
    "paper_holdings": {},
    # Ces paramètres sont mis à jour par le COLAB via GitHub
    "genome": {"rsi_buy": 30, "sl_mult": 2.0, "tp_mult": 3.0},
    "stats": {"generation": 0, "best_pnl": 0.0},
    "emotions": {"confidence": 50.0, "stress": 20.0},
    "karma": {s: 10.0 for s in WATCHLIST},
    "last_colab_update": "En attente..."
}

bot_state = {"status": "Booting V130...", "last_log": "Init...", "web_logs": []}
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

# --- SYNCHRO GITHUB (RECEPTION DU CERVEAU COLAB) ---
def load_brain():
    global brain
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        c = repo.get_contents("brain.json")
        loaded = json.loads(c.decoded_content.decode())
        
        # On met à jour le cash localement
        if "cash" in loaded: brain["cash"] = loaded["cash"]
        
        # On récupère l'intelligence du Colab
        if "genome" in loaded: 
            if loaded["genome"] != brain["genome"]:
                fast_log(f"🧬 **UPDATE COLAB:** Nouveaux paramètres reçus (RSI<{loaded['genome']['rsi_buy']})")
            brain["genome"] = loaded["genome"]
            brain["last_colab_update"] = datetime.now().strftime("%H:%M")
            
    except: pass

def save_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        
        # On lit d'abord pour ne pas écraser le travail du Colab
        try:
            c = repo.get_contents("brain.json")
            current_remote = json.loads(c.decoded_content.decode())
        except: current_remote = {}
        
        # On met à jour SEULEMENT nos données financières
        current_remote["cash"] = brain["cash"]
        current_remote["holdings"] = brain["holdings"]
        current_remote["paper_cash"] = brain["paper_cash"]
        
        content = json.dumps(current_remote, indent=4)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Render Update", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- MODULES SENSORIELS (RENDER) ---
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

def get_social_hype(symbol):
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).json()
        txt = " ".join([m['body'] for m in r['messages'][:10]])
        return TextBlob(txt).sentiment.polarity
    except: return 0

def consult_council(s, rsi, mc, vis, soc, whale):
    prompt = f"""
    DECISION {s}.
    1. Maths: {mc*100:.1f}% prob.
    2. Vision: {vis:.2f}.
    3. Social: {soc:.2f}.
    4. Baleine: {whale}.
    5. RSI: {rsi:.1f}.
    
    RÈGLE: (Maths > 0.6 ET Vision > 0.6) OU (Baleine ET RSI < 30) => BUY.
    JSON: {{"vote": "BUY/WAIT", "reason": "..."}}
    """
    try:
        res = model.generate_content(prompt)
        return json.loads(res.text.replace("```json","").replace("```",""))
    except: return {"vote": "WAIT", "reason": "Erreur"}

# --- MOTEUR TRADING ---
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
    
    fast_log("🌌 **OMNIVERSE V130:** Systèmes connectés. En attente du Colab.")
    
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            current_mode = "LIVE" if market_open else "REPLAY"
            bot_state['status'] = f"🟢 {current_mode}"
            
            # Resynchro mémoire fréquente
            if now.second < 5: load_brain()

            target_list = WATCHLIST if market_open else [random.choice(WATCHLIST)]
            
            for s in target_list:
                # VENTES (MAIN + PAPER)
                for pf_name, pf_holdings, pf_cash, hook in [("MAIN", brain['holdings'], 'cash', DISCORD_WEBHOOK_URL), ("PAPER", brain['paper_holdings'], 'paper_cash', PAPER_WEBHOOK_URL)]:
                    if pf_name == "MAIN" and not market_open: continue
                    
                    for sym in list(pf_holdings.keys()):
                        pos = pf_holdings[sym]
                        curr = broker.get_price(sym) if market_open else pos['entry'] * random.uniform(0.98, 1.03)
                        if not curr: continue
                        
                        exit_r = None
                        if curr < pos['stop']: exit_r = "STOP LOSS"
                        elif curr > pos['tp']: exit_r = "TAKE PROFIT"
                        
                        if exit_r:
                            pnl = (curr - pos['entry']) * pos['qty']
                            brain[pf_cash] += pos['qty'] * curr
                            del pf_holdings[sym]
                            col = 0x2ecc71 if pnl > 0 else 0xe74c3c
                            send_alert(hook, {"embeds": [{"title": f"{exit_r} : {sym}", "description": f"PnL: **{pnl:.2f}$**", "color": col}]})
                            save_brain()

                # ACHATS
                # On utilise les paramètres appris par le COLAB
                # Et on vérifie avec les sens du RENDER
                if len(brain['holdings']) < 5:
                    try:
                        df = None
                        if market_open:
                            df = yf.Ticker(s).history(period="1mo", interval="15m")
                        else:
                            df = yf.Ticker(s).history(period="1mo", interval="1h") # Replay data
                        
                        if df is not None and not df.empty:
                            df['RSI'] = ta.rsi(df['Close'], 14)
                            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)
                            row = df.iloc[-1] if market_open else df.iloc[random.randint(50, len(df)-10)]
                            
                            # LE FILTRE COLAB
                            if row['RSI'] < brain['genome']['rsi_buy']:
                                fast_log(f"🔎 **SCAN {s}:** RSI {row['RSI']:.1f} (Validé par Colab).")
                                
                                # ANALYSE RENDER
                                mc = run_monte_carlo(df['Close'])
                                if mc > 0.60:
                                    vis = get_vision_score(df) if market_open else 0.8
                                    soc = get_social_hype(s) if market_open else 0.0
                                    whl, wh_msg = check_whale(df)
                                    
                                    council = consult_council(s, row['RSI'], mc, vis, soc, whl)
                                    
                                    if council['vote'] == "BUY":
                                        price = row['Close']
                                        # Params Colab
                                        sl = price - (row['ATR'] * brain['genome']['sl_mult'])
                                        tp = price + (row['ATR'] * brain['genome']['tp_mult'])
                                        
                                        # Paper
                                        qty_p = 200 / price
                                        brain['paper_holdings'][s] = {"qty": qty_p, "entry": price, "stop": sl, "tp": tp}
                                        brain['paper_cash'] -= 200
                                        send_alert(PAPER_WEBHOOK_URL, {"title": f"🎮 {current_mode} PAPER : {s}", "description": council['reason'], "color": 0x3498db})
                                        
                                        # Main
                                        if market_open:
                                            qty = 1000 / price
                                            brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                                            brain['cash'] -= 1000
                                            send_alert(DISCORD_WEBHOOK_URL, {"title": f"🌌 ACHAT OMEGA : {s}", "description": council['reason'], "color": 0x2ecc71})
                                            
                                        save_brain()
                    except: pass
            
            time.sleep(10 if market_open else 5)
        except: time.sleep(10)

# --- DASHBOARD ---
@app.route('/')
def index():
    eq, pos = broker.get_portfolio()
    return render_template_string("""
    <html><body style='background:#000;color:#0f0;font-family:monospace'>
    <h1>OMNIVERSE V130</h1>
    <p>Status: {{s}}</p><p>Capital: ${{e}}</p>
    <p>Génome Colab: {{g}}</p>
    <div style='height:300px;overflow:auto;background:#111'>{% for l in logs %}<div>{{l}}</div>{% endfor %}</div>
    </body></html>
    """, s=bot_state['status'], e=f"{eq:,.2f}", g=brain['genome'], logs=bot_state['web_logs'])

def start_threads():
    threading.Thread(target=run_trading, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()
    threading.Thread(target=logger_worker, daemon=True).start()

load_brain()
start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
