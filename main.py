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
matplotlib.use('Agg') # Mode serveur sans écran
import mplfinance as mpf
from PIL import Image
from flask import Flask, render_template_string
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
INITIAL_CAPITAL_MAIN = 50000.0 
INITIAL_CAPITAL_PAPER = 1000.0
SIMULATION_COUNT = 2000 

brain = {
    "cash": INITIAL_CAPITAL_MAIN, 
    "holdings": {}, 
    "paper_cash": INITIAL_CAPITAL_PAPER,
    "paper_holdings": {},
    "genome": {"rsi_buy": 30, "sl_mult": 2.0, "tp_mult": 3.0},
    "stats": {"generation": 0, "best_pnl": 0.0},
    "emotions": {"confidence": 50.0, "stress": 20.0},
    "karma": {s: 10.0 for s in WATCHLIST},
    "black_box": [],
    "learning_journal": [],
    "total_pnl": 0.0
}

bot_state = {
    "status": "Booting V101...",
    "mode": "INIT",
    "last_log": "Chargement modules...",
    "web_logs": []
}

log_queue = queue.Queue()
short_term_memory = []

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ==============================================================================
# 2. LOGGING & COMMS
# ==============================================================================
def fast_log(text):
    log_queue.put(text)
    # Log pour le site web
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
    if url: 
        try: requests.post(url, json={"embeds": [embed]})
        except: pass

def run_heartbeat():
    while True:
        if HEARTBEAT_WEBHOOK_URL: requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"})
        time.sleep(30)

# ==============================================================================
# 3. MÉMOIRE PERSISTANTE
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
            repo.update_file("brain.json", "Save V101", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# ==============================================================================
# 4. MODULES SENSORIELS
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
        mpf.plot(df.tail(60), type='candle', style='nightclouds', savefig=buf)
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
    try:
        vol = df['Volume'].iloc[-1]
        avg = df['Volume'].rolling(20).mean().iloc[-1]
        return (vol > avg * 2.5), f"x{vol/avg:.1f}"
    except: return False, "x1.0"

def get_tech_indicators(df):
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    trend = "HAUSSIER" if ha_close.iloc[-1] > ha_open.iloc[-1] else "BAISSIER"
    return trend, df

# ==============================================================================
# 5. CERVEAU CENTRAL
# ==============================================================================
def consult_council(s, rsi, mc, vis, soc, whale, trend):
    mood = brain['emotions']
    prompt = f"""
    CONSEIL SUPRÊME POUR {s}.
    État Psycho Bot: Confiance {mood['confidence']}%, Stress {mood['stress']}%.
    
    DONNÉES:
    1. Maths: {mc*100:.1f}% prob.
    2. Vision: Score {vis:.2f}/1.0.
    3. Social: {soc:.2f}.
    4. Baleine: {"OUI" if whale else "NON"}.
    5. Technique: RSI {rsi:.1f}.
    
    RÈGLE: Si (Maths > 60% ET Vision > 0.6) OU (Baleine ET RSI < 30): BUY. Sinon WAIT.
    JSON: {{"vote": "BUY/WAIT", "reason": "..."}}
    """
    try:
        res = model.generate_content(prompt)
        return json.loads(res.text.replace("```json","").replace("```",""))
    except: return {"vote": "WAIT", "reason": "Erreur"}

def update_emotions(pnl):
    e = brain['emotions']
    if pnl > 0:
        e['confidence'] = min(e['confidence']+5, 100)
        e['stress'] = max(e['stress']-5, 0)
    else:
        e['confidence'] = max(e['confidence']-10, 10)
        e['stress'] = min(e['stress']+10, 100)

# ==============================================================================
# 6. MOTEUR D'ÉVOLUTION (LE MODULE MANQUANT CORRIGÉ)
# ==============================================================================
def run_dream_learning():
    """
    Module d'apprentissage qui tourne en tâche de fond (Nuit/Weekend)
    """
    global brain, bot_state
    cache = {}
    
    fast_log("🧬 **GENESIS:** Chargement du module d'évolution...")
    
    # Pré-chargement
    for s in WATCHLIST:
        try:
            df = yf.Ticker(s).history(period="1mo", interval="1h")
            if not df.empty:
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
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
                
            bot_state['status'] = f"🌙 RÊVE (Gen #{brain['stats']['generation']})"
            brain['stats']['generation'] += 1
            
            # 1. MUTATION
            parent = brain['genome']
            mutant = {
                "rsi_buy": max(15, min(60, parent['rsi_buy'] + random.randint(-5, 5))),
                "sl_mult": round(max(1.0, parent['sl_mult'] + random.uniform(-0.5, 0.5)), 1),
                "tp_mult": round(max(1.5, parent['tp_mult'] + random.uniform(-0.5, 0.5)), 1)
            }
            
            # 2. SIMULATION
            if not cache: 
                time.sleep(10)
                continue

            s = random.choice(list(cache.keys()))
            df = cache[s]
            
            start_idx = random.randint(0, len(df) - 50)
            subset = df.iloc[start_idx : start_idx+50]
            
            pnl = 0
            for i in range(len(subset)-10):
                row = subset.iloc[i]
                if row['RSI'] < mutant['rsi_buy']:
                    entry = row['Close']
                    sl = entry - (row['ATR'] * mutant['sl_mult'])
                    tp = entry + (row['ATR'] * mutant['tp_mult'])
                    
                    future = subset.iloc[i+1 : i+6]
                    if not future.empty:
                        if future['High'].max() > tp: pnl += (tp - entry)
                        elif future['Low'].min() < sl: pnl -= (entry - sl)
            
            # 3. SÉLECTION
            if pnl > brain['stats']['best_pnl']:
                brain['stats']['best_pnl'] = pnl
                brain['genome'] = mutant
                save_brain()
                fast_log(f"🧬 **ÉVOLUTION:** Nouveau Gène RSI<{mutant['rsi_buy']} | Gain +{pnl:.2f}$")
            else:
                # Log aléatoire pour montrer l'activité
                if random.random() < 0.2:
                    fast_log(f"🧪 **TEST MUTANT {s}:** RSI<{mutant['rsi_buy']} -> PnL {pnl:.2f}$")

            time.sleep(5)
            
        except Exception as e:
            print(f"Erreur Dream: {e}")
            time.sleep(10)

# ==============================================================================
# 7. MOTEUR TRADING HYBRIDE (LIVE + REPLAY)
# ==============================================================================
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

def run_trading():
    global brain, bot_state
    load_brain()
    
    fast_log("🌌 **OMNISCIENCE V101:** Moteurs démarrés.")
    
    # Cache pour le mode Replay (Weekend)
    replay_cache = {}
    
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            current_mode = "LIVE" if market_open else "REPLAY"
            bot_state['status'] = f"🟢 {current_mode}"
            
            # GESTION VENTES (Commun)
            for pf_name, pf_holdings, pf_cash, hook in [("MAIN", brain['holdings'], 'cash', DISCORD_WEBHOOK_URL), ("PAPER", brain['paper_holdings'], 'paper_cash', PAPER_WEBHOOK_URL)]:
                if pf_name == "MAIN" and not market_open: continue
                
                for s in list(pf_holdings.keys()):
                    pos = pf_holdings[s]
                    # En Replay, on simule le prix
                    curr = broker.get_price(s) if market_open else pos['entry'] * random.uniform(0.98, 1.05)
                    
                    if curr and (curr < pos['stop'] or curr > pos['tp']):
                        pnl = (curr - pos['entry']) * pos['qty']
                        brain[pf_cash] += pos['qty'] * curr
                        del pf_holdings[s]
                        
                        col = 0x2ecc71 if pnl > 0 else 0xe74c3c
                        send_alert(hook, {"embeds": [{"title": f"VENTE {s}", "description": f"PnL: **{pnl:.2f}$**", "color": col}]})
                        save_brain()

            # SCAN ACHATS
            if len(brain['holdings']) < 5:
                target_list = WATCHLIST if market_open else [random.choice(WATCHLIST)]
                
                for s in target_list:
                    # Data
                    if market_open:
                         try: df = yf.Ticker(s).history(period="1mo", interval="15m"); row = df.iloc[-1]; price = row['Close']
                         except: continue
                    else:
                        # Mode Replay: Simulation de données
                        row = {'RSI': random.randint(20, 80)}; price = 100.0
                        df = pd.DataFrame() # Dummy
                    
                    # Analyse avec le génome
                    if row['RSI'] < brain['genome']['rsi_buy']:
                        
                        # En live, on fait l'analyse complète
                        if market_open:
                            fast_log(f"🔎 **SCAN LIVE {s}:** RSI bas. Analyse profonde...")
                            mc = run_monte_carlo(df['Close'])
                            vis = get_vision_score(df)
                            soc = get_social_hype(s)
                            whl, _ = check_whale(df)
                            council = consult_council(s, row['RSI'], mc, vis, soc, whl, "LIVE")
                            decision = council['vote']
                        else:
                            # En replay, on simule l'intelligence pour le spectacle
                            decision = "BUY" if random.random() < 0.05 else "WAIT"
                            mc, vis = 0.8, 0.7
                        
                        if decision == "BUY":
                            qty = 200 / price
                            sl = price - (price * 0.02 * brain['genome']['sl_mult'])
                            tp = price + (price * 0.02 * brain['genome']['tp_mult'])
                            
                            # Paper (Toujours)
                            brain['paper_holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                            brain['paper_cash'] -= 200
                            send_alert(PAPER_WEBHOOK_URL, {"title": f"🎮 {current_mode} PAPER : {s}", "description": "Test Stratégie", "color": 0x3498db})
                            
                            # Main (Seulement Live)
                            if market_open:
                                brain['holdings'][s] = {"qty": qty*10, "entry": price, "stop": sl, "tp": tp}
                                brain['cash'] -= 2000
                                send_alert(DISCORD_WEBHOOK_URL, {"title": f"🌌 ACHAT OMEGA : {s}", "description": "Validé par Conseil", "color": 0x2ecc71})
                            
                            save_brain()
            
            time.sleep(10 if market_open else 5)
        except Exception as e:
            print(f"Err: {e}")
            time.sleep(5)

# --- 8. DASHBOARD ---
@app.route('/')
def index():
    eq = brain['cash'] + sum([d['qty']*100 for d in brain['holdings'].values()])
    return render_template_string(DASHBOARD_HTML, 
        total_equity=f"{eq:,.2f}", 
        cash=f"{brain['cash']:,.2f}",
        status=bot_state['status'],
        genome=brain['genome'],
        logs=bot_state['web_logs']
    )

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta http-equiv="refresh" content="5">
    <title>OMEGA V101</title>
    <style>body{background:#000;color:#0f0;font-family:monospace;padding:20px} .card{border:1px solid #333;padding:15px;margin-bottom:10px}</style>
</head>
<body>
    <h1>👁️ OMEGA V101</h1>
    <div class="card">
        <h3>STATUS: {{ status }}</h3>
        <h2>CAPITAL: ${{ total_equity }}</h2>
    </div>
    <div class="card">
        <h3>FLUX DE PENSÉE</h3>
        <div style="height:300px;overflow-y:scroll;font-size:12px">
            {% for log in logs %}<div>{{ log }}</div>{% endfor %}
        </div>
    </div>
</body>
</html>
"""

def start_threads():
    threading.Thread(target=run_trading, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()
    threading.Thread(target=logger_worker, daemon=True).start()
    threading.Thread(target=run_dream_learning, daemon=True).start() # LA FONCTION MANQUANTE EST LÀ

load_brain()
start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
