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
# 1. CLÉS & CONFIGURATION
# ==============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL")
LEARNING_WEBHOOK_URL = os.environ.get("LEARNING_WEBHOOK_URL")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR", "GOOG", "META"]
INITIAL_CAPITAL = 50000.0 
SIMULATION_COUNT = 2000 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    # ADN Évolutif
    "genome": {"rsi_buy": 30, "sl_mult": 2.0, "tp_mult": 3.5},
    # Mémoire
    "stats": {"generation": 0, "best_pnl": 0.0},
    "emotions": {"confidence": 50.0, "stress": 20.0, "euphoria": 0.0},
    "karma": {s: 10.0 for s in WATCHLIST},
    "black_box": [],
    "total_pnl": 0.0
}

bot_state = {
    "status": "Booting Omega...",
    "last_decision": "Init...",
    "last_log": "Chargement modules...",
    "web_logs": []
}

log_queue = queue.Queue()
short_term_memory = []

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ==============================================================================
# 2. DASHBOARD WEB (INTERFACE VISUELLE)
# ==============================================================================
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="5">
    <title>OMEGA SINGULARITY</title>
    <style>
        :root { --bg: #050505; --card: #0f1115; --text: #00ff41; --red: #ff0033; --accent: #00ccff; }
        body { background: var(--bg); color: var(--text); font-family: 'Consolas', monospace; padding: 20px; margin: 0; }
        .container { max_width: 1400px; margin: 0 auto; }
        .header { border-bottom: 1px solid #333; padding-bottom: 10px; margin-bottom: 20px; display: flex; justify-content: space-between; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }
        .card { background: var(--card); border: 1px solid #333; padding: 15px; border-radius: 4px; }
        h2 { margin: 0 0 10px 0; font-size: 16px; color: #fff; opacity: 0.7; }
        .big-val { font-size: 28px; font-weight: bold; color: #fff; }
        .log-box { height: 300px; overflow-y: auto; font-size: 12px; border-top: 1px solid #333; margin-top: 10px; padding-top: 5px; opacity: 0.8; }
        .log-line { margin-bottom: 4px; border-bottom: 1px solid #111; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th { text-align: left; color: #666; border-bottom: 1px solid #333; }
        td { padding: 5px 0; border-bottom: 1px solid #1a1a1a; }
        .green { color: var(--text); } .red { color: var(--red); }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>👁️ OMEGA SINGULARITY <span style="font-size:12px; color:#666">V90</span></h1>
            <div>{{ status }}</div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>CAPITAL TOTAL</h2>
                <div class="big-val">${{ total_equity }}</div>
                <div>Cash: ${{ cash }}</div>
                <div>PnL: <span class="{{ 'green' if total_pnl >= 0 else 'red' }}">{{ total_pnl }}$</span></div>
            </div>
            
            <div class="card">
                <h2>ÉTAT NEURONAL</h2>
                <div>Confiance: {{ confidence }}%</div>
                <div>Stress: {{ stress }}%</div>
                <div>Euphorie: {{ euphoria }}%</div>
            </div>
            
            <div class="card">
                <h2>GÉNÉTIQUE (Gen #{{ generation }})</h2>
                <div>RSI Trigger: <b>< {{ genome.rsi_buy }}</b></div>
                <div>Stop Loss: <b>{{ genome.sl_mult }} ATR</b></div>
                <div>Take Profit: <b>{{ genome.tp_mult }} ATR</b></div>
            </div>
        </div>

        <br>

        <div class="grid" style="grid-template-columns: 2fr 1fr;">
            <div class="card">
                <h2>POSITIONS ACTIVES</h2>
                {% if holdings %}
                <table>
                    <thead><tr><th>ACTIF</th><th>QTÉ</th><th>ENTRÉE</th><th>ACTUEL</th><th>PNL</th></tr></thead>
                    <tbody>
                        {% for p in holdings %}
                        <tr>
                            <td>{{ p.symbol }}</td>
                            <td>{{ p.qty }}</td>
                            <td>{{ p.entry }}</td>
                            <td>{{ p.current }}</td>
                            <td class="{{ 'green' if p.pnl >= 0 else 'red' }}">{{ p.pnl }}$</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                {% else %}
                <div style="text-align:center; padding:20px; color:#666">SCAN EN COURS... AUCUNE POSITION</div>
                {% endif %}
            </div>

            <div class="card">
                <h2>FLUX DE PENSÉE (LIVE)</h2>
                <div class="log-box">
                    {% for log in logs %}
                    <div class="log-line">{{ log }}</div>
                    {% endfor %}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

# ==============================================================================
# 3. SYSTÈME DE LOGGING & MÉMOIRE
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
                    requests.post(LEARNING_WEBHOOK_URL, json={"content": msg_block})
                last_send = time.time()
            time.sleep(0.2)
        except: time.sleep(1)

def send_trade_alert(embed):
    if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed]})

def run_heartbeat():
    while True:
        if HEARTBEAT_WEBHOOK_URL: requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"})
        time.sleep(30)

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
            repo.update_file("brain.json", "Omega Save", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# ==============================================================================
# 4. LES 6 SENS (ANALYSE MULTIMODALE)
# ==============================================================================
def run_monte_carlo(prices):
    try:
        returns = prices.pct_change().dropna()
        sims = prices.iloc[-1] * (1 + np.random.normal(returns.mean(), returns.std(), (SIMULATION_COUNT, 10)))
        return np.sum(sims[:, -1] > prices.iloc[-1]) / SIMULATION_COUNT
    except: return 0.5

def get_vision_score(df):
    try:
        buf = io.BytesIO()
        mpf.plot(df.tail(60), type='candle', style='nightclouds', savefig=buf)
        buf.seek(0)
        img = Image.open(buf)
        res = model.generate_content(["Score achat (0.0-1.0)?", img])
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
    high = df['High'].max()
    low = df['Low'].min()
    fibo = high - ((high - low) * 0.618)
    return fibo

# ==============================================================================
# 5. LE CERVEAU CENTRAL (DÉCISION)
# ==============================================================================
def consult_council(s, rsi, mc, vis, soc, whale):
    mood = brain['emotions']
    prompt = f"""
    DECISION TRADE {s}.
    Mood: Stress {mood['stress']}%, Confiance {mood['confidence']}%.
    Data:
    - Quantum: {mc*100:.0f}% prob.
    - Vision: {vis:.2f}/1.0.
    - Social: {soc:.2f}.
    - Baleine: {whale}.
    - RSI: {rsi:.1f}.
    
    Si (Quantum > 60% ET Vision > 0.6) OU (Baleine ET RSI < 30): BUY.
    Sinon WAIT.
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
        e['euphoria'] += 5
    else:
        e['confidence'] = max(e['confidence']-10, 10)
        e['stress'] = min(e['stress']+10, 100)
        e['euphoria'] = 0

def get_kelly_bet(score, capital):
    e = brain['emotions']
    factor = 0.5 if e['stress'] > 70 else (1.2 if e['confidence'] > 80 else 1.0)
    win_prob = score / 100.0
    if win_prob <= 0.5: return 0
    kelly = win_prob - (1 - win_prob)
    return min(capital * kelly * 0.5 * factor, capital * 0.20)

# ==============================================================================
# 6. MOTEUR D'APPRENTISSAGE (GÉNÉTIQUE & RÊVE)
# ==============================================================================
def run_dream_learning():
    global brain, bot_state
    cache = {}
    
    # Préchargement
    for s in WATCHLIST:
        try:
            df = yf.Ticker(s).history(period="1mo", interval="1h")
            if not df.empty:
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                cache[s] = df.dropna()
        except: pass

    while True:
        # Horaire
        nyc = pytz.timezone('America/New_York')
        now = datetime.now(nyc)
        market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
        
        if market_open:
            time.sleep(60)
            continue
            
        bot_state['status'] = f"🌙 RÊVE (Gen #{brain['stats']['generation']})"
        brain['stats']['generation'] += 1
        
        # Mutation
        parent = brain['genome']
        mutant = {
            "rsi_buy": max(15, min(60, parent['rsi_buy'] + random.randint(-5, 5))),
            "sl_mult": max(1.0, parent['sl_mult'] + random.uniform(-0.5, 0.5)),
            "tp_mult": max(1.5, parent['tp_mult'] + random.uniform(-0.5, 0.5))
        }
        
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
        
        if pnl > brain['stats']['best_pnl']:
            brain['stats']['best_pnl'] = pnl
            brain['genome'] = mutant
            save_brain()
            fast_log(f"🧬 **ÉVOLUTION:** Nouveau Gène RSI<{mutant['rsi_buy']} | Gain +{pnl:.2f}$")
        else:
            fast_log(f"🧪 **TEST MUTANT {s}:** PnL {pnl:.2f}$ (Inférieur au record).")
            
        time.sleep(5)

# ==============================================================================
# 7. GHOST BROKER & TRADING LIVE
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
                pct = (pnl / (d['qty'] * d['entry'])) * 100
                positions.append({"symbol": s, "qty": d['qty'], "entry": d['entry'], "current": p, "pnl": round(pnl, 2), "pct": round(pct, 2)})
        return equity, positions

broker = GhostBroker()

def run_trading():
    global brain, bot_state
    load_brain()
    
    fast_log("🌌 **OMEGA SINGULARITY:** Tous systèmes nominaux.")
    
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if market_open:
                bot_state['status'] = "🟢 TRADING ACTIF"
                
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
                        send_trade_alert({"embeds": [{"title": f"{exit_r} : {s}", "description": f"PnL: **{pnl:.2f}$**", "color": col}]})
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
                            
                            # Filtre Génétique
                            if row['RSI'] < brain['genome']['rsi_buy']:
                                fast_log(f"🔎 **SCAN {s}:** Signal RSI. Lancement Analyse Profonde...")
                                
                                mc = run_monte_carlo(df['Close'])
                                vis = get_vision_score(df)
                                soc = get_social_hype(s)
                                whl, wh_msg = check_whale(df)
                                
                                council = consult_council(s, row['RSI'], mc, vis, soc, whl)
                                fast_log(f"🧠 **{s}:** MC:{mc:.2f} | Vis:{vis:.2f} | {wh_msg} => {council['vote']}")
                                
                                if council['vote'] == "BUY":
                                    price = row['Close']
                                    bet = calculate_kelly_bet(0.7, brain['cash']) # Score simplifié
                                    if bet > 200:
                                        qty = bet / price
                                        sl = price - (row['ATR'] * brain['genome']['sl_mult'])
                                        tp = price + (row['ATR'] * brain['genome']['tp_mult'])
                                        
                                        brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                                        brain['cash'] -= bet
                                        
                                        send_trade_alert({
                                            "embeds": [{"title": f"🌌 ACHAT OMEGA : {s}", "description": council['reason'], "color": 0x2ecc71}]
                                        })
                                        save_brain()
                        except: pass
            else:
                bot_state['status'] = "🌙 NUIT"
            
            time.sleep(60)
        except: time.sleep(10)

# --- 8. ROUTE WEB ---
@app.route('/')
def index():
    eq, pos = broker.get_portfolio()
    return render_template_string(DASHBOARD_HTML, 
        total_equity=f"{eq:,.2f}", 
        cash=f"{brain['cash']:,.2f}", 
        total_pnl=f"{brain['total_pnl']:,.2f}",
        status=bot_state['status'],
        confidence=brain['emotions']['confidence'],
        stress=brain['emotions']['stress'],
        euphoria=brain['emotions']['euphoria'],
        generation=brain['stats']['generation'],
        genome=brain['genome'],
        holdings=pos,
        logs=bot_state['web_logs']
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
