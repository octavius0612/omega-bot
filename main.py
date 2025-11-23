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
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")      # Main Trading
PAPER_WEBHOOK_URL = os.environ.get("PAPER_WEBHOOK_URL")          # Paper/Replay
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL")  # Heartbeat
LEARNING_WEBHOOK_URL = os.environ.get("LEARNING_WEBHOOK_URL")    # Flux Cerveau
SUMMARY_WEBHOOK_URL = os.environ.get("SUMMARY_WEBHOOK_URL")      # Synthèse Prof
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
    
    # ADN Évolutif
    "genome": {"rsi_buy": 30, "sl_mult": 2.0, "tp_mult": 3.0},
    
    # Mémoires
    "stats": {"generation": 0, "best_pnl": 0.0},
    "emotions": {"confidence": 50.0, "stress": 20.0},
    "karma": {s: 10.0 for s in WATCHLIST},
    "black_box": [],
    
    # Journalisation
    "learning_journal": [],
    "total_pnl": 0.0
}

bot_state = {
    "status": "Booting Omniverse...",
    "mode": "INIT",
    "last_log": "Chargement modules...",
    "web_logs": []
}

log_queue = queue.Queue()
short_term_memory = [] # Pour la synthèse

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ==============================================================================
# 2. SYSTÈME DE LOGGING & COMMS
# ==============================================================================
def fast_log(text):
    """Log dans la console, la file Discord et le Web"""
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
            repo.update_file("brain.json", "Omniverse Save", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# ==============================================================================
# 4. MODULES SENSORIELS (LES 6 SENS)
# ==============================================================================

# A. QUANTIQUE
def run_monte_carlo(prices):
    try:
        returns = prices.pct_change().dropna()
        sims = prices.iloc[-1] * (1 + np.random.normal(returns.mean(), returns.std(), (SIMULATION_COUNT, 10)))
        prob = np.sum(sims[:, -1] > prices.iloc[-1]) / SIMULATION_COUNT
        return prob
    except: return 0.5

# B. VISION (Graphique)
def get_vision_score(df):
    try:
        buf = io.BytesIO()
        mpf.plot(df.tail(60), type='candle', style='nightclouds', savefig=buf)
        buf.seek(0)
        img = Image.open(buf)
        res = model.generate_content(["Analyse technique visuelle. Score achat 0.0-1.0 ?", img])
        return float(res.text.strip())
    except: return 0.5

# C. SOCIAL (StockTwits)
def get_social_hype(symbol):
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).json()
        txt = " ".join([m['body'] for m in r['messages'][:15]])
        return TextBlob(txt).sentiment.polarity
    except: return 0

# D. BALEINE
def check_whale(df):
    try:
        vol = df['Volume'].iloc[-1]
        avg = df['Volume'].rolling(20).mean().iloc[-1]
        return (vol > avg * 2.5), f"x{vol/avg:.1f}"
    except: return False, "x1.0"

# E. TECHNIQUE
def get_tech_indicators(df):
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['SMA200'] = ta.sma(df['Close'], length=200)
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    trend = "HAUSSIER" if ha_close.iloc[-1] > ha_open.iloc[-1] else "BAISSIER"
    return trend, df

# ==============================================================================
# 5. CERVEAU CENTRAL (DECISION & PSYCHO)
# ==============================================================================
def consult_council(s, rsi, mc, vis, soc, whale, trend):
    mood = brain['emotions']
    prompt = f"""
    CONSEIL OMNISCIENT {s}.
    PSYCHO: Stress {mood['stress']}%, Confiance {mood['confidence']}%.
    
    1. QUANTIQUE: {mc*100:.1f}% proba.
    2. VISION: Score {vis:.2f}/1.0.
    3. SOCIAL: {soc:.2f}.
    4. BALEINE: {"OUI" if whale else "NON"}.
    5. TECH: RSI {rsi:.1f}, Trend {trend}.
    
    RÈGLES:
    - Si (Maths > 60% ET Vision > 0.6) : BUY.
    - Si (Baleine ET RSI < 30) : STRONG BUY.
    - Sinon : WAIT.
    
    JSON: {{"vote": "BUY/WAIT", "reason": "..."}}
    """
    try:
        res = model.generate_content(prompt)
        return json.loads(res.text.replace("```json","").replace("```",""))
    except: return {"vote": "WAIT", "reason": "Erreur Conseil"}

def update_emotions(pnl):
    e = brain['emotions']
    if pnl > 0:
        e['confidence'] = min(e['confidence']+5, 100)
        e['stress'] = max(e['stress']-5, 0)
    else:
        e['confidence'] = max(e['confidence']-10, 10)
        e['stress'] = min(e['stress']+10, 100)

def get_kelly_bet(score, capital):
    e = brain['emotions']
    factor = 0.5 if e['stress'] > 70 else (1.2 if e['confidence'] > 80 else 1.0)
    win_prob = score / 100.0
    if win_prob <= 0.5: return 0
    kelly = win_prob - (1 - win_prob)
    return min(capital * kelly * 0.5 * factor, capital * 0.20)

# ==============================================================================
# 6. APPRENTISSAGE & SYNTHÈSE (LE PROFESSEUR)
# ==============================================================================
def generate_evolution_report(old, new, pnl, symbol):
    prompt = f"""
    Tu es un Architecte IA.
    Optimisation sur {symbol}.
    Ancien: RSI<{old['rsi_buy']} | Nouveau: RSI<{new['rsi_buy']}
    Gain Test: +{pnl:.2f}$
    
    Explique en 1 phrase pourquoi cette mutation est meilleure.
    """
    try:
        res = model.generate_content(prompt)
        return res.text.strip()
    except: return "Optimisation validée."

def run_learning_loop():
    global brain, short_term_memory
    cache = {}
    fast_log("🧬 **OMNI-LEARN:** Activation du module génétique.")
    
    # Pré-chargement
    for s in WATCHLIST:
        try:
            df = yf.Ticker(s).history(period="1mo", interval="1h")
            if not df.empty:
                _, df = get_tech_indicators(df)
                cache[s] = df.dropna()
        except: pass

    while True:
        try:
            s = random.choice(list(cache.keys()))
            df = cache[s]
            
            # 1. Mutation
            parent = brain['genome']
            mutant = {
                "rsi_buy": max(15, min(60, parent['rsi_buy'] + random.randint(-4, 4))),
                "sl_mult": round(max(1.0, parent['sl_mult'] + random.uniform(-0.3, 0.3)), 1),
                "tp_mult": round(max(1.5, parent['tp_mult'] + random.uniform(-0.3, 0.3)), 1)
            }
            
            # 2. Backtest Rapide
            idx = random.randint(0, len(df)-50)
            subset = df.iloc[idx : idx+50]
            
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
            
            # 3. Évolution
            if pnl > brain['stats']['best_pnl'] and pnl > 0:
                brain['stats']['best_pnl'] = pnl
                old_genome = brain['genome']
                brain['genome'] = mutant
                brain['stats']['generation'] += 1
                
                expl = generate_evolution_report(old_genome, mutant, pnl, s)
                
                entry = {
                    "gen": brain['stats']['generation'],
                    "time": datetime.now().strftime("%H:%M"),
                    "symbol": s,
                    "gain": pnl,
                    "reason": expl
                }
                brain['learning_journal'].insert(0, entry)
                if len(brain['learning_journal']) > 5: brain['learning_journal'].pop()
                
                save_brain()
                fast_log(f"🧬 **MUTATION GAGNANTE:** {expl}")

            # 4. SYNTHÈSE (Tous les 10 tests)
            if len(short_term_memory) < 10:
                short_term_memory.append(pnl)
            else:
                tot = sum(short_term_memory)
                wins = sum(1 for x in short_term_memory if x > 0)
                
                msg = {
                    "embeds": [{
                        "title": "🎓 RAPPORT SYNTHÈSE",
                        "color": 0xFFD700,
                        "fields": [
                            {"name": "Génération", "value": f"#{brain['stats']['generation']}", "inline": True},
                            {"name": "Précision", "value": f"{(wins/10)*100:.0f}%", "inline": True},
                            {"name": "Gènes Actuels", "value": f"RSI < {brain['genome']['rsi_buy']} | SL {brain['genome']['sl_mult']}", "inline": False}
                        ]
                    }]
                }
                send_alert(SUMMARY_WEBHOOK_URL, msg)
                short_term_memory = []

            time.sleep(5)
        except: time.sleep(10)

# ==============================================================================
# 7. MOTEUR TRADING HYBRIDE (LIVE + REPLAY PAPER)
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
            
            current_mode = "LIVE" if market_open else "REPLAY"
            bot_state['status'] = f"🟢 {current_mode}"
            
            # Pour le Replay (Week-end), on simule une action aléatoire
            target_list = WATCHLIST if market_open else [random.choice(WATCHLIST)]
            
            for s in target_list:
                # En Replay, on utilise le cache de l'apprentissage
                df = None
                if not market_open:
                    try: df = yf.Ticker(s).history(period="1mo", interval="1h"); df['RSI'] = ta.rsi(df['Close'], 14); df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14); df=df.dropna()
                    except: pass
                else:
                    # En Live, on prend les dernières données
                    try: df = yf.Ticker(s).history(period="1mo", interval="15m"); trend, df = get_tech_indicators(df)
                    except: pass
                
                if df is None or df.empty: continue
                
                # Simulation du temps qui passe en replay (on prend une ligne au hasard)
                row = df.iloc[-1] if market_open else df.iloc[random.randint(50, len(df)-10)]
                
                # FILTRE GÉNÉTIQUE
                if row['RSI'] < brain['genome']['rsi_buy']:
                    
                    fast_log(f"🔎 **SCAN {s} ({current_mode}):** RSI {row['RSI']:.1f} bas...")
                    
                    # ANALYSE COMPLÈTE
                    mc = run_monte_carlo(df['Close'])
                    
                    if mc > 0.60:
                        vis = get_vision_score(df) if market_open else 0.7 # Vision simulée en replay pour économiser quota
                        soc = get_social_hype(s) if market_open else 0.0
                        whl, wh_msg = check_whale(df)
                        
                        council = consult_council(s, row['RSI'], mc, vis, soc, whl)
                        fast_log(f"🧠 **{s}:** MC:{mc:.2f} | Vis:{vis:.2f} => {council['vote']}")
                        
                        if council['vote'] == "BUY":
                            price = row['Close']
                            
                            # A. COMPTE PAPER (1k) - Toujours actif (Replay ou Live)
                            if len(brain['paper_holdings']) < 5:
                                qty = 200 / price
                                sl = price - (row['ATR'] * brain['genome']['sl_mult'])
                                tp = price + (row['ATR'] * brain['genome']['tp_mult'])
                                brain['paper_holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                                brain['paper_cash'] -= 200
                                
                                tag = "🎬 REPLAY" if not market_open else "🔴 LIVE"
                                msg = {
                                    "title": f"🎮 {tag} PAPER ENTRY : {s}",
                                    "description": council['reason'],
                                    "color": 0x3498db,
                                    "footer": {"text": f"Solde Paper: {brain['paper_cash']:.2f}$"}
                                }
                                send_alert(PAPER_WEBHOOK_URL, msg)
                                save_brain()

                            # B. COMPTE MAIN (50k) - Seulement en LIVE
                            if market_open and len(brain['holdings']) < 5:
                                qty = 2000 / price
                                sl = price - (row['ATR'] * brain['genome']['sl_mult'])
                                tp = price + (row['ATR'] * brain['genome']['tp_mult'])
                                brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                                brain['cash'] -= 2000
                                
                                send_alert(DISCORD_WEBHOOK_URL, {
                                    "title": f"🌌 ACHAT OMEGA : {s}",
                                    "description": council['reason'],
                                    "color": 0x2ecc71
                                })
                                save_brain()
            
            # Pause : 1min en Live, 10s en Replay
            time.sleep(60 if market_open else 10)
            
        except Exception as e:
            time.sleep(10)

# --- 8. DASHBOARD ---
@app.route('/')
def index():
    eq = brain['cash'] + sum([d['qty']*100 for d in brain['holdings'].values()])
    return render_template_string(DASHBOARD_HTML, 
        total_equity=f"{eq:,.2f}", 
        cash=f"{brain['cash']:,.2f}",
        status=bot_state['status'],
        genome=brain['genome'],
        journal=brain['learning_journal'],
        logs=bot_state['web_logs']
    )

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta http-equiv="refresh" content="10">
    <title>OMNIVERSE V100</title>
    <style>
        body { background: #000; color: #0f0; font-family: monospace; padding: 20px; }
        .card { border: 1px solid #333; padding: 15px; margin-bottom: 20px; }
        table { width: 100%; border-collapse: collapse; color: #aaa; }
        td { border-bottom: 1px solid #222; padding: 5px; }
    </style>
</head>
<body>
    <h1>🌌 OMNIVERSE V100</h1>
    
    <div class="card">
        <h3>STATUS: {{ status }}</h3>
        <h2>CAPITAL: ${{ total_equity }}</h2>
        <p>Génome Actuel: RSI < {{ genome.rsi_buy }} | SL {{ genome.sl_mult }} ATR</p>
    </div>

    <div class="card">
        <h3>📜 JOURNAL D'ÉVOLUTION</h3>
        <table>
            <tr><th>GEN</th><th>HEURE</th><th>ACTIF</th><th>RAISON</th><th>GAIN</th></tr>
            {% for l in journal %}
            <tr>
                <td>#{{ l.gen }}</td>
                <td>{{ l.time }}</td>
                <td>{{ l.symbol }}</td>
                <td>{{ l.reason }}</td>
                <td style="color: #0f0">+{{ l.gain|round(1) }}$</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="card">
        <h3>FLUX NEURONAL</h3>
        <div style="height: 200px; overflow-y: auto; font-size: 12px;">
            {% for log in logs %}
            <div>{{ log }}</div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""

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
