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
matplotlib.use('Agg') # Mode serveur (sans écran)
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
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")      # Canal Trading
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL")  # Canal Status
LEARNING_WEBHOOK_URL = os.environ.get("LEARNING_WEBHOOK_URL")    # Canal Cerveau (Flux)
SUMMARY_WEBHOOK_URL = os.environ.get("SUMMARY_WEBHOOK_URL")      # Canal Synthèse
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR", "GOOG", "META"]
INITIAL_CAPITAL = 50000.0 
SIMULATION_COUNT = 5000 # Nombre de futurs simulés par analyse (Monte Carlo)

# --- STRUCTURE DU CERVEAU (MÉMOIRE) ---
brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    # ADN Évolutif (Le bot modifiera ces valeurs la nuit)
    "genome": {"rsi_buy": 30, "sl_mult": 2.0, "tp_mult": 3.5},
    # Stats d'apprentissage
    "stats": {"generation": 0, "best_pnl": 0.0},
    # État Psychologique
    "emotions": {"confidence": 50.0, "stress": 20.0, "euphoria": 0.0},
    # Mémoire Expérientielle
    "karma": {s: 10.0 for s in WATCHLIST},
    "black_box": [], # Configurations interdites
    "total_pnl": 0.0
}

bot_state = {
    "status": "Démarrage V94...",
    "last_decision": "Aucune",
    "last_log": "Initialisation...",
    "web_logs": []
}

log_queue = queue.Queue()
short_term_memory = []

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ==============================================================================
# 2. SYSTÈME NERVEUX (LOGGING & COMMS)
# ==============================================================================
def fast_log(text):
    """Ajoute un log dans la file d'attente et sur le site web"""
    log_queue.put(text)
    bot_state['web_logs'].insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] {text}")
    if len(bot_state['web_logs']) > 50: bot_state['web_logs'] = bot_state['web_logs'][:50]

def logger_worker():
    """Thread qui envoie les logs par paquets sur Discord (Anti-Ban)"""
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

# ==============================================================================
# 3. PERSISTANCE (MÉMOIRE GITHUB)
# ==============================================================================
def load_brain():
    global brain
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        c = repo.get_contents("brain.json")
        loaded = json.loads(c.decoded_content.decode())
        # Fusion intelligente pour ne pas écraser les nouvelles clés
        if "cash" in loaded: brain.update(loaded)
    except: pass

def save_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        content = json.dumps(brain, indent=4)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Omni Save", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# ==============================================================================
# 4. LES SENS (MODULES D'ANALYSE)
# ==============================================================================

# A. QUANTIQUE (Monte Carlo)
def run_monte_carlo(prices):
    try:
        returns = prices.pct_change().dropna()
        # Simulation vectorisée (NumPy) de 5000 avenirs possibles
        sims = prices.iloc[-1] * (1 + np.random.normal(returns.mean(), returns.std(), (SIMULATION_COUNT, 10)))
        prob = np.sum(sims[:, -1] > prices.iloc[-1]) / SIMULATION_COUNT
        return prob
    except: return 0.5

# B. VISION (Analyse Graphique IA)
def get_vision_score(df):
    try:
        buf = io.BytesIO()
        # Génération du graphique en mémoire
        mpf.plot(df.tail(60), type='candle', style='nightclouds', savefig=buf)
        buf.seek(0)
        img = Image.open(buf)
        # L'IA regarde l'image
        res = model.generate_content(["Tu es un expert chartiste. Analyse ce graphique. Donne un score d'achat de 0.0 (Vente) à 1.0 (Achat Fort). Réponds UNIQUEMENT le chiffre.", img])
        return float(res.text.strip())
    except: return 0.5

# C. SOCIAL (Sentiment de Foule)
def get_social_hype(symbol):
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).json()
        txt = " ".join([m['body'] for m in r['messages'][:10]])
        return TextBlob(txt).sentiment.polarity # Score entre -1 et 1
    except: return 0

# D. BALEINE (Détection Volume)
def check_whale(df):
    try:
        vol = df['Volume'].iloc[-1]
        avg = df['Volume'].rolling(20).mean().iloc[-1]
        return (vol > avg * 2.5), f"x{vol/avg:.1f}" # Ratio de volume
    except: return False, "x1.0"

# E. TECHNIQUE (Heikin Ashi + Fibo)
def get_tech_indicators(df):
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    trend = "HAUSSIER" if ha_close.iloc[-1] > ha_open.iloc[-1] else "BAISSIER"
    return trend

# ==============================================================================
# 5. LE CERVEAU CENTRAL (CONSEIL + PSYCHO)
# ==============================================================================
def consult_council(s, rsi, mc, vis, soc, whale, trend):
    """
    Réunit toutes les données et demande une décision finale à Gemini.
    """
    mood = brain['emotions']
    prompt = f"""
    CONSEIL SUPRÊME POUR {s}.
    État Psychologique Bot: Confiance {mood['confidence']}%, Stress {mood['stress']}%.
    
    DONNÉES ENTRÉE :
    1. ⚛️ MATHS (Monte Carlo): {mc*100:.1f}% probabilité de hausse.
    2. 👁️ VISION (Chartisme): Score {vis:.2f}/1.0.
    3. 🗣️ SOCIAL (Buzz): {soc:.2f} (-1 à 1).
    4. 🐋 BALEINE (Volume): {"OUI" if whale else "NON"}.
    5. 📉 TECH: RSI {rsi:.1f}, Tendance {trend}.
    
    ALGORITHME DE DÉCISION :
    - Si Maths > 60% ET Vision > 0.6 : ACHAT.
    - Si Baleine ET RSI < 30 : ACHAT FORT (Dip).
    - Sinon : ATTENTE.
    
    Réponds JSON: {{"vote": "BUY/WAIT", "reason": "Une phrase courte d'explication"}}
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
        e['euphoria'] += 5
    else:
        e['confidence'] = max(e['confidence']-10, 10)
        e['stress'] = min(e['stress']+10, 100)
        e['euphoria'] = 0

def calculate_position_size(score, capital):
    """Formule de Kelly ajustée par les émotions"""
    e = brain['emotions']
    # Si stressé, on mise moins. Si confiant, on mise plus.
    mood_factor = 0.5 if e['stress'] > 70 else (1.2 if e['confidence'] > 80 else 1.0)
    
    win_prob = score / 100.0
    if win_prob <= 0.5: return 0
    kelly = win_prob - (1 - win_prob)
    
    # Sécurité : Demi-Kelly cappé à 20% du capital
    bet = capital * kelly * 0.5 * mood_factor
    return min(bet, capital * 0.20)

# ==============================================================================
# 6. MOTEUR D'ÉVOLUTION (LE RÊVE NOCTURNE)
# ==============================================================================
def run_dream_learning():
    """
    Tourne quand le marché est fermé.
    Simule des mutations génétiques pour trouver les meilleurs paramètres.
    """
    global brain, bot_state
    cache = {}
    
    # Pré-chargement des données pour aller vite
    for s in WATCHLIST:
        try:
            df = yf.Ticker(s).history(period="1mo", interval="1h")
            if not df.empty:
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                cache[s] = df.dropna()
        except: pass

    while True:
        # Vérification horaire
        nyc = pytz.timezone('America/New_York')
        now = datetime.now(nyc)
        market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
        
        if market_open:
            time.sleep(60)
            continue
            
        bot_state['status'] = f"🌙 RÊVE (Gen #{brain['stats']['generation']})"
        brain['stats']['generation'] += 1
        
        # 1. MUTATION (Création d'un mutant)
        parent = brain['genome']
        mutant = {
            "rsi_buy": max(15, min(60, parent['rsi_buy'] + random.randint(-5, 5))),
            "sl_mult": max(1.0, parent['sl_mult'] + random.uniform(-0.5, 0.5)),
            "tp_mult": max(1.5, parent['tp_mult'] + random.uniform(-0.5, 0.5))
        }
        
        # 2. SIMULATION RAPIDE (Backtest sur cache)
        s = random.choice(list(cache.keys()))
        df = cache[s]
        start_idx = random.randint(0, len(df) - 50)
        subset = df.iloc[start_idx : start_idx+50]
        
        pnl = 0
        trades = 0
        for i in range(len(subset)-10):
            row = subset.iloc[i]
            if row['RSI'] < mutant['rsi_buy']:
                entry = row['Close']
                sl = entry - (row['ATR'] * mutant['sl_mult'])
                tp = entry + (row['ATR'] * mutant['tp_mult'])
                
                future = subset.iloc[i+1 : i+6]
                if not future.empty:
                    if future['High'].max() > tp: pnl += (tp - entry); trades += 1
                    elif future['Low'].min() < sl: pnl -= (entry - sl); trades += 1
        
        # 3. SÉLECTION NATURELLE
        if pnl > brain['stats']['best_pnl']:
            brain['stats']['best_pnl'] = pnl
            brain['genome'] = mutant
            save_brain()
            fast_log(f"🧬 **ÉVOLUTION:** Nouveau Gène RSI<{mutant['rsi_buy']} | Gain Simulé +{pnl:.2f}$")
        else:
            if random.random() < 0.2: # Log pas tout le temps pour éviter le spam
                fast_log(f"🧪 **TEST MUTANT {s}:** PnL {pnl:.2f}$ (Inférieur au record).")
            
        time.sleep(5) # Vitesse d'apprentissage

# ==============================================================================
# 7. MOTEUR TRADING LIVE & GHOST BROKER
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
    
    fast_log("🌌 **OMNISCIENCE:** Moteurs démarrés.")
    
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if market_open:
                bot_state['status'] = "🟢 TRADING ACTIF"
                
                # 1. GESTION VENTES
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
                        
                        send_trade_alert({
                            "embeds": [{"title": f"{exit_r} : {s}", "description": f"PnL: **{pnl:.2f}$**", "color": col}]
                        })
                        save_brain()

                # 2. SCAN ACHATS
                if len(brain['holdings']) < 5:
                    for s in WATCHLIST:
                        if s in brain['holdings']: continue
                        
                        try:
                            df = yf.Ticker(s).history(period="1mo", interval="1h")
                            if df.empty: continue
                            
                            df['RSI'] = ta.rsi(df['Close'], length=14)
                            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                            row = df.iloc[-1]
                            
                            # FILTRE RAPIDE (Paramètres Appris par Génétique)
                            if row['RSI'] < brain['genome']['rsi_buy']:
                                
                                fast_log(f"🔎 **SCAN {s}:** RSI {row['RSI']:.1f} bas. Analyse profonde...")
                                
                                # ANALYSE LOURDE
                                mc = run_monte_carlo(df['Close'])
                                vis = get_vision_score(df)
                                soc = get_social_hype(s)
                                whl, wh_msg = check_whale(df)
                                trend, _ = get_tech_indicators(df)
                                
                                # CONSEIL
                                council = consult_council(s, row['RSI'], mc, vis, soc, whl, trend)
                                bot_state['last_decision'] = f"{s}: {council['vote']}"
                                
                                fast_log(f"🧠 **{s}:** MC:{mc:.2f} | Vis:{vis:.2f} | {wh_msg} => {council['vote']}")
                                
                                if council['vote'] == "BUY":
                                    price = row['Close']
                                    
                                    # Score global pour la mise
                                    global_score = (mc + vis) * 50 
                                    bet = calculate_position_size(global_score, brain['cash'])
                                    
                                    if bet > 200:
                                        qty = bet / price
                                        # SL/TP issus du génome appris
                                        sl = price - (row['ATR'] * brain['genome']['sl_mult'])
                                        tp = price + (row['ATR'] * brain['genome']['tp_mult'])
                                        
                                        brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                                        brain['cash'] -= bet
                                        
                                        send_trade_alert({
                                            "embeds": [{
                                                "title": f"🌌 ACHAT OMNISCIENT : {s}",
                                                "description": council['reason'],
                                                "color": 0x2ecc71,
                                                "fields": [
                                                    {"name": "Vision", "value": f"{vis:.2f}", "inline": True},
                                                    {"name": "Quantique", "value": f"{mc:.2f}", "inline": True},
                                                    {"name": "Mise", "value": f"{bet:.0f}$", "inline": True}
                                                ]
                                            }]
                                        })
                                        save_brain()
                        except: pass
            else:
                bot_state['status'] = "🌙 NUIT"
            
            time.sleep(30)
        except: time.sleep(10)

# ==============================================================================
# 8. DASHBOARD WEB
# ==============================================================================
HTML = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta http-equiv="refresh" content="10">
    <title>OMNISCIENT V94</title>
    <style>
        :root { --bg: #050505; --card: #0f1115; --text: #00ff41; --red: #ff0033; }
        body { background: var(--bg); color: var(--text); font-family: 'Consolas', monospace; padding: 20px; margin: 0; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }
        .card { background: var(--card); border: 1px solid #333; padding: 15px; }
        h2 { margin: 0 0 10px 0; font-size: 16px; color: #fff; }
        .big { font-size: 28px; font-weight: bold; color: #fff; }
        .log-box { height: 300px; overflow-y: auto; font-size: 12px; border-top: 1px solid #333; margin-top: 10px; padding-top: 5px; opacity: 0.8; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        td { padding: 5px 0; border-bottom: 1px solid #1a1a1a; }
        .green { color: var(--text); } .red { color: var(--red); }
    </style>
</head>
<body>
    <h1>👁️ THE OMNISCIENT V94</h1>
    
    <div class="grid">
        <div class="card">
            <h2>STATUS</h2>
            <div class="big">{{ status }}</div>
            <p>Dernière Action: {{ last_dec }}</p>
        </div>
        <div class="card">
            <h2>FINANCE</h2>
            <div class="big">${{ equity }}</div>
            <p>Cash: ${{ cash }}</p>
        </div>
        <div class="card">
            <h2>PSYCHOLOGIE</h2>
            <p>Confiance: {{ conf }}%</p>
            <p>Stress: {{ stress }}%</p>
        </div>
    </div>
    <br>
    <div class="grid" style="grid-template-columns: 2fr 1fr;">
        <div class="card">
            <h2>POSITIONS</h2>
            <table>
                <tr><th>ACTIF</th><th>QTE</th><th>ENTRÉE</th><th>ACTUEL</th><th>PNL</th></tr>
                {% for p in positions %}
                <tr>
                    <td>{{ p.symbol }}</td>
                    <td>{{ p.qty|round(2) }}</td>
                    <td>{{ p.entry }}</td>
                    <td>{{ p.current }}</td>
                    <td class="{{ 'green' if p.pnl >= 0 else 'red' }}">{{ p.pnl }}$</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        <div class="card">
            <h2>LOGS CERVEAU</h2>
            <div class="log-box">
                {% for log in logs %}
                <div>{{ log }}</div>
                {% endfor %}
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    eq, pos = broker.get_portfolio()
    return render_template_string(HTML, 
        equity=f"{eq:,.2f}", cash=f"{brain['cash']:,.2f}",
        status=bot_state['status'], positions=pos,
        last_dec=bot_state['last_decision'],
        conf=brain['emotions']['confidence'], stress=brain['emotions']['stress'],
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
