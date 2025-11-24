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
matplotlib.use('Agg') # Indispensable pour serveur sans écran
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
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")      # Trading Live
PAPER_WEBHOOK_URL = os.environ.get("PAPER_WEBHOOK_URL")          # Paper/Replay
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL")  # Status
LEARNING_WEBHOOK_URL = os.environ.get("LEARNING_WEBHOOK_URL")    # Flux Cerveau
SUMMARY_WEBHOOK_URL = os.environ.get("SUMMARY_WEBHOOK_URL")      # Synthèse
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR", "GOOG", "META"]
INITIAL_CAPITAL_MAIN = 50000.0 
INITIAL_CAPITAL_PAPER = 1000.0
SIMULATION_COUNT = 2000 # Monte Carlo

brain = {
    "cash": INITIAL_CAPITAL_MAIN, 
    "holdings": {}, 
    "paper_cash": INITIAL_CAPITAL_PAPER,
    "paper_holdings": {},
    
    # ADN Évolutif (Optimisé par le moteur de rêve)
    "genome": {"rsi_buy": 30, "sl_mult": 2.0, "tp_mult": 3.5},
    
    # Mémoire Expérientielle
    "karma": {s: 10.0 for s in WATCHLIST},
    "black_box": [], # Patterns interdits
    
    # Stats & Psycho
    "stats": {"generation": 0, "best_pnl": 0.0},
    "emotions": {"confidence": 50.0, "stress": 20.0},
    
    "total_pnl": 0.0
}

bot_state = {
    "status": "Booting Singularity Alpha...",
    "mode": "INIT",
    "last_log": "Chargement...",
    "web_logs": []
}

log_queue = queue.Queue()
short_term_memory = [] # Pour le bilan 10 min

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ==============================================================================
# 2. SYSTÈME DE LOGGING & COMMS (MATRIX STREAM)
# ==============================================================================
def fast_log(text):
    """Ajoute un log dans la file d'attente et sur le site web"""
    log_queue.put(text)
    bot_state['web_logs'].insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] {text}")
    if len(bot_state['web_logs']) > 50: bot_state['web_logs'] = bot_state['web_logs'][:50]

def logger_worker():
    """Envoie les logs groupés sur Discord"""
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

def send_summary(msg):
    if SUMMARY_WEBHOOK_URL: requests.post(SUMMARY_WEBHOOK_URL, json=msg)

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
            repo.update_file("brain.json", "Singularity Save", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# ==============================================================================
# 4. MODULES SENSORIELS (LES 6 SENS)
# ==============================================================================

# A. QUANTIQUE (Monte Carlo)
def run_monte_carlo(prices):
    try:
        returns = prices.pct_change().dropna()
        # Simulation vectorisée NumPy
        sims = prices.iloc[-1] * (1 + np.random.normal(returns.mean(), returns.std(), (SIMULATION_COUNT, 10)))
        prob = np.sum(sims[:, -1] > prices.iloc[-1]) / SIMULATION_COUNT
        return prob
    except: return 0.5

# B. VISION (Gemini Vision)
def get_vision_score(df):
    try:
        buf = io.BytesIO()
        mpf.plot(df.tail(60), type='candle', style='nightclouds', savefig=buf)
        buf.seek(0)
        img = Image.open(buf)
        res = model.generate_content(["Score achat 0.0-1.0 ?", img])
        return float(res.text.strip())
    except: return 0.5

# C. SOCIAL (StockTwits)
def get_social_hype(symbol):
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).json()
        txt = " ".join([m['body'] for m in r['messages'][:10]])
        return TextBlob(txt).sentiment.polarity
    except: return 0

# D. BALEINES (Volume)
def check_whale(df):
    try:
        vol = df['Volume'].iloc[-1]
        avg = df['Volume'].rolling(20).mean().iloc[-1]
        return (vol > avg * 2.5), f"x{vol/avg:.1f}" if avg > 0 else "N/A"
    except: return False, "x1.0"

# E. TECHNIQUE (Heikin Ashi)
def get_tech_indicators(df):
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    trend = "HAUSSIER" if ha_close.iloc[-1] > ha_open.iloc[-1] else "BAISSIER"
    return trend

# ==============================================================================
# 5. CERVEAU CENTRAL (DÉCISION & PSYCHO)
# ==============================================================================
def consult_council(s, rsi, mc, vis, soc, whale, trend):
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
    else:
        e['confidence'] = max(e['confidence']-10, 10)
        e['stress'] = min(e['stress']+10, 100)

def get_kelly_bet(score, capital):
    e = brain['emotions']
    factor = 0.5 if e['stress'] > 70 else (1.2 if e['confidence'] > 80 else 1.0)
    win_prob = score / 100.0
    if win_prob <= 0.5: return 0
    kelly = win_prob - (1 - win_prob)
    # Sécurité : Demi-Kelly cappé à 20% du capital
    bet = capital * kelly * 0.5 * factor
    return min(bet, capital * 0.20)

# ==============================================================================
# 6. MOTEUR D'ÉVOLUTION (LE RÊVE NOCTURNE)
# ==============================================================================
def generate_gemini_summary(stats, best_run):
    """Analyse sémantique des résultats d'apprentissage"""
    try:
        prompt = f"Analyse trading: Profit {stats['total_pnl']:.2f}$. Top: {best_run['s']}. Conseil court ?"
        return model.generate_content(prompt).text.strip()
    except: return "Analyse terminée."

def run_dream_learning():
    global brain, bot_state, short_term_memory
    cache = {}
    
    fast_log("🧬 **GENESIS:** Chargement du module d'évolution...")
    
    # Pré-chargement des données pour aller vite
    for s in WATCHLIST:
        try:
            df = yf.Ticker(s).history(period="1mo", interval="1h")
            if not df.empty:
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                cache[s] = df.dropna()
        except: pass
    fast_log("✅ Données chargées. Prêt à évoluer.")

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
            
            # 1. Mutation Génétique
            parent = brain['genome']
            mutant = {
                "rsi_buy": max(15, min(60, parent['rsi_buy'] + random.randint(-5, 5))),
                "sl_mult": round(max(1.0, parent['sl_mult'] + random.uniform(-0.5, 0.5)), 1),
                "tp_mult": round(max(1.5, parent['tp_mult'] + random.uniform(-0.5, 0.5)), 1)
            }
            
            # 2. Simulation Rapide
            s = random.choice(list(cache.keys()))
            df = cache[s]
            
            idx = random.randint(0, len(df) - 50)
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
            
            # Logging
            if pnl != 0:
                short_term_memory.append({"s": s, "pnl": pnl, "win": pnl>0, "rsi": mutant['rsi_buy']})
                
                emoji = "✅" if pnl > 0 else "❌"
                fast_log(f"🧪 **TEST {s}:** RSI<{mutant['rsi_buy']} -> {emoji} {pnl:.1f}$")
                
                if pnl > brain['stats']['best_pnl']:
                    brain['stats']['best_pnl'] = pnl
                    brain['genome'] = mutant
                    save_brain()
                    fast_log(f"🧬 **ÉVOLUTION:** Nouveau Gène RSI<{mutant['rsi_buy']} | Gain +{pnl:.2f}$")
            
            # 3. Bilan Synthèse toutes les 10 simulations
            if len(short_term_memory) >= 10:
                wins = sum(1 for x in short_term_memory if x['win'])
                tot = sum(x['pnl'] for x in short_term_memory)
                best = max(short_term_memory, key=lambda x: x['pnl'])
                rate = (wins/10)*100
                
                ai_text = generate_gemini_summary({"total_pnl": tot, "win_rate": rate}, best)
                
                msg = {
                    "embeds": [{
                        "title": "🎓 RAPPORT D'ÉTUDE",
                        "color": 0xFFD700,
                        "description": f"**IA:** *{ai_text}*",
                        "fields": [
                            {"name": "Profit", "value": f"**{tot:.2f}$**", "inline": True},
                            {"name": "Top Config", "value": f"{best['s']} (Gain {best['pnl']:.0f}$)", "inline": False}
                        ]
                    }]
                }
                send_summary(msg)
                short_term_memory = []

            time.sleep(5)
            
        except Exception as e:
            print(f"Err Dream: {e}")
            time.sleep(10)

# ==============================================================================
# 7. MOTEUR TRADING HYBRIDE (LIVE & PAPER)
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
    
    fast_log("🌌 **OMNISCIENCE V115:** Moteurs démarrés.")
    
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            current_mode = "LIVE" if market_open else "REPLAY"
            bot_state['status'] = f"🟢 {current_mode}"
            
            # Gestion des cibles (Aléatoire en replay pour varier les simulations)
            target_list = WATCHLIST if market_open else [random.choice(WATCHLIST)]
            
            for s in target_list:
                # Gestion Data (Live vs Replay)
                df = None
                if not market_open:
                    # Replay cache logic (simplifiée)
                    try: df = yf.Ticker(s).history(period="1mo", interval="1h"); df['RSI'] = ta.rsi(df['Close'], 14); df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14); df=df.dropna()
                    except: pass
                else:
                    try: df = yf.Ticker(s).history(period="1mo", interval="15m"); df['RSI'] = ta.rsi(df['Close'], 14); df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14); df=df.dropna()
                    except: pass
                
                if df is None or df.empty: continue
                
                row = df.iloc[-1] if market_open else df.iloc[random.randint(50, len(df)-10)]
                
                # --- SCAN ACHAT ---
                if row['RSI'] < brain['genome']['rsi_buy']:
                    
                    if market_open:
                        fast_log(f"🔎 **SCAN LIVE {s}:** RSI {row['RSI']:.1f} bas. Analyse profonde...")
                    
                    # ANALYSE LOURDE
                    mc = run_monte_carlo(df['Close'])
                    if not market_open: mc = random.uniform(0.4, 0.9) # Simu en replay
                    
                    if mc > 0.60:
                        vis = get_vision_score(df) if market_open else 0.8
                        soc = get_social_hype(s) if market_open else 0.0
                        whl, _ = check_whale(df)
                        
                        # CONSEIL
                        council = consult_council(s, row['RSI'], mc, vis, soc, whl, "LIVE" if market_open else "SIM")
                        
                        if market_open:
                            fast_log(f"🧠 **{s}:** MC:{mc:.2f} | Vis:{vis:.2f} => {council['vote']}")
                        
                        if council['vote'] == "BUY":
                            price = row['Close']
                            
                            # PARAMETRES APPRIS
                            sl = price - (row['ATR'] * brain['genome']['sl_mult'])
                            tp = price + (row['ATR'] * brain['genome']['tp_mult'])
                            
                            # A. PAPER TRADING (Toujours actif)
                            if len(brain['paper_holdings']) < 5:
                                qty = 200 / price
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

                            # B. MAIN TRADING (Live Only)
                            if market_open and len(brain['holdings']) < 5:
                                qty = 2000 / price
                                brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                                brain['cash'] -= 2000
                                
                                send_alert(DISCORD_WEBHOOK_URL, {
                                    "title": f"🌌 ACHAT OMEGA : {s}",
                                    "description": council['reason'],
                                    "color": 0x2ecc71
                                })
                                save_brain()

            # --- GESTION VENTES (COMMUN) ---
            for pf_name, pf_holdings, pf_cash, hook in [("MAIN", brain['holdings'], 'cash', DISCORD_WEBHOOK_URL), ("PAPER", brain['paper_holdings'], 'paper_cash', PAPER_WEBHOOK_URL)]:
                if pf_name == "MAIN" and not market_open: continue
                
                for s in list(pf_holdings.keys()):
                    pos = pf_holdings[s]
                    # Prix réel en live, simulé en replay
                    curr = broker.get_price(s) if market_open else pos['entry'] * random.uniform(0.98, 1.05)
                    
                    if not curr: continue
                    
                    exit_r = None
                    if curr < pos['stop']: exit_r = "STOP LOSS"
                    elif curr > pos['tp']: exit_r = "TAKE PROFIT"
                    elif not market_open and random.random() < 0.1: exit_r = "SIM EXIT"
                    
                    if exit_r:
                        pnl = (curr - pos['entry']) * pos['qty']
                        brain[pf_cash] += pos['qty'] * curr
                        del pf_holdings[s]
                        
                        col = 0x2ecc71 if pnl > 0 else 0xe74c3c
                        tag = "[REPLAY]" if not market_open else "[LIVE]"
                        send_alert(hook, {"embeds": [{"title": f"{tag} VENTE {s} ({exit_r})", "description": f"PnL: **{pnl:.2f}$**", "color": col}]})
                        save_brain()
            
            time.sleep(10 if market_open else 5)
            
        except Exception as e:
            print(f"Err: {e}")
            time.sleep(10)

# --- 8. DASHBOARD WEB ---
@app.route('/')
def index():
    eq = brain['cash'] + sum([d['qty']*100 for d in brain['holdings'].values()])
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="fr">
    <head><meta http-equiv="refresh" content="5"><style>body{background:#000;color:#0f0;font-family:monospace;padding:20px}</style></head>
    <body>
        <h1>👁️ OMEGA V115</h1>
        <p>Status: {{ status }}</p>
        <p>Main Cash: ${{ cash }} | Paper Cash: ${{ paper }}</p>
        <hr>
        <div style="height:400px;overflow:auto;background:#111;padding:10px;font-size:12px">
            {% for l in logs %}<div>{{ l }}</div>{% endfor %}
        </div>
    </body>
    </html>
    """, status=bot_state['status'], cash=f"{brain['cash']:,.2f}", paper=f"{brain['paper_cash']:,.2f}", logs=bot_state['web_logs'])

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
