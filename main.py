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
matplotlib.use('Agg') # Indispensable pour serveur
import mplfinance as mpf
from PIL import Image
from flask import Flask, render_template_string
from datetime import datetime, time as dtime
import pytz
from github import Github

app = Flask(__name__)

# ==============================================================================
# 1. CONFIGURATION & CLÉS
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
SIMULATION_COUNT = 5000 # Monte Carlo (Equilibre Vitesse/Précision)

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    # L'ADN du bot (Paramètres qui évoluent)
    "genome": {"rsi_buy": 30, "sl_mult": 2.0, "tp_mult": 3.0},
    # Mémoire des expériences
    "memory_bank": [], # Stocke les patterns gagnants/perdants
    "black_box": [],   # Stocke les patterns interdits (Traumatismes)
    "stats": {"generation": 0, "best_pnl": 0.0},
    "emotions": {"confidence": 50.0, "stress": 20.0}
}

bot_status = "Éveil de la Singularité V88..."
log_queue = queue.Queue()

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ==============================================================================
# 2. LOGGING & COMMS
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
    except: pass

def save_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        content = json.dumps(brain, indent=4)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Singularity Update", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# ==============================================================================
# 4. MODULES SENSORIELS (LES SENS)
# ==============================================================================

# A. QUANTIQUE (Monte Carlo)
def run_monte_carlo(prices):
    try:
        returns = prices.pct_change().dropna()
        sims = prices.iloc[-1] * (1 + np.random.normal(returns.mean(), returns.std(), (SIMULATION_COUNT, 10)))
        prob = np.sum(sims[:, -1] > prices.iloc[-1]) / SIMULATION_COUNT
        return prob
    except: return 0.5

# B. VISION (Gemini)
def get_vision_score(df):
    try:
        buf = io.BytesIO()
        mpf.plot(df.tail(50), type='candle', style='nightclouds', savefig=buf)
        buf.seek(0)
        img = Image.open(buf)
        res = model.generate_content(["Analyse ce graphique boursier. Score d'achat (0.0 à 1.0) ? Réponds chiffre uniquement.", img])
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

# D. BALEINES (Volume)
def check_whale(df):
    try:
        vol = df['Volume'].iloc[-1]
        avg = df['Volume'].rolling(20).mean().iloc[-1]
        return (vol > avg * 2.5), f"x{vol/avg:.1f}"
    except: return False, "x1.0"

# E. TECHNIQUE (Fibonacci & Heikin Ashi)
def get_tech_indicators(df):
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    trend = "UP" if ha_close.iloc[-1] > ha_open.iloc[-1] else "DOWN"
    
    high = df['High'].max()
    low = df['Low'].min()
    fibo = high - ((high - low) * 0.618)
    
    return trend, fibo

# ==============================================================================
# 5. CERVEAU CENTRAL (CONSEIL + PSYCHO)
# ==============================================================================
def consult_council(s, rsi, mc, vis, soc, whale, trend):
    mood = brain['emotions']
    prompt = f"""
    CONSEIL SUPRÊME POUR {s}.
    État Psycho Bot: Confiance {mood['confidence']}%.
    
    1. QUANTIQUE: {mc*100:.1f}% chance hausse.
    2. VISION: {vis:.2f}/1.0.
    3. SOCIAL: {soc:.2f}.
    4. BALEINE: {"OUI" if whale else "NON"}.
    5. TECH: RSI {rsi:.1f}, Trend {trend}.
    
    RÈGLE:
    - Si (Quantique > 60% ET Vision > 0.6) OU (Baleine ET RSI < 30): ACHAT.
    - Sinon : ATTENTE.
    
    Réponds JSON: {{"vote": "BUY/WAIT", "reason": "..."}}
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

def get_kelly_bet(score, capital):
    e = brain['emotions']
    # Si stressé, on mise moins
    factor = 0.5 if e['stress'] > 70 else (1.2 if e['confidence'] > 80 else 1.0)
    
    win_prob = score / 100.0
    if win_prob <= 0.5: return 0
    kelly = win_prob - (1 - win_prob)
    return min(capital * kelly * 0.5 * factor, capital * 0.20)

# ==============================================================================
# 6. MOTEUR D'ÉVOLUTION (LE RÊVE)
# ==============================================================================
def run_dream_learning():
    """
    Tourne quand le marché est fermé.
    Simule des variations génétiques pour trouver le 'Gène Parfait'.
    """
    global brain, bot_status
    cache = {}
    
    # Préchargement Data
    for s in WATCHLIST:
        try:
            df = yf.Ticker(s).history(period="1mo", interval="1h")
            if not df.empty:
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                cache[s] = df.dropna()
        except: pass

    while True:
        # Check Horaire (On n'apprend que si le marché est fermé)
        nyc = pytz.timezone('America/New_York')
        now = datetime.now(nyc)
        market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
        
        if market_open:
            time.sleep(60)
            continue
            
        bot_status = f"🌙 RÊVE (Gen #{brain['stats']['generation']})"
        brain['stats']['generation'] += 1
        
        # 1. MUTATION : On crée un "Enfant" avec des paramètres légèrement différents
        parent = brain['genome']
        mutant = {
            "rsi_buy": max(15, min(60, parent['rsi_buy'] + random.randint(-5, 5))),
            "sl_mult": max(1.0, parent['sl_mult'] + random.uniform(-0.5, 0.5)),
            "tp_mult": max(1.5, parent['tp_mult'] + random.uniform(-0.5, 0.5))
        }
        
        # 2. SIMULATION (Backtest rapide sur le cache)
        s = random.choice(list(cache.keys()))
        df = cache[s]
        
        # On teste sur une portion aléatoire du passé
        start_idx = random.randint(0, len(df) - 50)
        subset = df.iloc[start_idx : start_idx+50]
        
        pnl = 0
        trades = 0
        
        for i in range(len(subset)-10):
            row = subset.iloc[i]
            # Test du mutant
            if row['RSI'] < mutant['rsi_buy']:
                entry = row['Close']
                sl = entry - (row['ATR'] * mutant['sl_mult'])
                tp = entry + (row['ATR'] * mutant['tp_mult'])
                
                future = subset.iloc[i+1 : i+6]
                if not future.empty:
                    if future['High'].max() > tp: 
                        pnl += (tp - entry)
                        trades += 1
                    elif future['Low'].min() < sl: 
                        pnl -= (entry - sl)
                        trades += 1
        
        # 3. SÉLECTION NATURELLE
        # Si le mutant est rentable et actif
        if pnl > 0 and trades > 0:
            # Si c'est un record, on l'adopte !
            if pnl > brain['stats']['best_pnl']:
                brain['stats']['best_pnl'] = pnl
                brain['genome'] = mutant
                save_brain()
                fast_log(f"🧬 **ÉVOLUTION RÉUSSIE:**\nNouveau Gène: RSI<{mutant['rsi_buy']} | SL {mutant['sl_mult']:.1f} ATR\nGain Simulé: +{pnl:.2f}$")
            else:
                fast_log(f"🧪 **TEST MUTANT:** {s} -> +{pnl:.2f}$ (Pas un record)")
        elif pnl < 0:
            # On enregistre la configuration toxique dans la Boîte Noire
            sig = f"RSI{mutant['rsi_buy']}|LOSS"
            if sig not in brain['black_box']: brain['black_box'].append(sig)
            fast_log(f"❌ **ÉCHEC:** Le mutant a perdu {pnl:.2f}$. Rejeté.")
            
        time.sleep(5) # Vitesse d'apprentissage

# ==============================================================================
# 7. MOTEUR TRADING LIVE (LE RÉVEIL)
# ==============================================================================
def run_trading():
    global brain, bot_status
    load_brain()
    
    fast_log("🌌 **SINGULARITÉ V88:** Démarrage des systèmes omniscients.")
    
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if market_open:
                bot_status = "🟢 TRADING ACTIF"
                
                # 1. GESTION VENTES
                for s in list(brain['holdings'].keys()):
                    pos = brain['holdings'][s]
                    # On récupère le prix via Yahoo pour la simplicité et fiabilité
                    try: curr = yf.Ticker(s).fast_info['last_price']
                    except: continue
                    
                    exit_reason = None
                    if curr < pos['stop']: exit_reason = "STOP LOSS"
                    elif curr > pos['tp']: exit_reason = "TAKE PROFIT"
                    
                    if exit_reason:
                        pnl = (curr - pos['entry']) * pos['qty']
                        brain['cash'] += pos['qty'] * curr
                        brain['total_pnl'] += pnl
                        del brain['holdings'][s]
                        
                        update_emotions(pnl)
                        col = 0x2ecc71 if pnl > 0 else 0xe74c3c
                        
                        send_trade_alert({
                            "title": f"{exit_reason} : {s}", 
                            "description": f"PnL: **{pnl:.2f}$**", 
                            "color": col
                        })
                        save_brain()

                # 2. SCAN ACHATS
                if len(brain['holdings']) < 5:
                    for s in WATCHLIST:
                        if s in brain['holdings']: continue
                        
                        try:
                            df = yf.Ticker(s).history(period="1mo", interval="15m")
                            if df.empty: continue
                            
                            df['RSI'] = ta.rsi(df['Close'], length=14)
                            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                            row = df.iloc[-1]
                            
                            # UTILISATION DU GÉNOME APPRIS
                            if row['RSI'] < brain['genome']['rsi_buy']:
                                
                                fast_log(f"🔎 **SCAN {s}:** Signal RSI ({row['RSI']:.1f}). Lancement Analyse Profonde...")
                                
                                # Lancement de TOUS les modules
                                mc = run_monte_carlo(df['Close'])
                                vis = get_vision_score(df)
                                soc = get_social_hype(s)
                                whl, wh_msg = check_whale(df)
                                trend, _ = get_tech_indicators(df)
                                
                                # Le Conseil
                                council = consult_council(s, row['RSI'], mc, vis, soc, whl, trend)
                                
                                fast_log(f"🧠 **{s}:** MC:{mc:.2f} | Vis:{vis:.2f} | {wh_msg} => {council['vote']}")
                                
                                if council['vote'] == "BUY":
                                    price = row['Close']
                                    # Mise ajustée par Kelly + Emotions
                                    bet = calculate_kelly_bet(0.7, brain['cash'])
                                    
                                    if bet > 200:
                                        qty = bet / price
                                        # SL/TP issus du génome appris
                                        sl = price - (row['ATR'] * brain['genome']['sl_mult'])
                                        tp = price + (row['ATR'] * brain['genome']['tp_mult'])
                                        
                                        brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                                        brain['cash'] -= bet
                                        
                                        msg = {
                                            "title": f"🌌 ACHAT SINGULARITÉ : {s}",
                                            "description": council['reason'],
                                            "color": 0x2ecc71,
                                            "fields": [
                                                {"name": "Analyse", "value": f"Vis:{vis:.2f} | MC:{mc:.2f}", "inline": True},
                                                {"name": "Gestion", "value": f"Mise: {bet:.0f}$", "inline": True}
                                            ]
                                        }
                                        send_trade_alert(msg)
                                        save_brain()
                        except: pass
            
            time.sleep(30) # Cycle de trading
        except: time.sleep(10)

@app.route('/')
def index(): return f"<h1>SINGULARITY V88</h1><p>{bot_status}</p><p>Genome: {brain['genome']}</p>"

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
