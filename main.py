import websocket
import os
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
from scipy.stats import norm, linregress
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
matplotlib.use('Agg') # Mode sans écran pour serveur
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
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")      # Alertes Trading
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL")  # Battement Cœur
LEARNING_WEBHOOK_URL = os.environ.get("LEARNING_WEBHOOK_URL")    # Flux de Pensée
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR", "GOOG", "META"]
INITIAL_CAPITAL = 50000.0 
SIMULATION_COUNT = 2000 # Nombre de futurs simulés (Monte Carlo)

# LA MÉMOIRE TOTALE
brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    # Mémoire Émotionnelle
    "emotions": {"confidence": 50.0, "stress": 20.0, "euphoria": 0.0},
    # Mémoire Expérientielle (Karma des actions)
    "karma": {s: 10.0 for s in WATCHLIST}, 
    # Mémoire Traumatique (Configurations interdites)
    "black_box": [], 
    "total_pnl": 0.0,
    "last_prices": {}
}

bot_state = {
    "status": "Éveil...",
    "last_decision": "Aucune",
    "last_log": "Initialisation des systèmes..."
}

log_queue = queue.Queue()
short_term_memory = []

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ==============================================================================
# 2. SYSTÈME DE LOGGING "MATRIX"
# ==============================================================================
def fast_log(text):
    log_queue.put(text)

def logger_worker():
    """Mitraillette à logs pour Discord"""
    buffer = []
    last_send = time.time()
    while True:
        try:
            while not log_queue.empty():
                buffer.append(log_queue.get())
            
            # Envoi groupé toutes les 1.5s ou si buffer plein
            if buffer and (len(buffer) > 5 or time.time() - last_send > 1.5):
                msg_block = "\n".join(buffer[:12])
                buffer = buffer[12:]
                if LEARNING_WEBHOOK_URL:
                    try: requests.post(LEARNING_WEBHOOK_URL, json={"content": msg_block})
                    except: pass
                last_send = time.time()
            time.sleep(0.2)
        except: time.sleep(1)

def send_trade_alert(embed):
    if DISCORD_WEBHOOK_URL:
        try: requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed]})
        except: pass

def run_heartbeat():
    while True:
        if HEARTBEAT_WEBHOOK_URL: requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"})
        time.sleep(30)

# ==============================================================================
# 3. PERSISTANCE (GITHUB)
# ==============================================================================
def load_brain():
    global brain
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        c = repo.get_contents("brain.json")
        loaded = json.loads(c.decoded_content.decode())
        # Fusion prudente
        if "cash" in loaded: brain["cash"] = loaded["cash"]
        if "karma" in loaded: brain["karma"] = loaded["karma"]
        if "black_box" in loaded: brain["black_box"] = loaded["black_box"]
        if "emotions" in loaded: brain["emotions"] = loaded["emotions"]
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

# --- SENS 1 : TECHNIQUE (Fibonacci & Heikin Ashi) ---
def get_technical_data(df):
    # Heikin Ashi
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    ha_trend = "HAUSSIER" if ha_close.iloc[-1] > ha_open.iloc[-1] else "BAISSIER"
    
    # Fibonacci (Retracement depuis le haut)
    high = df['High'].max()
    low = df['Low'].min()
    fib_618 = high - ((high - low) * 0.618)
    
    return ha_trend, fib_618

# --- SENS 2 : BALEINES (Volume) ---
def check_whale(df):
    try:
        vol = df['Volume'].iloc[-1]
        avg = df['Volume'].rolling(20).mean().iloc[-1]
        ratio = vol / avg if avg > 0 else 0
        return (ratio > 2.5), f"Vol x{ratio:.1f}"
    except: return False, "Vol --"

# --- SENS 3 : QUANTIQUE (Monte Carlo) ---
def run_monte_carlo(prices):
    try:
        returns = prices.pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        last = prices.iloc[-1]
        # Simulation vectorisée NumPy
        sims = last * (1 + np.random.normal(mu, sigma, (SIMULATION_COUNT, 10)))
        prob = np.sum(sims[:, -1] > last) / SIMULATION_COUNT
        return prob
    except: return 0.5

# --- SENS 4 : SOCIAL (StockTwits) ---
def get_social_sentiment(symbol):
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers).json()
        txt = " ".join([m['body'] for m in r['messages'][:10]])
        return TextBlob(txt).sentiment.polarity
    except: return 0

# --- SENS 5 : VISION (Gemini) ---
def get_vision_score(df):
    try:
        buf = io.BytesIO()
        mpf.plot(df.tail(60), type='candle', style='nightclouds', savefig=buf)
        buf.seek(0)
        img = Image.open(buf)
        res = model.generate_content(["Tu es un trader chartiste. Analyse ce graphique. Donne un score d'achat de 0.0 (Vente) à 1.0 (Achat Fort). Réponds UNIQUEMENT le chiffre.", img])
        return float(res.text.strip())
    except: return 0.5

# ==============================================================================
# 5. CERVEAU CENTRAL & PSYCHOLOGIE
# ==============================================================================

def get_market_signature(rsi, trend, vol):
    """Crée une empreinte unique de la situation"""
    return f"RSI{int(rsi/10)*10}|{trend}|{'HIGH' if vol else 'LOW'}"

def update_emotions(pnl):
    e = brain['emotions']
    if pnl > 0:
        e['confidence'] = min(e['confidence'] + 5, 100)
        e['stress'] = max(e['stress'] - 5, 0)
        e['euphoria'] += 5
    else:
        e['confidence'] = max(e['confidence'] - 10, 10)
        e['stress'] = min(e['stress'] + 10, 100)
        e['euphoria'] = 0

def calculate_kelly_bet(score, capital, symbol):
    # Facteur émotionnel
    e = brain['emotions']
    mood_factor = 0.5 if e['stress'] > 70 else (1.2 if e['confidence'] > 80 else 1.0)
    
    # Facteur Karma (Expérience passée sur ce titre)
    karma = brain['karma'].get(symbol, 10.0) / 10.0
    
    win_prob = score / 100.0
    if win_prob <= 0.5: return 0
    
    # Formule Kelly
    kelly = win_prob - (1 - win_prob)
    bet = capital * kelly * 0.5 * mood_factor * karma # Demi-Kelly sécurisé
    return min(bet, capital * 0.20) # Max 20%

def consult_council(s, rsi, mc, vis, soc, whale):
    """Le Débat des IAs"""
    mood = brain['emotions']
    prompt = f"""
    CONSEIL SUPRÊME POUR {s}.
    État Psychologique du Bot: Stress {mood['stress']}%, Confiance {mood['confidence']}%.
    
    DONNÉES:
    1. Maths (Quantique): {mc*100:.1f}% proba hausse.
    2. Vision (Yeux): Score {vis:.2f}/1.0.
    3. Social (Oreilles): Sentiment {soc:.2f}.
    4. Baleine (Toucher): {"OUI" if whale else "NON"}.
    5. Technique: RSI {rsi:.1f}.
    
    DÉBAT :
    Fais dialoguer 3 agents (Chartiste, Quant, Psycho) en 1 ligne chacun.
    
    DECISION :
    Vote Final (OUI/NON) et Score (0-100).
    JSON: {{"vote": "OUI", "score": 85, "reason": "Synthèse..."}}
    """
    try:
        res = model.generate_content(prompt)
        return json.loads(res.text.replace("```json","").replace("```",""))
    except: return {"vote": "WAIT", "score": 0, "reason": "Erreur Conseil"}

# ==============================================================================
# 6. APPRENTISSAGE CONTINU (DOJO)
# ==============================================================================
def run_learning_loop():
    global brain
    cache = {}
    fast_log("🧠 **OMNI-LEARN:** Module d'auto-amélioration actif.")
    
    while True:
        try:
            s = random.choice(WATCHLIST)
            # Update data (10% chance)
            if s not in cache or random.random() < 0.1:
                try:
                    df = yf.Ticker(s).history(period="1mo", interval="1h")
                    if not df.empty:
                        df['RSI'] = ta.rsi(df['Close'], length=14)
                        df['EMA50'] = ta.ema(df['Close'], length=50)
                        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                        cache[s] = df.dropna()
                except: pass
            
            if s in cache:
                df = cache[s]
                # On prend un point aléatoire du passé
                idx = random.randint(50, len(df)-10)
                row = df.iloc[idx]
                
                # On regarde ce qui s'est passé ensuite
                future = df.iloc[idx+1:idx+6]
                if future.empty: continue
                
                # Création Signature
                trend = "UP" if row['Close'] > row['EMA50'] else "DOWN"
                vol = row['Volume'] > df['Volume'].mean()
                sig = get_market_signature(row['RSI'], trend, vol)
                
                # Résultat Réel
                pnl = future['Close'].iloc[-1] - row['Close']
                outcome = "WIN" if pnl > 0 else "LOSS"
                
                # Si c'était une grosse perte, on l'ajoute à la Boîte Noire
                if pnl < - (row['Close']*0.05): # -5%
                    if sig not in brain['black_box']:
                        brain['black_box'].append(sig)
                        fast_log(f"⛔ **APPRENTISSAGE:** Config `{sig}` ajoutée à la Boîte Noire (Toxique).")
                
                # Simulation Visuelle pour le log
                if random.random() < 0.2: # Pas tout le temps
                    emoji = "✅" if outcome == "WIN" else "❌"
                    fast_log(f"🧪 **TEST {s}:** Config `{sig}` -> {emoji} {outcome}")

            time.sleep(5)
        except: time.sleep(10)

# ==============================================================================
# 7. MOTEUR TRADING LIVE
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
                bot_state['status'] = "🟢 LIVE OMNISCIENT"
                
                # 1. GESTION VENTES
                for s in list(brain['holdings'].keys()):
                    pos = brain['holdings'][s]
                    curr = brain['last_prices'].get(s, 0)
                    if curr == 0: continue # Pas de prix reçu encore
                    
                    exit_r = None
                    if curr < pos['stop']: exit_r = "STOP LOSS"
                    elif curr > pos['tp']: exit_r = "TAKE PROFIT"
                    
                    if exit_r:
                        pnl = (curr - pos['entry']) * pos['qty']
                        brain['cash'] += pos['qty'] * curr
                        brain['total_pnl'] += pnl
                        del brain['holdings'][s]
                        
                        update_emotions(pnl)
                        # Mise à jour Karma
                        brain['karma'][s] = min(brain['karma'][s] + (1 if pnl>0 else -2), 20)
                        
                        col = 0x2ecc71 if pnl > 0 else 0xe74c3c
                        send_trade_alert({
                            "title": f"{exit_r} {s}", 
                            "description": f"PnL: **{pnl:.2f}$**", 
                            "color": col
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
                            df['EMA50'] = ta.ema(df['Close'], length=50)
                            row = df.iloc[-1]
                            brain['last_prices'][s] = row['Close'] # Backup prix
                            
                            # CHECK BOITE NOIRE
                            trend = "UP" if row['Close'] > row['EMA50'] else "DOWN"
                            sig = get_market_signature(row['RSI'], trend, False)
                            if sig in brain['black_box']:
                                # fast_log(f"🛡️ Rejet {s} (Config `{sig}` en Boîte Noire)")
                                continue

                            # PRE-FILTRE
                            if row['RSI'] < 40:
                                # LANCEMENT ANALYSE LOURDE
                                mc = run_monte_carlo(df['Close'])
                                whale, wh_msg = check_whale(df)
                                social = get_social_sentiment(s)
                                vis = get_vision_score(df)
                                
                                # CONSEIL
                                council = consult_god_council(s, row['RSI'], mc, vis, social, whale)
                                bot_state['last_decision'] = f"{s}: {council['vote']}"
                                
                                fast_log(f"👁️ **ANALYSE {s}:** MC:{mc:.2f} | Vis:{vis:.2f} | Vote:{council['vote']}")
                                
                                if council['vote'] == "OUI":
                                    price = row['Close']
                                    bet = calculate_kelly_bet(council['score'], brain['cash'], s)
                                    
                                    if bet > 200:
                                        qty = bet / price
                                        sl = price - (row['ATR'] * 2.0)
                                        tp = price + (row['ATR'] * 3.5)
                                        
                                        brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                                        brain['cash'] -= bet
                                        
                                        send_trade_alert({
                                            "title": f"🌌 ACHAT OMNISCIENT : {s}",
                                            "description": f"**Raison:** {council['reason']}",
                                            "color": 0x2ecc71,
                                            "fields": [
                                                {"name": "Vision", "value": f"{vis:.2f}", "inline": True},
                                                {"name": "Quantique", "value": f"{mc:.2f}", "inline": True},
                                                {"name": "Mise", "value": f"{bet:.0f}$", "inline": True}
                                            ]
                                        })
                                        save_brain()
                        except: pass
            else:
                bot_state['status'] = "🌙 NUIT (Apprentissage)"
            
            time.sleep(60)
        except Exception as e:
            print(f"Err Loop: {e}")
            time.sleep(10)

# --- 8. WEBSOCKET (MILLISECONDE) ---
def on_message(ws, message):
    try:
        data = json.loads(message)
        if data['type'] == 'trade':
            for t in data['data']:
                s = t['s']
                p = t['p']
                brain['last_prices'][s] = p
                # Check Stop Loss Ultra Rapide
                if s in brain['holdings']:
                    if p < brain['holdings'][s]['stop']:
                        # Trigger Vente dans le thread principal via flag ou direct
                        pass # Simplifié ici, géré par main loop pour sécurité
    except: pass

def start_socket():
    ws = websocket.WebSocketApp(f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}", on_message=on_message)
    ws.run_forever()

# --- 9. DASHBOARD ---
@app.route('/')
def index():
    hold_html = "".join([f"<li>{s}: {d['qty']:.2f} @ {d['entry']:.2f}$</li>" for s,d in brain['holdings'].items()])
    return f"""
    <style>body{{background:#000;color:#0f0;font-family:monospace}}</style>
    <h1>OMNI-GOD V79</h1>
    <p>Status: {bot_state['status']}</p>
    <p>Capital: {brain['cash']:.2f}$</p>
    <p>Positions: <ul>{hold_html}</ul></p>
    <p>Log: {bot_state['last_log']}</p>
    """

def start_threads():
    threading.Thread(target=run_trading, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()
    threading.Thread(target=logger_worker, daemon=True).start()
    threading.Thread(target=run_learning_loop, daemon=True).start()
    threading.Thread(target=start_socket, daemon=True).start()

load_brain()
start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
