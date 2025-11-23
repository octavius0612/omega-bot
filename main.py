import websocket
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
from flask import Flask
from datetime import datetime, time as dtime
import pytz
from github import Github

app = Flask(__name__)

# --- 1. CLÉS DU POUVOIR ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")      # Trading
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL")  # Coeur
LEARNING_WEBHOOK_URL = os.environ.get("LEARNING_WEBHOOK_URL")    # Cerveau (Flux)
SUMMARY_WEBHOOK_URL = os.environ.get("SUMMARY_WEBHOOK_URL")      # Synthèse (Bilan)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

# --- 2. CONFIGURATION ---
WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR", "GOOG", "META"]
INITIAL_CAPITAL = 50000.0 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "q_table": {}, 
    "karma": {s: 10.0 for s in WATCHLIST},
    "emotions": {"confidence": 50.0, "stress": 20.0, "euphoria": 0.0},
    "best_params": {"rsi_buy": 30, "sl": 2.0}, # Paramètres évolutifs
    "learning_stats": {"tests": 0, "wins": 0}
}
bot_status = "Activation du Protocole DIEU..."
log_queue = queue.Queue()
short_term_memory = [] # Pour le résumé

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- 3. LOGGING INTELLIGENT ---
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
                msg_block = "\n".join(buffer[:15])
                buffer = buffer[15:]
                if LEARNING_WEBHOOK_URL:
                    requests.post(LEARNING_WEBHOOK_URL, json={"content": msg_block})
                last_send = time.time()
            time.sleep(0.5)
        except: time.sleep(1)

def send_summary(msg):
    if SUMMARY_WEBHOOK_URL: requests.post(SUMMARY_WEBHOOK_URL, json=msg)

def send_trade_alert(msg):
    if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json=msg)

def run_heartbeat():
    while True:
        if HEARTBEAT_WEBHOOK_URL: requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"})
        time.sleep(30)

# --- 4. MÉMOIRE ---
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
            repo.update_file("brain.json", "God Save", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 5. MODULES TECHNIQUES AVANCÉS ---
def calculate_fibonacci(df):
    high = df['High'].max()
    low = df['Low'].min()
    fib_618 = high - ((high - low) * 0.618)
    return fib_618

def check_whale(df):
    vol = df['Volume'].iloc[-1]
    avg = df['Volume'].rolling(20).mean().iloc[-1]
    return (vol > avg * 2.5), f"Vol x{vol/avg:.1f}"

def get_social_hype(symbol):
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).json()
        msgs = r['messages']
        txt = " ".join([m['body'] for m in msgs[:10]])
        polarity = TextBlob(txt).sentiment.polarity
        return polarity
    except: return 0

def run_monte_carlo(prices):
    returns = prices.pct_change().dropna()
    sims = prices.iloc[-1] * (1 + np.random.normal(returns.mean(), returns.std(), (1000, 10)))
    prob = np.sum(sims[:, -1] > prices.iloc[-1]) / 1000
    return prob

def get_vision_score(df, symbol):
    try:
        buf = io.BytesIO()
        mpf.plot(df.tail(50), type='candle', style='nightclouds', savefig=buf)
        buf.seek(0)
        img = Image.open(buf)
        res = model.generate_content(["Analyse chartiste visuelle (patterns). Score 0.0-1.0 ?", img])
        return float(res.text.strip())
    except: return 0.5

# --- 6. CERVEAU CENTRAL (LE CONSEIL) ---
def consult_god_council(symbol, rsi, fibo, whale, social, mc_prob, vision):
    prompt = f"""
    CONSEIL SUPRÊME POUR {symbol}.
    
    DONNÉES :
    1. Technique: RSI {rsi:.1f}, Fibo 61.8% (Support).
    2. Flux: Baleine={whale}, Social={social:.2f}.
    3. Futur (Quantique): {mc_prob*100:.1f}% hausse.
    4. Vision: {vision:.2f}/1.0.
    
    DÉBAT ENTRE IAs :
    - Chartiste: "La vision et Fibo disent..."
    - Quant: "Les stats Monte Carlo disent..."
    - Psychologue: "Le sentiment social dit..."
    
    DECISION FINALE (OUI/NON) et SCORE (0-100).
    JSON: {{"vote": "OUI", "score": 85, "reason": "..."}}
    """
    try:
        res = model.generate_content(prompt)
        return json.loads(res.text.replace("```json","").replace("```",""))
    except: return {"vote": "NON", "score": 0}

# --- 7. PSYCHOLOGIE & GESTION ---
def update_emotions(pnl):
    e = brain['emotions']
    if pnl > 0:
        e['confidence'] = min(e['confidence']+5, 100)
        e['stress'] = max(e['stress']-5, 0)
    else:
        e['confidence'] = max(e['confidence']-10, 10)
        e['stress'] = min(e['stress']+10, 100)

def calculate_position_size(score):
    # Formule Kelly simplifiée modérée par les émotions
    e = brain['emotions']
    
    # Si stressé, on divise par 2
    stress_factor = 0.5 if e['stress'] > 70 else 1.0
    # Si confiant, on multiplie par 1.2
    conf_factor = 1.2 if e['confidence'] > 80 else 1.0
    
    base_bet = 500.0 * (score/100) # Plus le score IA est haut, plus on mise
    return base_bet * stress_factor * conf_factor

# --- 8. MODULE APPRENTISSAGE CONTINU ---
def run_learning_loop():
    global brain, short_term_memory
    cache = {}
    
    while True:
        try:
            # A. TEST RAPIDE (Toutes les minutes)
            s = random.choice(WATCHLIST)
            
            # Mise à jour data
            if s not in cache or random.random() < 0.2:
                try:
                    df = yf.Ticker(s).history(period="1mo", interval="1h")
                    if not df.empty:
                        df['RSI'] = ta.rsi(df['Close'], length=14)
                        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                        cache[s] = df.dropna()
                except: pass
            
            if s in cache:
                df = cache[s]
                # On teste une hypothèse
                t_rsi = random.randint(20, 50)
                t_sl = round(random.uniform(1.5, 4.0), 1)
                
                # Simulation rapide
                pnl = 0
                trades = 0
                for i in range(len(df)-50, len(df)-1):
                    if df.iloc[i]['RSI'] < t_rsi:
                        entry = df.iloc[i]['Close']
                        sl = entry - (df.iloc[i]['ATR'] * t_sl)
                        # Vérif futur proche
                        fut = df.iloc[i+1:i+6]
                        if not fut.empty:
                            if fut['Low'].min() < sl: pnl -= (entry - sl)
                            else: pnl += (fut['Close'].iloc[-1] - entry)
                            trades += 1
                
                # Feedback Flux
                emoji = "✅" if pnl > 0 else "❌"
                fast_log(f"🧪 Test {s} (RSI<{t_rsi}, SL={t_sl}) -> {emoji} {pnl:.1f}$")
                
                short_term_memory.append({"pnl": pnl, "rsi": t_rsi, "sl": t_sl, "win": pnl>0})
                
                # Si super résultat, on adopte
                if pnl > 500:
                    brain['best_params'] = {"rsi_buy": t_rsi, "sl": t_sl}
                    save_brain()

            # B. BILAN (Toutes les 10 iterations)
            if len(short_term_memory) >= 10:
                wins = sum(1 for x in short_term_memory if x['win'])
                best = max(short_term_memory, key=lambda x: x['pnl'])
                
                msg = {
                    "embeds": [{
                        "title": "🎓 RAPPORT DU PROFESSEUR",
                        "color": 0xFFD700,
                        "fields": [
                            {"name": "Tests", "value": "10", "inline": True},
                            {"name": "Succès", "value": f"{wins}0%", "inline": True},
                            {"name": "Meilleur Paramètre", "value": f"RSI < {best['rsi']} (PnL +{best['pnl']:.0f}$)", "inline": False}
                        ]
                    }]
                }
                send_summary(msg)
                short_term_memory = []
            
            time.sleep(60)
        except Exception as e:
            print(f"Learn Error: {e}")
            time.sleep(60)

# --- 9. MOTEUR TRADING ---
def run_trading():
    global brain, bot_status
    load_brain()
    
    fast_log("🌌 **GOD PROTOCOL:** Tous les systèmes en ligne.")
    
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if market_open:
                bot_status = "🟢 LIVE TRADING"
                
                # VENTES
                for s in list(brain['holdings'].keys()):
                    pos = brain['holdings'][s]
                    curr = yf.Ticker(s).fast_info['last_price']
                    
                    exit = None
                    if curr < pos['stop']: exit = "STOP LOSS"
                    elif curr > pos['tp']: exit = "TAKE PROFIT"
                    
                    if exit:
                        pnl = (curr - pos['entry']) * pos['qty']
                        brain['cash'] += pos['qty'] * curr
                        del brain['holdings'][s]
                        update_emotions(pnl)
                        
                        col = 0x2ecc71 if pnl > 0 else 0xe74c3c
                        send_trade_alert({"embeds": [{"title": f"VENTE {s} ({exit})", "description": f"PnL: {pnl:.2f}$", "color": col}]})
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
                            
                            # Utilisation des paramètres appris par le "Professor"
                            if row['RSI'] < brain['best_params']['rsi_buy']:
                                
                                # Analyse Approfondie (Lourde)
                                whale, _ = check_whale(df)
                                social = get_social_hype(s)
                                fibo = calculate_fibonacci(df)
                                mc = run_monte_carlo(df['Close'])
                                
                                # Si Monte Carlo valide (Maths), on demande à l'IA
                                if mc > 0.60:
                                    vis = get_vision_score(df, s)
                                    council = consult_god_council(s, row['RSI'], fibo, whale, social, mc, vis)
                                    
                                    if council['vote'] == "OUI":
                                        price = row['Close']
                                        bet = calculate_position_size(council['score'])
                                        qty = bet / price
                                        sl = price - (row['ATR'] * brain['best_params']['sl'])
                                        tp = price + (row['ATR'] * 3.0)
                                        
                                        brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                                        brain['cash'] -= bet
                                        
                                        send_trade_alert({
                                            "embeds": [{
                                                "title": f"🌌 ACHAT DIVIN : {s}",
                                                "description": council['reason'],
                                                "color": 0x2ecc71,
                                                "fields": [
                                                    {"name": "Vision", "value": f"{vis:.2f}", "inline": True},
                                                    {"name": "Quantique", "value": f"{mc:.2f}", "inline": True},
                                                    {"name": "Social", "value": f"{social:.2f}", "inline": True}
                                                ]
                                            }]
                                        })
                                        save_brain()
                        except: pass
            else:
                bot_status = "🌙 Veille"
            
            time.sleep(60)
        except: time.sleep(10)

@app.route('/')
def index(): return f"<h1>GOD PROTOCOL V66</h1><p>{bot_status}</p>"

def start_threads():
    threading.Thread(target=run_trading, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()
    threading.Thread(target=logger_worker, daemon=True).start()
    threading.Thread(target=run_learning_loop, daemon=True).start()

load_brain()
start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
