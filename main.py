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

# --- 1. TOUTES LES CLÉS ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL")
LEARNING_WEBHOOK_URL = os.environ.get("LEARNING_WEBHOOK_URL")
SUMMARY_WEBHOOK_URL = os.environ.get("SUMMARY_WEBHOOK_URL")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

# --- 2. CONFIGURATION ---
WATCHLIST_LIVE = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR"]
INITIAL_CAPITAL = 50000.0 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    # Paramètres évolutifs (L'IA les modifiera)
    "params": {"rsi_buy": 30, "sl_atr": 2.0, "tp_atr": 4.0},
    # Psychologie
    "emotions": {"confidence": 50.0, "stress": 20.0, "euphoria": 0.0},
    # Mémoire
    "karma": {s: 10.0 for s in WATCHLIST_LIVE},
    "trade_history": [],
    "total_pnl": 0.0,
    "last_prices": {}
}
bot_status = "Démarrage OMNI-GOD..."
log_queue = queue.Queue()

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- 3. LOGGING ---
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

def send_trade_alert(msg):
    if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json=msg)

def send_summary(msg):
    if SUMMARY_WEBHOOK_URL: requests.post(SUMMARY_WEBHOOK_URL, json=msg)

def run_heartbeat():
    while True:
        if HEARTBEAT_WEBHOOK_URL: requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"})
        time.sleep(30)

# --- 4. MÉMOIRE & PERSISTANCE ---
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
    return high - ((high - low) * 0.618)

def get_heikin_ashi(df):
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    return "UP" if ha_close.iloc[-1] > ha_open.iloc[-1] else "DOWN"

def check_whale(df):
    vol = df['Volume'].iloc[-1]
    avg = df['Volume'].rolling(20).mean().iloc[-1]
    return (vol > avg * 2.5), f"Vol x{vol/avg:.1f}" if avg > 0 else "Vol Normal"

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
        res = model.generate_content(["Score achat (0.0-1.0) chartiste ?", img])
        return float(res.text.strip())
    except: return 0.5

# --- 6. CERVEAU CENTRAL (CONSEIL + PSYCHO) ---
def consult_god_council(symbol, rsi, fibo, whale, social, mc, vision):
    # Ajustement selon l'humeur
    mood = brain['emotions']
    context_mood = f"Trader Stress: {mood['stress']}%, Confiance: {mood['confidence']}%"
    
    prompt = f"""
    CONSEIL SUPRÊME POUR {symbol}.
    ÉTAT PSYCHOLOGIQUE: {context_mood}.
    
    DONNÉES :
    1. Technique: RSI {rsi:.1f}, Fibo (Support).
    2. Flux: Baleine={whale}, Social={social:.2f}.
    3. Futur (Quantique): {mc*100:.1f}% hausse.
    4. Vision: {vision:.2f}/1.0.
    
    DÉBAT ENTRE IAs :
    - Chartiste: "La vision et Fibo disent..."
    - Quant: "Les stats Monte Carlo disent..."
    - Psychologue: "Le sentiment social et ton stress disent..."
    
    Vote Final (OUI/NON) pour Achat.
    JSON: {{"vote": "OUI", "score": 85, "reason": "Synthèse"}}
    """
    try:
        res = model.generate_content(prompt)
        return json.loads(res.text.replace("```json","").replace("```",""))
    except: return {"vote": "NON", "score": 0}

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
    # Ajustement psycho
    e = brain['emotions']
    factor = 0.5 if e['stress'] > 70 else (1.2 if e['confidence'] > 80 else 1.0)
    
    win_prob = score / 100.0
    if win_prob <= 0.5: return 0
    kelly = win_prob - (1 - win_prob)
    return min(capital * kelly * 0.5 * factor, capital * 0.25)

# --- 7. AUTO-AMÉLIORATION (NUIT) ---
def run_optimization_cycle():
    fast_log("🧬 **OPTIMISEUR:** Analyse des trades passés pour améliorer les paramètres...")
    
    # On prend les derniers trades
    history = brain.get('trade_history', [])[-20:]
    if not history: return
    
    prompt = f"""
    Historique récent: {json.dumps(history)}
    Paramètres actuels: {json.dumps(brain['params'])}
    
    Si beaucoup de pertes, rends les paramètres plus stricts.
    Si gains, optimise pour plus de profit.
    Réponds JSON uniquement avec les nouveaux params.
    """
    try:
        res = model.generate_content(prompt)
        new_params = json.loads(res.text.replace("```json","").replace("```",""))
        brain['params'] = new_params
        
        msg = {
            "embeds": [{
                "title": "🧬 ÉVOLUTION PARAMÉTRIQUE",
                "description": f"Nouveaux réglages adoptés :\n{json.dumps(new_params, indent=2)}",
                "color": 0x9b59b6
            }]
        }
        send_summary(msg)
        save_brain()
    except: pass

# --- 8. MOTEUR TEMPS RÉEL (WEBSOCKET) ---
def on_message(ws, message):
    global brain
    try:
        data = json.loads(message)
        if data['type'] == 'trade':
            for trade in data['data']:
                symbol = trade['s']
                price = trade['p']
                brain['last_prices'][symbol] = price
                
                # CHECK VENTE FLASH (STOP LOSS)
                if symbol in brain['holdings']:
                    pos = brain['holdings'][symbol]
                    if price < pos['stop']:
                        pnl = (price - pos['entry']) * pos['qty']
                        brain['cash'] += pos['qty'] * price
                        brain['total_pnl'] += pnl
                        del brain['holdings'][symbol]
                        
                        update_emotions(pnl)
                        brain['trade_history'].append({"pnl": pnl, "reason": "STOP_LOSS"})
                        
                        send_trade_alert({"content": f"⚡ **FLASH SELL {symbol}** (Stop) | PnL: {pnl:.2f}$"})
                        save_brain()
    except: pass

def on_error(ws, error): print("Reconnexion...")
def on_close(ws, a, b): time.sleep(5); start_websocket()

def start_websocket():
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}",
                              on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = lambda ws: [ws.send(json.dumps({"type": "subscribe", "symbol": s})) for s in WATCHLIST_LIVE]
    ws.run_forever()

# --- 9. BOUCLE ANALYSE (SCANNER) ---
def get_full_data(s):
    try:
        df = yf.Ticker(s).history(period="1mo", interval="1h")
        if df.empty: return None
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        return df
    except: return None

def run_scanner():
    global brain, bot_status
    load_brain()
    
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if not market_open:
                bot_status = "🌙 Nuit (Optimisation)"
                run_optimization_cycle()
                time.sleep(600) # Pause 10 min la nuit
                continue
            
            bot_status = "🟢 SCAN COMPLET"
            
            # ACHAT (Scan toutes les 60s, exécution flash via websocket pour la vente)
            if len(brain['holdings']) < 5:
                for s in WATCHLIST_LIVE:
                    if s in brain['holdings']: continue
                    
                    df = get_full_data(s)
                    if df is None: continue
                    row = df.iloc[-1]
                    
                    # Filtre Rapide (Paramètres Appris)
                    if row['RSI'] < brain['params']['rsi_buy']:
                        
                        # ANALYSE COMPLETE
                        whale, _ = check_whale(df)
                        social = get_social_hype(s)
                        mc = run_monte_carlo(df['Close'])
                        
                        if mc > 0.60:
                            vis = get_vision_score(df, s)
                            fibo = calculate_fibonacci(df)
                            
                            council = consult_god_council(s, row['RSI'], fibo, whale, social, mc, vis)
                            
                            fast_log(f"🧠 **{s}:** Score {council['score']} | Vote: {council['vote']}")
                            
                            if council['vote'] == "OUI":
                                price = row['Close']
                                bet = calculate_kelly_bet(council['score'], brain['cash'])
                                
                                if bet > 200:
                                    qty = bet / price
                                    sl = price - (row['ATR'] * brain['params']['sl_atr'])
                                    tp = price + (row['ATR'] * brain['params']['tp_atr'])
                                    
                                    brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                                    brain['cash'] -= bet
                                    
                                    send_trade_alert({
                                        "embeds": [{
                                            "title": f"🌌 ACHAT DIVIN : {s}",
                                            "description": council['reason'],
                                            "color": 0x2ecc71,
                                            "fields": [
                                                {"name": "Stats", "value": f"MC:{mc:.2f} | Vis:{vis:.2f}", "inline": True},
                                                {"name": "Gestion", "value": f"Mise: {bet:.0f}$", "inline": True}
                                            ]
                                        }]
                                    })
                                    save_brain()
            
            time.sleep(60)
        except: time.sleep(10)

@app.route('/')
def index(): return f"<h1>OMNI-GOD V72</h1><p>{bot_status}</p>"

def start_threads():
    threading.Thread(target=start_websocket, daemon=True).start()
    threading.Thread(target=run_scanner, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()
    threading.Thread(target=logger_worker, daemon=True).start()

load_brain()
start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
