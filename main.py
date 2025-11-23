import os
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
import requests
import google.generativeai as genai
import json
import time
import threading
import queue
import io
import random
# --- VISUALISATION ---
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
from PIL import Image
# --- SERVEUR ---
from flask import Flask
from datetime import datetime, time as dtime
import pytz
from github import Github

app = Flask(__name__)

# --- 1. CLÉS ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL")
LEARNING_WEBHOOK_URL = os.environ.get("LEARNING_WEBHOOK_URL")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")

# --- 2. CONFIG ---
WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR"]
INITIAL_CAPITAL = 50000.0 

# STRUCTURE MÉMOIRE COMPLEXE
brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "q_table": {}, 
    "total_pnl": 0.0,
    # NOUVEAU : Le Karma (Mémoire affective par action)
    "karma": {s: 10.0 for s in WATCHLIST}, # Tout le monde commence avec 10 points
    # NOUVEAU : Le Grimoire (Historique détaillé)
    "grimoire": [] 
}
bot_status = "Initialisation Mémoire Éternelle..."
log_queue = queue.Queue()

# MODÈLE FLASH
if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- 3. SYSTÈME DE LOG ---
def fast_log(text):
    log_queue.put(text)

def logger_worker():
    buffer = []
    last_send = time.time()
    while True:
        try:
            while not log_queue.empty():
                buffer.append(log_queue.get())
            if buffer and (len(buffer) > 5 or time.time() - last_send > 2.5):
                msg_block = "\n".join(buffer[:15])
                buffer = buffer[15:]
                if LEARNING_WEBHOOK_URL:
                    requests.post(LEARNING_WEBHOOK_URL, json={"content": msg_block})
                last_send = time.time()
            time.sleep(0.5)
        except: time.sleep(1)

def run_heartbeat():
    while True:
        if HEARTBEAT_WEBHOOK_URL: requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"})
        time.sleep(30)

# --- 4. GESTION AVANCÉE DE LA MÉMOIRE (HIPPOCAMPE) ---
def load_brain():
    global brain
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        c = repo.get_contents("brain.json")
        loaded = json.loads(c.decoded_content.decode())
        
        # Fusion intelligente pour ne rien perdre
        if "karma" in loaded: brain["karma"].update(loaded["karma"])
        if "grimoire" in loaded: brain["grimoire"] = loaded["grimoire"]
        if "q_table" in loaded: brain["q_table"] = loaded["q_table"]
        if "cash" in loaded: brain["cash"] = loaded["cash"]
        if "holdings" in loaded: brain["holdings"] = loaded["holdings"]
        
    except: pass

def save_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        content = json.dumps(brain, indent=4)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Eternal Save", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

def update_karma(symbol, pnl):
    """
    Punit ou Récompense une action selon le résultat.
    """
    current_karma = brain["karma"].get(symbol, 10.0)
    
    if pnl > 0:
        # Récompense : On aime bien cette action
        brain["karma"][symbol] = min(current_karma + 2.0, 20.0) # Max 20
        fast_log(f"🧠 **MÉMOIRE:** J'ai gagné sur {symbol}. J'aime cette action (Karma: {brain['karma'][symbol]:.1f}).")
    else:
        # Punition : On déteste cette action
        brain["karma"][symbol] = max(current_karma - 5.0, 0.0) # Min 0
        fast_log(f"🧠 **TRAUMATISME:** J'ai perdu sur {symbol}. Je vais m'en méfier (Karma: {brain['karma'][symbol]:.1f}).")

def check_memory_before_trade(symbol):
    """
    Consulte le passé avant d'agir.
    """
    karma = brain["karma"].get(symbol, 10.0)
    
    if karma < 5.0:
        fast_log(f"⛔ **BLOCAGE MÉMOIRE:** Mon Karma sur {symbol} est trop bas ({karma}). Je refuse de trader par sécurité.")
        return False
    return True

def add_to_grimoire(action, symbol, price, reason, pnl=0):
    """Écrit l'histoire dans le livre"""
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "action": action,
        "symbol": symbol,
        "price": price,
        "reason": reason,
        "pnl": pnl
    }
    brain["grimoire"].append(entry)
    # On garde seulement les 100 dernières entrées pour ne pas saturer GitHub
    if len(brain["grimoire"]) > 100:
        brain["grimoire"] = brain["grimoire"][-100:]

# --- 5. MODULE QUANTIQUE (MATHS) ---
def run_monte_carlo(prices, sims=1000):
    returns = prices.pct_change().dropna()
    mu, sigma = returns.mean(), returns.std()
    last = prices.iloc[-1]
    future_paths = last * (1 + np.random.normal(mu, sigma, (sims, 10)))
    final_prices = future_paths[:, -1]
    return np.sum(final_prices > last) / sims

def get_q_score(rsi, trend):
    state = f"RSI{int(rsi/10)*10}|{trend}"
    return brain['q_table'].get(state, 0.0)

# --- 6. MODULE VISION ---
def get_vision_score(df, symbol):
    try:
        buf = io.BytesIO()
        mpf.plot(df.tail(50), type='candle', style='nightclouds', savefig=buf)
        buf.seek(0)
        image = Image.open(buf)
        prompt = "Analyse technique visuelle. Donne un score d'achat de 0.0 à 1.0."
        res = model.generate_content([prompt, image])
        return float(res.text.strip())
    except: return 0.5

# --- 7. GESTION ---
def calculate_kelly_bet(win_prob, capital, symbol):
    # Le Karma influence la mise !
    karma = brain["karma"].get(symbol, 10.0)
    karma_factor = karma / 10.0 # Si karma=5, on divise la mise par 2
    
    if win_prob <= 0.5: return 0
    kelly_fraction = win_prob - (1 - win_prob)
    safe_kelly = kelly_fraction * 0.5
    bet_amount = capital * safe_kelly * karma_factor # Application du Karma
    return min(bet_amount, capital * 0.20)

# --- 8. MOTEUR TRADING ---
def get_live_data(s):
    try:
        df = yf.Ticker(s).history(period="1mo", interval="1h")
        if df.empty: return None
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['SMA200'] = ta.sma(df['Close'], length=200)
        return df
    except: return None

def run_trading():
    global brain, bot_status
    load_brain()
    
    fast_log("📚 **MÉMOIRE ÉTERNELLE:** Accès au Grimoire et au Karma.")
    
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            if now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0):
                bot_status = "🟢 SCAN AVEC MÉMOIRE..."
                
                # VENTES
                for s in list(brain['holdings'].keys()):
                    pos = brain['holdings'][s]
                    curr = yf.Ticker(s).fast_info['last_price']
                    
                    if curr < pos['stop']:
                        brain['cash'] += pos['qty'] * curr
                        del brain['holdings'][s]
                        
                        pnl = (curr - pos['entry']) * pos['qty']
                        update_karma(s, pnl) # MISE A JOUR DE LA MÉMOIRE
                        add_to_grimoire("SELL", s, curr, "Stop Loss", pnl)
                        
                        if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json={"content": f"🔴 VENTE {s} (Stop Loss) | PnL: {pnl:.2f}$"})
                        save_brain()
                        
                    elif curr > pos['tp']:
                        brain['cash'] += pos['qty'] * curr
                        del brain['holdings'][s]
                        
                        pnl = (curr - pos['entry']) * pos['qty']
                        update_karma(s, pnl) # MISE A JOUR DE LA MÉMOIRE
                        add_to_grimoire("SELL", s, curr, "Take Profit", pnl)
                        
                        if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json={"content": f"🟢 VENTE {s} (Take Profit) | PnL: {pnl:.2f}$"})
                        save_brain()

                # ACHATS
                if len(brain['holdings']) < 5:
                    for s in WATCHLIST:
                        if s in brain['holdings']: continue
                        
                        # 1. Appel à la Mémoire (Karma Check)
                        if not check_memory_before_trade(s):
                            continue # On passe à l'action suivante
                        
                        df = get_live_data(s)
                        if df is None: continue
                        row = df.iloc[-1]
                        
                        mc_prob = run_monte_carlo(df['Close'])
                        trend = "UP" if row['Close'] > row['SMA200'] else "DOWN"
                        
                        if mc_prob > 0.60:
                            vision_score = get_vision_score(df, s)
                            q_score = get_q_score(row['RSI'], trend)
                            
                            final_score = (mc_prob * 0.4) + (vision_score * 0.3) + (min(max(q_score/10,0),1) * 0.3)
                            
                            fast_log(f"🧠 **REFLEXION {s}:** Score {final_score:.2f} | Karma: {brain['karma'].get(s, 10):.1f}")
                            
                            if final_score > 0.75:
                                price = row['Close']
                                bet_size = calculate_kelly_bet(final_score, brain['cash'], s)
                                
                                if bet_size > 200:
                                    qty = bet_size / price
                                    sl = price - (row['ATR'] * 2.0)
                                    tp = price + (row['ATR'] * 4.0)
                                    
                                    brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                                    brain['cash'] -= bet_size
                                    
                                    add_to_grimoire("BUY", s, price, f"Score {final_score:.2f}")
                                    
                                    msg = f"🌌 **ACHAT V56 : {s}**\nScore: {final_score:.2f}\nKarma: {brain['karma'][s]}\nMise: {bet_size:.2f}$"
                                    if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
                                    save_brain()
            
            else:
                bot_status = "🌙 Nuit (Classement des souvenirs)"
                time.sleep(60)
            
            time.sleep(60)
        except Exception as e:
            print(e)
            time.sleep(10)

@app.route('/')
def index(): 
    karma_html = "<br>".join([f"{k}: {v:.1f}" for k,v in brain['karma'].items()])
    return f"<h1>MEMORY V56</h1><p>{bot_status}</p><h3>Karma Actions:</h3>{karma_html}"

def start_threads():
    threading.Thread(target=run_trading, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()
    threading.Thread(target=logger_worker, daemon=True).start()

load_brain()
start_threads()

if
