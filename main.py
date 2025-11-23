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

# --- 2. CONFIGURATION LEVIATHAN ---
WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR", "GOOG", "META"]
INITIAL_CAPITAL = 50000.0 

# Cartographie des secteurs (Qui est le chef de qui ?)
SECTOR_MAP = {
    "NVDA": "QQQ", "AAPL": "QQQ", "MSFT": "QQQ", "AMD": "QQQ", "GOOG": "QQQ", "META": "QQQ", # Tech
    "TSLA": "XLY", "AMZN": "XLY", # Conso Discrétionnaire
    "COIN": "BTC-USD", "MSTR": "BTC-USD" # Crypto
}

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "q_table": {},
    "total_pnl": 0.0
}
bot_status = "Activation du Sonar..."
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

# --- 4. MÉMOIRE ---
def load_brain():
    global brain
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        c = repo.get_contents("brain.json")
        loaded = json.loads(c.decoded_content.decode())
        if "q_table" in loaded: brain.update(loaded)
    except: pass

def save_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        content = json.dumps(brain, indent=4)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Leviathan Save", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 5. MODULE BALEINE (WHALE WATCHER) ---
def check_whale_activity(df):
    """
    Détecte si le volume est anormalement élevé (signe institutionnel).
    """
    current_vol = df['Volume'].iloc[-1]
    avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
    
    # Si le volume est 2.5x supérieur à la moyenne
    if current_vol > (avg_vol * 2.5):
        return True, f"WHALE DETECTED (Vol x{current_vol/avg_vol:.1f})"
    return False, "Volume Normal"

# --- 6. MODULE SECTORIEL (CORRÉLATION) ---
def check_sector_trend(symbol):
    """
    Vérifie si le 'Père' du secteur est d'accord.
    """
    sector_ticker = SECTOR_MAP.get(symbol, "SPY") # SPY par défaut
    try:
        df = yf.Ticker(sector_ticker).history(period="5d", interval="15m")
        if df.empty: return True # On laisse passer si pas de data
        
        ema50 = ta.ema(df['Close'], length=50).iloc[-1]
        price = df['Close'].iloc[-1]
        
        if price > ema50:
            return True # Le secteur est haussier, feu vert
        else:
            return False # Le secteur chute, feu rouge
    except: return True

# --- 7. MODULE QUANTIQUE & VISION (Héritage V55) ---
def run_monte_carlo(prices, sims=1000):
    returns = prices.pct_change().dropna()
    mu, sigma = returns.mean(), returns.std()
    last = prices.iloc[-1]
    future_paths = last * (1 + np.random.normal(mu, sigma, (sims, 10)))
    final_prices = future_paths[:, -1]
    return np.sum(final_prices > last) / sims

def get_vision_score(df, symbol):
    try:
        buf = io.BytesIO()
        mpf.plot(df.tail(50), type='candle', style='nightclouds', savefig=buf)
        buf.seek(0)
        image = Image.open(buf)
        prompt = "Analyse chartiste. Score d'achat 0.0 à 1.0. Réponds chiffre uniquement."
        res = model.generate_content([prompt, image])
        return float(res.text.strip())
    except: return 0.5

# --- 8. GESTION (KELLY) ---
def calculate_kelly_bet(win_prob, capital):
    if win_prob <= 0.5: return 0
    kelly = win_prob - (1 - win_prob)
    return min(capital * kelly * 0.5, capital * 0.25)

# --- 9. MOTEUR TRADING LEVIATHAN ---
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
    
    fast_log("🐋 **LEVIATHAN:** Sonar actif. Recherche de mouvements institutionnels.")
    
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            if now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0):
                bot_status = "🟢 CHASSE EN COURS..."
                
                # VENTES
                for s in list(brain['holdings'].keys()):
                    pos = brain['holdings'][s]
                    curr = yf.Ticker(s).fast_info['last_price']
                    if curr < pos['stop']:
                        brain['cash'] += pos['qty'] * curr
                        del brain['holdings'][s]
                        if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json={"content": f"🔴 SELL {s} (Stop)"})
                        save_brain()
                    elif curr > pos['tp']:
                        brain['cash'] += pos['qty'] * curr
                        del brain['holdings'][s]
                        if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json={"content": f"🟢 SELL {s} (Target)"})
                        save_brain()

                # ACHATS AVEC FILTRE BALEINE & SECTEUR
                if len(brain['holdings']) < 5:
                    for s in WATCHLIST:
                        if s in brain['holdings']: continue
                        
                        df = get_live_data(s)
                        if df is None: continue
                        row = df.iloc[-1]
                        
                        # 1. CHECK SECTORIEL (Nouveau)
                        sector_ok = check_sector_trend(s)
                        if not sector_ok:
                            # fast_log(f"🛡️ {s} bloqué : Secteur baissier.")
                            continue 
                        
                        # 2. CHECK BALEINE (Nouveau)
                        is_whale, whale_msg = check_whale_activity(df)
                        
                        # 3. ANALYSE QUANTIQUE
                        mc_prob = run_monte_carlo(df['Close'])
                        
                        # CONDITION D'ENTRÉE RENFORCÉE :
                        # Il faut (Maths OK) ET (Baleine OU Vision excellente)
                        if mc_prob > 0.60:
                            vision_score = get_vision_score(df, s)
                            
                            # Bonus de score si une Baleine est détectée
                            whale_bonus = 0.2 if is_whale else 0.0
                            
                            final_score = (mc_prob * 0.4) + (vision_score * 0.4) + whale_bonus
                            
                            # fast_log(f"🔎 {s}: Score {final_score:.2f} | {whale_msg}")
                            
                            if final_score > 0.75:
                                price = row['Close']
                                bet = calculate_kelly_bet(final_score, brain['cash'])
                                
                                if bet > 200:
                                    qty = bet / price
                                    sl = price - (row['ATR'] * 2.0)
                                    tp = price + (row['ATR'] * 4.0)
                                    
                                    brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                                    brain['cash'] -= bet
                                    
                                    # Message complet
                                    whale_icon = "🐋" if is_whale else "🌊"
                                    msg = f"{whale_icon} **LEVIATHAN ENTRY : {s}**\nScore: {final_score:.2f}\nVolume: {whale_msg}\nSecteur: ✅ Confirmé"
                                    
                                    if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
                                    save_brain()
            else:
                bot_status = "🌙 Nuit (Analyse Fondamentale)"
                time.sleep(60)
            
            time.sleep(60)
        except Exception as e:
            time.sleep(10)

@app.route('/')
def index(): return f"<h1>LEVIATHAN V59</h1><p>{bot_status}</p>"

def start_threads():
    threading.Thread(target=run_trading, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()
    threading.Thread(target=logger_worker, daemon=True).start()

load_brain()
start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
