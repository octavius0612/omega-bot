import websocket
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
import random
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
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

# --- 2. CONFIGURATION AGRESSIVE ---
WATCHLIST_LIVE = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN"] 
INITIAL_CAPITAL = 50000.0 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "params": {"rsi_buy": 45, "stop_loss_atr": 2.0, "tp_atr": 3.0}, # Valeurs par défaut plus larges
    "last_prices": {},
    "generation": 0
}
bot_status = "V36 Aggressive Boot..."

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)

# --- 3. OUTILS ---
def log_thought(emoji, text):
    if LEARNING_WEBHOOK_URL: requests.post(LEARNING_WEBHOOK_URL, json={"content": f"{emoji} **V36:** {text}"})

def run_heartbeat():
    while True:
        try:
            if HEARTBEAT_WEBHOOK_URL: requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"})
            time.sleep(30)
        except: time.sleep(30)

def load_brain():
    global brain
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        c = repo.get_contents("brain.json")
        brain.update(json.loads(c.decoded_content.decode()))
    except: pass

def save_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        content = json.dumps(brain, indent=4)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "V36 Update", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 4. MOTEUR GÉNÉTIQUE CORRIGÉ (ANTI-STAGNATION) ---
def backtest_strategy(df, params):
    """
    Simulation vectorisée optimisée
    """
    rsi_limit = params['rsi_buy']
    sl_mult = params['stop_loss_atr']
    tp_mult = params['tp_atr']
    
    # RÈGLE V36 : On achète si RSI bas ET Tendance Haussière (Prix > EMA50)
    # Cela permet d'acheter plus souvent mais plus sûrement
    buy_signals = (df['RSI'] < rsi_limit) & (df['Close'] > df['EMA50'])
    
    pnl = 0
    trades = 0
    
    indices = df.index[buy_signals]
    
    for i in indices:
        try:
            row = df.loc[i]
            entry = row['Close']
            stop = entry - (row['ATR'] * sl_mult)
            target = entry + (row['ATR'] * tp_mult)
            
            future = df.loc[i:].head(30) # On regarde 30 bougies devant
            
            hit_tp = future[future['High'] > target].index.min()
            hit_sl = future[future['Low'] < stop].index.min()
            
            if hit_tp and (not hit_sl or hit_tp < hit_sl):
                pnl += (target - entry)
                trades += 1
            elif hit_sl:
                pnl -= (entry - stop)
                trades += 1
        except: pass
    
    # PÉNALITÉ D'INACTION : Si 0 trade, on donne un score horrible
    if trades == 0:
        return -1000, 0 
        
    return pnl, trades

def run_evolution_loop():
    global brain
    log_thought("🧬", "Démarrage Évolution V36 (Mode Agressif).")
    
    # Cache Data
    cache = {}
    for s in ["NVDA", "TSLA", "AAPL"]:
        try:
            df = yf.Ticker(s).history(period="1mo", interval="30m") # 30m pour moins de bruit
            if not df.empty:
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                df['EMA50'] = ta.ema(df['Close'], length=50)
                cache[s] = df.dropna()
        except: pass

    while True:
        # Check horaires (On entraîne la nuit/weekend)
        nyc = pytz.timezone('America/New_York')
        now = datetime.now(nyc)
        market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
        
        if market_open:
            time.sleep(300)
            continue
            
        brain['generation'] = brain.get('generation', 0) + 1
        
        # Population : On force des RSI plus hauts (jusqu'à 65)
        population = []
        for _ in range(15):
            population.append({
                "rsi_buy": np.random.randint(30, 65), # ON FORCE L'ACTION ICI
                "stop_loss_atr": np.random.uniform(1.5, 4.0),
                "tp_atr": np.random.uniform(2.0, 6.0)
            })
            
        scores = []
        for genome in population:
            total_pnl = 0
            total_trades = 0
            for s, df in cache.items():
                pnl, tr = backtest_strategy(df, genome)
                total_pnl += pnl
                total_trades += tr
            scores.append((total_pnl, total_trades, genome))
            
        # Tri par PnL
        scores.sort(key=lambda x: x[0], reverse=True)
        champion = scores[0]
        
        # On ne sauvegarde que si le champion a fait de l'argent ET des trades
        if champion[0] > -900: # -900 car mieux vaut une petite perte que l'inaction (-1000)
            brain['params'] = champion[2]
            
            emoji = "🟢" if champion[0] > 0 else "🔴"
            msg = f"Génération {brain['generation']} : PnL {emoji} **{champion[0]:.2f}$** ({champion[1]} trades).\nParams: RSI<{champion[2]['rsi_buy']} / SL={champion[2]['stop_loss_atr']:.1f}ATR"
            log_thought("🧬", msg)
            save_brain()
            
        time.sleep(10) # Vitesse rapide

# --- 5. WEBSOCKET FLASH ---
def on_message(ws, message):
    global brain
    try:
        data = json.loads(message)
        if data['type'] == 'trade':
            for trade in data['data']:
                symbol = trade['s']
                price = trade['p']
                brain['last_prices'][symbol] = price
                check_flash_triggers(symbol, price)
    except: pass

def on_error(ws, error): print("Reconnexion...")
def on_close(ws, a, b): time.sleep(5); start_websocket()

def check_flash_triggers(symbol, price):
    # 1. Stop Loss
    if symbol in brain['holdings']:
        pos = brain['holdings'][symbol]
        if price < pos['stop']:
            brain['cash'] += pos['qty'] * price
            del brain['holdings'][symbol]
            if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json={"content": f"⚡ FLASH SELL {symbol} (Stop Loss)"})
            save_brain()
            return

    # 2. Achat (Scan aléatoire pour économiser CPU)
    if len(brain['holdings']) < 3 and brain['cash'] > 500 and random.random() < 0.02:
        try:
            # On utilise les params appris par l'évolution
            rsi_limit = brain['params']['rsi_buy']
            sl_atr = brain['params']['stop_loss_atr']
            
            df = yf.Ticker(symbol).history(period="5d", interval="15m")
            rsi = ta.rsi(df['Close'], length=14).iloc[-1]
            atr = ta.atr(df['High'], df['Low'], df['Close'], length=14).iloc[-1]
            
            if rsi < rsi_limit:
                bet = brain['cash'] * 0.15
                brain['cash'] -= bet
                qty = bet / price
                sl = price - (atr * sl_atr)
                brain['holdings'][symbol] = {"qty": qty, "entry": price, "stop": sl}
                
                if DISCORD_WEBHOOK_URL: 
                    requests.post(DISCORD_WEBHOOK_URL, json={"content": f"⚡ **FLASH BUY {symbol}** à {price:.2f}$\n🧬 Paramètre Appris: RSI < {rsi_limit}"})
                save_brain()
        except: pass

def start_websocket():
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}",
                              on_message = on_message, on_error = on_error, on_close = on_close)
    ws.on_open = lambda ws: [ws.send(json.dumps({"type": "subscribe", "symbol": s})) for s in WATCHLIST_LIVE]
    ws.run_forever()

# --- 6. LANCEMENT ---
def start_threads():
    threading.Thread(target=start_websocket, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()
    threading.Thread(target=run_evolution_loop, daemon=True).start()

load_brain()
start_threads()

@app.route('/')
def index(): return f"<h1>V36 AGGRESSIVE</h1><p>Gen: {brain.get('generation', 0)}</p>"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
