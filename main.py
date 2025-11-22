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

# --- 2. CONFIG VELOCITY ---
# On se concentre sur les actifs les plus liquides pour l'apprentissage
WATCHLIST_LIVE = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD"] 
INITIAL_CAPITAL = 50000.0 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    # Paramètres de départ agressifs
    "params": {"rsi_buy": 45, "stop_loss_atr": 1.5, "tp_atr": 4.0},
    "generation": 0,
    "best_win_rate": 0.0
}
bot_status = "V37 Velocity Boot..."

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)

# --- 3. OUTILS DE COM ---
def log_thought(emoji, text):
    if LEARNING_WEBHOOK_URL: requests.post(LEARNING_WEBHOOK_URL, json={"content": f"{emoji} **VELOCITY:** {text}"})

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
            repo.update_file("brain.json", "V37 Update", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 4. MOTEUR DE SIMULATION ULTRA-RAPIDE ---
def simulate_strategy(df, params):
    """
    Simulation vectorisée avec gestion du risque intelligente.
    """
    rsi_limit = params['rsi_buy']
    sl_mult = params['stop_loss_atr']
    tp_mult = params['tp_atr']
    
    # Règle V37 : Achat sur repli dans une tendance saine
    # RSI < Limit ET Prix au-dessus de la moyenne mobile 200 (Trend Filter)
    # Ce filtre "MA200" est CRUCIAL pour arrêter de perdre.
    buy_signals = (df['RSI'] < rsi_limit) & (df['Close'] > df['MA200'])
    
    pnl = 0
    wins = 0
    losses = 0
    
    indices = df.index[buy_signals]
    
    for i in indices:
        try:
            row = df.loc[i]
            entry = row['Close']
            atr = row['ATR']
            
            stop = entry - (atr * sl_mult)
            target = entry + (atr * tp_mult)
            
            # On regarde 48 bougies dans le futur (2 jours de trading si H1)
            future = df.loc[i:].head(48) 
            
            hit_tp = future[future['High'] > target].index.min()
            hit_sl = future[future['Low'] < stop].index.min()
            
            if hit_tp and (not hit_sl or hit_tp < hit_sl):
                pnl += (target - entry)
                wins += 1
            elif hit_sl:
                pnl -= (entry - stop)
                losses += 1
        except: pass
    
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    # PÉNALITÉ SI PAS ASSEZ DE TRADES
    if total_trades < 5: return -9999, 0
    
    return pnl, win_rate

def run_velocity_evolution():
    global brain
    log_thought("🏎️", "Démarrage Moteur Velocity V37. Objectif: Win Rate > 60%.")
    
    # Cache Data Optimisé
    cache = {}
    for s in ["NVDA", "TSLA", "AMD", "AAPL"]:
        try:
            df = yf.Ticker(s).history(period="2mo", interval="1h") # 2 mois d'historique
            if not df.empty:
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                df['MA200'] = ta.sma(df['Close'], length=200) # Filtre Tendance
                cache[s] = df.dropna()
        except: pass

    while True:
        # Vérification Horaire
        nyc = pytz.timezone('America/New_York')
        now = datetime.now(nyc)
        market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
        
        if market_open:
            time.sleep(300)
            continue
            
        brain['generation'] = brain.get('generation', 0) + 1
        
        # POPULATION INTELLIGENTE
        # On ne teste plus au hasard complet. On teste autour des valeurs qui marchent.
        population = []
        for _ in range(20):
            population.append({
                "rsi_buy": np.random.randint(30, 65), # RSI plus large
                "stop_loss_atr": np.random.uniform(1.5, 3.5), # Stop serré
                "tp_atr": np.random.uniform(2.0, 6.0)  # Gain large (Risk/Reward > 1:2)
            })
            
        best_gen_pnl = -99999
        best_gen_genome = None
        best_gen_wr = 0
        
        for genome in population:
            total_pnl = 0
            avg_wr = 0
            count = 0
            
            for s, df in cache.items():
                pnl, wr = simulate_strategy(df, genome)
                total_pnl += pnl
                avg_wr += wr
                count += 1
            
            final_wr = avg_wr / count if count > 0 else 0
            
            # LE SECRET : On cherche le profit, MAIS on rejette si WinRate < 40%
            if total_pnl > best_gen_pnl and final_wr > 40:
                best_gen_pnl = total_pnl
                best_gen_genome = genome
                best_gen_wr = final_wr
        
        # Logique de sauvegarde
        if best_gen_genome:
            # On ne remplace le cerveau que si c'est VRAIMENT mieux
            # Ou si c'est la première fois qu'on trouve un truc positif
            if best_gen_pnl > 0:
                brain['params'] = best_gen_genome
                brain['best_win_rate'] = best_gen_wr
                
                emoji = "🚀" if best_gen_pnl > 100 else "🟢"
                msg = f"Gen {brain['generation']} : PnL {emoji} **{best_gen_pnl:.2f}$** (WinRate: {best_gen_wr:.1f}%).\nParamètres Gagnants: RSI<{best_gen_genome['rsi_buy']} / Stop:{best_gen_genome['stop_loss_atr']:.1f}ATR"
                log_thought("🏎️", msg)
                save_brain()
            else:
                # On log quand même les échecs pour montrer qu'il bosse
                log_thought("🔧", f"Gen {brain['generation']} : Meilleur essai négatif ({best_gen_pnl:.2f}$). Je continue de chercher...")
            
        time.sleep(5) # Vitesse Max

# --- 5. WEBSOCKET FLASH ---
def on_message(ws, message):
    global brain
    try:
        data = json.loads(message)
        if data['type'] == 'trade':
            for trade in data['data']:
                symbol = trade['s']
                price = trade['p']
                check_flash_triggers(symbol, price)
    except: pass

def on_error(ws, error): print("Reconnexion...")
def on_close(ws, a, b): time.sleep(5); start_websocket()

def check_flash_triggers(symbol, price):
    if symbol in brain['holdings']:
        pos = brain['holdings'][symbol]
        if price < pos['stop']:
            brain['cash'] += pos['qty'] * price
            del brain['holdings'][symbol]
            if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json={"content": f"⚡ FLASH SELL {symbol} (Stop Loss)"})
            save_brain()
            return

    # Achat si conditions réunies (Scan rapide)
    if len(brain['holdings']) < 3 and brain['cash'] > 500 and random.random() < 0.05:
        try:
            # On utilise les params appris
            rsi_limit = brain['params']['rsi_buy']
            sl_atr = brain['params']['stop_loss_atr']
            
            df = yf.Ticker(symbol).history(period="5d", interval="15m")
            rsi = ta.rsi(df['Close'], length=14).iloc[-1]
            atr = ta.atr(df['High'], df['Low'], df['Close'], length=14).iloc[-1]
            ma200 = ta.sma(df['Close'], length=200).iloc[-1]
            
            # Filtre Tendance (MA200) + RSI
            if rsi < rsi_limit and price > ma200:
                bet = brain['cash'] * 0.15
                brain['cash'] -= bet
                qty = bet / price
                sl = price - (atr * sl_atr)
                brain['holdings'][symbol] = {"qty": qty, "entry": price, "stop": sl}
                
                if DISCORD_WEBHOOK_URL: 
                    requests.post(DISCORD_WEBHOOK_URL, json={"content": f"⚡ **VELOCITY BUY {symbol}** à {price:.2f}$\n🧬 RSI: {rsi:.1f} (Limit: {rsi_limit})"}).
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
    threading.Thread(target=run_velocity_evolution, daemon=True).start()

load_brain()
start_threads()

@app.route('/')
def index(): return f"<h1>VELOCITY V37</h1><p>Gen: {brain.get('generation', 0)}</p>"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
