import websocket
import os
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
from scipy.stats import linregress
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

# --- 2. CONFIG ALPHA ---
WATCHLIST_LIVE = ["NVDA", "TSLA", "AMD", "COIN", "MSTR", "AAPL"] 
INITIAL_CAPITAL = 50000.0 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    # La "Knowledge Base" : La mémoire des situations
    # Format: "RSI_LOW|TREND_UP|VOL_LOW": {"wins": 5, "losses": 1}
    "knowledge_base": {},
    "total_xp": 0
}
bot_status = "V39 Alpha Zero Boot..."

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)

# --- 3. OUTILS ---
def log_thought(emoji, text):
    if LEARNING_WEBHOOK_URL: requests.post(LEARNING_WEBHOOK_URL, json={"content": f"{emoji} **ALPHA:** {text}"})

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
        loaded = json.loads(c.decoded_content.decode())
        # Fusion intelligente pour ne pas perdre l'expérience
        if "knowledge_base" in loaded:
            brain["knowledge_base"] = loaded["knowledge_base"]
            brain["total_xp"] = loaded.get("total_xp", 0)
    except: pass

def save_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        content = json.dumps(brain, indent=4)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Alpha Learning", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 4. LE MOTEUR DE RECONNAISSANCE DE PATTERNS ---
def get_market_state(row, df_subset):
    """
    Transforme les chiffres en une "Signature" unique.
    Exemple: "RSI:BAS|PENTE:HAUSSE|VOL:CALME"
    """
    # 1. RSI
    rsi = row['RSI']
    if rsi < 30: rsi_state = "LOW"
    elif rsi > 70: rsi_state = "HIGH"
    else: rsi_state = "MID"
    
    # 2. Pente (Trend) - Calculée sur les 5 dernières bougies
    try:
        y = df_subset['Close'].values
        x = np.arange(len(y))
        slope, _, _, _, _ = linregress(x, y)
        if slope > 0.5: trend_state = "BULL_STRONG"
        elif slope > 0: trend_state = "BULL_WEAK"
        elif slope < -0.5: trend_state = "BEAR_STRONG"
        else: trend_state = "BEAR_WEAK"
    except: trend_state = "FLAT"
    
    # 3. Volatilité (ATR relatif au prix)
    atr_pct = (row['ATR'] / row['Close']) * 100
    if atr_pct > 1.5: vol_state = "HIGH"
    else: vol_state = "LOW"
    
    # Signature Unique
    return f"{rsi_state}|{trend_state}|{vol_state}"

def run_alpha_learning():
    """
    Apprend en analysant le passé. Il ne cherche pas des paramètres,
    il cherche des CONFIGURATIONS GAGNANTES.
    """
    global brain
    log_thought("🧠", "Démarrage Alpha Zero. Je cartographie les patterns gagnants...")
    
    cache = {}
    for s in ["NVDA", "TSLA", "MSTR", "AMD"]:
        try:
            df = yf.Ticker(s).history(period="2mo", interval="1h")
            if not df.empty:
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                cache[s] = df.dropna()
        except: pass

    while True:
        nyc = pytz.timezone('America/New_York')
        now = datetime.now(nyc)
        if now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0):
            time.sleep(300)
            continue
            
        # APPRENTISSAGE PROFOND
        # On parcourt l'historique trade par trade
        new_xp = 0
        
        for s, df in cache.items():
            # On prend 100 points au hasard dans le passé pour apprendre
            sample_indices = random.sample(range(20, len(df)-24), min(100, len(df)-50))
            
            for i in sample_indices:
                # Contexte à l'instant T
                current_row = df.iloc[i]
                subset = df.iloc[i-5:i+1] # Pour calculer la pente
                
                signature = get_market_state(current_row, subset)
                
                # Résultat Futur (24h plus tard)
                future_row = df.iloc[i+12] # +12 heures (environ)
                pnl_pct = (future_row['Close'] - current_row['Close']) / current_row['Close']
                
                # Enregistrement dans la base de connaissance
                if signature not in brain['knowledge_base']:
                    brain['knowledge_base'][signature] = {"wins": 0, "total": 0}
                
                brain['knowledge_base'][signature]["total"] += 1
                if pnl_pct > 0.01: # Si gain > 1%
                    brain['knowledge_base'][signature]["wins"] += 1
                
                new_xp += 1
        
        brain['total_xp'] += new_xp
        
        # Analyse des découvertes
        best_pattern = max(brain['knowledge_base'], key=lambda k: brain['knowledge_base'][k]['wins'] if brain['knowledge_base'][k]['total'] > 5 else 0)
        stats = brain['knowledge_base'][best_pattern]
        wr = (stats['wins'] / stats['total'] * 100) if stats['total'] > 0 else 0
        
        msg = f"Cycle terminé (+{new_xp} simulations).\nPattern Roi: **{best_pattern}**\nWin Rate Historique: **{wr:.1f}%** ({stats['wins']}/{stats['total']})"
        log_thought("🎓", msg)
        
        save_brain()
        time.sleep(60) # Pause

# --- 5. WEBSOCKET & TRADING (APPLICATION DU SAVOIR) ---
def on_message(ws, message):
    # (Code WebSocket simplifié pour l'exemple, reste actif)
    pass 

def on_error(ws, error): print("Reconnexion...")
def on_close(ws, a, b): time.sleep(5); start_websocket()

def start_websocket():
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}",
                              on_message = on_message, on_error = on_error, on_close = on_close)
    ws.on_open = lambda ws: [ws.send(json.dumps({"type": "subscribe", "symbol": s})) for s in WATCHLIST_LIVE]
    ws.run_forever()

def run_trading_logic():
    """
    Applique les connaissances en temps réel
    """
    global brain, bot_status
    load_brain()
    
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if not market_open:
                bot_status = "🌙 Mode Apprentissage"
                time.sleep(60)
                continue
                
            bot_status = "🟢 Recherche Patterns Connus..."
            
            # Gestion Ventes (Classique)
            for s in list(brain['holdings'].keys()):
                pos = brain['holdings'][s]
                curr = yf.Ticker(s).fast_info['last_price']
                if curr < pos['stop']:
                    brain['cash'] += pos['qty'] * curr
                    del brain['holdings'][s]
                    if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json={"content": f"🔴 VENTE {s}"})
                    save_brain()

            # Scan Achats INTELLIGENT
            if len(brain['holdings']) < 3:
                for s in WATCHLIST_LIVE:
                    if s in brain['holdings']: continue
                    
                    try:
                        # Analyse Temps Réel
                        df = yf.Ticker(s).history(period="5d", interval="15m")
                        if df.empty: continue
                        
                        df['RSI'] = ta.rsi(df['Close'], length=14)
                        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                        
                        current_row = df.iloc[-1]
                        subset = df.iloc[-6:]
                        
                        # Quelle est la signature actuelle ?
                        signature = get_market_state(current_row, subset)
                        
                        # Est-ce qu'on connait cette signature ?
                        memory = brain['knowledge_base'].get(signature)
                        
                        if memory and memory['total'] > 5: # Il faut au moins 5 exemples passés
                            win_rate = (memory['wins'] / memory['total']) * 100
                            
                            # ON ACHÈTE SEULEMENT SI LE PASSÉ PROUVE QUE ÇA GAGNE > 70% DU TEMPS
                            if win_rate > 70:
                                price = current_row['Close']
                                bet = brain['cash'] * 0.15
                                brain['cash'] -= bet
                                qty = bet / price
                                sl = price - (current_row['ATR'] * 2.0)
                                
                                brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl}
                                
                                msg = f"🟢 **ACHAT ALPHA : {s}**\nPattern Reconnu: `{signature}`\nProbabilité Historique: **{win_rate:.1f}%** ({memory['wins']} victoires connues)."
                                if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
                                save_brain()
                    except: pass
            
            time.sleep(60)
        except: time.sleep(10)

def start_threads():
    threading.Thread(target=start_websocket, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()
    threading.Thread(target=run_alpha_learning, daemon=True).start()
    threading.Thread(target=run_trading_logic, daemon=True).start()

load_brain()
start_threads()

@app.route('/')
def index(): return f"<h1>ALPHA ZERO V39</h1><p>Patterns Connus: {len(brain.get('knowledge_base', {}))}</p>"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
