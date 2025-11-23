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

# --- 1. CLÉS ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL")
LEARNING_WEBHOOK_URL = os.environ.get("LEARNING_WEBHOOK_URL")
SUMMARY_WEBHOOK_URL = os.environ.get("SUMMARY_WEBHOOK_URL")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

# --- 2. CONFIG ---
WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR"]
INITIAL_CAPITAL = 50000.0 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    # Ici, on stocke le CODE source des stratégies inventées par l'IA
    "active_strategies": {}, 
    "total_pnl": 0.0,
    "evolution_log": []
}
bot_status = "Démarrage Évolution..."
log_queue = queue.Queue()
short_term_memory = []

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
            if buffer and (len(buffer) > 3 or time.time() - last_send > 1.5):
                msg_block = "\n".join(buffer[:10]) 
                buffer = buffer[10:]
                if LEARNING_WEBHOOK_URL:
                    requests.post(LEARNING_WEBHOOK_URL, json={"content": msg_block})
                last_send = time.time()
            time.sleep(0.5)
        except: time.sleep(1)

def send_summary(msg):
    if SUMMARY_WEBHOOK_URL: requests.post(SUMMARY_WEBHOOK_URL, json=msg)

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
            repo.update_file("brain.json", "Evolution Save", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 5. L'ARCHITECTE DE CODE (AUTO-CODING) ---
def generate_strategy_code():
    """Demande à Gemini d'écrire une nouvelle stratégie en Python"""
    
    indicators = "RSI, EMA50, EMA200, ATR, ADX, VOLUME, CLOSE, OPEN, HIGH, LOW"
    
    prompt = f"""
    Agis comme un Trader Algorithmique Expert.
    TA MISSION : Inventer une logique de trading court terme innovante.
    
    Données disponibles dans le dataframe 'df' (dernière ligne 'row') :
    {indicators}
    
    Écris un snippet Python qui définit une variable 'signal'.
    - signal = 100 (ACHAT)
    - signal = -100 (VENTE/STOP)
    - signal = 0 (RIEN)
    
    Sois créatif ! Utilise la volatilité, les croisements, ou le volume.
    Ne mets PAS de commentaires, juste le code.
    """
    try:
        res = model.generate_content(prompt)
        code = res.text.replace("```python", "").replace("```", "").strip()
        # Nettoyage basique pour sécurité
        if "import" in code or "os." in code: return None
        return code
    except: return None

def explain_improvement(old_pnl, new_pnl, code):
    """Demande à Gemini d'expliquer pourquoi la nouvelle stratégie est meilleure"""
    prompt = f"""
    Tu viens de créer une stratégie de trading.
    Ancienne Performance : {old_pnl}$
    Nouvelle Performance : {new_pnl}$
    
    Voici le code de la nouvelle stratégie :
    {code}
    
    Explique en 1 phrase simple ce que tu as appris et pourquoi c'est mieux.
    Exemple : "J'ai appris à utiliser l'ATR pour filtrer la volatilité, ce qui réduit les faux signaux."
    """
    try:
        res = model.generate_content(prompt)
        return res.text.strip()
    except: return "Amélioration statistique détectée."

# --- 6. MOTEUR DE TEST (BACKTEST) ---
def backtest_strategy(code, symbol):
    try:
        df = yf.Ticker(symbol).history(period="1mo", interval="1h")
        if df.empty: return -999
        
        # Calcul indicateurs
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        df['EMA200'] = ta.ema(df['Close'], length=200)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
        df = df.dropna()
        
        capital = 10000
        position = 0
        entry = 0
        trades = 0
        
        for i, row in df.iterrows():
            loc = {'row': row, 'signal': 0}
            try: exec(code, {}, loc)
            except: pass
            
            # Logique simple pour tester la qualité du signal d'entrée
            if position == 0 and loc['signal'] == 100:
                position = capital / row['Close']
                entry = row['Close']
                capital = 0
            elif position > 0:
                # Sortie fixe pour test standardisé
                if row['High'] > entry * 1.05 or row['Low'] < entry * 0.97:
                    capital = position * row['Close']
                    position = 0
                    trades += 1
                    
        final = capital + (position * df['Close'].iloc[-1])
        return final - 10000
    except: return -999

# --- 7. BOUCLE D'ÉVOLUTION ---
def run_evolution_loop():
    global brain
    fast_log("🧬 **GENESIS:** Démarrage du cycle d'auto-programmation.")
    
    while True:
        # On travaille quand le marché est calme ou fermé
        nyc = pytz.timezone('America/New_York')
        now = datetime.now(nyc)
        market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
        
        if not market_open or len(brain['active_strategies']) < 1:
            s = random.choice(WATCHLIST)
            
            # 1. Création
            code = generate_strategy_code()
            if code:
                # 2. Test
                pnl = backtest_strategy(code, s)
                
                # Comparaison avec la performance moyenne actuelle
                avg_pnl = 0 # Placeholder
                
                if pnl > 200: # Si la stratégie est viable
                    # 3. Explication (Méta-Cognition)
                    explanation = explain_improvement(0, pnl, code)
                    
                    # Stockage
                    strat_id = f"STRAT_{int(time.time())}"
                    brain['active_strategies'][strat_id] = {"code": code, "pnl": pnl, "symbol": s}
                    
                    # Rapport
                    msg = {
                        "embeds": [{
                            "title": "🧬 ÉVOLUTION DU CODE",
                            "color": 0x9b59b6,
                            "description": f"**J'ai réécrit mon propre code pour {s}.**",
                            "fields": [
                                {"name": "Performance Test", "value": f"+{pnl:.2f}$", "inline": True},
                                {"name": "Ce que j'ai appris", "value": explanation, "inline": False},
                                {"name": "Code Généré", "value": f"```python\n{code[:200]}...\n```", "inline": False}
                            ]
                        }]
                    }
                    send_summary(msg)
                    save_brain()
                else:
                    fast_log(f"🗑️ Code testé et rejeté (PnL: {pnl:.2f}$). J'essaie autre chose.")
            
            time.sleep(30)
        else:
            time.sleep(60)

# --- 8. TRADING LIVE (Utilise le code généré) ---
def run_trading():
    global brain, bot_status
    load_brain()
    
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if market_open:
                bot_status = "🟢 Trading Auto-Codé"
                
                # VENTES (Standard)
                for s in list(brain['holdings'].keys()):
                    # ... (Gestion classique SL/TP pour sécurité) ...
                    pass # Simplifié pour l'espace

                # ACHATS (Basés sur les stratégies générées)
                for s in WATCHLIST:
                    if s in brain['holdings']: continue
                    
                    # On cherche si on a une stratégie spécialisée pour cette action
                    # Ou on utilise une générique
                    try:
                        df = yf.Ticker(s).history(period="1mo", interval="15m")
                        if df.empty: continue
                        # Indicateurs
                        df['RSI'] = ta.rsi(df['Close'], length=14)
                        df['EMA50'] = ta.ema(df['Close'], length=50)
                        df['EMA200'] = ta.ema(df['Close'], length=200)
                        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                        df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
                        row = df.iloc[-1]
                        
                        # On teste toutes les stratégies actives
                        for strat_id, strat in brain['active_strategies'].items():
                            loc = {'row': row, 'signal': 0}
                            try:
                                exec(strat['code'], {}, loc)
                                if loc['signal'] == 100:
                                    # SIGNAL !
                                    price = row['Close']
                                    qty = 500 / price
                                    sl = price - (row['ATR'] * 2)
                                    tp = price + (row['ATR'] * 3)
                                    
                                    brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                                    brain['cash'] -= 500
                                    
                                    if DISCORD_WEBHOOK_URL:
                                        requests.post(DISCORD_WEBHOOK_URL, json={"content": f"🧬 **ACHAT ÉVOLUTIF : {s}**\nStratégie: `{strat_id}`"})
                                    save_brain()
                                    break
                            except: pass
                    except: pass
            else:
                bot_status = "🌙 Nuit (Codage)"
            
            time.sleep(60)
        except: time.sleep(10)

@app.route('/')
def index(): return f"<h1>EVOLUTION V69</h1><p>{bot_status}</p>"

def start_threads():
    threading.Thread(target=run_trading, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()
    threading.Thread(target=logger_worker, daemon=True).start()
    threading.Thread(target=run_evolution_loop, daemon=True).start()

load_brain()
start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
