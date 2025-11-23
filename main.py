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
import traceback
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

# --- 2. CONFIGURATION ---
WATCHLIST_LIVE = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR"]
INITIAL_CAPITAL = 50000.0 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    # Ici, on stocke les stratégies codées par l'IA
    "strategies": {}, 
    "generation": 0,
    "total_pnl": 0.0
}
bot_status = "Démarrage de la Ruche..."

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)

# --- 3. COMMS ---
def log_thought(emoji, text):
    print(f"{emoji} {text}")
    if LEARNING_WEBHOOK_URL: requests.post(LEARNING_WEBHOOK_URL, json={"content": f"{emoji} **HIVE:** {text}"})

def send_alert(msg):
    if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json=msg)

def run_heartbeat():
    while True:
        try:
            if HEARTBEAT_WEBHOOK_URL: requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"})
            time.sleep(30)
        except: time.sleep(30)

# --- 4. MÉMOIRE ---
def load_brain():
    global brain
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        c = repo.get_contents("brain.json")
        loaded = json.loads(c.decoded_content.decode())
        if "strategies" in loaded: brain.update(loaded)
    except: pass

def save_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        content = json.dumps(brain, indent=4)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Hive Update", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 5. L'ARCHITECTE (GÉNÉRATEUR DE CODE) ---
def ai_architect_generate():
    """L'IA écrit du code Python pour créer une stratégie"""
    indicators = "RSI, EMA50, EMA200, BBU (Bollinger Haut), BBL (Bollinger Bas), ATR, ADX, VOLUME, CLOSE"
    
    prompt = f"""
    Tu es un Architecte Quantitatif. Invente une stratégie de trading court terme.
    Données dispo dans le dataframe 'row': {indicators}.
    
    Rédige un code Python qui définit la variable 'signal'.
    signal = 100 (Achat), signal = 0 (Rien).
    
    Exemple de créativité attendue :
    "Acheter si le prix casse la Bollinger Basse ET que le RSI remonte."
    
    Réponds UNIQUEMENT par le code Python (sans markdown).
    """
    try:
        model = genai.GenerativeModel('gemini-pro')
        res = model.generate_content(prompt)
        code = res.text.replace("```python", "").replace("```", "").strip()
        name = f"STRAT_GEN_{brain['generation']}_{random.randint(100,999)}"
        return name, code
    except: return None, None

def backtest_strategy(code):
    """Le Dojo : On teste le code de l'IA sur le passé"""
    try:
        df = yf.Ticker("NVDA").history(period="1mo", interval="30m")
        if df.empty: return -999
        
        # Calcul indicateurs pour le code généré
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        df['EMA200'] = ta.ema(df['Close'], length=200)
        bb = ta.bbands(df['Close'], length=20)
        df['BBU'] = bb['BBU_20_2.0']
        df['BBL'] = bb['BBL_20_2.0']
        df = df.dropna()
        
        balance = 10000
        pos = 0
        entry = 0
        
        for i, row in df.iterrows():
            loc = {'row': row, 'signal': 0}
            try: exec(code, {}, loc)
            except: pass
            
            if pos == 0 and loc['signal'] == 100:
                pos = balance / row['Close']
                entry = row['Close']
                balance = 0
            elif pos > 0:
                # TP 5% / SL 3%
                if row['High'] > entry * 1.05:
                    balance = pos * entry * 1.05
                    pos = 0
                elif row['Low'] < entry * 0.97:
                    balance = pos * entry * 0.97
                    pos = 0
                    
        final = balance + (pos * df['Close'].iloc[-1])
        return final - 10000
    except: return -999

def run_learning_loop():
    global brain, bot_status
    
    while True:
        # On apprend si marché fermé OU si on a moins de 3 stratégies
        nyc = pytz.timezone('America/New_York')
        now = datetime.now(nyc)
        market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
        
        if not market_open or len(brain['strategies']) < 3:
            bot_status = "🧠 Création Stratégies..."
            brain['generation'] += 1
            
            # 1. L'IA écrit du code
            name, code = ai_architect_generate()
            if code:
                # 2. On teste le code (Backtest)
                pnl = backtest_strategy(code)
                
                if pnl > 200: # Si rentable
                    log_thought("🔥", f"**NOUVELLE STRATÉGIE CRÉÉE !**\nNom: {name}\nPerformance Test: +{pnl:.2f}$")
                    brain['strategies'][name] = {"code": code, "score": pnl}
                    save_brain()
                else:
                    log_thought("🗑️", f"Stratégie {name} rejetée (PnL: {pnl:.2f}$). Je recommence.")
            
            # Nettoyage des mauvaises strats existantes
            if len(brain['strategies']) > 3:
                worst = min(brain['strategies'], key=lambda k: brain['strategies'][k]['score'])
                del brain['strategies'][worst]
                log_thought("♻️", f"Optimisation : Suppression de la stratégie la plus faible ({worst}).")
                
            time.sleep(10)
        else:
            time.sleep(60)

# --- 6. LE CONSEIL DES SAGES (MULTI-IA) ---
def consult_the_council(symbol, rsi, trend):
    """
    Réunit 3 experts IA pour valider un trade.
    """
    prompt = f"""
    C'est une RÉUNION DE CRISE pour valider un trade sur {symbol}.
    Données : RSI={rsi:.1f}, Tendance={trend}.
    
    Intervenants :
    1. [L'Analyste] (Prudent, regarde la tendance)
    2. [Le Scalper] (Agressif, regarde le RSI)
    3. [Le Risk Manager] (Paranoïaque, protège le capital)
    
    Fais-les débattre en 1 phrase chacun.
    Puis Vote Final : OUI ou NON.
    
    Format JSON : {{"debat": "...", "vote_final": "OUI/NON"}}
    """
    try:
        model = genai.GenerativeModel('gemini-pro')
        res = model.generate_content(prompt)
        return json.loads(res.text.replace("```json","").replace("```",""))
    except: return {"vote_final": "NON", "debat": "Erreur Conseil"}

# --- 7. MOTEUR TRADING ---
def get_live_data(s):
    try:
        df = yf.Ticker(s).history(period="1mo", interval="15m")
        if df.empty: return None
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        df['BBU'] = ta.bbands(df['Close'], length=20)['BBU_20_2.0']
        df['BBL'] = ta.bbands(df['Close'], length=20)['BBL_20_2.0']
        return df.iloc[-1]
    except: return None

def run_trading():
    global brain, bot_status
    load_brain()
    
    log_thought("👋", "Hive Mind V50 en ligne. L'IA apprend et se concerte.")
    
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if market_open:
                bot_status = "🟢 Trading & Concertation..."
                
                # Gestion Ventes
                for s in list(brain['holdings'].keys()):
                    pos = brain['holdings'][s]
                    curr = yf.Ticker(s).fast_info['last_price']
                    
                    exit = None
                    if curr < pos['stop']: exit = "STOP LOSS"
                    elif curr > pos['entry'] * 1.05: exit = "TAKE PROFIT"
                    
                    if exit:
                        pnl = (curr - pos['entry']) * pos['qty']
                        brain['cash'] += pos['qty'] * curr
                        brain['total_pnl'] += pnl
                        del brain['holdings'][s]
                        
                        emoji = "💰" if pnl > 0 else "💸"
                        send_alert({"content": f"{emoji} **VENTE {s}** ({exit}) | PnL: {pnl:.2f}$"})
                        save_brain()

                # Scan Achats
                if len(brain['holdings']) < 5:
                    for s in WATCHLIST_LIVE:
                        if s in brain['holdings']: continue
                        
                        row = get_live_data(s)
                        if row is None: continue
                        
                        # 1. On teste avec nos stratégies générées par l'IA
                        algo_signal = False
                        used_strat = ""
                        
                        for name, strat in brain['strategies'].items():
                            loc = {'row': row, 'signal': 0}
                            try:
                                exec(strat['code'], {}, loc)
                                if loc['signal'] == 100:
                                    algo_signal = True
                                    used_strat = name
                                    break
                            except: pass
                        
                        # 2. Si un Algo dit OUI, on convoque le CONSEIL (IA)
                        if algo_signal:
                            trend = "HAUSSIER" if row['Close'] > row['EMA50'] else "BAISSIER"
                            council = consult_the_council(s, row['RSI'], trend)
                            
                            if council['vote_final'] == "OUI":
                                price = row['Close']
                                qty = 200 / price
                                sl = price * 0.97
                                
                                brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl}
                                brain['cash'] -= 200
                                
                                msg = {
                                    "embeds": [{
                                        "title": f"🏛️ ORDRE CONCERTÉ : {s}",
                                        "description": f"**Stratégie:** `{used_strat}`\n\n**🗣️ Débat du Conseil :**\n{council['debat']}",
                                        "color": 0x2ecc71,
                                        "fields": [{"name": "Prix", "value": f"{price:.2f}$", "inline": True}]
                                    }]
                                }
                                send_alert(msg)
                                save_brain()
                            else:
                                log_thought("✋", f"Opportunité sur {s} bloquée par le Conseil (Vote: NON).")

            else:
                bot_status = "🌙 Apprentissage Nuit"
            
            time.sleep(60)
        except Exception as e:
            print(e)
            time.sleep(10)

def start_threads():
    threading.Thread(target=run_trading, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()
    threading.Thread(target=run_learning_loop, daemon=True).start()

load_brain()
start_threads()

@app.route('/')
def index(): 
    strats = "<br>".join([f"{k}: {v['score']:.0f}pts" for k,v in brain['strategies'].items()])
    return f"<h1>HIVE V50</h1><p>{bot_status}</p><p>Stratégies:</p>{strats}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
