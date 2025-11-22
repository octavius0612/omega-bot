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

# --- 2. CONFIG FINANCIÈRE ---
WATCHLIST_LIVE = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR"]
INITIAL_CAPITAL = 50000.0 

brain = {
    "cash": INITIAL_CAPITAL,
    "holdings": {},
    "active_strategies": {},
    "graveyard": [],
    "generation": 0,
    # Compteur global historique
    "total_realized_pnl": 0.0
}
bot_status = "Audit Financier en cours..."

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)

# --- 3. OUTILS COMMS ---
def log_thought(emoji, text):
    print(f"{emoji} {text}")
    if LEARNING_WEBHOOK_URL: requests.post(LEARNING_WEBHOOK_URL, json={"content": f"{emoji} **FACTORY:** {text}"})

def send_financial_report():
    """Calcule et envoie le bilan total"""
    global brain
    
    # Valeur du Cash
    total_value = brain['cash']
    
    # Valeur des Actions détenues (au prix actuel estimé)
    unrealized_pnl = 0
    details = ""
    
    if brain['holdings']:
        for s, pos in brain['holdings'].items():
            try:
                current_price = yf.Ticker(s).fast_info['last_price']
                pos_value = pos['qty'] * current_price
                total_value += pos_value
                
                # Gain latent
                trade_pnl = pos_value - (pos['qty'] * pos['entry'])
                unrealized_pnl += trade_pnl
                
                emoji = "🟢" if trade_pnl >= 0 else "🔴"
                details += f"• {s}: {emoji} {trade_pnl:+.2f}$ ({pos['strategy_origin']})\n"
            except: pass
            
    # Calcul PnL Total (Latent + Réalisé)
    total_pnl = total_value - INITIAL_CAPITAL
    
    color = 0x2ecc71 if total_pnl >= 0 else 0xe74c3c
    
    msg = {
        "embeds": [{
            "title": "💰 BILAN FINANCIER",
            "color": color,
            "fields": [
                {"name": "Profit Total (Net)", "value": f"**{total_pnl:+.2f} $**", "inline": True},
                {"name": "Capital Actuel", "value": f"{total_value:.2f} $", "inline": True},
                {"name": "Positions en cours", "value": details if details else "Aucune (100% Cash)", "inline": False},
                {"name": "Gains déjà encaissés", "value": f"{brain['total_realized_pnl']:+.2f} $", "inline": True}
            ],
            "footer": {"text": f"Cash dispo: {brain['cash']:.2f}$"}
        }]
    }
    if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json=msg)

def run_heartbeat():
    count = 0
    while True:
        try:
            if HEARTBEAT_WEBHOOK_URL: requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"})
            
            # Toutes les 60 minutes (120 * 30s), on envoie le bilan financier
            count += 1
            if count >= 120: 
                send_financial_report()
                count = 0
                
            time.sleep(30)
        except: time.sleep(30)

def load_brain():
    global brain
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        c = repo.get_contents("brain.json")
        loaded = json.loads(c.decoded_content.decode())
        # Fusion prudente
        if "cash" in loaded: brain["cash"] = loaded["cash"]
        if "holdings" in loaded: brain["holdings"] = loaded["holdings"]
        if "active_strategies" in loaded: brain["active_strategies"] = loaded["active_strategies"]
        if "total_realized_pnl" in loaded: brain["total_realized_pnl"] = loaded["total_realized_pnl"]
    except: pass

def save_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        content = json.dumps(brain, indent=4)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Financial Update", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 4. USINE A CODE (GENERATION) ---
def generate_new_strategy_idea():
    prompt = f"""
    Agis comme un Quants Developer.
    TA MISSION : Inventer une stratégie d'achat agressive mais sûre.
    Variables: row['RSI'], row['EMA50'], row['EMA200'], row['BBU'], row['BBL'], row['ADX'], row['Close']
    
    Format de sortie (Code Python brut uniquement) :
    signal = 0
    if condition: signal = 100
    """
    try:
        model = genai.GenerativeModel('gemini-pro')
        res = model.generate_content(prompt)
        code = res.text.replace("```python", "").replace("```", "").strip()
        name = f"STRAT_V43_{brain['generation']}_{random.randint(10,99)}"
        return name, code
    except: return None, None

def backtest_code(strategy_code):
    try:
        df = yf.Ticker("NVDA").history(period="1mo", interval="30m")
        if df.empty: return -999, 0, 0
        
        # Indicateurs
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        df['EMA200'] = ta.ema(df['Close'], length=200)
        df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
        bb = ta.bbands(df['Close'], length=20)
        df['BBU'] = bb['BBU_20_2.0']
        df['BBL'] = bb['BBL_20_2.0']
        df = df.dropna()
        
        capital = 10000
        position = 0
        entry = 0
        trades = 0
        wins = 0
        
        for index, row in df.iterrows():
            local_vars = {'row': row, 'signal': 0}
            try:
                exec(strategy_code, {}, local_vars)
            except: pass
            
            if position == 0 and local_vars['signal'] == 100:
                position = capital / row['Close']
                entry = row['Close']
                capital = 0
            elif position > 0:
                # TP 4% / SL 2%
                if row['High'] > entry * 1.04:
                    capital = position * entry * 1.04
                    position = 0; trades += 1; wins += 1
                elif row['Low'] < entry * 0.98:
                    capital = position * entry * 0.98
                    position = 0; trades += 1
                    
        final = capital + (position * df['Close'].iloc[-1])
        return final - 10000, trades, (wins/trades*100 if trades else 0)
    except: return -999, 0, 0

def manage_strategies_lifecycle():
    global brain
    
    # 1. Élimination des pertes
    to_delete = []
    for name, stats in brain['active_strategies'].items():
        if stats['real_pnl'] < -50: # Tolérance zéro
            log_thought("💸", f"Stratégie {name} virée (Perte: {stats['real_pnl']}$).")
            to_delete.append(name)
    for name in to_delete: del brain['active_strategies'][name]
    
    # 2. Recrutement
    if len(brain['active_strategies']) < 3:
        brain['generation'] += 1
        name, code = generate_new_strategy_idea()
        if code:
            pnl, trades, wr = backtest_code(code)
            if pnl > 0 and trades > 2:
                log_thought("🤑", f"**NOUVELLE STRATÉGIE RENTABLE !**\nNom: `{name}`\nBacktest PnL: +{pnl:.2f}$ (WinRate: {wr:.1f}%)")
                brain['active_strategies'][name] = {"code": code, "real_pnl": 0.0, "trades": 0}
                save_brain()

# --- 5. TRADING REEL ---
def get_live_data(s):
    try:
        df = yf.Ticker(s).history(period="1mo", interval="15m")
        if df.empty: return None
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        df['EMA200'] = ta.ema(df['Close'], length=200)
        df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
        bb = ta.bbands(df['Close'], length=20)
        df['BBU'] = bb['BBU_20_2.0']
        df['BBL'] = bb['BBL_20_2.0']
        return df.iloc[-1]
    except: return None

def run_trading_engine():
    global brain, bot_status
    load_brain()
    
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if not market_open:
                bot_status = "🌙 Recrutement..."
                manage_strategies_lifecycle()
                time.sleep(10)
                continue
            
            bot_status = "🟢 Trading..."
            
            # ACHATS
            for s in WATCHLIST_LIVE:
                if s in brain['holdings']: continue
                row = get_live_data(s)
                if row is None: continue
                
                for name, strat in brain['active_strategies'].items():
                    loc = {'row': row, 'signal': 0}
                    try:
                        exec(strat['code'], {}, loc)
                        if loc['signal'] == 100:
                            price = row['Close']
                            qty = 200 / price
                            sl = price * 0.98
                            
                            brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "strategy_origin": name}
                            brain['cash'] -= 200
                            
                            if DISCORD_WEBHOOK_URL:
                                requests.post(DISCORD_WEBHOOK_URL, json={"content": f"🟢 **ACHAT {s}** ({name})\nPrix: {price:.2f}$"})
                            save_brain()
                            break
                    except: pass
            
            # VENTES (AVEC CALCUL DE PROFIT EXACT)
            for s in list(brain['holdings'].keys()):
                pos = brain['holdings'][s]
                row = get_live_data(s)
                if row is None: continue
                curr = row['Close']
                
                exit = None
                if curr < pos['stop']: exit = "STOP LOSS"
                elif curr > pos['entry'] * 1.04: exit = "TAKE PROFIT"
                
                if exit:
                    revenue = pos['qty'] * curr
                    cost = pos['qty'] * pos['entry']
                    pnl = revenue - cost
                    
                    brain['cash'] += revenue
                    brain['total_realized_pnl'] += pnl # On ajoute au compteur global
                    
                    # Mise à jour score stratégie
                    s_name = pos['strategy_origin']
                    if s_name in brain['active_strategies']:
                        brain['active_strategies'][s_name]['real_pnl'] += pnl
                    
                    del brain['holdings'][s]
                    
                    # Message Spécial Argent
                    emoji = "💰" if pnl > 0 else "💸"
                    msg = f"{emoji} **VENTE {s} ({exit})**\nRésultat: **{pnl:+.2f} $**\nStratégie: {s_name}"
                    
                    if DISCORD_WEBHOOK_URL:
                        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
                    save_brain()
            
            time.sleep(60)
        except: time.sleep(10)

@app.route('/')
def index(): return f"<h1>CFO V43</h1><p>Profit Total: {brain.get('total_realized_pnl', 0):.2f}$</p>"

def start_threads():
    t1 = threading.Thread(target=run_trading_engine, daemon=True)
    t1.start()
    t2 = threading.Thread(target=run_heartbeat, daemon=True)
    t2.start()

start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
