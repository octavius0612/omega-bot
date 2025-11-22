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

# --- 2. CONFIG SCALPING ---
WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMD", "AMZN"] # Les plus liquides pour le scalping
INITIAL_CAPITAL = 50000.0 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "history": [],
    "total_pnl": 0.0
}
bot_status = "Démarrage Scalper..."

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)

# --- 3. COMMS ---
def send_discord(msg):
    if DISCORD_WEBHOOK_URL:
        try: requests.post(DISCORD_WEBHOOK_URL, json=msg)
        except: pass

def log_thought(emoji, text):
    if LEARNING_WEBHOOK_URL: requests.post(LEARNING_WEBHOOK_URL, json={"content": f"{emoji} **SCALPER:** {text}"})

# --- 4. MÉMOIRE ---
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
            repo.update_file("brain.json", "Scalp Update", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 5. LE MOTEUR SCALPING (1 MINUTE) ---
def get_scalp_data(s):
    try:
        # On prend les données 1 minute (Ultra court terme)
        df = yf.Ticker(s).history(period="1d", interval="1m")
        if df.empty: return None
        
        df['RSI'] = ta.rsi(df['Close'], length=7) # RSI rapide (7 au lieu de 14)
        df['EMA20'] = ta.ema(df['Close'], length=20) # Tendance immédiate
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        return df.iloc[-1]
    except: return None

def execute_trade(s, price, side, reason, is_replay=False):
    global brain
    
    tag = "[REPLAY]" if is_replay else "[LIVE]"
    color = 0x2ecc71 if side == "BUY" else 0xe74c3c
    
    if side == "BUY":
        # On mise 5% du capital pour scalper vite
        qty = (brain['cash'] * 0.05) / price
        sl = price * 0.99 # Stop Loss très serré (1%)
        tp = price * 1.015 # Take Profit rapide (1.5%)
        
        brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
        brain['cash'] -= (qty * price)
        
        msg = {
            "embeds": [{
                "title": f"⚡ {tag} SCALP BUY : {s}",
                "description": f"Raison: {reason}",
                "color": color,
                "fields": [
                    {"name": "Prix", "value": f"{price:.2f}$", "inline": True},
                    {"name": "Cible", "value": f"{tp:.2f}$ (+1.5%)", "inline": True}
                ]
            }]
        }
        send_discord(msg)
        
    elif side == "SELL":
        if s in brain['holdings']:
            pos = brain['holdings'][s]
            revenue = pos['qty'] * price
            cost = pos['qty'] * pos['entry']
            pnl = revenue - cost
            brain['cash'] += revenue
            brain['total_pnl'] += pnl
            del brain['holdings'][s]
            
            emoji = "💰" if pnl > 0 else "🩸"
            msg = {
                "embeds": [{
                    "title": f"{emoji} {tag} SCALP SELL : {s}",
                    "description": f"Raison: {reason}",
                    "color": color,
                    "fields": [
                        {"name": "Résultat", "value": f"**{pnl:+.2f}$**", "inline": True},
                        {"name": "Total PnL", "value": f"{brain['total_pnl']:+.2f}$", "inline": True}
                    ]
                }]
            }
            send_discord(msg)
    
    if not is_replay: save_brain()

# --- 6. MODE REPLAY (POUR VOIR L'ACTION LE WEEKEND) ---
def run_weekend_replay():
    """
    Simule des trades basés sur les données de Vendredi pour montrer l'activité.
    """
    log_thought("🎬", "Marché Fermé. Lancement du mode REPLAY (Simulation des trades de Vendredi).")
    
    # On charge les données de vendredi
    cache = {}
    for s in WATCHLIST:
        try:
            df = yf.Ticker(s).history(period="5d", interval="1m") # 5 derniers jours
            cache[s] = df
        except: pass
        
    while True:
        # On prend une action au hasard et un moment au hasard
        s = random.choice(list(cache.keys()))
        df = cache[s]
        
        # Simulation d'une décision
        row = df.sample(1).iloc[-1]
        rsi = random.randint(20, 80) # Simulation RSI pour l'exemple visuel
        
        # Simulation Achat
        if s not in brain['holdings'] and rsi < 30:
            execute_trade(s, row['Close'], "BUY", f"RSI survendu ({rsi})", is_replay=True)
            
        # Simulation Vente
        elif s in brain['holdings']:
            execute_trade(s, row['Close'], "SELL", "Target Replay atteinte", is_replay=True)
            
        time.sleep(random.randint(10, 30)) # Un trade toutes les 10-30 secondes

# --- 7. THREADS ---
def run_market_engine():
    global brain, bot_status
    load_brain()
    
    while True:
        nyc = pytz.timezone('America/New_York')
        now = datetime.now(nyc)
        market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
        
        if not market_open:
            bot_status = "🎬 Mode Replay"
            run_weekend_replay() # On lance le replay infini tant que c'est fermé
            # Note: run_weekend_replay est une boucle infinie, donc on ne sort pas d'ici avant redémarrage
            # C'est voulu pour le weekend.
        else:
            bot_status = "⚡ Trading Live"
            # LOGIQUE LIVE (Lundi)
            for s in WATCHLIST:
                row = get_scalp_data(s)
                if row is None: continue
                
                # Stratégie Scalping Pure (RSI Rapide)
                if s not in brain['holdings']:
                    if row['RSI'] < 25: # Très survendu en 1 minute
                        execute_trade(s, row['Close'], "BUY", f"Scalp RSI {row['RSI']:.1f}")
                else:
                    # Gestion Sortie
                    pos = brain['holdings'][s]
                    if row['Close'] < pos['stop']:
                        execute_trade(s, row['Close'], "SELL", "Stop Loss")
                    elif row['Close'] > pos['tp']:
                        execute_trade(s, row['Close'], "SELL", "Take Profit")
            
            time.sleep(10) # Scan toutes les 10 secondes (très rapide)

def run_reporting():
    """Envoie le bilan toutes les 10 minutes pile"""
    while True:
        time.sleep(600) # 600 secondes = 10 minutes
        
        val = brain['cash']
        # Valeur estimée (sommaire)
        assets_val = 0
        txt = "Aucune position."
        if brain['holdings']:
            txt = ""
            for s, pos in brain['holdings'].items():
                assets_val += pos['qty'] * pos['entry']
                txt += f"• {s}\n"
        
        total = val + assets_val
        pnl = brain['total_pnl']
        color = 0x2ecc71 if pnl >= 0 else 0xe74c3c
        
        msg = {
            "embeds": [{
                "title": "⏱️ BILAN 10 MINUTES",
                "color": color,
                "fields": [
                    {"name": "Profit Réalisé", "value": f"**{pnl:+.2f} $**", "inline": True},
                    {"name": "Capital Total", "value": f"{total:.2f} $", "inline": True},
                    {"name": "Positions Actives", "value": txt, "inline": False}
                ],
                "footer": {"text": "Mode Scalping V45"}
            }]
        }
        send_discord(msg)

def run_heartbeat():
    while True:
        if HEARTBEAT_WEBHOOK_URL: requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"})
        time.sleep(30)

@app.route('/')
def index(): return f"<h1>SCALPER V45</h1><p>{bot_status}</p>"

def start_threads():
    threading.Thread(target=run_market_engine, daemon=True).start()
    threading.Thread(target=run_reporting, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()

start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
