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

# --- 2. CONFIG ---
WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN"]
INITIAL_CAPITAL = 50000.0 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "total_pnl": 0.0,
    "trades_count": 0
}
bot_status = "Démarrage V47..."
last_replay_log = 0 # Pour éviter le spam

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)

# --- 3. COMMS ---
def send_discord(msg):
    if DISCORD_WEBHOOK_URL:
        try: requests.post(DISCORD_WEBHOOK_URL, json=msg)
        except: pass

def log_thought(emoji, text):
    print(f"{emoji} {text}")
    if LEARNING_WEBHOOK_URL: 
        try: requests.post(LEARNING_WEBHOOK_URL, json={"content": f"{emoji} **ENGINE:** {text}"})
        except: pass

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
        brain.update(json.loads(c.decoded_content.decode()))
    except: pass

def save_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        content = json.dumps(brain, indent=4)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Save", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 5. MARKET DATA ---
def generate_fake_market_data():
    price = 100.0
    data = []
    for _ in range(60):
        change = np.random.normal(0, 0.5)
        price += change
        data.append(price)
    df = pd.DataFrame(data, columns=['Close'])
    return df

def get_data_or_fake(symbol):
    try:
        # On essaie Yahoo
        df = yf.Ticker(symbol).history(period="1d", interval="1m")
        if not df.empty: return df
    except: pass
    # Si échec, fake data
    return generate_fake_market_data()

# --- 6. MOTEUR DE TRADING ---
def execute_trade_logic(symbol, price, is_replay=False):
    global brain
    
    # VENTE
    if symbol in brain['holdings']:
        pos = brain['holdings'][symbol]
        pnl_pct = (price - pos['entry']) / pos['entry']
        
        exit_reason = None
        if price < pos['stop']: exit_reason = "STOP LOSS"
        elif price > pos['tp']: exit_reason = "TAKE PROFIT"
        elif is_replay and random.random() < 0.1: exit_reason = "SIM EXIT"
        
        if exit_reason:
            revenue = pos['qty'] * price
            cost = pos['qty'] * pos['entry']
            pnl = revenue - cost
            brain['cash'] += revenue
            brain['total_pnl'] += pnl
            brain['trades_count'] += 1
            del brain['holdings'][symbol]
            
            tag = "[REPLAY]" if is_replay else "[LIVE]"
            color = 0x2ecc71 if pnl > 0 else 0xe74c3c
            emoji = "💰" if pnl > 0 else "🩸"
            
            msg = {
                "embeds": [{
                    "title": f"{emoji} {tag} SELL : {symbol}",
                    "description": f"Raison: {exit_reason}",
                    "color": color,
                    "fields": [{"name": "PnL", "value": f"**{pnl:+.2f}$**", "inline": True}]
                }]
            }
            send_discord(msg)
            if not is_replay: save_brain()

    # ACHAT
    elif len(brain['holdings']) < 5 and brain['cash'] > 1000:
        force_buy = is_replay and random.random() < 0.05
        # En Live, on ajouterait ici une condition technique RSI
        if force_buy:
            qty = (brain['cash'] * 0.10) / price
            sl = price * 0.99
            tp = price * 1.02
            brain['holdings'][symbol] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
            brain['cash'] -= (qty * price)
            
            tag = "[REPLAY]" if is_replay else "[LIVE]"
            msg = {
                "embeds": [{
                    "title": f"⚡ {tag} BUY : {symbol}",
                    "color": 0x3498db,
                    "fields": [{"name": "Prix", "value": f"{price:.2f}$", "inline": True}]
                }]
            }
            send_discord(msg)
            if not is_replay: save_brain()

# --- 7. BOUCLE PRINCIPALE ---
def run_engine():
    global brain, bot_status, last_replay_log
    load_brain()
    
    # Message de démarrage unique
    log_thought("🔥", "Moteur V47 (Non-Bloquant) démarré.")
    
    while True:
        try:
            # Check Horaire
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if market_open:
                bot_status = "🟢 LIVE TRADING"
                for s in WATCHLIST:
                    try:
                        price = yf.Ticker(s).fast_info['last_price']
                        execute_trade_logic(s, price, is_replay=False)
                    except: pass
                time.sleep(10)
                
            else:
                bot_status = "🎬 REPLAY"
                
                # On ne log le message "Replay" qu'une fois par heure pour ne pas spammer
                if time.time() - last_replay_log > 3600:
                    log_thought("🎬", "Mode Replay actif (Simulation weekend).")
                    last_replay_log = time.time()

                # On prend un symbole au hasard
                s = random.choice(WATCHLIST)
                # On récupère ou génère des données
                df = get_data_or_fake(s)
                # On prend un prix au hasard dans ces données
                sim_price = random.choice(df['Close'].tolist())
                
                execute_trade_logic(s, sim_price, is_replay=True)
                
                # Pause aléatoire entre les actions simulées
                time.sleep(random.randint(5, 15))
                
        except Exception as e:
            print(f"Erreur boucle: {e}")
            time.sleep(5)

# --- 8. REPORTING ---
def run_reporting():
    while True:
        time.sleep(600) # 10 minutes
        val = brain['cash']
        hold_val = sum([v['qty'] * v['entry'] for k,v in brain['holdings'].items()])
        total = val + hold_val
        pnl = brain.get('total_pnl', 0)
        
        msg = {
            "embeds": [{
                "title": "⏱️ BILAN 10 MINUTES",
                "color": 0xf1c40f,
                "fields": [
                    {"name": "PnL Réalisé", "value": f"**{pnl:+.2f} $**", "inline": True},
                    {"name": "Capital Total", "value": f"{total:.2f} $", "inline": True}
                ],
                "footer": {"text": f"Status: {bot_status}"}
            }]
        }
        send_discord(msg)

@app.route('/')
def index(): return f"<h1>UNSTOPPABLE V47</h1><p>{bot_status}</p>"

def start_threads():
    threading.Thread(target=run_engine, daemon=True).start()
    threading.Thread(target=run_reporting, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()

start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
