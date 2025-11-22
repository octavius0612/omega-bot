import os
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
from textblob import TextBlob
import requests
import google.generativeai as genai
import json
import time
import threading
from flask import Flask
from datetime import datetime, time as dtime
import pytz
from github import Github

app = Flask(__name__)

# --- 1. LES CLÉS SECRÈTES (Render les remplira) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")

# --- 2. RÉGLAGES ---
WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "MSFT", "AMD", "PLTR", "META", "GOOG", "NFLX", "COIN", "MSTR"]
CHECK_INTERVAL = 60       
INITIAL_CAPITAL = 25000.0 

brain = {"cash": INITIAL_CAPITAL, "holdings": {}, "stats": {"wins": 0, "losses": 0}, "strategies": {"QUANTUM": {"score": 10}, "MEAN": {"score": 10}}}
bot_status = "Démarrage..."

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)

# --- 3. MÉMOIRE GITHUB ---
def load_brain():
    global brain
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        c = repo.get_contents("brain.json")
        brain = json.loads(c.decoded_content.decode())
    except: pass

def save_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        content = json.dumps(brain, indent=4)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Update", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 4. ANALYSE MONTE CARLO & QUANTIQUE ---
def get_data(s):
    try:
        df = yf.Ticker(s).history(period="1mo", interval="15m")
        if df.empty: return None
        
        # Monte Carlo (Prédiction Futur)
        returns = df['Close'].pct_change().dropna()
        last = df['Close'].iloc[-1]
        sims = last * (1 + np.random.normal(returns.mean(), returns.std(), 1000))
        prob_up = np.sum(sims > last) / 10 # Pourcentage
        
        # Z-Score (Statistique)
        mean = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        z = (last - mean.iloc[-1]) / std.iloc[-1]
        
        return {
            "s": s, "p": last, "prob": prob_up, "z": z,
            "rsi": ta.rsi(df['Close'], length=14).iloc[-1],
            "atr": ta.atr(df['High'], df['Low'], df['Close'], length=14).iloc[-1],
            "inst": yf.Ticker(s).info.get('heldPercentInstitutions', 0)*100
        }
    except: return None

# --- 5. CERVEAU GEMINI ---
def ask_omega(d):
    prompt = f"Action: {d['s']}. Monte Carlo: {d['prob']:.1f}% hausse. Z-Score: {d['z']:.2f}. Inst: {d['inst']:.1f}%. Si Monte Carlo > 60% et Z-Score ok: BUY. Sinon WAIT. JSON: {{'decision':'BUY/WAIT', 'reason':'short'}}"
    try:
        model = genai.GenerativeModel('gemini-pro')
        res = model.generate_content(prompt)
        return json.loads(res.text.replace("```json","").replace("```",""))
    except: return {"decision": "WAIT"}

# --- 6. MOTEUR ---
def run():
    global brain, bot_status
    load_brain()
    count = 0
    while True:
        try:
            count += 1
            if count >= 5: save_brain(); count = 0
            
            # Gestion Vente
            for s in list(brain['holdings'].keys()):
                pos = brain['holdings'][s]
                curr = yf.Ticker(s).fast_info['last_price']
                
                # Stop Loss Dynamique
                if curr < pos['stop']:
                    brain['cash'] += pos['qty'] * curr
                    del brain['holdings'][s]
                    if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json={"content": f"🔴 VENTE {s} (Stop Loss)"})
                    save_brain()

            # Scan Achat
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            if now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0):
                bot_status = "🟢 Scan..."
                if len(brain['holdings']) < 5:
                    for s in WATCHLIST:
                        if s in brain['holdings']: continue
                        d = get_data(s)
                        if d:
                            dec = ask_omega(d)
                            if dec['decision'] == "BUY" and d['prob'] > 60 and brain['cash'] > 500:
                                # Mise Kelly
                                bet = brain['cash'] * 0.15 
                                brain['cash'] -= bet
                                qty = bet / d['p']
                                sl = d['p'] - (d['atr']*2)
                                brain['holdings'][s] = {"qty": qty, "entry": d['p'], "stop": sl}
                                if DISCORD_WEBHOOK_URL: 
                                    requests.post(DISCORD_WEBHOOK_URL, json={"content": f"🟢 ACHAT OMEGA {s} | Proba: {d['prob']:.1f}% | Mise: {bet:.2f}$"})
                                save_brain()
            else:
                bot_status = "🌙 Fermé"
            
            time.sleep(60)
        except: time.sleep(10)

@app.route('/')
def index(): return f"<h1>OMEGA V26</h1><p>{bot_status}</p><p>Cash: {brain['cash']:.2f}</p>"

def start():
    t = threading.Thread(target=run)
    t.daemon = True
    t.start()
start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
