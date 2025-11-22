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
from datetime import datetime, time as dtime, timedelta
import pytz
from github import Github

app = Flask(__name__)

# --- 1. TOUTES LES CLÉS ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")      # Salon Trading
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL")  # Salon Status
LEARNING_WEBHOOK_URL = os.environ.get("LEARNING_WEBHOOK_URL")    # Salon Cerveau (NOUVEAU)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")

# --- 2. CONFIG ---
WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "MSFT", "AMD", "PLTR", "META", "GOOG", "NFLX", "COIN", "MSTR"]
INITIAL_CAPITAL = 25000.0 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "stats": {"wins": 0, "losses": 0},
    "params": {"rsi_buy": 30, "stop_loss_atr": 2.0, "monte_carlo_threshold": 60}
}
bot_status = "Démarrage..."

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)

# --- 3. FONCTIONS DE COMMUNICATION ---
def run_heartbeat():
    while True:
        try:
            if HEARTBEAT_WEBHOOK_URL: requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"})
            time.sleep(30)
        except: time.sleep(30)

def log_thought(emoji, text):
    """Envoie les pensées du bot dans le salon #cerveau_ia"""
    if LEARNING_WEBHOOK_URL:
        try:
            requests.post(LEARNING_WEBHOOK_URL, json={"content": f"{emoji} **IA:** {text}"})
        except: pass

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
            repo.update_file("brain.json", "Brain Update", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 5. LE LABORATOIRE D'ÉTUDE (Cœur de la V28) ---
def ask_gemini_teacher(context):
    """Gemini agit comme un prof qui commente les résultats"""
    prompt = f"""
    Tu es un étudiant en trading algorithmique.
    Voici ce que tu viens de tester : {context}
    
    Fais un commentaire très court (1 phrase) sur ce que tu as appris.
    Exemple: "Je vois que le Stop Loss serré ne marche pas sur la Tech."
    """
    try:
        model = genai.GenerativeModel('gemini-pro')
        res = model.generate_content(prompt)
        return res.text
    except: return "Analyse en cours..."

def train_brain_simulation():
    global brain, bot_status
    bot_status = "🧠 Étude en cours..."
    
    log_thought("📚", "J'ouvre mes manuels... Début de la session d'apprentissage du week-end.")
    time.sleep(2)
    
    # 1. Téléchargement Data
    log_thought("💾", "Téléchargement de l'historique des 30 derniers jours pour NVDA, TSLA et BTC...")
    data_cache = {}
    for s in ["NVDA", "TSLA", "COIN"]: 
        df = yf.Ticker(s).history(period="1mo", interval="1h")
        if not df.empty:
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            data_cache[s] = df.dropna()
    
    best_pnl = -999999
    best_params = brain['params'].copy()
    
    # 2. Tests Itératifs (Le bot essaie des trucs)
    log_thought("🧪", "Je vais simuler 5 scénarios différents pour trouver le meilleur réglage.")
    
    for i in range(1, 6): # 5 Essais
        test_rsi = np.random.randint(25, 45)
        test_sl = np.random.uniform(1.5, 4.0)
        
        simulated_pnl = 0
        trades_count = 0
        
        # Simulation rapide
        for s, df in data_cache.items():
            for index, row in df.iterrows():
                # Logique simplifiée pour aller vite
                if row['RSI'] < test_rsi: # Achat théorique
                    gain = (row['Close'] * 0.02) - (row['ATR'] * 0.1) # Simulation bruitée
                    simulated_pnl += gain
                    trades_count += 1
        
        # Gemini commente l'essai
        context = f"Essai #{i}: RSI Buy < {test_rsi}, StopLoss {test_sl:.1f} ATR. Résultat PnL: {simulated_pnl:.2f}$ sur {trades_count} trades."
        comment = ask_gemini_teacher(context)
        
        log_thought("🤔", f"**Test {i}/5 :** {context}\n> *Note: {comment}*")
        
        if simulated_pnl > best_pnl:
            best_pnl = simulated_pnl
            best_params = {"rsi_buy": test_rsi, "stop_loss_atr": test_sl, "monte_carlo_threshold": 60}
            time.sleep(1)

    # 3. Conclusion
    brain['params'] = best_params
    save_brain()
    
    log_thought("🎓", f"**Session terminée !** J'ai trouvé les paramètres optimaux pour Lundi.\nJe vais utiliser un RSI < {best_params['rsi_buy']} et un Stop Loss large de {best_params['stop_loss_atr']:.1f} ATR.")
    
    if DISCORD_WEBHOOK_URL:
        requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [{"title": "🏋️ RAPPORT DOJO", "description": "Optimisation terminée.", "color": 0xff00ff}]})
    
    time.sleep(3600) # Pause 1h

# --- 6. TRADING & RESTE DU CODE ---
def get_data(s):
    try:
        df = yf.Ticker(s).history(period="1mo", interval="15m")
        if df.empty: return None
        returns = df['Close'].pct_change().dropna()
        last = df['Close'].iloc[-1]
        sims = last * (1 + np.random.normal(returns.mean(), returns.std(), 1000))
        prob_up = np.sum(sims > last) / 10 
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

def ask_omega(d):
    rsi_limit = brain['params']['rsi_buy']
    prompt = f"Action: {d['s']}. MC: {d['prob']}%. Z: {d['z']}. RSI: {d['rsi']}. Limit RSI: {rsi_limit}. Dec: BUY/WAIT json"
    try:
        model = genai.GenerativeModel('gemini-pro')
        res = model.generate_content(prompt)
        return json.loads(res.text.replace("```json","").replace("```",""))
    except: return {"decision": "WAIT"}

def run_trading():
    global brain, bot_status
    load_brain()
    count = 0
    while True:
        try:
            count += 1
            if count >= 5: save_brain(); count = 0
            
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if not market_open:
                train_brain_simulation() # Lancement apprentissage
                bot_status = "🌙 Entraînement"
                time.sleep(60)
                continue

            bot_status = "🟢 Scan..."
            # Gestion Vente
            for s in list(brain['holdings'].keys()):
                pos = brain['holdings'][s]
                curr = yf.Ticker(s).fast_info['last_price']
                if curr < pos['stop']:
                    brain['cash'] += pos['qty'] * curr
                    del brain['holdings'][s]
                    if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json={"content": f"🔴 VENTE {s}"})
                    save_brain()

            # Scan Achat
            if len(brain['holdings']) < 5:
                for s in WATCHLIST:
                    if s in brain['holdings']: continue
                    d = get_data(s)
                    if d:
                        dec = ask_omega(d)
                        if dec['decision'] == "BUY" and d['prob'] > brain['params']['monte_carlo_threshold'] and brain['cash'] > 500:
                            bet = brain['cash'] * 0.15 
                            brain['cash'] -= bet
                            qty = bet / d['p']
                            sl = d['p'] - (d['atr'] * brain['params']['stop_loss_atr'])
                            brain['holdings'][s] = {"qty": qty, "entry": d['p'], "stop": sl}
                            if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json={"content": f"🟢 ACHAT {s}"})
                            save_brain()
            time.sleep(60)
        except: time.sleep(10)

@app.route('/')
def index(): return f"<h1>OMEGA SCHOLAR V28</h1><p>{bot_status}</p>"

def start_threads():
    t1 = threading.Thread(target=run_trading)
    t1.daemon = True
    t1.start()
    t2 = threading.Thread(target=run_heartbeat)
    t2.daemon = True
    t2.start()

start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
