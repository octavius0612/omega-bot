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
import queue
import io
import random
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
from PIL import Image
from flask import Flask
from datetime import datetime, time as dtime
from github import Github

app = Flask(__name__)

# --- 1. CLÉS ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL")
LEARNING_WEBHOOK_URL = os.environ.get("LEARNING_WEBHOOK_URL")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")

# --- 2. CONFIG ---
WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR"]
INITIAL_CAPITAL = 50000.0 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "emotions": {
        "confidence": 50.0, 
        "stress": 20.0, # Stress de base faible
        "euphoria": 0.0,
        "streak": 0
    }
}
bot_status = "Calibrage Psychologique..."
log_queue = queue.Queue()

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- 3. LOGS & CŒUR ---
def fast_log(text):
    log_queue.put(text)

def logger_worker():
    buffer = []
    last_send = time.time()
    while True:
        try:
            while not log_queue.empty():
                buffer.append(log_queue.get())
            if buffer and (len(buffer) > 5 or time.time() - last_send > 2.5):
                msg_block = "\n".join(buffer[:15])
                buffer = buffer[15:]
                if LEARNING_WEBHOOK_URL:
                    requests.post(LEARNING_WEBHOOK_URL, json={"content": msg_block})
                last_send = time.time()
            time.sleep(0.5)
        except: time.sleep(1)

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
        if "emotions" in loaded: brain.update(loaded)
    except: pass

def save_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        content = json.dumps(brain, indent=4)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Psych Update", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 5. SYSTÈME ÉMOTIONNEL ADAPTATIF ---
def update_emotions(event_type):
    emo = brain['emotions']
    
    if event_type == "WIN":
        emo['confidence'] = min(emo['confidence'] + 5, 100)
        emo['stress'] = max(emo['stress'] - 10, 0)
        emo['streak'] += 1
        if emo['streak'] > 3: emo['euphoria'] += 10
    
    elif event_type == "LOSS":
        emo['confidence'] = max(emo['confidence'] - 10, 10) # Min 10 pour pas déprimer total
        emo['stress'] = min(emo['stress'] + 15, 100)
        emo['streak'] = 0
        emo['euphoria'] = 0

def get_psychological_modifiers():
    """
    Retourne les modificateurs de trading basés sur l'humeur.
    Retour: (Multiplicateur_Mise, Multiplicateur_StopLoss, Nom_du_Mode)
    """
    emo = brain['emotions']
    
    # 1. MODE BUNKER (Stress élevé)
    if emo['stress'] > 70:
        # On divise la mise par 2, on serre le Stop Loss (0.8x distance)
        return 0.5, 0.8, "🛡️ BUNKER (Stress)"
    
    # 2. MODE LION (Confiance élevée, pas d'euphorie)
    if emo['confidence'] > 70 and emo['euphoria'] < 50:
        # On augmente la mise x1.2, Stop Loss normal
        return 1.2, 1.0, "🦁 LION (Confiance)"
    
    # 3. MODE RENARD (Euphorie : Attention au piège)
    if emo['euphoria'] > 80:
        # On revient à mise normale (ne pas flamber), on serre le Take Profit
        return 1.0, 0.9, "🦊 RENARD (Prudence Euphorie)"
    
    # 4. MODE ROBOT (Normal)
    return 1.0, 1.0, "🤖 ROBOT (Normal)"

# --- 6. CONSULTATION IA CONTEXTUELLE ---
def consult_adaptive_ai(symbol, rsi, trend):
    """Gemini analyse avec le filtre émotionnel du bot"""
    
    # On récupère l'état actuel
    _, _, mood_name = get_psychological_modifiers()
    
    prompt = f"""
    Tu es un Trader IA. Ton état psychologique actuel est : {mood_name}.
    Actif : {symbol}. RSI: {rsi:.1f}. Tendance: {trend}.
    
    CONSIGNES SELON TON ÉTAT :
    - Si BUNKER (Stress) : Sois paranoïaque. Cherche la perfection. Rejette tout doute.
    - Si LION (Confiance) : Sois audacieux mais logique.
    - Si RENARD (Euphorie) : Méfie-toi des pièges "Bull Trap".
    
    Décision : ACHETER (100) ou ATTENDRE (0) ?
    Réponds juste le chiffre.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        res = model.generate_content(prompt)
        score = float(res.text.strip())
        return score
    except: return 0

# --- 7. MOTEUR TRADING ---
def get_live_data(s):
    try:
        df = yf.Ticker(s).history(period="5d", interval="15m")
        if df.empty: return None
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['SMA200'] = ta.sma(df['Close'], length=200)
        return df
    except: return None

def run_trading():
    global brain, bot_status
    load_brain()
    
    fast_log("🧠 **PSYCHÉ V58:** Système émotionnel connecté. Je ne dors plus, je m'adapte.")
    
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            
            # MODE NUIT / WEEKEND : Rêve et Simulation (Pour garder les émotions actives)
            if not (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0)):
                bot_status = "🌙 Rêve (Simulation interne)..."
                
                # Simulation de trade pour faire varier les émotions même la nuit
                if random.random() < 0.1:
                    sim_result = random.choice(["WIN", "LOSS"])
                    update_emotions(sim_result)
                    fast_log(f"💤 **RÊVE:** Simulation mentale ({sim_result}). Stress: {brain['emotions']['stress']}%")
                
                time.sleep(60)
                continue
            
            # MODE JOUR : Trading
            size_mult, sl_mult, mood = get_psychological_modifiers()
            bot_status = f"🟢 {mood}"
            
            # VENTES
            for s in list(brain['holdings'].keys()):
                pos = brain['holdings'][s]
                curr = yf.Ticker(s).fast_info['last_price']
                
                exit = None
                if curr < pos['stop']: exit = "LOSS"
                elif curr > pos['tp']: exit = "WIN"
                
                if exit:
                    pnl = (curr - pos['entry']) * pos['qty']
                    brain['cash'] += pos['qty'] * curr
                    del brain['holdings'][s]
                    
                    update_emotions(exit) # Impact émotionnel réel
                    
                    color = 0x2ecc71 if exit == "WIN" else 0xe74c3c
                    msg = f"**VENTE {s}** | PnL: {pnl:.2f}$ | Nouvel État: {mood}"
                    if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
                    save_brain()

            # ACHATS
            if len(brain['holdings']) < 5:
                for s in WATCHLIST:
                    if s in brain['holdings']: continue
                    
                    df = get_live_data(s)
                    if df is None: continue
                    row = df.iloc[-1]
                    
                    trend = "UP" if row['Close'] > row['SMA200'] else "DOWN"
                    
                    # 1. Consultation IA (Affectée par l'humeur)
                    ia_score = consult_adaptive_ai(s, row['RSI'], trend)
                    
                    # 2. Filtre Mathématique (Q-Learning simplifié ici)
                    math_ok = (row['RSI'] < 35 and trend == "UP")
                    
                    if math_ok and ia_score == 100:
                        price = row['Close']
                        
                        # APPLICATION DES MODIFICATEURS ÉMOTIONNELS
                        # Mise de base 500$ * Multiplicateur Emotionnel (0.5x si stress, 1.2x si confiant)
                        bet_size = 500 * size_mult
                        
                        # Stop Loss ajusté par l'émotion (plus serré si stress)
                        sl_dist = row['ATR'] * 2.0 * sl_mult
                        sl = price - sl_dist
                        tp = price + (sl_dist * 2.0)
                        
                        brain['holdings'][s] = {"qty": bet_size/price, "entry": price, "stop": sl, "tp": tp}
                        brain['cash'] -= bet_size
                        
                        fast_log(f"🧠 **DÉCISION {mood}:** J'achète {s}.\nMise ajustée: {bet_size:.2f}$ (Facteur {size_mult}x)")
                        
                        if DISCORD_WEBHOOK_URL: 
                            requests.post(DISCORD_WEBHOOK_URL, json={"content": f"🟢 **ACHAT {s}**\nMode: {mood}\nMise: {bet_size:.2f}$"})
                        save_brain()
            
            time.sleep(60)
        except Exception as e:
            print(e)
            time.sleep(10)

@app.route('/')
def index(): 
    emo = brain['emotions']
    return f"<h1>ADAPTIVE V58</h1><p>Mood: {emo['confidence']}% Conf | {emo['stress']}% Stress</p>"

def start_threads():
    threading.Thread(target=run_trading, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()
    threading.Thread(target=logger_worker, daemon=True).start()

load_brain()
start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
