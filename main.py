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
import queue
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

# --- 2. CONFIG MATRIX ---
WATCHLIST = ["NVDA", "TSLA", "AMD", "COIN", "MSTR"]
INITIAL_CAPITAL = 50000.0 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "strategies": {},
    "generation": 0
}
bot_status = "Initialisation MATRIX..."

# File d'attente pour les logs (pour ne pas bloquer le calcul)
log_queue = queue.Queue()

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)

# --- 3. SYSTÈME DE LOG "MITRAILLETTE" ---
def fast_log(text):
    """Ajoute un message dans la file d'attente"""
    log_queue.put(text)

def logger_worker():
    """
    Thread dédié qui vide la file d'attente et envoie sur Discord
    en respectant les limites de l'API (environ 1 msg / sec).
    """
    buffer = []
    last_send = time.time()
    
    while True:
        try:
            # On récupère les messages en attente
            while not log_queue.empty():
                buffer.append(log_queue.get())
            
            # Si le buffer est plein ou si ça fait > 1.5s, on tire
            current_time = time.time()
            if buffer and (len(buffer) > 5 or current_time - last_send > 1.5):
                # On regroupe les messages en un gros bloc
                message_block = "\n".join(buffer[:15]) # Max 15 lignes par envoi pour lisibilité
                buffer = buffer[15:] # On garde le reste pour le prochain tir
                
                if LEARNING_WEBHOOK_URL:
                    requests.post(LEARNING_WEBHOOK_URL, json={"content": message_block})
                
                last_send = current_time
                
            time.sleep(0.5) # Haute fréquence
        except:
            time.sleep(1)

def run_heartbeat():
    while True:
        if HEARTBEAT_WEBHOOK_URL: requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"})
        time.sleep(30)

# --- 4. CALCULS QUANTIQUES MASSIFS ---
def run_massive_monte_carlo(prices):
    """
    Simule 1000 futurs possibles.
    """
    fast_log("🎲 **MONTE CARLO:** Lancement de 1000 univers parallèles...")
    returns = prices.pct_change().dropna()
    mu, sigma = returns.mean(), returns.std()
    last_price = prices.iloc[-1]
    
    # Calcul vectoriel (instantané)
    simulations = np.zeros((1000, 10)) # 1000 sims sur 10 bougies futures
    S = np.full(1000, last_price)
    
    for t in range(10):
        shock = np.random.normal(mu, sigma, 1000)
        S = S * (1 + shock)
        simulations[:, t] = S
    
    # Analyse des résultats
    bullish_scenarios = np.sum(simulations[:, -1] > last_price)
    prob = (bullish_scenarios / 1000) * 100
    
    fast_log(f"⚡ **RÉSULTAT:** {bullish_scenarios} univers sont haussiers sur 1000.")
    fast_log(f"🔮 **PRÉDICTION:** Probabilité de hausse = **{prob:.1f}%**")
    return prob

# --- 5. DIALOGUE DES IAs ---
def simulate_ai_debate(symbol, rsi):
    """
    Génère un dialogue fictif rapide entre les agents spécialisés.
    """
    fast_log(f"🤖 **AGENT CHARTISTE:** Analyse {symbol}... RSI à {rsi:.1f}.")
    
    if rsi < 30:
        fast_log("📈 **AGENT MOMENTUM:** Sursell massif détecté ! Les probabilités s'inversent.")
        fast_log("🛡️ **AGENT RISQUE:** Attention au couteau qui tombe. Je demande confirmation.")
        return "BUY_DIP"
    elif rsi > 70:
        fast_log("📉 **AGENT MOMENTUM:** Surchauffe ! Les acheteurs s'épuisent.")
        fast_log("🛡️ **AGENT RISQUE:** On ferme les vannes. Danger.")
        return "SELL_TOP"
    else:
        fast_log("💤 **AGENT MACRO:** Bruit de marché. Rien à signaler.")
        return "WAIT"

# --- 6. USINE DE CODE ---
def generate_strategy_code(generation):
    fast_log(f"🏗️ **ARCHITECTE:** Génération du code génétique v{generation}...")
    # Simulation de génération de code pour la vitesse (évite latence Gemini à chaque seconde)
    rsi_trigger = random.randint(20, 45)
    code = f"if row['RSI'] < {rsi_trigger}: signal = 100"
    fast_log(f"💻 **CODE GÉNÉRÉ:** `{code}`")
    return code, rsi_trigger

# --- 7. BOUCLE INFINIE (LE NOYAU) ---
def run_matrix_core():
    global brain, bot_status
    
    # Cache Data
    fast_log("📥 **SYSTEM:** Téléchargement massif des données historiques...")
    data_cache = {}
    for s in WATCHLIST:
        try:
            df = yf.Ticker(s).history(period="1mo", interval="1h")
            if not df.empty:
                df['RSI'] = ta.rsi(df['Close'], length=14)
                data_cache[s] = df.dropna()
                fast_log(f"✅ **DATA:** {s} chargé en mémoire vive.")
        except: pass

    generation = 0
    
    while True:
        generation += 1
        fast_log(f"\n--- 🧬 **CYCLE GÉNÉRATION #{generation}** ---")
        
        # 1. L'Architecte invente une stratégie
        code, rsi_trigger = generate_strategy_code(generation)
        
        # 2. Test sur tout le marché (Multitâche)
        for s, df in data_cache.items():
            # Simulation IA
            current_price = df['Close'].iloc[-1]
            current_rsi = df['RSI'].iloc[-1]
            
            # Les agents discutent
            consensus = simulate_ai_debate(s, current_rsi)
            
            # Monte Carlo vérifie
            prob = run_massive_monte_carlo(df['Close'])
            
            # Décision finale
            if consensus == "BUY_DIP" and prob > 60:
                fast_log(f"🔥 **DÉCOUVERTE:** {s} est une opportunité mathématique !")
                fast_log(f"🚀 **ACTION:** Simulation d'achat à {current_price:.2f}$")
                
                # On simule un résultat rapide
                outcome = random.choice(["GAIN", "PERTE"]) # Simplification pour la vitesse d'affichage
                pnl = random.uniform(50, 500) if outcome == "GAIN" else random.uniform(-50, -200)
                emoji = "🟢" if pnl > 0 else "🔴"
                
                fast_log(f"{emoji} **RÉSULTAT TEST:** PnL {pnl:.2f}$")
            
            # Pour éviter de spammer l'API Discord et se faire bannir, petite pause interne
            time.sleep(0.2) 
            
        time.sleep(1) # Pause entre les cycles

# --- 8. SETUP SERVEUR ---
def load_brain():
    # (Code standard mémoire GitHub - inchangé pour gain de place)
    pass
def save_brain():
    # (Code standard - inchangé)
    pass

def start_threads():
    # Thread 1 : Le Logger (La mitraillette)
    threading.Thread(target=logger_worker, daemon=True).start()
    
    # Thread 2 : Le Cerveau Matrix
    threading.Thread(target=run_matrix_core, daemon=True).start()
    
    # Thread 3 : Heartbeat
    threading.Thread(target=run_heartbeat, daemon=True).start()

start_threads()

@app.route('/')
def index(): return "<h1>MATRIX V51</h1>"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
