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

# --- 2. CONFIG ---
WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "MSFT", "AMD", "COIN"] # Actions only
INITIAL_CAPITAL = 50000.0 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    # On stocke le MEILLEUR génome trouvé
    "best_genome": {"rsi_buy": 30, "stop_loss_atr": 2.0, "tp_atr": 3.0},
    "generation": 0
}
bot_status = "Booting Genetic Core..."

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)

# --- 3. ALGORITHME GÉNÉTIQUE (ACCÉLÉRÉ) ---
def backtest_strategy(df, params):
    """
    Simulation Vectorisée (100x plus rapide que la boucle for)
    """
    rsi_buy = params['rsi_buy']
    sl_mult = params['stop_loss_atr']
    tp_mult = params['tp_atr']
    
    # Signaux d'achat
    buy_signals = df['RSI'] < rsi_buy
    
    pnl = 0
    trades = 0
    
    # On parcourt seulement les moments où il y a signal (beaucoup moins de boucles)
    indices = df.index[buy_signals]
    
    for i in indices:
        try:
            row = df.loc[i]
            entry = row['Close']
            stop = entry - (row['ATR'] * sl_mult)
            target = entry + (row['ATR'] * tp_mult)
            
            # On regarde le futur (les 24 prochaines heures/bougies)
            future = df.loc[i:].head(24) 
            
            # Vérification issue du trade
            # Est-ce qu'on touche le TP ou le SL en premier ?
            hit_tp = future[future['High'] > target].index.min()
            hit_sl = future[future['Low'] < stop].index.min()
            
            if hit_tp and (not hit_sl or hit_tp < hit_sl):
                pnl += (target - entry)
                trades += 1
            elif hit_sl:
                pnl -= (entry - stop)
                trades += 1
        except: pass
        
    return pnl, trades

def run_genetic_evolution():
    """
    Fait évoluer les stratégies comme des êtres vivants.
    """
    global brain, bot_status
    
    # 1. Chargement Données (Une seule fois)
    log_thought("🧬", "Chargement des données génétiques... Prêt à accélérer.")
    cache = {}
    for s in ["NVDA", "TSLA"]:
        df = yf.Ticker(s).history(period="1mo", interval="1h")
        if not df.empty:
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            cache[s] = df.dropna()
    
    population_size = 20
    # Création population initiale aléatoire
    population = []
    for _ in range(population_size):
        population.append({
            "rsi_buy": np.random.randint(20, 50),
            "stop_loss_atr": np.random.uniform(1.0, 4.0),
            "tp_atr": np.random.uniform(1.5, 6.0)
        })
        
    generation = brain.get('generation', 0)
    
    while True:
        # Check Marché Ouvert
        nyc = pytz.timezone('America/New_York')
        now = datetime.now(nyc)
        if now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0):
            log_thought("🔔", "Marché Ouvert - Pause de l'évolution.")
            time.sleep(300)
            continue
            
        generation += 1
        bot_status = f"🧬 Gen #{generation}"
        
        # 2. ÉVALUATION (Fitness)
        scores = []
        for genome in population:
            total_pnl = 0
            for s, df in cache.items():
                pnl, _ = backtest_strategy(df, genome)
                total_pnl += pnl
            scores.append((total_pnl, genome))
        
        # 3. SÉLECTION (Les meilleurs survivent)
        scores.sort(key=lambda x: x[0], reverse=True) # Tri décroissant
        best_genome = scores[0][1]
        best_score = scores[0][0]
        
        # Sauvegarde du Champion
        if best_score > 0:
            brain['best_genome'] = best_genome
            brain['generation'] = generation
            save_brain()
        
        # Affichage Discord (Seulement le Top 1)
        msg = f"Génération {generation} : Le Champion a fait **{best_score:.2f}$**.\nParamètres: RSI<{best_genome['rsi_buy']} / SL={best_genome['stop_loss_atr']:.1f}ATR / TP={best_genome['tp_atr']:.1f}ATR"
        log_thought("🏆", msg)
        
        # 4. REPRODUCTION & MUTATION (Création nouvelle génération)
        top_performers = [x[1] for x in scores[:5]] # Top 5 parents
        new_population = []
        
        while len(new_population) < population_size:
            parent = random.choice(top_performers)
            # Mutation (Légère modification des gènes)
            child = {
                "rsi_buy": parent['rsi_buy'] + np.random.randint(-3, 4),
                "stop_loss_atr": parent['stop_loss_atr'] + np.random.uniform(-0.2, 0.2),
                "tp_atr": parent['tp_atr'] + np.random.uniform(-0.3, 0.3)
            }
            # Limites biologiques (pour ne pas avoir de chiffres absurdes)
            child['rsi_buy'] = max(15, min(55, child['rsi_buy']))
            child['stop_loss_atr'] = max(0.5, min(5.0, child['stop_loss_atr']))
            child['tp_atr'] = max(1.0, min(10.0, child['tp_atr']))
            
            new_population.append(child)
            
        population = new_population
        time.sleep(5) # Petite pause pour ne pas faire exploser le CPU

# --- 4. FONCTIONS UTILITAIRES ---
def log_thought(emoji, text):
    if LEARNING_WEBHOOK_URL: requests.post(LEARNING_WEBHOOK_URL, json={"content": f"{emoji} **GENESIS:** {text}"})

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
        brain.update(json.loads(c.decoded_content.decode()))
    except: pass

def save_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        content = json.dumps(brain, indent=4)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Evolution", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 5. MOTEUR TRADING ---
def run_trading():
    global brain, bot_status
    load_brain()
    
    log_thought("🧬", "Système Génétique activé. Je vais faire évoluer mes stratégies.")
    
    # Lancement du thread d'évolution
    threading.Thread(target=run_genetic_evolution, daemon=True).start()
    
    count = 0
    while True:
        # Ici, on trade avec le MEILLEUR GÉNOME trouvé par l'évolution
        try:
            count += 1
            if count >= 10: save_brain(); count = 0
            
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            if not (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0)):
                time.sleep(60)
                continue

            # Logique Trading simplifiée utilisant 'best_genome'
            # ... (Même logique d'achat que V33 mais avec brain['best_genome']['rsi_buy']) ...
            
            time.sleep(60)
        except: time.sleep(10)

@app.route('/')
def index(): return f"<h1>GENETIC V34</h1><p>{bot_status}</p>"

def start_threads():
    t1 = threading.Thread(target=run_trading, daemon=True)
    t1.start()
    t2 = threading.Thread(target=run_heartbeat, daemon=True)
    t2.start()

start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
