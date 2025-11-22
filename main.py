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

# --- 2. CONFIG ---
WATCHLIST_LIVE = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR"]
INITIAL_CAPITAL = 50000.0 

# Structure de Mémoire Avancée
brain = {
    "cash": INITIAL_CAPITAL,
    "holdings": {},
    # Le portefeuille de stratégies actives (Max 3)
    "active_strategies": {}, 
    # Cimetière des mauvaises idées (pour ne pas les refaire)
    "graveyard": [],
    "generation": 0
}
bot_status = "Initialisation du Fonds Quantique..."

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)

# --- 3. OUTILS ---
def log_thought(emoji, text):
    print(f"{emoji} {text}")
    if LEARNING_WEBHOOK_URL: requests.post(LEARNING_WEBHOOK_URL, json={"content": f"{emoji} **QUANTUM FUND:** {text}"})

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
        loaded = json.loads(c.decoded_content.decode())
        if "active_strategies" in loaded: brain.update(loaded)
    except: pass

def save_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        content = json.dumps(brain, indent=4)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Quantum Update", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 4. USINE A STRATÉGIES (L'ARCHITECTE) ---
def generate_new_strategy_idea():
    """Demande à Gemini d'inventer une stratégie complexe en Python"""
    
    # On donne des briques de base à l'IA
    indicators = "RSI, EMA50, EMA200, BB_UPPER, BB_LOWER, ATR, ADX, VOLUME"
    
    prompt = f"""
    Agis comme un Quants Developer Senior.
    Ta mission : Inventer une stratégie de trading algorithmique inédite.
    
    Variables disponibles (déjà calculées dans le DataFrame 'row') :
    - row['RSI'] (0-100)
    - row['EMA50'], row['EMA200'] (Tendances)
    - row['BBU'], row['BBL'] (Bandes Bollinger Haute/Basse)
    - row['ADX'] (Force tendance 0-100)
    - row['Close'] (Prix actuel)
    
    Tâche :
    Écris SEULEMENT le corps d'une fonction Python qui retourne :
    - 100 pour ACHAT FORT
    - 0 pour NEUTRE
    
    Le code doit être créatif. Utilise des croisements, des seuils dynamiques.
    Exemple format attendu (sans markdown) :
    
    signal = 0
    if row['Close'] < row['BBL'] and row['RSI'] < 30: signal = 100
    if row['ADX'] > 25 and row['Close'] > row['EMA50']: signal = 100
    """
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        res = model.generate_content(prompt)
        code = res.text.replace("```python", "").replace("```", "").strip()
        # On donne un nom cool à la strat
        name = f"STRAT_GEN_{brain['generation']}_{random.randint(100,999)}"
        return name, code
    except: return None, None

def backtest_code(strategy_code):
    """Teste le code généré sur des données historiques réelles"""
    try:
        # Données de test (NVDA sur 1 mois)
        df = yf.Ticker("NVDA").history(period="1mo", interval="1h")
        if df.empty: return -999, 0, 0
        
        # Calcul des indicateurs pour que le code de l'IA fonctionne
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
        entry_price = 0
        trades = 0
        wins = 0
        
        # Simulation
        for index, row in df.iterrows():
            # Environnement d'exécution sécurisé
            local_vars = {'row': row, 'signal': 0}
            try:
                exec(strategy_code, {}, local_vars)
                signal = local_vars['signal']
            except: signal = 0
            
            # Logique Trading simple pour test
            if position == 0 and signal == 100:
                position = capital / row['Close']
                entry_price = row['Close']
                capital = 0
            elif position > 0:
                # Sortie fixe (Take Profit 5% / Stop Loss 3%) pour valider la qualité de l'entrée
                if row['High'] > entry_price * 1.05:
                    capital = position * (entry_price * 1.05)
                    position = 0
                    trades += 1
                    wins += 1
                elif row['Low'] < entry_price * 0.97:
                    capital = position * (entry_price * 0.97)
                    position = 0
                    trades += 1
        
        final_value = capital + (position * df['Close'].iloc[-1])
        pnl = final_value - 10000
        win_rate = (wins / trades * 100) if trades > 0 else 0
        
        return pnl, trades, win_rate
        
    except Exception as e:
        return -999, 0, 0

def manage_strategies_lifecycle():
    """
    Le DRH du Hedge Fund : Embauche et Vire des stratégies.
    """
    global brain, bot_status
    
    log_thought("👔", "Réunion du Comité Stratégique. Analyse des performances...")
    
    # 1. Élimination des faibles
    to_delete = []
    for name, stats in brain['active_strategies'].items():
        # Si une stratégie perd de l'argent après 10 trades réels -> Poubelle
        if stats['real_pnl'] < -100:
            log_thought("🗑️", f"La stratégie {name} a échoué (PnL: {stats['real_pnl']}$). Licenciée.")
            to_delete.append(name)
            brain['graveyard'].append(stats['code'])
    
    for name in to_delete:
        del brain['active_strategies'][name]
    
    # 2. Recrutement (Si on a de la place, max 3 strats)
    while len(brain['active_strategies']) < 3:
        brain['generation'] += 1
        name, code = generate_new_strategy_idea()
        if not code: continue
        
        log_thought("🧪", f"Test du candidat {name}...")
        pnl, trades, wr = backtest_code(code)
        
        if pnl > 200 and trades > 5: # Critères d'embauche stricts
            log_thought("🤝", f"**EMBAUCHÉ !** {name} a généré +{pnl:.2f}$ en backtest (WinRate: {wr:.1f}%).")
            brain['active_strategies'][name] = {
                "code": code,
                "backtest_pnl": pnl,
                "real_pnl": 0.0, # PnL réel commence à 0
                "trades": 0
            }
            save_brain()
        else:
            log_thought("❌", f"Candidat {name} rejeté (PnL faible ou bug).")
            time.sleep(2)

# --- 5. TRADING TEMPS RÉEL ---
def run_trading_engine():
    global brain, bot_status
    load_brain()
    
    while True:
        # Gestion Horaire
        nyc = pytz.timezone('America/New_York')
        now = datetime.now(nyc)
        market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
        
        if not market_open:
            bot_status = "🌙 Bourse Fermée - Recrutement Stratégies"
            manage_strategies_lifecycle()
            time.sleep(60)
            continue
            
        bot_status = "🟢 Trading Actif (Multi-Stratégies)"
        
        # Scan du marché avec TOUTES les stratégies actives
        for s in WATCHLIST_LIVE:
            # On ignore si déjà en portefeuille
            if s in brain['holdings']: continue
            
            try:
                # Récup Data Live
                df = yf.Ticker(s).history(period="1mo", interval="15m")
                if df.empty: continue
                
                # Calcul Indicateurs
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['EMA50'] = ta.ema(df['Close'], length=50)
                df['EMA200'] = ta.ema(df['Close'], length=200)
                df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
                bb = ta.bbands(df['Close'], length=20)
                df['BBU'] = bb['BBU_20_2.0']
                df['BBL'] = bb['BBL_20_2.0']
                row = df.iloc[-1]
                
                # On demande à chaque stratégie son avis
                for strat_name, strat_data in brain['active_strategies'].items():
                    local_vars = {'row': row, 'signal': 0}
                    try:
                        exec(strat_data['code'], {}, local_vars)
                        if local_vars['signal'] == 100:
                            # BINGO ! Une stratégie veut acheter
                            price = row['Close']
                            qty = 200 / price # Mise fixe 200$ pour tester
                            
                            # SL/TP Standard
                            atr = ta.atr(df['High'], df['Low'], df['Close'], length=14).iloc[-1]
                            sl = price - (atr * 2)
                            
                            brain['holdings'][s] = {
                                "qty": qty, "entry": price, "stop": sl,
                                "strategy_origin": strat_name # On note qui a pris la décision
                            }
                            brain['cash'] -= 200
                            
                            msg = f"🟢 **ACHAT {s}** par la stratégie `{strat_name}`\nPrix: {price:.2f}$"
                            if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
                            save_brain()
                            break # Une seule stratégie suffit pour déclencher
                    except: pass
            except: pass
            
        # Gestion des Ventes (Commune à toutes les strats)
        for s in list(brain['holdings'].keys()):
            pos = brain['holdings'][s]
            curr = yf.Ticker(s).fast_info['last_price']
            pnl_val = (curr - pos['entry']) * pos['qty']
            
            exit = False
            if curr < pos['stop']: exit = "STOP LOSS"
            elif curr > pos['entry'] * 1.05: exit = "TAKE PROFIT"
            
            if exit:
                brain['cash'] += pos['qty'] * curr
                strat_name = pos['strategy_origin']
                
                # Mise à jour du score de la stratégie responsable
                if strat_name in brain['active_strategies']:
                    brain['active_strategies'][strat_name]['real_pnl'] += pnl_val
                    brain['active_strategies'][strat_name]['trades'] += 1
                
                del brain['holdings'][s]
                if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json={"content": f"🔴 VENTE {s} ({exit}) | PnL: {pnl_val:.2f}$"})
                save_brain()
        
        time.sleep(60)

@app.route('/')
def index(): 
    strats = "<br>".join([f"<b>{k}</b>: PnL Réel {v['real_pnl']:.2f}$" for k,v in brain['active_strategies'].items()])
    return f"<h1>QUANTUM FUND V41</h1><p>{bot_status}</p><h3>Stratégies Actives:</h3>{strats}"

def start_threads():
    t1 = threading.Thread(target=run_trading_engine, daemon=True)
    t1.start()
    t2 = threading.Thread(target=run_heartbeat, daemon=True)
    t2.start()

load_brain()
start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
