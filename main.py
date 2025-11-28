import os
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import time
import threading
from flask import Flask
from github import Github

app = Flask(__name__)

# CLÉS
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")

WATCHLIST = ["NVDA", "TSLA", "AAPL", "BTC-USD", "ETH-USD"]
bot_state = {"status": "Attente Cerveau...", "model_version": "Aucune"}

# 1. TÉLÉCHARGEMENT DU CERVEAU (Synchronisation)
def update_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        
        # On cherche le fichier model.zip
        contents = repo.get_contents("omega_model.zip")
        
        # On le télécharge en local
        with open("omega_model.zip", "wb") as f:
            f.write(contents.decoded_content)
            
        bot_state['model_version'] = contents.last_modified
        return True
    except: return False

# 2. PRÉPARATION DES DONNÉES (Mêmes yeux que le Colab)
def get_observation(symbol):
    try:
        df = yf.Ticker(symbol).history(period="60d", interval="1h")
        if len(df) < 50: return None
        
        # Indicateurs Techniques (Inputs du Neurone)
        df['RSI'] = ta.rsi(df['Close'], 14)
        df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)
        df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], 14)
        
        df = df.dropna()
        row = df.iloc[-1]
        
        # Normalisation (Indispensable pour l'IA)
        obs = np.array([
            row['RSI'] / 100,
            row['MACD'],
            row['ATR'] / row['Close'],
            row['CCI'] / 100
        ], dtype=np.float32)
        
        return obs, row['Close']
    except: return None, 0

# 3. MOTEUR D'EXÉCUTION
def run_ai_trader():
    print("🟢 Démarrage du Trader Neuronal...")
    model = None
    
    while True:
        # Mise à jour du cerveau si disponible
        if update_brain():
            try:
                model = PPO.load("omega_model")
                print("🧠 Nouveau Cerveau Chargé !")
            except: pass
        
        if model:
            bot_state['status'] = "🧠 IA ACTIVE"
            for s in WATCHLIST:
                obs, price = get_observation(s)
                if obs is not None:
                    # L'IA DÉCIDE (0=Wait, 1=Buy, 2=Sell)
                    action, _ = model.predict(obs)
                    
                    if action == 1: # BUY
                        msg = f"🧠 **SIGNAL NEURONAL : {s}**\nAction: ACHAT\nPrix: {price:.2f}$"
                        # (Code d'achat réel ici...)
                        print(msg)
        else:
            bot_state['status'] = "⚠️ Pas de cerveau détecté"
        
        time.sleep(60) # Scan chaque minute

@app.route('/')
def index(): return f"<h1>NEURAL HOST V128</h1><p>Status: {bot_state['status']}</p>"

threading.Thread(target=run_ai_trader, daemon=True).start()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
