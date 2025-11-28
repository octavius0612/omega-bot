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
import io
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
from datetime import datetime
from github import Github

app = Flask(__name__)

# --- CLÉS ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")

brain = {
    "cash": 50000.0, 
    "holdings": {}, 
    "colab_signal": None, 
    "logs": []
}

bot_state = {"status": "En attente du Cerveau Colab...", "last_log": "Init"}

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- FONCTIONS ALGORITHMIQUES (RENDER) ---
def check_whale(symbol):
    try:
        df = yf.Ticker(symbol).history(period="5d", interval="1h")
        vol = df['Volume'].iloc[-1]
        avg = df['Volume'].mean()
        return (vol > avg * 3.0), f"Vol x{vol/avg:.1f}"
    except: return False, "N/A"

def check_fibonacci(symbol):
    try:
        df = yf.Ticker(symbol).history(period="1mo", interval="1d")
        high = df['High'].max()
        low = df['Low'].min()
        curr = df['Close'].iloc[-1]
        fib618 = high - ((high - low) * 0.618)
        return (abs(curr - fib618) / curr < 0.02), f"Fibo {fib618:.2f}"
    except: return False, "N/A"

def get_gemini_vision(symbol):
    try:
        df = yf.Ticker(symbol).history(period="1mo", interval="1h")
        buf = io.BytesIO()
        mpf.plot(df.tail(50), type='candle', style='nightclouds', savefig=buf)
        buf.seek(0)
        img = Image.open(buf)
        res = model.generate_content(["Score achat 0.0-1.0 ?", img])
        return float(res.text.strip())
    except: return 0.5

# --- RÉCEPTION DU COLAB ---
@app.route('/receive_super_signal', methods=['POST'])
def receive_super_signal():
    data = request.json
    symbol = data.get('symbol')
    score_ai = data.get('score')
    
    log_msg = f"🧠 **COLAB SIGNAL REÇU:** {symbol} (Score IA: {score_ai:.2f})"
    print(log_msg)
    brain['colab_signal'] = {"s": symbol, "score": score_ai, "time": time.time()}
    
    # Lancement de la vérification de sécurité Render
    threading.Thread(target=process_colab_order, args=(symbol, score_ai)).start()
    
    return jsonify({"status": "ACK"}), 200

def process_colab_order(symbol, ai_score):
    """
    Le Colab a dit d'acheter. Render vérifie si c'est prudent.
    """
    is_whale, whale_msg = check_whale(symbol)
    is_fibo, fibo_msg = check_fibonacci(symbol)
    vis_score = get_gemini_vision(symbol)
    
    final_decision = False
    reason = ""
    
    if ai_score > 0.9:
        final_decision = True
        reason = "IA Confidence Max"
    elif ai_score > 0.7 and (is_whale or is_fibo or vis_score > 0.7):
        final_decision = True
        reason = f"IA + Confirmation ({whale_msg}, Vis:{vis_score:.2f})"
        
    if final_decision:
        try:
            price = yf.Ticker(symbol).fast_info['last_price']
            qty = 2000 / price
            brain['holdings'][symbol] = {"qty": qty, "entry": price}
            brain['cash'] -= 2000
            
            msg = {
                "embeds": [{
                    "title": f"🌌 HYBRID GOD ORDER : {symbol}",
                    "description": f"**Raison:** {reason}",
                    "color": 0x2ecc71,
                    "fields": [
                        {"name": "Cerveau Colab", "value": f"{ai_score*100:.1f}%", "inline": True},
                        {"name": "Yeux Render", "value": f"{vis_score:.2f}", "inline": True},
                        {"name": "Algorithme", "value": f"{whale_msg} | {fibo_msg}", "inline": False}
                    ]
                }]
            }
            if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json=msg)
        except: pass

@app.route('/')
def index():
    return f"<h1>RENDER BODY V128 (FIXED)</h1><p>Status: {bot_state['status']}</p><p>Cash: {brain['cash']}</p>"

# Cette fonction sert juste à garder le thread principal actif
def keep_alive():
    while True:
        time.sleep(60)

if __name__ == "__main__":
    # On lance un thread de fond pour éviter que Gunicorn ne tue le processus s'il n'y a pas de requêtes
    threading.Thread(target=keep_alive, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
