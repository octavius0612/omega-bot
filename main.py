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
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
from flask import Flask
from datetime import datetime, time as dtime
import pytz
from github import Github
from textblob import TextBlob

app = Flask(__name__)

# --- 1. CLÉS ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL")
LEARNING_WEBHOOK_URL = os.environ.get("LEARNING_WEBHOOK_URL")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")

# --- 2. CONFIGURATION 100% ACTIONS ---
# Uniquement les géants de Wall Street
WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "MSFT", "AMD", "PLTR", "META", "GOOG", "NFLX", "JPM", "LLY"]
INITIAL_CAPITAL = 50000.0 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "params": {"rsi_buy": 30, "stop_loss_atr": 2.0},
    "best_pnl": 0
}
bot_status = "Ouverture du terminal Wall Street..."

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)

# --- 3. MODULE SOCIAL (ACTIONS UNIQUEMENT) ---
def get_social_sentiment(symbol):
    """
    Analyse StockTwits pour les ACTIONS.
    """
    try:
        # Pas besoin de conversion .X pour les actions, le symbole suffit (ex: AAPL)
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        data = r.json()
        
        messages = data['messages']
        texts = [m['body'] for m in messages[:15]]
        full_text = " | ".join(texts)
        
        polarity = TextBlob(full_text).sentiment.polarity
        
        # Calcul Hype
        recent = sum(1 for m in messages if "less than a minute" in m['created_at'] or "minutes ago" in m['created_at'])
        hype = recent / 15 * 100 
        
        return {"text": full_text[:1000], "polarity": polarity, "hype": hype}
    except:
        return {"text": "R.A.S", "polarity": 0, "hype": 0}

# --- 4. GRAPHIQUE ---
def generate_chart(symbol, df, entry, sl, tp):
    try:
        filename = f"/tmp/{symbol}_chart.png"
        data = df.tail(50)
        # Style professionnel Wall Street
        mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc, style='yahoo')
        
        mpf.plot(data, type='candle', style=s, title=f"{symbol} - EQUITY TRADE",
                 hlines=dict(hlines=[entry, sl, tp], colors=['blue','red','green'], linewidths=[1,1,1], linestyle='-.'),
                 savefig=filename)
        
        with open(filename, 'rb') as f:
            requests.post(DISCORD_WEBHOOK_URL, 
                          data={"content": f"📊 **ORDRE BOURSE : {symbol}**"}, 
                          files={"file": f})
    except: pass

# --- 5. CERVEAU ---
def consult_titan(symbol, tech, social):
    prompt = f"""
    Tu es un Trader Senior à Wall Street (Equity Strategist).
    Action : {symbol}.
    
    SENTIMENT MARCHÉ :
    - Foule (StockTwits) : {social['polarity']:.2f} (-1=Peur, 1=Greed).
    - Hype : {social['hype']:.0f}%.
    
    TECHNIQUE :
    - RSI : {tech['rsi']:.1f}.
    - Tendance : {tech['trend']}.
    - Support Institutionnel : {tech['inst']:.1f}%.
    
    DÉCISION :
    Nous cherchons des configurations "Blue Chip" parfaites.
    - Achat sur repli (Dip) si RSI < 35 et Sentiment > 0 (La foule est confiante malgré la baisse).
    - Achat sur force (Breakout) si Tendance HAUSSIERE et Institutions > 60%.
    
    JSON : {{"decision": "BUY/WAIT", "score": 0-100, "reason": "Analyse Wall Street"}}
    """
    try:
        model = genai.GenerativeModel('gemini-pro')
        res = model.generate_content(prompt)
        return json.loads(res.text.replace("```json","").replace("```",""))
    except: return {"decision": "WAIT", "score": 0}

# --- 6. UTILS ---
def log_thought(emoji, text):
    if LEARNING_WEBHOOK_URL: requests.post(LEARNING_WEBHOOK_URL, json={"content": f"{emoji} **TITAN:** {text}"})

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
            repo.update_file("brain.json", "Update", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 7. TRAINING ---
def run_eternal_learning():
    log_thought("📚", "Module d'analyse financière activé en arrière-plan.")
    while True:
        time.sleep(300)

# --- 8. MOTEUR ---
def get_tech_data(s):
    try:
        df = yf.Ticker(s).history(period="1mo", interval="15m")
        if df.empty: return None
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        trend = "HAUSSIER" if df['Close'].iloc[-1] > df['EMA50'].iloc[-1] else "BAISSIER"
        
        return {
            "p": df['Close'].iloc[-1], "rsi": df['RSI'].iloc[-1], 
            "atr": df['ATR'].iloc[-1], "trend": trend, "df": df,
            "inst": yf.Ticker(s).info.get('heldPercentInstitutions', 0)*100
        }
    except: return None

def run_trading():
    global brain, bot_status
    load_brain()
    
    log_thought("🏛️", "TITAN connecté au NYSE/NASDAQ. Prêt pour l'ouverture.")
    
    count = 0
    while True:
        try:
            count += 1
            if count >= 10: save_brain(); count = 0
            
            # Horaires Bourse US (New York)
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            
            # Lundi(0) à Vendredi(4), de 9h30 à 16h00
            is_market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if not is_market_open:
                bot_status = "🌙 Bourse Fermée (Analyse de fond)"
                # Ici, on pourrait lancer une analyse de bilan comptable...
                time.sleep(60)
                continue

            bot_status = "🔔 Marché Ouvert - Scan..."
            
            # Gestion Positions (Vente)
            for s in list(brain['holdings'].keys()):
                pos = brain['holdings'][s]
                curr = yf.Ticker(s).fast_info['last_price']
                if curr < pos['stop']:
                    brain['cash'] += pos['qty'] * curr
                    del brain['holdings'][s]
                    if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json={"content": f"🔴 VENTE {s} (Protection)"})
                    save_brain()

            # Scan Achat
            if len(brain['holdings']) < 5:
                for s in WATCHLIST:
                    if s in brain['holdings']: continue
                    
                    tech = get_tech_data(s)
                    if not tech: continue
                    
                    social = get_social_sentiment(s)
                    decision = consult_titan(s, tech, social)
                    
                    if decision['decision'] == "BUY" and decision['score'] >= 85 and brain['cash'] > 500:
                        bet = brain['cash'] * 0.15 
                        brain['cash'] -= bet
                        qty = bet / tech['p']
                        sl = tech['p'] - (tech['atr'] * brain['params']['stop_loss_atr'])
                        tp = tech['p'] + (tech['atr'] * 3.0)
                        
                        brain['holdings'][s] = {"qty": qty, "entry": tech['p'], "stop": sl}
                        
                        msg = {
                            "embeds": [{
                                "title": f"🏦 ORDRE WALL STREET : {s}",
                                "description": decision['reason'],
                                "color": 0x2ecc71,
                                "fields": [
                                    {"name": "Social", "value": f"{social['polarity']:.2f}", "inline": True},
                                    {"name": "Institutions", "value": f"{tech['inst']:.1f}%", "inline": True}
                                ]
                            }]
                        }
                        if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json=msg)
                        
                        generate_chart(s, tech['df'], tech['p'], sl, tp)
                        save_brain()
            
            time.sleep(60)
        except Exception as e:
            print(f"Erreur: {e}")
            time.sleep(10)

@app.route('/')
def index(): return f"<h1>WALL STREET TITAN V33</h1><p>{bot_status}</p>"

def start_threads():
    threading.Thread(target=run_trading, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()
    threading.Thread(target=run_eternal_learning, daemon=True).start()

start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
