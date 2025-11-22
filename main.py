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

# --- 2. CONFIG ---
WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "MSFT", "AMD", "COIN", "MSTR", "ETH.X", "BTC-USD"]
INITIAL_CAPITAL = 50000.0 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "params": {"rsi_buy": 30, "stop_loss_atr": 2.0},
    "social_cache": {} # Mémoire sociale
}
bot_status = "Connexion au Hivemind..."

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)

# --- 3. MODULE SOCIAL (ANALYSIS TWITTER/STOCKTWITS) ---
def get_social_sentiment(symbol):
    """
    Aspire les discussions en temps réel sur StockTwits (Proxy Twitter Finance).
    Analyse la psychologie des foules.
    """
    try:
        # Nettoyage du symbole (Yahoo utilise '-' mais StockTwits non)
        clean_symbol = symbol.replace("-USD", ".X").replace("BTC", "BTC.X").replace("ETH", "ETH.X")
        
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{clean_symbol}.json"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        data = r.json()
        
        messages = data['messages']
        
        # 1. Extraction du texte brut
        texts = [m['body'] for m in messages[:15]] # Les 15 derniers messages
        full_text = " | ".join(texts)
        
        # 2. Analyse Sentiment Base (TextBlob)
        polarity = TextBlob(full_text).sentiment.polarity # -1 (Peur) à +1 (Euphorie)
        
        # 3. Analyse Volume (Hype)
        # On regarde si les messages sont récents (moins de 1h)
        recent_count = sum(1 for m in messages if "less than a minute" in m['created_at'] or "minutes ago" in m['created_at'])
        hype_score = recent_count / 15 * 100 # Pourcentage de messages très récents
        
        return {
            "text_sample": full_text[:1000], # On coupe pour pas saturer Gemini
            "polarity": polarity,
            "hype": hype_score,
            "volume": len(messages)
        }
    except Exception as e:
        print(f"Erreur Social: {e}")
        return {"text_sample": "Pas de données", "polarity": 0, "hype": 0}

# --- 4. SERVEUR GRAPHIQUE ---
def generate_chart(symbol, df, entry, sl, tp):
    try:
        filename = f"/tmp/{symbol}_chart.png"
        data = df.tail(60)
        mc = mpf.make_marketcolors(up='#2ecc71', down='#e74c3c', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc, style='nightclouds', grid_style=':')
        
        mpf.plot(data, type='candle', style=s, title=f"{symbol} - SIGNAL HIVEMIND",
                 hlines=dict(hlines=[entry, sl, tp], colors=['#3498db','#e74c3c','#2ecc71'], linewidths=[2,2,2], alpha=0.8),
                 volume=True, savefig=filename)
        
        with open(filename, 'rb') as f:
            requests.post(DISCORD_WEBHOOK_URL, 
                          data={"content": f"📸 **PREUVE GRAPHIQUE : {symbol}**"}, 
                          files={"file": f})
    except: pass

# --- 5. L'INTELLIGENCE SUPRÊME (GEMINI) ---
def consult_hivemind(symbol, tech_data, social_data):
    """
    Fusionne l'Analyse Technique et l'Analyse Sociale.
    """
    prompt = f"""
    Tu es THE HIVEMIND, une IA de Trading Neuro-Sociale.
    
    ACTIF : {symbol}
    
    1. ANALYSE SOCIALE (Psychologie des Foules) :
    - Ambiance : {social_data['polarity']:.2f} (-1=Panique, 1=Euphorie).
    - Hype (Vitesse des messages) : {social_data['hype']:.0f}% (Si > 50%, tout le monde en parle).
    - Échantillon des discussions : "{social_data['text_sample']}..."
    
    2. ANALYSE TECHNIQUE (Froid) :
    - RSI : {tech_data['rsi']:.2f}
    - Tendance : {tech_data['trend']}
    - Institutions : {tech_data['inst']:.1f}%
    
    MISSION :
    Détecte les opportunités.
    - Si RSI bas + Hype Sociale (Les gens disent "Buy the dip") -> ACHAT FORT.
    - Si RSI haut + Euphorie (Les gens disent "Moon") -> ATTENTION (Vente probable).
    
    Réponds JSON : {{"decision": "BUY/WAIT", "score": 0-100, "analysis": "Ton analyse psycho-technique"}}
    """
    try:
        model = genai.GenerativeModel('gemini-pro')
        res = model.generate_content(prompt)
        return json.loads(res.text.replace("```json","").replace("```",""))
    except: return {"decision": "WAIT", "score": 0}

# --- 6. FONCTIONS STANDARD ---
def log_thought(emoji, text):
    if LEARNING_WEBHOOK_URL: requests.post(LEARNING_WEBHOOK_URL, json={"content": f"{emoji} **HIVEMIND:** {text}"})

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

# --- 7. TRAINING EN CONTINU ---
def run_eternal_learning():
    log_thought("🧬", "Modules d'analyse sociale activés en arrière-plan.")
    while True:
        # Ici le bot pourrait optimiser ses poids entre Social vs Technique
        # Pour l'instant, il veille.
        time.sleep(300)

# --- 8. MOTEUR TRADING ---
def get_tech_data(s):
    try:
        df = yf.Ticker(s).history(period="1mo", interval="15m")
        if df.empty: return None
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        trend = "HAUSSIER" if df['Close'].iloc[-1] > df['EMA50'].iloc[-1] else "BAISSIER"
        
        return {
            "p": df['Close'].iloc[-1], 
            "rsi": df['RSI'].iloc[-1], 
            "atr": df['ATR'].iloc[-1],
            "trend": trend,
            "df": df,
            "inst": yf.Ticker(s).info.get('heldPercentInstitutions', 0)*100
        }
    except: return None

def run_trading():
    global brain, bot_status
    load_brain()
    
    log_thought("🛸", "Connection aux flux sociaux (StockTwits) et financiers (Yahoo).")
    
    count = 0
    while True:
        try:
            count += 1
            if count >= 10: save_brain(); count = 0
            
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            # Crypto tourne 24/7, Actions Lundi-Vendredi
            is_crypto_weekend = (now.weekday() >= 5)
            
            bot_status = "🟢 Analyse Psycho-Sociale..."
            
            for s in list(brain['holdings'].keys()):
                # Gestion Vente (inchangé pour stabilité)
                pos = brain['holdings'][s]
                curr = yf.Ticker(s).fast_info['last_price']
                if curr < pos['stop']:
                    brain['cash'] += pos['qty'] * curr
                    del brain['holdings'][s]
                    if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json={"content": f"🔴 VENTE {s} (Stop Loss)"})
                    save_brain()

            if len(brain['holdings']) < 5:
                for s in WATCHLIST:
                    # Filtre Crypto Weekend
                    if is_crypto_weekend and s not in ["BTC-USD", "ETH.X", "COIN"]: continue
                    if s in brain['holdings']: continue
                    
                    # 1. Données Techniques
                    tech = get_tech_data(s)
                    if not tech: continue
                    
                    # 2. Données Sociales (NOUVEAU)
                    social = get_social_sentiment(s)
                    
                    # 3. Fusion des Cerveaux
                    decision = consult_hivemind(s, tech, social)
                    
                    # Achat si Score > 85 (Très exigeant)
                    if decision['decision'] == "BUY" and decision['score'] >= 85 and brain['cash'] > 500:
                        bet = brain['cash'] * 0.15 
                        brain['cash'] -= bet
                        qty = bet / tech['p']
                        sl = tech['p'] - (tech['atr'] * brain['params']['stop_loss_atr'])
                        tp = tech['p'] + (tech['atr'] * 3.0)
                        
                        brain['holdings'][s] = {"qty": qty, "entry": tech['p'], "stop": sl}
                        
                        # Message Discord Enrichi
                        msg = {
                            "embeds": [{
                                "title": f"🧠 SIGNAL HIVEMIND : {s}",
                                "description": decision['analysis'],
                                "color": 0x9b59b6, # Violet
                                "fields": [
                                    {"name": "🗣️ Ambiance Sociale", "value": f"Sentiment: {social['polarity']:.2f}\nHype: {social['hype']:.0f}%", "inline": True},
                                    {"name": "📈 Technique", "value": f"RSI: {tech['rsi']:.1f}\nTrend: {tech['trend']}", "inline": True},
                                    {"name": "💰 Ordre", "value": f"Mise: {bet:.2f}$\nCible: {tp:.2f}$", "inline": False}
                                ],
                                "footer": {"text": f"Score IA: {decision['score']}/100"}
                            }]
                        }
                        if DISCORD_WEBHOOK_URL: requests.post(DISCORD_WEBHOOK_URL, json=msg)
                        
                        # Génération Image
                        log_thought("🎨", f"Je dessine le plan de trade pour {s}...")
                        generate_chart(s, tech['df'], tech['p'], sl, tp)
                        
                        save_brain()
            
            time.sleep(60)
        except Exception as e:
            print(f"Erreur: {e}")
            time.sleep(10)

@app.route('/')
def index(): return f"<h1>HIVEMIND V31</h1><p>{bot_status}</p>"

def start_threads():
    t1 = threading.Thread(target=run_trading, daemon=True)
    t1.start()
    t2 = threading.Thread(target=run_heartbeat, daemon=True)
    t2.start()
    t3 = threading.Thread(target=run_eternal_learning, daemon=True)
    t3.start()

start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
