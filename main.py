import os
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
from scipy.stats import norm
from textblob import TextBlob
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
from flask import Flask, render_template_string
from datetime import datetime, time as dtime
import pytz
from github import Github

app = Flask(__name__)

# --- 1. CLÉS ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")

# --- 2. CONFIGURATION ---
WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR", "GOOG"]
INITIAL_CAPITAL = 50000.0 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "trade_history": [],
    "total_pnl": 0.0,
    "emotions": {"confidence": 50.0, "stress": 20.0}
}

bot_state = {
    "status": "Démarrage...",
    "last_decision": "Aucune",
    "last_log": "Initialisation...",
    "equity": INITIAL_CAPITAL,
    "positions": []
}

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- 3. CLASSE "GHOST BROKER" ---
class GhostBroker:
    def get_price(self, symbol):
        try: return yf.Ticker(symbol).fast_info['last_price']
        except: return None

    def get_portfolio(self):
        equity = brain['cash']
        positions = []
        for s, d in brain['holdings'].items():
            p = self.get_price(s)
            if p:
                val = d['qty'] * p
                equity += val
                pnl = val - (d['qty'] * d['avg_price'])
                pnl_pct = (pnl / (d['qty'] * d['avg_price'])) * 100
                positions.append({"symbol": s, "qty": d['qty'], "entry": d['avg_price'], "current": p, "pnl": round(pnl, 2), "pct": round(pnl_pct, 2)})
        return equity, positions

    def execute(self, side, symbol, price, qty, reason):
        if side == "BUY":
            cost = qty * price
            if brain['cash'] >= cost:
                brain['cash'] -= cost
                if symbol in brain['holdings']:
                    old = brain['holdings'][symbol]
                    new_qty = old['qty'] + qty
                    new_avg = ((old['qty']*old['avg_price']) + cost) / new_qty
                    brain['holdings'][symbol] = {"qty": new_qty, "avg_price": new_avg}
                else:
                    brain['holdings'][symbol] = {"qty": qty, "avg_price": price}
                
                self.log(f"🟢 ACHAT {symbol}", reason, price)
                return True

        elif side == "SELL":
            if symbol in brain['holdings']:
                pos = brain['holdings'][symbol]
                revenue = pos['qty'] * price
                cost = pos['qty'] * pos['avg_price']
                pnl = revenue - cost
                
                brain['cash'] += revenue
                brain['total_pnl'] += pnl
                del brain['holdings'][symbol]
                
                self.log(f"🔴 VENTE {symbol} ({pnl:+.2f}$)", reason, price)
                return True
        return False

    def log(self, title, desc, price):
        entry = f"{title} @ {price:.2f}$ | {desc}"
        bot_state['last_log'] = entry
        brain['trade_history'].append(entry)
        if DISCORD_WEBHOOK_URL:
            try: requests.post(DISCORD_WEBHOOK_URL, json={"content": f"**{title}**\n{desc}"})
            except: pass
        save_brain()

broker = GhostBroker()

# --- 4. MODULES INTELLIGENTS ---
def run_monte_carlo(prices):
    try:
        returns = prices.pct_change().dropna()
        sims = prices.iloc[-1] * (1 + np.random.normal(returns.mean(), returns.std(), (1000, 10)))
        prob = np.sum(sims[:, -1] > prices.iloc[-1]) / 1000
        return prob
    except: return 0.5

def get_vision_score(df):
    try:
        buf = io.BytesIO()
        mpf.plot(df.tail(50), type='candle', style='nightclouds', savefig=buf)
        buf.seek(0)
        img = Image.open(buf)
        res = model.generate_content(["Score achat 0.0-1.0 ?", img])
        return float(res.text.strip())
    except: return 0.5

def get_social_hype(symbol):
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers).json()
        txt = " ".join([m['body'] for m in r['messages'][:10]])
        return TextBlob(txt).sentiment.polarity
    except: return 0

def check_whale(df):
    try:
        vol = df['Volume'].iloc[-1]
        avg = df['Volume'].rolling(20).mean().iloc[-1]
        return vol > (avg * 2.5)
    except: return False

def consult_council(s, rsi, mc, vis, social, whale):
    prompt = f"""
    CONSEIL SUPRÊME POUR {s}.
    1. Maths: MonteCarlo {mc:.0%} hausse.
    2. Vision: Score {vis:.2f}/1.0.
    3. Social: Sentiment {social:.2f}.
    4. Baleine: {"OUI" if whale else "NON"}.
    5. Technique: RSI {rsi:.1f}.
    
    Si Maths > 60% ET Vision > 0.6 : ACHAT.
    Si Baleine ET RSI < 30 : ACHAT FORT.
    Sinon : ATTENTE.
    
    JSON: {{"vote": "BUY/WAIT", "reason": "Synthèse courte"}}
    """
    try:
        res = model.generate_content(prompt)
        return json.loads(res.text.replace("```json","").replace("```",""))
    except: return {"vote": "WAIT", "reason": "Erreur IA"}

# --- 5. PERSISTANCE ---
def load_brain():
    global brain
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        c = repo.get_contents("brain.json")
        loaded = json.loads(c.decoded_content.decode())
        if "cash" in loaded: brain.update(loaded)
    except: pass

def save_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        content = json.dumps(brain, indent=4)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Save", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# --- 6. MOTEUR PRINCIPAL (C'est ici que j'ai corrigé le nom) ---
def run_trading_engine():
    global bot_state, brain
    load_brain()
    print("MOTEUR DÉMARRÉ.")
    
    while True:
        try:
            # Mise à jour Dashboard
            eq, positions = broker.get_portfolio()
            bot_state['equity'] = eq
            bot_state['positions'] = positions
            
            # Horaires Bourse
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            if not (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0)):
                bot_state['status'] = "🌙 Marché Fermé"
                time.sleep(60)
                continue
                
            bot_state['status'] = "👁️ SCANNING..."
            
            # 1. VENTES
            for s in list(brain['holdings'].keys()):
                pos = brain['holdings'][s]
                curr = broker.get_price(s)
                if not curr: continue
                
                if curr < pos['avg_price'] * 0.97:
                    broker.execute("SELL", s, curr, 0, "Stop Loss")
                elif curr > pos['avg_price'] * 1.06:
                    broker.execute("SELL", s, curr, 0, "Take Profit")

            # 2. ACHATS
            if len(brain['holdings']) < 5:
                for s in WATCHLIST:
                    if s in brain['holdings']: continue
                    
                    try:
                        df = yf.Ticker(s).history(period="1mo", interval="1h")
                        if df.empty: continue
                        
                        rsi = ta.rsi(df['Close'], length=14).iloc[-1]
                        
                        if rsi < 40: 
                            mc = run_monte_carlo(df['Close'])
                            vis = get_vision_score(df)
                            soc = get_social_hype(s)
                            whl = check_whale(df)
                            
                            council = consult_council(s, rsi, mc, vis, soc, whl)
                            bot_state['last_decision'] = f"{s}: {council['vote']}"
                            
                            if council['vote'] == "BUY":
                                qty = 500 / df['Close'].iloc[-1]
                                broker.execute("BUY", s, df['Close'].iloc[-1], qty, council['reason'])
                    except Exception as e: print(e)
                    time.sleep(2)
            
            time.sleep(60)
        except Exception as e:
            print(f"Erreur: {e}")
            time.sleep(10)

# --- 7. DASHBOARD WEB ---
HTML = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta http-equiv="refresh" content="10">
    <title>SUPREME ENTITY</title>
    <style>
        body { background: #000; color: #0f0; font-family: monospace; padding: 20px; }
        .card { border: 1px solid #333; padding: 15px; margin-bottom: 20px; }
        h1 { margin: 0; color: #fff; }
        table { width: 100%; border-collapse: collapse; color: #aaa; }
        td { border-bottom: 1px solid #222; padding: 5px; }
    </style>
</head>
<body>
    <h1>👁️ THE SUPREME ENTITY V65</h1>
    <div class="card">
        <h3>STATUS: {{ status }}</h3>
        <h2>CAPITAL: ${{ equity }}</h2>
        <p>Dernière Décision: {{ last_dec }}</p>
        <p>Log: {{ last_log }}</p>
    </div>
    <div class="card">
        <table>
            <tr><th>ACTIF</th><th>QTE</th><th>PRIX MOY</th><th>VALEUR</th><th>PNL</th></tr>
            {% for p in positions %}
            <tr>
                <td>{{ p.symbol }}</td>
                <td>{{ p.qty|round(2) }}</td>
                <td>{{ p.entry }}</td>
                <td>{{ (p.qty * p.current)|round(2) }}</td>
                <td style="color: {{ 'red' if p.pnl < 0 else '#0f0' }}">{{ p.pnl }}$</td>
            </tr>
            {% endfor %}
        </table>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML, 
        equity=f"{bot_state['equity']:,.2f}", 
        status=bot_state['status'], 
        positions=bot_state['positions'],
        last_log=bot_state['last_log'],
        last_dec=bot_state['last_decision']
    )

def start():
    # C'est ici que j'ai corrigé le nom de la fonction appelée
    threading.Thread(target=run_trading_engine, daemon=True).start()

start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
