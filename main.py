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

# ==============================================================================
# 1. CLÉS & CONFIGURATION
# ==============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")      # Trading Live
PAPER_WEBHOOK_URL = os.environ.get("PAPER_WEBHOOK_URL")          # Paper/Replay
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL")  # Status
LEARNING_WEBHOOK_URL = os.environ.get("LEARNING_WEBHOOK_URL")    # Cerveau (Détails)
SUMMARY_WEBHOOK_URL = os.environ.get("SUMMARY_WEBHOOK_URL")      # Synthèse (Bilan)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR", "GOOG", "META"]
INITIAL_CAPITAL_MAIN = 50000.0 
INITIAL_CAPITAL_PAPER = 1000.0
SIMULATION_COUNT = 2000 

brain = {
    "cash": INITIAL_CAPITAL_MAIN, 
    "holdings": {}, 
    "paper_cash": INITIAL_CAPITAL_PAPER,
    "paper_holdings": {},
    "genome": {"rsi_buy": 32, "sl_mult": 2.0, "tp_mult": 3.5}, # ADN de départ optimisé
    "stats": {"generation": 0, "best_pnl": 0.0},
    "emotions": {"confidence": 50.0, "stress": 20.0},
    "karma": {s: 10.0 for s in WATCHLIST},
    "total_pnl": 0.0
}

bot_state = {
    "status": "Booting V104...",
    "mode": "INIT",
    "last_log": "Chargement...",
    "web_logs": []
}

log_queue = queue.Queue()
short_term_memory = []

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ==============================================================================
# 2. LOGGING & COMMS
# ==============================================================================
def fast_log(text):
    log_queue.put(text)
    bot_state['web_logs'].insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] {text}")
    if len(bot_state['web_logs']) > 50: bot_state['web_logs'] = bot_state['web_logs'][:50]

def logger_worker():
    buffer = []
    last_send = time.time()
    while True:
        try:
            while not log_queue.empty():
                buffer.append(log_queue.get())
            
            if buffer and (len(buffer) > 3 or time.time() - last_send > 1.5):
                msg_block = "\n".join(buffer[:12])
                buffer = buffer[12:]
                if LEARNING_WEBHOOK_URL:
                    try: requests.post(LEARNING_WEBHOOK_URL, json={"content": msg_block})
                    except: pass
                last_send = time.time()
            time.sleep(0.2)
        except: time.sleep(1)

def send_alert(url, embed):
    if url: 
        try: requests.post(url, json={"embeds": [embed]})
        except: pass

def run_heartbeat():
    while True:
        if HEARTBEAT_WEBHOOK_URL: requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"})
        time.sleep(30)

# ==============================================================================
# 3. MÉMOIRE
# ==============================================================================
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
            repo.update_file("brain.json", "V104 Save", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# ==============================================================================
# 4. MODULES D'INTELLIGENCE (LES SENS)
# ==============================================================================
def run_monte_carlo(prices):
    """Maths Quantiques"""
    try:
        returns = prices.pct_change().dropna()
        sims = prices.iloc[-1] * (1 + np.random.normal(returns.mean(), returns.std(), (SIMULATION_COUNT, 10)))
        prob = np.sum(sims[:, -1] > prices.iloc[-1]) / SIMULATION_COUNT
        return prob
    except: return 0.5

def get_vision_score(df):
    """Vision Artificielle"""
    try:
        buf = io.BytesIO()
        mpf.plot(df.tail(60), type='candle', style='nightclouds', savefig=buf)
        buf.seek(0)
        img = Image.open(buf)
        res = model.generate_content(["Score achat 0.0-1.0 ?", img])
        return float(res.text.strip())
    except: return 0.5

def get_social_hype(symbol):
    """Sentiment Social"""
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).json()
        txt = " ".join([m['body'] for m in r['messages'][:10]])
        return TextBlob(txt).sentiment.polarity
    except: return 0

def check_whale(df):
    """Détection Volume"""
    try:
        vol = df['Volume'].iloc[-1]
        avg = df['Volume'].rolling(20).mean().iloc[-1]
        return (vol > avg * 2.5), f"x{vol/avg:.1f}"
    except: return False, "x1.0"

def consult_council(s, rsi, mc, vis, soc, whale):
    """Cerveau Central (Synthèse)"""
    prompt = f"""
    CONSEIL SUPRÊME {s}.
    Maths: {mc*100:.1f}%. Vision: {vis:.2f}. Social: {soc:.2f}. Baleine: {whale}. RSI: {rsi:.1f}.
    Si Maths > 60% ET Vision > 0.6 : ACHAT.
    JSON: {{"vote": "BUY/WAIT", "reason": "Synthèse"}}
    """
    try:
        res = model.generate_content(prompt)
        return json.loads(res.text.replace("```json","").replace("```",""))
    except: return {"vote": "WAIT", "reason": "Erreur"}

# ==============================================================================
# 5. MOTEUR D'APPRENTISSAGE (DÉTAILLÉ)
# ==============================================================================
def run_dream_learning():
    global brain, short_term_memory
    cache = {}
    
    fast_log("🧠 **OMNI-LEARN:** Chargement des données pour l'entraînement...")
    for s in WATCHLIST:
        try:
            df = yf.Ticker(s).history(period="1mo", interval="1h")
            if not df.empty:
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                df['EMA200'] = ta.ema(df['Close'], length=200) # FILTRE DE TENDANCE CRUCIAL
                cache[s] = df.dropna()
        except: pass
    fast_log("✅ Données prêtes.")

    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if market_open:
                time.sleep(60)
                continue
            
            # 1. Mutation
            brain['stats']['generation'] += 1
            parent = brain['genome']
            # On varie légèrement autour du meilleur connu
            mutant = {
                "rsi_buy": max(20, min(60, parent['rsi_buy'] + random.randint(-4, 4))),
                "sl_mult": round(max(1.0, parent['sl_mult'] + random.uniform(-0.3, 0.3)), 1),
                "tp_mult": round(max(1.5, parent['tp_mult'] + random.uniform(-0.3, 0.3)), 1)
            }
            
            # 2. Simulation (Sur un actif aléatoire)
            s = random.choice(list(cache.keys()))
            df = cache[s]
            
            # On cherche un point d'entrée valide (RSI bas + Tendance Haussière)
            # C'est le secret pour éviter les pertes
            valid_entries = df[(df['RSI'] < mutant['rsi_buy']) & (df['Close'] > df['EMA200'])]
            
            if valid_entries.empty:
                # Si pas de signal parfait, on force un test aléatoire pour voir
                idx = random.randint(0, len(df)-50)
            else:
                # On prend un vrai signal du passé
                idx = df.index.get_loc(valid_entries.index[random.randint(0, len(valid_entries)-1)])
                if idx > len(df) - 50: idx = len(df) - 50
            
            row = df.iloc[idx]
            
            # Agents Simulés (Pour l'affichage)
            mc_prob = run_monte_carlo(df['Close'].iloc[:idx])
            is_whale = row['Volume'] > df['Volume'].mean() * 2
            whale_icon = "🐋" if is_whale else "🐟"
            
            # Résultat du Trade (Futur)
            entry = row['Close']
            sl = entry - (row['ATR'] * mutant['sl_mult'])
            tp = entry + (row['ATR'] * mutant['tp_mult'])
            
            future = df.iloc[idx+1 : idx+10]
            pnl = 0
            outcome = "NEUTRE"
            
            if not future.empty:
                if future['Low'].min() < sl:
                    pnl = sl - entry
                    outcome = "STOP LOSS"
                elif future['High'].max() > tp:
                    pnl = tp - entry
                    outcome = "TAKE PROFIT"
                else:
                    pnl = future['Close'].iloc[-1] - entry
                    outcome = "EN COURS"
            
            emoji = "✅" if pnl > 0 else "❌"
            
            # 3. LOG DÉTAILLÉ (CE QUE TU VEUX VOIR)
            log_block = (
                f"🧪 **SIMULATION {s}** (Gen {brain['stats']['generation']})\n"
                f"🧬 Params: RSI<{mutant['rsi_buy']} | SL {mutant['sl_mult']} ATR\n"
                f"├─ 📉 Tech: RSI {row['RSI']:.1f}\n"
                f"├─ ⚛️ Quant: Proba {mc_prob*100:.0f}% | {whale_icon} Vol High\n"
                f"└─ 🏁 **RÉSULTAT:** {emoji} {outcome} (**{pnl:+.2f}$**)"
            )
            fast_log(log_block)
            
            # 4. ÉVOLUTION
            if pnl > 0:
                short_term_memory.append(pnl)
                # Si c'est un super trade confirmé par les maths
                if pnl > 500 and mc_prob > 0.6:
                    brain['genome'] = mutant # On adopte !
                    save_brain()
                    fast_log(f"🧬 **ÉVOLUTION VALIDÉE !** Nouveaux paramètres adoptés.")

            # 5. BILAN (Tous les 10 gains)
            if len(short_term_memory) >= 10:
                total = sum(short_term_memory)
                msg = {
                    "embeds": [{
                        "title": "🎓 RAPPORT D'ÉTUDE",
                        "color": 0xFFD700,
                        "description": f"La session d'entraînement est fructueuse.",
                        "fields": [
                            {"name": "Profit Cumulé", "value": f"**{total:.2f}$**", "inline": True},
                            {"name": "Gènes Actuels", "value": f"RSI < {brain['genome']['rsi_buy']}", "inline": True}
                        ]
                    }]
                }
                if SUMMARY_WEBHOOK_URL: requests.post(SUMMARY_WEBHOOK_URL, json=msg)
                short_term_memory = []
            
            time.sleep(5)
        except Exception as e:
            print(f"Learn Err: {e}")
            time.sleep(10)

# ==============================================================================
# 6. MOTEUR TRADING LIVE
# ==============================================================================
class GhostBroker:
    def get_price(self, symbol):
        try: return yf.Ticker(symbol).fast_info['last_price']
        except: return None
    def get_portfolio(self):
        return brain['cash'], [{"symbol": s, "qty": d['qty'], "pnl": 0} for s, d in brain['holdings'].items()]

broker = GhostBroker()

def run_trading():
    global brain, bot_state
    load_brain()
    
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if market_open:
                bot_state['status'] = "🟢 LIVE"
                
                # VENTES
                for s in list(brain['holdings'].keys()):
                    pos = brain['holdings'][s]
                    curr = broker.get_price(s)
                    if not curr: continue
                    
                    if curr < pos['stop']:
                        brain['cash'] += pos['qty'] * curr
                        del brain['holdings'][s]
                        send_trade_alert({"embeds": [{"title": f"🔴 VENTE {s} (Stop)", "description": "Perte limitée", "color": 0xe74c3c}]})
                        save_brain()
                    elif curr > pos['tp']:
                        brain['cash'] += pos['qty'] * curr
                        del brain['holdings'][s]
                        send_trade_alert({"embeds": [{"title": f"🟢 VENTE {s} (Target)", "description": "Profit encaissé", "color": 0x2ecc71}]})
                        save_brain()

                # ACHATS
                if len(brain['holdings']) < 5:
                    for s in WATCHLIST:
                        if s in brain['holdings']: continue
                        try:
                            df = yf.Ticker(s).history(period="1mo", interval="1h")
                            if df.empty: continue
                            df['RSI'] = ta.rsi(df['Close'], length=14)
                            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                            row = df.iloc[-1]
                            
                            # FILTRE GÉNÉTIQUE APPRIS
                            if row['RSI'] < brain['genome']['rsi_buy']:
                                
                                mc = run_monte_carlo(df['Close'])
                                vis = get_vision_score(df)
                                soc = get_social_hype(s)
                                whl, _ = check_whale(df)
                                council = consult_council(s, row['RSI'], mc, vis, soc, whl)
                                
                                if council['vote'] == "BUY":
                                    price = row['Close']
                                    qty = 500 / price
                                    sl = price - (row['ATR'] * brain['genome']['sl_mult'])
                                    tp = price + (row['ATR'] * brain['genome']['tp_mult'])
                                    
                                    brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                                    brain['cash'] -= 500
                                    
                                    msg = {
                                        "embeds": [{
                                            "title": f"🌌 ACHAT OMEGA : {s}",
                                            "description": council['reason'],
                                            "color": 0x2ecc71,
                                            "fields": [
                                                {"name": "Quantique", "value": f"{mc:.2f}", "inline": True},
                                                {"name": "Vision", "value": f"{vis:.2f}", "inline": True}
                                            ]
                                        }]
                                    }
                                    send_trade_alert(msg)
                                    save_brain()
                        except: pass
            else:
                bot_state['status'] = "🌙 NUIT"
            time.sleep(30)
        except: time.sleep(10)

# ==============================================================================
# 7. DASHBOARD
# ==============================================================================
@app.route('/')
def index():
    eq, pos = broker.get_portfolio()
    return f"""
    <h1>OMNISCIENT V104</h1>
    <p>Status: {bot_state['status']}</p>
    <p>Capital: ${eq:,.2f}</p>
    <div style="background:#000;color:#ccc;padding:10px;height:300px;overflow:auto">
    {'<br>'.join(bot_state['web_logs'])}
    </div>
    """

def start_threads():
    threading.Thread(target=run_trading, daemon=True).start()
    threading.Thread(target=run_heartbeat, daemon=True).start()
    threading.Thread(target=logger_worker, daemon=True).start()
    threading.Thread(target=run_dream_learning, daemon=True).start()

load_brain()
start_threads()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)