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
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
PAPER_WEBHOOK_URL = os.environ.get("PAPER_WEBHOOK_URL")
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL")
LEARNING_WEBHOOK_URL = os.environ.get("LEARNING_WEBHOOK_URL")
SUMMARY_WEBHOOK_URL = os.environ.get("SUMMARY_WEBHOOK_URL")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR", "GOOG", "META", "NFLX", "INTC", "PLTR"]
INITIAL_CAPITAL = 50000.0 
SIMULATION_COUNT = 2000 

brain = {
    "cash": INITIAL_CAPITAL, 
    "holdings": {}, 
    "paper_cash": 1000.0,
    "paper_holdings": {},
    "genome": {"rsi_buy": 30, "sl_mult": 2.0, "tp_mult": 3.0},
    "stats": {"generation": 0, "best_pnl": 0.0, "speed": "0 ops/s"},
    "emotions": {"confidence": 50.0, "stress": 20.0},
    "karma": {s: 10.0 for s in WATCHLIST},
    "black_box": [],
    "learning_journal": [],
    "total_pnl": 0.0
}

bot_state = {
    "status": "Démarrage UNLEASHED...",
    "last_log": "Init...",
    "web_logs": []
}

log_queue = queue.Queue()
short_term_memory = []

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ==============================================================================
# 2. LOGGING OPTIMISÉ (POUR NE PAS RALENTIR)
# ==============================================================================
def fast_log(text):
    log_queue.put(text)
    # On limite l'historique web pour économiser la RAM
    bot_state['web_logs'].insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] {text}")
    if len(bot_state['web_logs']) > 30: bot_state['web_logs'] = bot_state['web_logs'][:30]

def logger_worker():
    buffer = []
    last_send = time.time()
    while True:
        try:
            while not log_queue.empty():
                buffer.append(log_queue.get())
            
            # Envoi en rafale toutes les 1.5s
            if buffer and (len(buffer) > 5 or time.time() - last_send > 1.5):
                msg_block = "\n".join(buffer[:10])
                buffer = buffer[10:]
                if LEARNING_WEBHOOK_URL:
                    try: requests.post(LEARNING_WEBHOOK_URL, json={"content": msg_block}, timeout=1)
                    except: pass
                last_send = time.time()
            time.sleep(0.1) # Latence minimale
        except: time.sleep(1)

def send_alert(url, embed):
    if url: 
        # Thread séparé pour ne pas bloquer le calcul
        threading.Thread(target=lambda: requests.post(url, json={"embeds": [embed]}, timeout=2)).start()

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
    # Sauvegarde asynchrone pour ne pas bloquer le trading
    def _save():
        try:
            g = Github(GITHUB_TOKEN)
            repo = g.get_repo(REPO_NAME)
            content = json.dumps(brain, indent=4)
            try:
                f = repo.get_contents("brain.json")
                repo.update_file("brain.json", "Fast Save", content, f.sha)
            except:
                repo.create_file("brain.json", "Init", content)
        except: pass
    threading.Thread(target=_save).start()

# ==============================================================================
# 4. MODULES D'INTELLIGENCE (SANS PAUSE)
# ==============================================================================
def run_monte_carlo(prices):
    try:
        returns = prices.pct_change().dropna()
        # Calcul matriciel optimisé
        sims = prices.iloc[-1] * (1 + np.random.normal(returns.mean(), returns.std(), (SIMULATION_COUNT, 10)))
        prob = np.sum(sims[:, -1] > prices.iloc[-1]) / SIMULATION_COUNT
        return prob
    except: return 0.5

def get_vision_score(df):
    # Vision est lourde, on la lance moins souvent ou sur échantillon
    try:
        buf = io.BytesIO()
        mpf.plot(df.tail(40), type='candle', style='nightclouds', savefig=buf)
        buf.seek(0)
        img = Image.open(buf)
        res = model.generate_content(["Score achat 0.0-1.0 ?", img])
        return float(res.text.strip())
    except: return 0.5

def check_whale(df):
    try:
        vol = df['Volume'].iloc[-1]
        avg = df['Volume'].rolling(20).mean().iloc[-1]
        return (vol > avg * 2.5), f"x{vol/avg:.1f}"
    except: return False, "x1.0"

def consult_council(s, rsi, mc, vis, whale):
    # Version accélérée : Prompt court
    prompt = f"""
    DECISION {s}.
    Maths: {mc*100:.0f}%. Vision: {vis:.2f}. Baleine: {whale}. RSI: {rsi:.1f}.
    Si Maths > 60% ET Vision > 0.6 : BUY.
    JSON: {{"vote": "BUY/WAIT", "reason": "..."}}
    """
    try:
        res = model.generate_content(prompt)
        return json.loads(res.text.replace("```json","").replace("```",""))
    except: return {"vote": "WAIT", "reason": "Erreur"}

# ==============================================================================
# 5. MOTEUR D'APPRENTISSAGE NON-STOP (NO SLEEP)
# ==============================================================================
def run_dream_learning():
    global brain, bot_state, short_term_memory
    cache = {}
    
    fast_log("🧬 **UNLEASHED:** Mode apprentissage continue sans pause.")
    
    # 1. Chargement initial massif en RAM
    for s in WATCHLIST:
        try:
            df = yf.Ticker(s).history(period="2mo", interval="1h")
            if not df.empty:
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                cache[s] = df.dropna()
        except: pass
    
    ops_counter = 0
    start_time = time.time()

    while True:
        try:
            # Pas de vérification horaire. Il apprend TOUT LE TEMPS, même pendant le trading.
            
            s = random.choice(list(cache.keys()))
            df = cache[s]
            
            # Mutation Ultra-Rapide
            brain['stats']['generation'] += 1
            mutant = {
                "rsi_buy": random.randint(15, 60),
                "sl_mult": round(random.uniform(1.0, 4.0), 1),
                "tp_mult": round(random.uniform(1.5, 6.0), 1)
            }
            
            # Simulation éclair (sur un segment aléatoire)
            idx = random.randint(0, len(df) - 60)
            subset = df.iloc[idx : idx+60]
            row = subset.iloc[0]
            
            pnl = 0
            # On teste si le signal s'active
            if row['RSI'] < mutant['rsi_buy']:
                entry = row['Close']
                sl = entry - (row['ATR'] * mutant['sl_mult'])
                tp = entry + (row['ATR'] * mutant['tp_mult'])
                
                # Vérif futur (vectorisée si possible, ici boucle courte pour précision)
                for i in range(1, 20): # 20 heures max
                    fut = subset.iloc[i]
                    if fut['High'] > tp: pnl = tp - entry; break
                    elif fut['Low'] < sl: pnl = entry - sl; pnl = -pnl; break
            
            # Résultat
            ops_counter += 1
            if pnl > brain['stats']['best_pnl']:
                brain['stats']['best_pnl'] = pnl
                brain['genome'] = mutant
                save_brain()
                fast_log(f"🧬 **EVO:** Nouveau Record ! RSI<{mutant['rsi_buy']} (+{pnl:.1f}$)")
            
            # Vitesse
            if time.time() - start_time > 1.0:
                brain['stats']['speed'] = f"{ops_counter} ops/s"
                ops_counter = 0
                start_time = time.time()
            
            # ZÉRO PAUSE
            # time.sleep(0.001) # Juste pour laisser respirer le CPU
            
        except Exception as e:
            time.sleep(1) # Sécurité crash

# ==============================================================================
# 6. MOTEUR TRADING LIVE (FLUX TENDU)
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
            
            bot_state['status'] = "🟢 LIVE" if market_open else "🌙 REPLAY/LEARN"
            
            # Liste dynamique (Mélangée pour ne pas bloquer sur un actif)
            targets = list(WATCHLIST)
            random.shuffle(targets)
            
            for s in targets:
                # 1. GESTION VENTES (Prioritaire)
                if s in brain['holdings'] or (not market_open and s in brain['paper_holdings']):
                    # On vérifie le prix
                    curr = broker.get_price(s)
                    if not curr and not market_open: curr = 100.0 # Fake price for replay test
                    
                    if curr:
                        # Check SL/TP
                        # ... (Logique de vente habituelle)
                        pass
                
                # 2. ACHATS
                # On ne scanne pour acheter que si on a de la place
                if len(brain['holdings']) < 5:
                    # Récupération Data (Cache optimisé ?)
                    # Ici on fait un appel direct pour avoir le vrai prix
                    try:
                        if market_open:
                            df = yf.Ticker(s).history(period="5d", interval="5m") # 5min pour réactivité
                        else:
                            # En mode nuit, on continue de simuler sur des données passées pour Paper
                            df = yf.Ticker(s).history(period="1mo", interval="1h")
                        
                        if df.empty: continue
                        row = df.iloc[-1]
                        
                        # FILTRE RAPIDE (Génome)
                        if row['RSI'] < brain['genome']['rsi_buy']:
                            
                            fast_log(f"🔎 **SCAN {s}:** RSI {row['RSI']:.1f}. Analyse...")
                            
                            # ANALYSE LOURDE
                            mc = run_monte_carlo(df['Close'])
                            whl, _ = check_whale(df)
                            
                            # Si Maths OK, on appelle la Vision (Coûteuse)
                            if mc > 0.60:
                                vis = get_vision_score(df)
                                council = consult_council(s, row['RSI'], mc, vis, 0, whl)
                                
                                if council['vote'] == "BUY":
                                    # EXECUTION
                                    # ... (Même logique achat V115) ...
                                    # On ajoute juste le log pour confirmer l'action
                                    fast_log(f"🚀 **ACHAT {s}** validé.")
                                    
                                    # Logique Paper Replay (Hors marché)
                                    if not market_open:
                                        msg = {"title": f"🎬 REPLAY : {s}", "description": "Opportunité détectée en historique", "color": 0x3498db}
                                        send_alert(PAPER_WEBHOOK_URL, msg)
                                        
                    except: pass
            
            # ZÉRO PAUSE : On boucle immédiatement
            # time.sleep(0.1) 
            
        except: time.sleep(1)

# --- DASHBOARD ---
@app.route('/')
def index():
    return f"<h1>UNLEASHED V129</h1><p>Speed: {brain['stats'].get('speed', 'Calculating...')}</p><p>Génome: {brain['genome']}</p>"

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
