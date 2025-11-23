import os
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
from scipy.stats import norm, linregress
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
matplotlib.use('Agg') # Mode sans écran (Serveur)
import mplfinance as mpf
from PIL import Image
from flask import Flask, render_template_string
from datetime import datetime, time as dtime
import pytz
from github import Github

app = Flask(__name__)

# ==============================================================================
# 1. CONFIGURATION & ACCÈS (LE SYSTÈME NERVEUX)
# ==============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")      # Canal Principal (50k)
PAPER_WEBHOOK_URL = os.environ.get("PAPER_WEBHOOK_URL")          # Canal Replay (1k)
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL")  # Canal Status
LEARNING_WEBHOOK_URL = os.environ.get("LEARNING_WEBHOOK_URL")    # Canal Cerveau (Logs)
SUMMARY_WEBHOOK_URL = os.environ.get("SUMMARY_WEBHOOK_URL")      # Canal Synthèse
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = os.environ.get("REPO_NAME")

WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMZN", "AMD", "COIN", "MSTR", "GOOG", "META"]
INITIAL_CAPITAL_MAIN = 50000.0 
INITIAL_CAPITAL_PAPER = 1000.0
SIMULATION_COUNT = 5000 # Nombre de futurs simulés par analyse

# --- STRUCTURE CÉRÉBRALE (MÉMOIRE) ---
brain = {
    "cash": INITIAL_CAPITAL_MAIN, 
    "holdings": {}, 
    "paper_cash": INITIAL_CAPITAL_PAPER,
    "paper_holdings": {},
    
    # ADN ÉVOLUTIF (Paramètres optimisés par le moteur génétique)
    "genome": {"rsi_buy": 30, "sl_mult": 2.0, "tp_mult": 3.5},
    
    # MÉMOIRE EXPÉRIENTIELLE
    "q_table": {},           # Apprentissage par renforcement
    "black_box": [],         # Patterns interdits (Traumatismes)
    "karma": {s: 10.0 for s in WATCHLIST}, # Préférences par actif
    
    # ÉTAT PSYCHOLOGIQUE
    "emotions": {"confidence": 50.0, "stress": 20.0, "euphoria": 0.0},
    
    "stats": {"generation": 0, "best_pnl": 0.0},
    "total_pnl": 0.0
}

bot_state = {
    "status": "Démarrage V94...",
    "mode": "INIT",
    "last_log": "Chargement..."
}

log_queue = queue.Queue()

if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ==============================================================================
# 2. MODULES DE COMMUNICATION (LA VOIX)
# ==============================================================================
def fast_log(text):
    """Ajoute un message dans la file d'attente pour Discord"""
    log_queue.put(text)

def logger_worker():
    """Thread dédié qui envoie les logs par paquets pour éviter le ban Discord"""
    buffer = []
    last_send = time.time()
    while True:
        try:
            while not log_queue.empty():
                buffer.append(log_queue.get())
            
            # Envoi toutes les 1.5s
            if buffer and (len(buffer) > 5 or time.time() - last_send > 1.5):
                msg_block = "\n".join(buffer[:12])
                buffer = buffer[12:]
                if LEARNING_WEBHOOK_URL:
                    try: requests.post(LEARNING_WEBHOOK_URL, json={"content": msg_block})
                    except: pass
                last_send = time.time()
            time.sleep(0.2)
        except: time.sleep(1)

def send_alert(url, embed):
    """Envoie une alerte riche (Embed) sur un webhook spécifique"""
    if url: 
        try: requests.post(url, json={"embeds": [embed]})
        except: pass

def run_heartbeat():
    """Battement de cœur pour confirmer que le bot est vivant"""
    while True:
        if HEARTBEAT_WEBHOOK_URL: requests.post(HEARTBEAT_WEBHOOK_URL, json={"content": "💓"})
        time.sleep(30)

# ==============================================================================
# 3. GESTION MÉMOIRE (PERSISTANCE GITHUB)
# ==============================================================================
def load_brain():
    global brain
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        c = repo.get_contents("brain.json")
        loaded = json.loads(c.decoded_content.decode())
        # Fusion prudente pour ne pas écraser les nouvelles clés du code
        if "cash" in loaded: brain.update(loaded)
        if "paper_cash" not in brain: brain["paper_cash"] = 1000.0
    except: pass

def save_brain():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        content = json.dumps(brain, indent=4)
        try:
            f = repo.get_contents("brain.json")
            repo.update_file("brain.json", "Omniscient Save", content, f.sha)
        except:
            repo.create_file("brain.json", "Init", content)
    except: pass

# ==============================================================================
# 4. LES 5 SENS (ANALYSE DE DONNÉES AVANCÉE)
# ==============================================================================

# --- SENS 1 : MATHÉMATIQUES (Quantique) ---
def run_monte_carlo(prices):
    """Simule 5000 avenirs possibles selon la volatilité passée"""
    try:
        returns = prices.pct_change().dropna()
        # Génération de matrices aléatoires (NumPy)
        sims = prices.iloc[-1] * (1 + np.random.normal(returns.mean(), returns.std(), (SIMULATION_COUNT, 10)))
        # Calcul de la probabilité que le prix finisse plus haut
        prob = np.sum(sims[:, -1] > prices.iloc[-1]) / SIMULATION_COUNT
        return prob
    except: return 0.5

# --- SENS 2 : VISION (Analyse d'Image IA) ---
def get_vision_score(df):
    """Génère un graphique et demande à Gemini de le juger"""
    try:
        buf = io.BytesIO()
        # Style 'nightclouds' pour un look pro
        mpf.plot(df.tail(60), type='candle', style='nightclouds', savefig=buf)
        buf.seek(0)
        img = Image.open(buf)
        
        prompt = "Tu es un expert chartiste. Analyse ce graphique. Donne un score d'achat entre 0.0 (Vente) et 1.0 (Achat Fort). Réponds UNIQUEMENT le chiffre."
        res = model.generate_content([prompt, img])
        return float(res.text.strip())
    except: return 0.5

# --- SENS 3 : SOCIAL (Psychologie des foules) ---
def get_social_hype(symbol):
    """Scrape StockTwits pour sentir l'humeur"""
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers).json()
        txt = " ".join([m['body'] for m in r['messages'][:15]])
        # Analyse de sentiment basique (-1 à +1)
        return TextBlob(txt).sentiment.polarity
    except: return 0

# --- SENS 4 : BALEINES (Volume Institutionnel) ---
def check_whale(df):
    """Détecte les anomalies de volume"""
    try:
        vol = df['Volume'].iloc[-1]
        avg = df['Volume'].rolling(20).mean().iloc[-1]
        # Si volume > 2.5x la moyenne, c'est une baleine
        return (vol > avg * 2.5), f"x{vol/avg:.1f}"
    except: return False, "x1.0"

# --- SENS 5 : TECHNIQUE (Indicateurs Classiques) ---
def get_tech_indicators(df):
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['SMA200'] = ta.sma(df['Close'], length=200) # Tendance de fond
    
    # Heikin Ashi (Lissage)
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    trend = "HAUSSIER" if ha_close.iloc[-1] > ha_open.iloc[-1] else "BAISSIER"
    
    return trend, df

# ==============================================================================
# 5. LE CERVEAU CENTRAL (PRISE DE DÉCISION)
# ==============================================================================
def consult_council(s, rsi, mc, vis, soc, whale, trend):
    """
    Réunit toutes les données et demande une décision finale à Gemini.
    """
    mood = brain['emotions']
    prompt = f"""
    DÉCISION CRITIQUE POUR {s}.
    État Psychologique Bot: Confiance {mood['confidence']}%, Stress {mood['stress']}%.
    
    DONNÉES ENTRÉE :
    1. ⚛️ MATHS (Monte Carlo): {mc*100:.1f}% probabilité de hausse.
    2. 👁️ VISION (Chartisme): Score {vis:.2f}/1.0.
    3. 🗣️ SOCIAL (Buzz): {soc:.2f} (-1 à 1).
    4. 🐋 BALEINE (Volume): {"OUI" if whale else "NON"}.
    5. 📉 TECH: RSI {rsi:.1f}, Tendance {trend}.
    
    ALGORITHME DE DÉCISION :
    - Si Maths > 60% ET Vision > 0.6 : ACHAT.
    - Si Baleine présente ET RSI < 35 : ACHAT FORT (Dip).
    - Sinon : ATTENTE.
    
    Réponds JSON: {{"vote": "BUY/WAIT", "reason": "Une phrase courte d'explication"}}
    """
    try:
        res = model.generate_content(prompt)
        return json.loads(res.text.replace("```json","").replace("```",""))
    except: return {"vote": "WAIT", "reason": "Erreur Conseil"}

# ==============================================================================
# 6. MOTEUR D'APPRENTISSAGE (GENETICS & REPLAY)
# ==============================================================================
def run_dream_learning():
    """
    Tourne quand le marché est fermé.
    Utilise des données historiques pour optimiser les paramètres (Génétique).
    """
    global brain, bot_state
    cache = {}
    
    # Pré-chargement pour vitesse
    for s in WATCHLIST:
        try:
            df = yf.Ticker(s).history(period="1mo", interval="1h")
            if not df.empty:
                _, df = get_tech_indicators(df)
                cache[s] = df.dropna()
        except: pass

    while True:
        nyc = pytz.timezone('America/New_York')
        now = datetime.now(nyc)
        market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
        
        if market_open:
            time.sleep(60)
            continue
            
        # --- MODE RÊVE ---
        bot_state['status'] = f"🌙 RÊVE (Gen #{brain['stats']['generation']})"
        brain['stats']['generation'] += 1
        
        # 1. Mutation Génétique (On teste de nouveaux paramètres)
        parent = brain['genome']
        mutant = {
            "rsi_buy": max(15, min(60, parent['rsi_buy'] + random.randint(-5, 5))),
            "sl_mult": max(1.0, parent['sl_mult'] + random.uniform(-0.5, 0.5)),
            "tp_mult": max(1.5, parent['tp_mult'] + random.uniform(-0.5, 0.5))
        }
        
        # 2. Simulation Rapide (Replay)
        s = random.choice(list(cache.keys()))
        df = cache[s]
        idx = random.randint(0, len(df)-50)
        subset = df.iloc[idx : idx+50]
        
        pnl = 0
        for i in range(len(subset)-10):
            row = subset.iloc[i]
            # Test du mutant
            if row['RSI'] < mutant['rsi_buy']:
                entry = row['Close']
                sl = entry - (row['ATR'] * mutant['sl_mult'])
                tp = entry + (row['ATR'] * mutant['tp_mult'])
                
                future = subset.iloc[i+1 : i+6]
                if not future.empty:
                    if future['High'].max() > tp: pnl += (tp - entry)
                    elif future['Low'].min() < sl: pnl -= (entry - sl)
        
        # 3. Sélection Naturelle
        if pnl > brain['stats']['best_pnl']:
            brain['stats']['best_pnl'] = pnl
            brain['genome'] = mutant
            save_brain()
            fast_log(f"🧬 **ÉVOLUTION:** Nouveau Gène RSI<{mutant['rsi_buy']} | Gain Simulé +{pnl:.2f}$")
            
        # 4. Mode Replay Visuel (Pour divertir sur le canal Paper)
        if random.random() < 0.1: # 1 fois sur 10
            row = subset.iloc[-1]
            # On simule un signal visuel
            msg = {
                "title": f"🎬 REPLAY SCÉNARIO : {s}",
                "description": f"Test paramètre RSI < {mutant['rsi_buy']}.\nRésultat Simulation: **{pnl:+.2f}$**",
                "color": 0x3498db,
                "footer": {"text": f"Solde Paper (Simulé): {brain['paper_cash']:.2f}$"}
            }
            send_alert(PAPER_WEBHOOK_URL, msg)

        time.sleep(5)

# ==============================================================================
# 7. MOTEUR TRADING HYBRIDE (LIVE + PAPER)
# ==============================================================================
def run_trading_engine():
    global brain, bot_state
    load_brain()
    
    fast_log("🌌 **OMNISCIENCE:** Moteurs démarrés. Prêt.")
    
    while True:
        try:
            nyc = pytz.timezone('America/New_York')
            now = datetime.now(nyc)
            market_open = (now.weekday() < 5 and dtime(9,30) <= now.time() <= dtime(16,0))
            
            if market_open:
                bot_state['status'] = "🟢 LIVE TRADING"
                
                # --- GESTION VENTES (MAIN + PAPER) ---
                # (Simplifié : On applique la même logique aux deux portefeuilles)
                for pf_name, pf_holdings, pf_cash in [("MAIN", brain['holdings'], 'cash'), ("PAPER", brain['paper_holdings'], 'paper_cash')]:
                    for s in list(pf_holdings.keys()):
                        pos = pf_holdings[s]
                        try: curr = yf.Ticker(s).fast_info['last_price']
                        except: continue
                        
                        exit_r = None
                        if curr < pos['stop']: exit_r = "STOP LOSS"
                        elif curr > pos['tp']: exit_r = "TAKE PROFIT"
                        
                        if exit_r:
                            pnl = (curr - pos['entry']) * pos['qty']
                            brain[pf_cash] += pos['qty'] * curr
                            del pf_holdings[s]
                            
                            webhook = DISCORD_WEBHOOK_URL if pf_name == "MAIN" else PAPER_WEBHOOK_URL
                            col = 0x2ecc71 if pnl > 0 else 0xe74c3c
                            send_alert(webhook, {"title": f"{exit_r} : {s}", "description": f"PnL: **{pnl:.2f}$**", "color": col})
                            save_brain()

                # --- SCAN ACHATS ---
                if len(brain['holdings']) < 5:
                    for s in WATCHLIST:
                        if s in brain['holdings']: continue
                        
                        try:
                            df = yf.Ticker(s).history(period="1mo", interval="1h")
                            if df.empty: continue
                            trend, df = get_tech_indicators(df) # Ajoute RSI/ATR
                            row = df.iloc[-1]
                            
                            # FILTRE RAPIDE (Basé sur la génétique)
                            if row['RSI'] < brain['genome']['rsi_buy']:
                                
                                fast_log(f"🔎 **SCAN {s}:** RSI {row['RSI']:.1f} bas. Analyse profonde...")
                                
                                # ANALYSE LOURDE
                                mc = run_monte_carlo(df['Close'])
                                vis = get_vision_score(df)
                                soc = get_social_hype(s)
                                whl, wh_msg = check_whale(df)
                                
                                # CONSEIL
                                council = consult_council(s, row['RSI'], mc, vis, soc, whl, trend)
                                
                                fast_log(f"🧠 **{s}:** MC:{mc:.2f} | Vis:{vis:.2f} | {wh_msg} => {council['vote']}")
                                
                                if council['vote'] == "BUY":
                                    price = row['Close']
                                    
                                    # --- EXECUTION MAIN (50k) ---
                                    qty = 1000 / price
                                    sl = price - (row['ATR'] * brain['genome']['sl_mult'])
                                    tp = price + (row['ATR'] * brain['genome']['tp_mult'])
                                    
                                    brain['holdings'][s] = {"qty": qty, "entry": price, "stop": sl, "tp": tp}
                                    brain['cash'] -= 1000
                                    
                                    # Alerte Main
                                    msg = {
                                        "title": f"🌌 ACHAT OMNISCIENT : {s}",
                                        "description": council['reason'],
                                        "color": 0x2ecc71,
                                        "fields": [
                                            {"name": "Vision", "value": f"{vis:.2f}", "inline": True},
                                            {"name": "Quantique", "value": f"{mc:.2f}", "inline": True},
                                            {"name": "Social", "value": f"{soc:.2f}", "inline": True}
                                        ]
                                    }
                                    send_alert(DISCORD_WEBHOOK_URL, msg)
                                    
                                    # --- EXECUTION PAPER (1k) ---
                                    # Le compte paper suit le maître
                                    qty_p = 100 / price
                                    brain['paper_holdings'][s] = {"qty": qty_p, "entry": price, "stop": sl, "tp": tp}
                                    brain['paper_cash'] -= 100
                                    
                                    send_alert(PAPER_WEBHOOK_URL, {"title": f"📋 COPY TRADE : {s}", "description": "Suivi du compte principal.", "color": 0x3498db})
                                    
                                    save_brain()
                        except: pass
            else:
                bot_state['status'] = "🌙 NUIT (Apprentissage)"
            
            time.sleep(30)
        except: time.sleep(10)

# --- 8. DASHBOARD ---
@app.route('/')
def index():
    # Calcul Equity
    eq_main = brain['cash'] + sum([d['qty']*100 for d in brain['holdings'].values()]) # Approx
    return f"""
    <style>body{{background:#000;color:#0f0;font-family:monospace;padding:20px}}</style>
    <h1>OMNI-GOD V90</h1>
    <p>Status: {bot_state['status']}</p>
    <p>Main Equity: ${eq_main:,.2f}</p>
    <p>Paper Equity: ${brain.get('paper_cash', 0):,.2f}</p>
    <p>Log: {bot_state.get('last_log', 'Running...')}</p>
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
