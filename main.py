import aiohttp, asyncio, pandas as pd, ta, datetime, numpy as np
from textblob import TextBlob
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

TELEGRAM_TOKEN = "8543111323:AAFc6NjDmzAr3E0WQqndWTmT8xM-PVI3z9s"
TWELVEDATA_KEY = "ca1acbf0cedb4488b130c59252891c5e"
ALPHAVANTAGE_KEY = "EOGVA134GOOP2UMU"
NEWS_API_KEY = "qDGIzb9o2OttTxWNvBLMDyZD9KbdQ0qaPHvupsjH"

MIN_CANDLES = 150
MIN_ATR_PCT = 0.3
MIN_BACKTEST_WR = 55
SCAN_INTERVAL = 3600
TRACK_INTERVAL = 60
NEWS_LIMIT = 10
MIN_VOLUME_USD = 500000

CRYPTOS = {
    "ETHUSDT":"ETH/USD","BNBUSDT":"BNB/USD","XRPUSDT":"XRP/USD","SOLUSDT":"SOL/USD",
    "ADAUSDT":"ADA/USD","DOGEUSDT":"DOGE/USD","AVAXUSDT":"AVAX/USD","DOTUSDT":"DOT/USD",
    "MATICUSDT":"MATIC/USD","LTCUSDT":"LTC/USD","LINKUSDT":"LINK/USD","TRXUSDT":"TRX/USD",
    "ATOMUSDT":"ATOM/USD","UNIUSDT":"UNI/USD","SHIBUSDT":"SHIB/USD","FTMUSDT":"FTM/USD",
    "NEARUSDT":"NEAR/USD","AAVEUSDT":"AAVE/USD","EOSUSDT":"EOS/USD","XLMUSDT":"XLM/USD",
    "SUSHIUSDT":"SUSHI/USD","ALGOUSDT":"ALGO/USD","CHZUSDT":"CHZ/USD","KSMUSDT":"KSM/USD",
    "ZILUSDT":"ZIL/USD","ENJUSDT":"ENJ/USD","GRTUSDT":"GRT/USD","BATUSDT":"BAT/USD","RVNUSDT":"RVN/USD"
}

active_trades = {}
stats = {"wins":0,"losses":0,"be":0}
ml_scaler = StandardScaler()
ml_model = SGDClassifier(max_iter=1000, tol=1e-3)

def evaluate_signal(score):
    if score >= 3:
        return "Strong Signal 🚀"
    elif score >= 2:
        return "Moderate Signal ⚡"
    elif score >= 1:
        return "Weak Signal ⚠️"
    elif score >= 0.1:
        return "Very Weak Signal 💤"
    else:
        return "No strong signals right now"

def session_ok():
    h = datetime.datetime.utcnow().hour
    return 7 <= h <= 20

async def fetch(session, symbol, interval="15min", outputsize=1000):
    try:
        async with session.get("https://api.twelvedata.com/time_series",
            params={"symbol":CRYPTOS[symbol],"interval":interval,"outputsize":outputsize,"apikey":TWELVEDATA_KEY},
            timeout=10) as r:
            j = await r.json()
            if "values" in j:
                rows=[{"c":float(v["close"]),"h":float(v["high"]),"l":float(v["low"]),"v":float(v.get("volume",0))} for v in reversed(j["values"])]
                return pd.DataFrame(rows)
    except:
        pass
    return pd.DataFrame()

def enrich(df):
    if len(df)<MIN_CANDLES:
        return pd.DataFrame()
    df["RSI"]=ta.momentum.RSIIndicator(df["c"],14).rsi()
    df["EMA50"]=ta.trend.EMAIndicator(df["c"],50).ema_indicator()
    df["EMA200"]=ta.trend.EMAIndicator(df["c"],200).ema_indicator()
    df["MACD"]=ta.trend.MACD(df["c"]).macd_diff()
    df["ADX"]=ta.trend.ADXIndicator(df["h"],df["l"],df["c"]).adx()
    df["ATR"]=ta.volatility.AverageTrueRange(df["h"],df["l"],df["c"]).average_true_range()
    df["BOLL_H"]=ta.volatility.BollingerBands(df["c"]).bollinger_hband()
    df["BOLL_L"]=ta.volatility.BollingerBands(df["c"]).bollinger_lband()
    df["STOCH_K"]=ta.momentum.StochasticOscillator(df["h"],df["l"],df["c"]).stoch()
    df["CCI"]=ta.trend.CCIIndicator(df["h"],df["l"],df["c"]).cci()
    df["MFI"]=ta.volume.MFIIndicator(df["h"],df["l"],df["c"],df["v"]).money_flow_index()
    df["OBV"]=ta.volume.OnBalanceVolumeIndicator(df["c"],df["v"]).on_balance_volume()
    df["VOLUSD"]=df["v"]*df["c"]
    return df.dropna()

def signal(df, news_sentiment=0, weight=1):
    last=df.iloc[-1]
    s=0
    s+=2.5 if last["EMA50"]>last["EMA200"] else -2.5
    s+=1.2 if last["RSI"]>55 else -1.2 if last["RSI"]<45 else 0
    s+=1.5 if last["MACD"]>0 else -1.5
    s+=1.2 if last["ADX"]>20 else 0
    s+=1.5 if news_sentiment>0.05 else -1.5 if news_sentiment<-0.05 else 0
    s*=weight

    if s >= 3:
        d="BUY"; strength=3
    elif s >= 2:
        d="BUY"; strength=2
    elif s >= 1:
        d="BUY"; strength=1
    elif s >= 0.1:
        d="BUY"; strength=0.1
    elif s <= -3:
        d="SELL"; strength=3
    elif s <= -2:
        d="SELL"; strength=2
    elif s <= -1:
        d="SELL"; strength=1
    elif s <= -0.1:
        d="SELL"; strength=0.1
    else:
        return None

    p=last["c"]; atr=last["ATR"]
    return {"dir":d,"entry":p,"sl":p-atr if d=="BUY" else p+atr,"tp":p+atr*2 if d=="BUY" else p-atr*2,"score":s}

async def scan(context):
    async with aiohttp.ClientSession() as s:
        signals=[]
        for sym in CRYPTOS:
            if sym in active_trades:
                continue
            df=enrich(await fetch(s,sym,"1h"))
            if df.empty:
                continue
            sig=signal(df)
            if sig:
                signals.append((sym,sig))
        signals=sorted(signals,key=lambda x:abs(x[1]["score"]),reverse=True)[:3]
        if signals:
            msg="🚀 TOP SIGNALS\n\n"
            for sym,sig in signals:
                msg+=f"{sym}\n{sig['dir']} | Score {sig['score']} | {evaluate_signal(sig['score'])}\n\n"
                active_trades[sym]=sig
            await context.bot.send_message(chat_id=context.job.chat_id,text=msg)

async def track(context):
    pass

async def start(update:Update,context:ContextTypes.DEFAULT_TYPE):
    kb=[[InlineKeyboardButton(k,callback_data=k)] for k in CRYPTOS]
    await update.message.reply_text("📡 Select a coin or wait for auto scan",reply_markup=InlineKeyboardMarkup(kb))
    context.job_queue.run_repeating(scan,SCAN_INTERVAL,chat_id=update.effective_chat.id)

async def analyze_callback(update:Update,context:ContextTypes.DEFAULT_TYPE):
    q=update.callback_query
    await q.answer()
    sym=q.data
    async with aiohttp.ClientSession() as s:
        df=enrich(await fetch(s,sym,"1h"))
        if df.empty:
            await q.edit_message_text("No data")
            return
        sig=signal(df)
        if not sig:
            await q.edit_message_text("No signal")
            return
        await q.edit_message_text(f"{sym}\n{sig['dir']}\nEntry {sig['entry']}\nSL {sig['sl']}\nTP {sig['tp']}\nScore {sig['score']}")

if __name__=="__main__":
    app=ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",start))
    app.add_handler(CallbackQueryHandler(analyze_callback))
    app.run_polling()