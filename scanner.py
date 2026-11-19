import os, io, json, time, urllib.request, datetime as dt
import numpy as np, pandas as pd, requests, yfinance as yf

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
URLS = ["https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"]
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
N = 10


def universe():
    for u in URLS:
        try:
            r = requests.get(u, headers=HDR, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            if len(df) > 300:
                df.to_csv("nifty500_backup.csv", index=False)
                return df
        except Exception as e:
            print("Nifty 500 list fetch failed:", u.split("/")[2], type(e).__name__)
    if os.path.exists("nifty500_backup.csv"):
        print("Using nifty500_backup.csv")
        return pd.read_csv("nifty500_backup.csv")
    raise SystemExit("Could not get the Nifty 500 list. Download it from NSE and upload it as nifty500_backup.csv")


def prices(symbols):
    out = {}
    for i in range(0, len(symbols), 100):
        chunk = symbols[i:i + 100]
        d = yf.download([s + ".NS" for s in chunk], period="1y", interval="1d", group_by="ticker",
                        progress=False, auto_adjust=False, threads=True)
        for s in chunk:
            try:
                x = d[s + ".NS"][["Close", "High", "Volume"]].dropna()
                if len(x) >= 120:
                    out[s] = x
            except Exception:
                pass
    return out


def metrics(x):
    c, h, v = x.Close, x.High, x.Volume
    n, last = len(c), float(c.iloc[-1])
    ret = lambda k: (last / float(c.iloc[-1 - k]) - 1) * 100 if n > k else np.nan
    d200 = float(c.tail(200).mean()) if n >= 200 else np.nan
    return dict(
        price=last, day=ret(1), ret1m=ret(21), ret3m=ret(63), ret6m=ret(126),
        ret12m=(last / float(c.iloc[0]) - 1) * 100,
        from_high=(last / float(h.max()) - 1) * 100,
        vol_surge=float(v.tail(5).mean() / max(v.tail(50).mean(), 1)),
        above50=last > float(c.tail(50).mean()), above200=bool(last > d200) if d200 == d200 else False,
        dist200=(last / d200 - 1) * 100 if d200 == d200 else np.nan,
        maxdd=float(((c / c.cummax()) - 1).min() * 100),
        turnover=float((c * v).tail(50).mean() / 1e7))  # crore


def nifty():
    try:
        x = yf.download("^NSEI", period="1y", interval="1d", progress=False, auto_adjust=False)["Close"].squeeze().dropna()
        return float(x.iloc[-1]), bool(x.iloc[-1] > x.tail(200).mean())
    except Exception:
        return None, None


def tg_send(text):
    t, c = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip(), os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if t and c:
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://api.telegram.org/bot{t}/sendMessage",
                data=json.dumps({"chat_id": c, "text": text[:4000]}).encode(), headers={"Content-Type": "application/json"}), timeout=20)
        except Exception as e:
            print("Telegram failed:", type(e).__name__)


def track(df, lists, nv):
    """Save today's picks, score older picks vs Nifty 50, and list what changed since last run."""
    hist = json.load(open("docs/history.json")) if os.path.exists("docs/history.json") else []
    today = dt.datetime.now(IST).date()
    hist = [h for h in hist if h["d"] != today.isoformat()]
    rows = []
    for name in ("short", "long", "weak"):
        for lb in (7, 28, 84):
            rs, ex = [], []
            for h in hist:
                if (today - dt.date.fromisoformat(h["d"])).days < lb or not h.get("nifty") or not nv:
                    continue
                cur = [df.price[k] / p - 1 for k, p in h[name].items() if k in df.index]
                if cur:
                    r = float(np.mean(cur)) * 100
                    rs.append(r); ex.append(r - (nv / h["nifty"] - 1) * 100)
            if rs:
                rows.append(dict(list=name, days=lb, runs=len(rs), avg=clean(np.mean(rs)), excess=clean(np.mean(ex))))
    prev = hist[-1] if hist else None
    msg = ""
    if prev:
        for name, label in (("short", "Short-term"), ("long", "Long-term"), ("weak", "Weak")):
            new, gone = set(lists[name]) - set(prev[name]), set(prev[name]) - set(lists[name])
            if new or gone:
                msg += f"{label}: " + (f"+ {', '.join(sorted(new))} " if new else "") + (f"- {', '.join(sorted(gone))}" if gone else "") + "\n"
    hist.append(dict(d=today.isoformat(), nifty=nv, **{k: {s: round(float(df.price[s]), 2) for s in v} for k, v in lists.items()}))
    os.makedirs("docs", exist_ok=True)
    json.dump(hist[-400:], open("docs/history.json", "w"))
    return rows, msg


def pr(s, higher=True):
    return s.rank(pct=True, ascending=higher) * 100


def fundamentals(sym):
    try:
        i = yf.Ticker(sym + ".NS").info
        return dict(roe=i.get("returnOnEquity"), de=i.get("debtToEquity"), eg=i.get("earningsGrowth"),
                    pm=i.get("profitMargins"))
    except Exception:
        return {}


def clean(v):
    if isinstance(v, (np.floating, float)):
        return None if v != v else round(float(v), 2)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


def main():
    uni = universe()
    uni["Symbol"] = uni["Symbol"].astype(str).str.strip()
    px = prices(list(uni.Symbol))
    print(f"Prices for {len(px)} of {len(uni)} stocks")
    if len(px) < 100:
        raise SystemExit("Too little price data, aborting so the old dashboard stays.")
    df = pd.DataFrame({s: metrics(x) for s, x in px.items()}).T
    for col in df.columns:
        if col in ("above50", "above200"):
            df[col] = df[col].astype(bool)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    meta = uni.set_index("Symbol")
    df["name"] = [meta["Company Name"].get(s, s) for s in df.index]
    df["industry"] = [meta["Industry"].get(s, "") for s in df.index]
    liquid = df.turnover >= 1

    # Short-term: momentum + rising volume + near highs, above 50-day average
    df["short"] = 0.30 * pr(df.ret1m) + 0.25 * pr(df.ret3m) + 0.25 * pr(df.vol_surge) + 0.20 * pr(df.from_high)
    short = df[liquid & df.above50].nlargest(N, "short").copy()
    short["score"] = short.short
    short["note"] = np.where(short.ret1m > 25, "⚠️ stretched, up fast", "")

    # Long-term: strong steady trend, then check profitability on the best 60
    df["trend"] = 0.4 * pr(df.ret12m) + 0.3 * pr(df.maxdd) + 0.3 * pr(df.from_high)
    pool = df[liquid & df.above200 & (df.ret12m > 0) & (df.ret6m > 0)].nlargest(60, "trend").copy()
    fund = {s: fundamentals(s) for s in pool.index}
    f = pd.DataFrame(fund).T.reindex(index=pool.index, columns=["roe", "de", "eg", "pm"])
    for col in f.columns:
        f[col] = pd.to_numeric(f[col], errors="coerce")
    fin = pool.industry.str.contains("Financial", case=False, na=False)
    f.loc[fin, "de"] = np.nan  # debt/equity is not meaningful for banks and lenders
    q = pd.concat([pr(f.roe), pr(f.eg), pr(f.pm), pr(f.de, higher=False)], axis=1)
    pool["quality"] = q.mean(axis=1)
    pool["nq"] = q.notna().sum(axis=1)
    pool["score"] = np.where(pool.nq >= 2, 0.5 * pool.trend + 0.5 * pool.quality, 0.6 * pool.trend)
    pool["note"] = [("ROE %.0f%%" % (f.roe[s] * 100) if f.roe[s] == f.roe[s] else "fundamentals n/a") for s in pool.index]
    long_ = pool.nlargest(N, "score")

    # Weak / neglected: below long-term average, falling, far from highs
    df["weak"] = 100 - (0.30 * pr(df.ret3m) + 0.25 * pr(df.ret6m) + 0.25 * pr(df.from_high) + 0.20 * pr(df.dist200))
    df["weak_flag"] = (~df.above200) & (df.ret6m < 0) & (df.from_high < -25)
    weak = df[liquid & ~df.above200 & (df.ret6m < 0)].nlargest(N, "weak").copy()
    weak["score"] = weak.weak
    weak["note"] = [", ".join(filter(None, ["below 200-day avg", "%.0f%% off high" % -r.from_high,
                    "volume fading" if r.vol_surge < 0.8 else ""])) for r in weak.itertuples()]

    sec = df.groupby("industry").agg(stocks=("price", "size"), ret1m=("ret1m", "median"), ret3m=("ret3m", "median"),
                                      above200=("above200", "mean")).reset_index()
    sec["above200"] = sec.above200 * 100
    sec = sec[sec.stocks >= 3].sort_values("ret3m", ascending=False)

    tag = {s: "short" for s in short.index}
    tag.update({s: "long" for s in long_.index})
    tag.update({s: "weak" for s in df.index[df.weak_flag]})  # weak flag wins
    df["tag"] = [tag.get(s, "") for s in df.index]
    df["symbol"] = df.index
    nv, nup = nifty()
    b200 = df.above200.mean() * 100
    mood = ("Weak", "Market looks weak: Nifty is below its 200-day average and few stocks are above theirs. Momentum picks fail more often in falling markets.") if (nup is False and b200 < 45) else \
           ("Healthy", "Market looks healthy: Nifty is above its 200-day average and most stocks are in uptrends.") if (nup and b200 >= 55) else \
           ("Mixed", "Market is mixed. Be selective and check each pick yourself.")
    trk, chg = track(df, dict(short=list(short.index), long=list(long_.index), weak=list(weak.index)), nv)
    if chg:
        tg_send("📊 Scanner changes " + dt.datetime.now(IST).strftime("%d %b") + "\n" + chg + "(+ new, - dropped)\nMarket mood: " + mood[0])
    longest = max(px.values(), key=len)
    sdates = [d.strftime("%d %b %y") for d in longest.index[::-5][::-1]]
    df["spark"] = [[round(float(v), 1) for v in px[s].Close.iloc[::-5][::-1]] for s in df.index]
    cols = ["symbol", "spark", "name", "industry", "price", "day", "ret1m", "ret3m", "ret6m", "ret12m", "from_high",
            "vol_surge", "dist200", "weak_flag", "tag"]
    rows = lambda d, extra=("score", "note"): [{k: clean(v) for k, v in r.items()} for r in
                                               d.assign(symbol=d.index)[["symbol", "name", "industry", "price", "ret1m", "ret3m",
                                                                         "ret6m", "from_high", "vol_surge", *extra]].to_dict("records")]
    out = dict(
        generated=dt.datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST"), scanned=len(df),
        breadth=dict(above50=clean(df.above50.mean() * 100), above200=clean(df.above200.mean() * 100),
                     advancers=clean((df.day > 0).mean() * 100), median1m=clean(df.ret1m.median()),
                     median3m=clean(df.ret3m.median())),
        mood=dict(label=mood[0], text=mood[1]), track=trk, sdates=sdates,
        short=rows(short), long=rows(long_), weak=rows(weak),
        sectors=[{k: clean(v) for k, v in r.items()} for r in sec.to_dict("records")],
        all=[{k: clean(v) for k, v in r.items()} for r in df[cols].to_dict("records")])
    os.makedirs("docs", exist_ok=True)
    js = json.dumps(out, ensure_ascii=False, allow_nan=False)
    open("docs/data.json", "w", encoding="utf-8").write(js)
    open("docs/index.html", "w", encoding="utf-8").write(HTML.replace("__DATA__", js.replace("</", "<\\/")))
    print("Done:", out["generated"])


HTML = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Nifty 500 Scanner</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--card:#f2f3f5;--up:#0a8a3a;--dn:#d0302f;--ac:#2557d6}
@media(prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;--mut:#999;--card:#1d1f23;--up:#3ddc84;--dn:#ff6b6b;--ac:#7aa2ff}}
body{font:14px system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:auto;padding:12px;max-width:1100px}
h1{font-size:20px;margin:4px 0}.m{color:var(--mut);font-size:12px}
.cards{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}.c{background:var(--card);border-radius:8px;padding:8px 12px}
.c b{display:block;font-size:18px}
.tabs button{background:var(--card);color:var(--fg);border:0;padding:8px 12px;margin:2px;border-radius:6px;cursor:pointer}
.tabs button.on{background:var(--ac);color:#fff}
.w{overflow-x:auto}table{border-collapse:collapse;width:100%;margin-top:8px}
th,td{padding:6px 8px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--card)}
th:first-child,td:first-child,td:nth-child(2){text-align:left}th{cursor:pointer;color:var(--mut)}
.st{cursor:pointer;color:#e0a800}.lk{color:var(--ac);cursor:pointer;text-decoration:underline}#md{display:none;position:fixed;inset:0;background:#0008;padding:16px;overflow:auto}#md div{background:var(--bg);max-width:420px;margin:auto;padding:14px;border-radius:10px}
.up{color:var(--up)}.dn{color:var(--dn)}input{width:100%;padding:8px;box-sizing:border-box;margin-top:8px}
</style></head><body>
<h1>Nifty 500 Scanner</h1><div class=m id=gen></div><div id=mood class=c></div><div class=cards id=br></div>
<div id=md></div><div class=tabs id=tabs></div><div id=out></div>
<p class=m>Data from Yahoo Finance, computed by a free script. Rankings are statistical screens, not investment advice or predictions.
Check a company's business, valuation and news yourself before acting.</p>
<script id=d type=application/json>__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('d').textContent);
const F=(x,d)=>x==null?'–':(+x).toFixed(d),C=x=>x==null?'':x>=0?'up':'dn';
const S=[['Stock','symbol'],['Industry','industry'],['Price','price',2,'',0],['1M','ret1m',1,'%',1],['3M','ret3m',1,'%',1],
['6M','ret6m',1,'%',1],['vs 52w high','from_high',1,'%',1],['Vol ×','vol_surge',2,'×',0],['Score','score',0,'',0],['Note','note']];
const A=S.filter(c=>c[1]!='score'&&c[1]!='note').concat([['Tag','tag']]);
const SEC=[['Industry','industry'],['Stocks','stocks',0,'',0],['1M median','ret1m',1,'%',1],['3M median','ret3m',1,'%',1],['% above 200-day','above200',0,'%',0]];
const W=new Set(),K='wl';try{JSON.parse(localStorage.getItem(K)||'[]').forEach(x=>W.add(x))}catch(e){}
const save=()=>{try{localStorage.setItem(K,JSON.stringify([...W]))}catch(e){}};
function T(rows,cols){return '<div class=w><table><tr>'+cols.map(c=>'<th data-k="'+c[1]+'">'+c[0]+'</th>').join('')+'</tr>'+
rows.map(r=>'<tr>'+cols.map(c=>{const v=r[c[1]];return c[1]=='symbol'?'<td title="'+(r.name||'')+'"><span class=st data-s="'+v+'">'+(W.has(v)?'★':'☆')+'</span> <a class=lk data-s="'+v+'">'+v+'</a></td>':c[2]===undefined?'<td title="'+(r.name||'')+'">'+(v??'')+'</td>':
'<td class="'+(c[4]?C(v):'')+'">'+F(v,c[2])+(v==null?'':c[3])+'</td>'}).join('')+'</tr>').join('')+'</table></div>'}
document.getElementById('gen').textContent='Updated '+D.generated+' · '+D.scanned+' stocks scanned';
const b=D.breadth;document.getElementById('br').innerHTML=[['Above 50-day',F(b.above50,0)+'%'],['Above 200-day',F(b.above200,0)+'%'],
['Up today',F(b.advancers,0)+'%'],['Median 1M',F(b.median1m,1)+'%'],['Median 3M',F(b.median3m,1)+'%']].map(x=>'<div class=c>'+x[0]+'<b>'+x[1]+'</b></div>').join('');
const TR=[['List','list'],['Picks at least (days old)','days',0,'',0],['Days sampled','runs',0,'',0],['Avg return since listed','avg',1,'%',1],['vs Nifty 50','excess',1,'%',1]];
const TABS={'Watchlist':()=>W.size?T(D.all.filter(x=>W.has(x.symbol)),A):'<p class=m>Tap ☆ next to a stock to add it here. Saved in this browser only.</p>',
'Track record':()=>(D.track.length?T(D.track,TR):'<p class=m>Needs a few weeks of daily history.</p>')+'<p class=m>Average return of past picks since they were listed, versus the Nifty 50. Early numbers are noisy.</p>','Short-term':()=>T(D.short,S),'Long-term':()=>T(D.long,S),'Weak / fading':()=>T(D.weak,S),
'Sectors':()=>T(D.sectors,SEC),'All stocks':null};
let cur='Short-term',sk='ret3m',sd=-1;
function show(){document.getElementById('tabs').innerHTML=Object.keys(TABS).map(k=>'<button class="'+(k==cur?'on':'')+'" data-t="'+k+'">'+k+'</button>').join('');
const o=document.getElementById('out');
if(cur=='All stocks'){o.innerHTML='<input id=q placeholder="Search symbol, name or industry"><div id=tb></div>';
const q=document.getElementById('q');q.oninput=all;all()}else o.innerHTML=TABS[cur]()}
function all(){const q=document.getElementById('q').value.toLowerCase();
let r=D.all.filter(x=>(x.symbol+x.name+x.industry).toLowerCase().includes(q));
r.sort((a,b)=>sd*((a[sk]??-1e9)-(b[sk]??-1e9))||0);document.getElementById('tb').innerHTML=T(r.slice(0,150),A)}
document.getElementById('mood').innerHTML='<b>Market mood: '+D.mood.label+'</b><div class=m>'+D.mood.text+'</div>';
function detail(s){const x=D.all.find(a=>a.symbol==s),sp=x.spark||[],ds=D.sdates.slice(-sp.length),sc=D.sectors.find(a=>a.industry==x.industry);
document.getElementById('md').innerHTML='<div><b>'+x.symbol+'</b> · '+x.name+'<div class=m>'+x.industry+'</div>'+chart(sp)+'<div id=tip class=m>Hover or touch the chart for date and price</div><p>Price '+F(x.price,2)+' · 1M '+F(x.ret1m,1)+'% · 3M '+F(x.ret3m,1)+'% · 6M '+F(x.ret6m,1)+'%<br>'+F(x.from_high,1)+'% from 52w high · volume '+F(x.vol_surge,2)+'× normal · '+F(x.dist200,1)+'% vs 200-day avg<br>Sector 3M median: '+(sc?F(sc.ret3m,1)+'%':'–')+' · Tag: '+(x.tag||'none')+'</p><button class="on" data-x=1>Close</button></div>';
document.getElementById('md').style.display='block';hover(sp,ds)}
const CW=300,CH=110,lo_=a=>Math.min(...a),hi_=a=>Math.max(...a);
const cx=(i,n)=>i*CW/(n-1||1),cy=(v,a)=>8+(CH-26)*(1-(v-lo_(a))/((hi_(a)-lo_(a))||1));
function chart(sp){const n=sp.length,hi=hi_(sp),lo=lo_(sp),up=sp[n-1]>=sp[0],col=up?'var(--up)':'var(--dn)',p=sp.map((v,i)=>cx(i,n).toFixed(1)+','+cy(v,sp).toFixed(1)).join(' ');
const mk=(v,t)=>{const i=sp.indexOf(v),X=cx(i,n),Y=cy(v,sp);return '<circle cx='+X+' cy='+Y+' r=3 fill="'+col+'"/><text x='+X+' y='+(t=='H'?Y-5:Y+12)+' font-size=9 fill="var(--mut)" text-anchor='+(X>150?'end':'start')+'>'+t+' '+v.toFixed(1)+'</text>'};
return '<svg id=ch viewBox="0 0 300 110" width=100% style="touch-action:none;cursor:crosshair"><polygon fill="'+col+'" opacity=.13 points="0,'+(CH-18)+' '+p+' '+CW+','+(CH-18)+'"/><polyline fill=none stroke="'+col+'" stroke-width=2 points="'+p+'"/>'+mk(hi,'H')+mk(lo,'L')+'<line id=cl y1=4 y2='+(CH-18)+' stroke="var(--mut)" stroke-dasharray="3 3" opacity=0 /><circle id=cd r=4.5 fill="'+col+'" stroke="var(--bg)" stroke-width=2 opacity=0 /></svg>'}
function hover(sp,ds){const s=document.getElementById('ch');if(!s)return;const n=sp.length,hi=hi_(sp);
const mv=e=>{const r=s.getBoundingClientRect(),i=Math.max(0,Math.min(n-1,Math.round((e.clientX-r.left)/r.width*(n-1)))),X=cx(i,n),Y=cy(sp[i],sp),l=document.getElementById('cl'),d=document.getElementById('cd'),ch=(sp[i]/sp[0]-1)*100;
l.setAttribute('x1',X);l.setAttribute('x2',X);l.setAttribute('opacity',1);d.setAttribute('cx',X);d.setAttribute('cy',Y);d.setAttribute('opacity',1);
document.getElementById('tip').innerHTML='<b>'+(ds[i]||'')+'</b> · ₹'+F(sp[i],2)+' · '+(ch>=0?'+':'')+F(ch,1)+'% vs a year ago · '+F((sp[i]/hi-1)*100,1)+'% from the high'};
s.addEventListener('pointermove',mv);s.addEventListener('pointerdown',mv)}
document.addEventListener('click',e=>{const t=e.target;
if(t.classList.contains('st')){W.has(t.dataset.s)?W.delete(t.dataset.s):W.add(t.dataset.s);save();show();return}
if(t.classList.contains('lk')){detail(t.dataset.s);return}
if(t.dataset.x||t.id=='md'){document.getElementById('md').style.display='none';return}
if(t.dataset.t){cur=t.dataset.t;show()}
else if(t.tagName=='TH'&&cur=='All stocks'&&t.dataset.k){sd=sk==t.dataset.k?-sd:-1;sk=t.dataset.k;all()}});
show();
</script></body></html>"""

if __name__ == "__main__":
    main()
