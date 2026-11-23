import os, sys, json, smtplib, urllib.request, datetime as dt
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd, pyotp, yfinance as yf
from growwapi import GrowwAPI

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
MODE = os.environ.get("MODE", "hourly")          # hourly | alert | weekly | monthly
ALERT_PCT = float(os.environ.get("ALERT_PCT", "5"))  # alert threshold in % (change here or via env)
TOP_N = 10
HR_N = 5
CONC_WARN = float(os.environ.get("CONC_WARN", "8"))  # warn if one stock is above this % of portfolio
BIG_MOVE = {"hourly": ALERT_PCT, "weekly": 10, "monthly": 15}


def inr(x, sign=False, dec=0, sym="₹"):
    """Indian digit grouping: 1234567 -> ₹12,34,567"""
    neg = round(abs(x), dec) != 0 and x < 0
    whole, _, frac = f"{abs(x):.{dec}f}".partition(".")
    if len(whole) > 3:
        head, tail, parts = whole[:-3], whole[-3:], []
        while len(head) > 2:
            parts.insert(0, head[-2:]); head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    out = whole + ("." + frac if dec else "")
    return f"{sym}{'-' if neg else ('+' if sign else '')}{out}"


def get_holdings():
    totp = pyotp.TOTP(os.environ["GROWW_TOTP_SECRET"].strip().strip('"').replace(" ", "")).now()
    token = GrowwAPI.get_access_token(api_key=os.environ["GROWW_API_KEY"], totp=totp)
    rows = GrowwAPI(token).get_holdings_for_user(timeout=20)["holdings"]
    df = pd.DataFrame(rows)[["trading_symbol", "quantity", "average_price"]]
    return df[df.quantity > 0].reset_index(drop=True)


def closes(symbols, period="7d", interval="1d"):
    """symbol -> Series of closing prices. Tries NSE (.NS) first, then BSE (.BO)."""
    out = {}
    for suffix in (".NS", ".BO"):
        todo = [s for s in symbols if s not in out]
        if not todo:
            break
        data = yf.download([s + suffix for s in todo], period=period, interval=interval,
                           group_by="ticker", progress=False, auto_adjust=False, threads=True)
        for s in todo:
            try:
                c = data[s + suffix]["Close"].dropna()
                if len(c) >= 2:
                    out[s] = c
            except Exception:
                pass
    return out


def prices_from(cl):
    """Build {symbol: (end_price, base_price, end_date)} for the current mode."""
    out = {}
    today = dt.datetime.now(IST).date()
    first_this = today.replace(day=1)
    first_prev = (first_this - dt.timedelta(days=1)).replace(day=1)
    for s, c in cl.items():
        try:
            if MODE in ("hourly", "alert"):
                out[s] = (float(c.iloc[-1]), float(c.iloc[-2]), c.index[-1].date())
            elif MODE == "weekly":
                if len(c) >= 6:
                    out[s] = (float(c.iloc[-1]), float(c.iloc[-6]), c.index[-1].date())
            elif MODE == "monthly":
                d = pd.Series(c.values, index=[x.date() for x in c.index])
                end = d[d.index < first_this]
                base = d[d.index < first_prev]
                if base.empty:
                    base = d[d.index >= first_prev]
                if len(end) and len(base):
                    out[s] = (float(end.iloc[-1]), float(base.iloc[-1] if len(base) else base.iloc[0]), end.index[-1])
        except Exception:
            pass
    return out


def build(df, prices):
    df = df[df.trading_symbol.isin(prices)].copy()
    df["ltp"] = df.trading_symbol.map(lambda s: prices[s][0])
    df["prev"] = df.trading_symbol.map(lambda s: prices[s][1])
    df["day_pct"] = (df.ltp / df.prev - 1) * 100
    df["day_pnl"] = (df.ltp - df.prev) * df.quantity
    df["value"] = df.ltp * df.quantity
    df["invested"] = df.average_price * df.quantity
    df["total_pnl"] = df.value - df.invested
    df["total_pct"] = df.total_pnl / df.invested * 100
    return df


def hour_change(df, now):
    """Change over roughly the last hour of trading (stateless, from 15-min bars)."""
    t = pd.Timestamp(now)
    t = min(t, t.normalize().replace(hour=15, minute=30))
    ref = t - pd.Timedelta(minutes=60)
    intra = closes(list(df.trading_symbol), period="1d", interval="15m")
    refp = {}
    for s, c in intra.items():
        cut = ref if c.index.tz is not None else ref.tz_localize(None)
        e = c[c.index <= cut]
        if len(e):
            refp[s] = float(e.iloc[-1])
    if len(refp) < 5:
        return None
    h = df[df.trading_symbol.isin(refp)].copy()
    h["h_ref"] = h.trading_symbol.map(refp)
    h["h_pct"] = (h.ltp / h.h_ref - 1) * 100
    h["h_pnl"] = (h.ltp - h.h_ref) * h.quantity
    return h


def hour_totals(hr):
    pnl = hr.h_pnl.sum()
    base = (hr.h_ref * hr.quantity).sum()
    return pnl, pnl / base * 100


def colr(x):
    return "#0a8a3a" if x >= 0 else "#d0302f"


def _tbl(title, head, rows):
    rows = rows.replace("<tr>", "<tr style='border-bottom:1px solid #e6e8eb'>").replace("<td", "<td style='padding:7px 6px'", 1) if False else rows.replace("<tr>", "<tr style='border-bottom:1px solid #e6e8eb'>")
    return (f"<h3 style='margin:20px 0 6px;font-size:15px;border-left:4px solid #2557d6;padding-left:8px'>{title}</h3><div style='overflow-x:auto'><table cellpadding=6 width=100% style='border-collapse:collapse;font-size:13px'>"
            f"<tr style='background:#f1f3f5;color:#555;font-size:12px'>{head}</tr>{rows}</table></div>")


def table_hr(d, title):
    rows = "".join(
        f"<tr><td>{r.trading_symbol}</td><td align=right>{inr(r.ltp, dec=2, sym='')}</td>"
        f"<td align=right style='color:{colr(r.h_pct)}'>{r.h_pct:+.2f}%</td>"
        f"<td align=right>{inr(r.h_pnl, True)}</td></tr>" for r in d.itertuples())
    return _tbl(title, "<th align=left>Stock</th><th>Price</th><th>Hour %</th><th>Hour ₹</th>", rows)


def table_ret(d, title):
    rows = "".join(
        f"<tr><td>{r.trading_symbol}</td><td align=right>{inr(r.average_price, dec=2, sym='')}</td>"
        f"<td align=right>{inr(r.ltp, dec=2, sym='')}</td>"
        f"<td align=right style='color:{colr(r.total_pct)}'>{r.total_pct:+.1f}%</td>"
        f"<td align=right>{inr(r.total_pnl, True)}</td></tr>" for r in d.itertuples())
    return _tbl(title, "<th align=left>Stock</th><th>Avg buy</th><th>Price</th><th>Return %</th><th>P&amp;L ₹</th>", rows)


def table_conc(d, total, title):
    rows = "".join(
        f"<tr><td>{'⚠️ ' if r.value/total*100 > CONC_WARN else ''}{r.trading_symbol}</td>"
        f"<td align=right>{inr(r.value)}</td><td align=right><b>{r.value/total*100:.1f}%</b></td>"
        f"<td align=right style='color:{colr(r.total_pct)}'>{r.total_pct:+.1f}%</td></tr>" for r in d.itertuples())
    return _tbl(title, "<th align=left>Stock</th><th>Value</th><th>Share</th><th>Return %</th>", rows)


def table(d, title):
    g = lambda x: "#0a8a3a" if x >= 0 else "#d0302f"
    lab = {"weekly": "This week", "monthly": "This month"}.get(MODE, "Today")
    rows = "".join(
        "<tr style='border-bottom:1px solid #e6e8eb'>"
        f"<td style='padding:8px 6px'><b style='font-size:15px'>{r.trading_symbol}</b><br>"
        f"<span style='color:#777;font-size:12px'>{r.quantity:g} sh · avg {inr(r.average_price, dec=2, sym='')} → {inr(r.ltp, dec=2, sym='')}</span></td>"
        f"<td align=right style='padding:8px 6px;color:{g(r.day_pct)}'><b>{r.day_pct:+.2f}%</b><br><span style='font-size:12px'>{inr(r.day_pnl, True)}</span></td>"
        f"<td align=right style='padding:8px 6px;color:{g(r.total_pnl)}'><b>{inr(r.total_pnl, True)}</b><br><span style='font-size:12px'>{r.total_pct:+.1f}%</span></td></tr>"
        for r in d.itertuples())
    return (f"<h3 style='margin:20px 0 6px;font-size:15px;border-left:4px solid #2557d6;padding-left:8px'>{title}</h3><table width=100% style='border-collapse:collapse;font-size:13px'>"
            f"<tr style='background:#f1f3f5;color:#555;font-size:12px'><th align=left style='padding:6px'>Stock · shares · avg → price</th>"
            f"<th align=right style='padding:6px'>{lab}</th><th align=right style='padding:6px'>Overall return</th></tr>{rows}</table>")


def html_report(df, missing, now, heading, period, unpriced=None, hr=None):
    dp = df.day_pnl.sum()
    dpct = dp / (df.value.sum() - dp) * 100
    cost = float((unpriced.quantity * unpriced.average_price).sum()) if unpriced is not None and len(unpriced) else 0.0
    v, inv = df.value.sum() + cost, df.invested.sum() + cost
    up, down = int((df.day_pct > 0).sum()), int((df.day_pct < 0).sum())
    col = "#0a8a3a" if dp >= 0 else "#d0302f"
    hr_p, hr_pct = hour_totals(hr) if hr is not None else (0, 0)
    card = lambda label, val, sub="", c="#111": (f"<td width=50% style='background:#f6f7f9;border-radius:8px;padding:10px 12px'>"
        f"<div style='color:#777;font-size:12px'>{label}</div><div style='font-size:19px;font-weight:bold;color:{c}'>{val}</div>"
        f"<div style='font-size:12px;color:{c}'>{sub or '&nbsp;'}</div></td>")
    h = (f"<div style='font-family:Arial,sans-serif;max-width:640px;margin:auto;color:#111'>"
         f"<h2 style='margin:0'>{heading}</h2><div style='color:#777;font-size:12px;margin-bottom:8px'>{now:%d %b %Y, %I:%M %p} IST</div>"
         f"<table width=100% cellspacing=6 style='border-collapse:separate'><tr>{card('Invested', inr(inv))}{card('Current value', inr(v))}</tr>"
         f"<tr>{card('Overall P&amp;L', inr(v-inv, True), f'{(v/inv-1)*100:+.2f}%', colr(v-inv))}{card(period, inr(dp, True), f'{dpct:+.2f}%', col)}</tr></table>"
         + (f"<div style='font-size:13px;color:{colr(hr_p)};margin:2px 0'><b>Last hour: {inr(hr_p, True)} ({hr_pct:+.2f}%)</b></div>" if hr is not None else "")
         + f"<div style='font-size:13px;color:#555'>{up} stocks up · {down} down · {len(df)-up-down} flat</div>")
    h += table(df.nlargest(TOP_N, "day_pct"), f"🟢 Top {TOP_N} gainers (%)")
    h += table(df.nsmallest(TOP_N, "day_pct"), f"🔴 Top {TOP_N} losers (%)")
    h += table(df.nlargest(TOP_N, "day_pnl"), f"💰 Top {TOP_N} ₹ contributors")
    h += table(df.nsmallest(TOP_N, "day_pnl"), f"📉 Top {TOP_N} ₹ drags")
    t = BIG_MOVE.get(MODE, 5)
    big = df[df.day_pct.abs() >= t]
    if len(big):
        h += table(big.sort_values("day_pct"), f"⚠️ Moves of {t:g}% or more")
    if hr is not None:
        h += table_hr(hr.nlargest(HR_N, "h_pct"), f"⏱️ Last hour: top {HR_N} gainers")
        h += table_hr(hr.nsmallest(HR_N, "h_pct"), f"⏱️ Last hour: top {HR_N} losers")
    h += table_ret(df.nlargest(TOP_N, "total_pct"), f"🏆 Since you bought: top {TOP_N} winners")
    h += table_ret(df.nsmallest(TOP_N, "total_pct"), f"🧊 Since you bought: bottom {TOP_N}")
    top = df.nlargest(10, "value")
    h += (f"<h3>🎯 Concentration: top 10 holdings = {top.value.sum()/v*100:.1f}% of portfolio</h3>"
          f"<p style='color:#888;margin:0'>⚠️ marks a stock above {CONC_WARN:g}% of your portfolio</p>")
    h += table_conc(top, v, "Largest holdings")
    if MODE in ("weekly", "monthly"):
        h += "<p style='color:#888'>Based on your current holdings and quantities.</p>"
    if missing:
        h += (f"<p style='color:#888'>No live price for: {', '.join(missing)}. "
              f"Counted at your buy price ({inr(cost)}), so they don't affect P&amp;L or today's change.</p>")
    return h + "</div>"


def tg_send(text):
    """Send a Telegram message (skipped if the two Telegram secrets are not set)."""
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip(), os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat:
        return
    try:
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=json.dumps({"chat_id": chat, "text": text[:4000]}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=20).read()
    except Exception as e:  # never let Telegram problems stop the email
        print("Telegram failed:", type(e).__name__)


def send(html, subject, tg=None, email=True):
    if email and html:
        msg = MIMEMultipart("alternative")
        msg["Subject"], msg["From"], msg["To"] = subject, os.environ["GMAIL_USER"], os.environ["EMAIL_TO"]
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(os.environ["GMAIL_USER"], os.environ["GMAIL_APP_PASSWORD"])
            s.send_message(msg)
    if tg:
        tg_send(tg)


def tg_summary(df, now, heading, period, unpriced=None, hr=None):
    dp = df.day_pnl.sum()
    dpct = dp / (df.value.sum() - dp) * 100
    cost = float((unpriced.quantity * unpriced.average_price).sum()) if unpriced is not None and len(unpriced) else 0.0
    v, inv = df.value.sum() + cost, df.invested.sum() + cost
    up, down = int((df.day_pct > 0).sum()), int((df.day_pct < 0).sum())
    line = lambda r: (f"{r.trading_symbol}  {r.day_pct:+.2f}%  ({inr(r.day_pnl, True)})\n"
                      f"   {r.quantity:g} sh · avg {inr(r.average_price, dec=2)} → {inr(r.ltp, dec=2)} · return {inr(r.total_pnl, True)} ({r.total_pct:+.1f}%)")
    return (f"📊 {heading} · {now:%d %b, %I:%M %p}\n\n"
            f"Invested: {inr(inv)}\nCurrent Value: {inr(v)}\n"
            f"Overall P&L: {inr(v-inv, True)} ({(v/inv-1)*100:+.2f}%)\n"
            f"{period}: {inr(dp, True)} ({dpct:+.2f}%)\n"
            + (f"Last hour: {inr(hour_totals(hr)[0], True)} ({hour_totals(hr)[1]:+.2f}%)\n" if hr is not None else "")
            + f"{up} up · {down} down · {len(df)-up-down} flat\n\n"
            f"🟢 Top {TOP_N} gainers\n" + "\n".join(line(r) for r in df.nlargest(TOP_N, "day_pct").itertuples()) +
            f"\n\n🔴 Top {TOP_N} losers\n" + "\n".join(line(r) for r in df.nsmallest(TOP_N, "day_pct").itertuples()))


def run_alert(df, now):
    """Alert only for stocks that JUST crossed the threshold (not ones already past it 25+ min ago)."""
    hits = df[df.day_pct.abs() >= ALERT_PCT]
    if hits.empty:
        print("No stock beyond threshold.")
        return
    intra = closes(list(hits.trading_symbol), period="1d", interval="5m")
    new = []
    for r in hits.itertuples():
        c = intra.get(r.trading_symbol)
        fresh = True
        if c is not None:
            cutoff = pd.Timestamp(now)
            if c.index.tz is None:
                cutoff = cutoff.tz_localize(None)
            earlier = c[c.index <= cutoff - pd.Timedelta(minutes=25)]
            if len(earlier):
                pct_then = (float(earlier.iloc[-1]) / r.prev - 1) * 100
                fresh = abs(pct_then) < ALERT_PCT
        if fresh:
            new.append(r.Index)
    if not new:
        print("Moves already alerted earlier.")
        return
    d = hits.loc[new].sort_values("day_pct")
    rows = "".join(
        f"<tr><td><b>{r.trading_symbol}</b></td><td align=right>{inr(r.ltp, dec=2, sym='')}</td>"
        f"<td align=right style='color:{'#0a8a3a' if r.day_pct>=0 else '#d0302f'}'><b>{r.day_pct:+.2f}%</b></td>"
        f"<td align=right>{inr(r.day_pnl, True)}</td><td align=right>{inr(r.value)}</td></tr>" for r in d.itertuples())
    html = (f"<div style='font-family:Arial'><h2>🚨 Price alert · {now:%d %b, %I:%M %p} IST</h2>"
            f"<p>Moved {ALERT_PCT:g}% or more today:</p>"
            f"<table cellpadding=5 style='border-collapse:collapse;font-size:14px'>"
            f"<tr style='background:#eee'><th align=left>Stock</th><th>Price</th><th>Day %</th><th>Impact ₹</th><th>Holding ₹</th></tr>"
            f"{rows}</table></div>")
    names = ", ".join(f"{r.trading_symbol} {r.day_pct:+.1f}%" for r in d.itertuples())
    tg = f"🚨 Price alert · {now:%d %b, %I:%M %p}\n\n" + "\n".join(
        f"{r.trading_symbol}  {r.day_pct:+.2f}%  @ {inr(r.ltp, dec=2, sym='₹')}  (impact {inr(r.day_pnl, True)})" for r in d.itertuples())
    send(html, f"🚨 Alert: {names}"[:150], tg)
    print("Alert sent:", names)


def run_review(holdings, now):
    """Compare your holdings with the public Nifty 500 scanner results."""
    url = os.environ.get("SCANNER_DATA_URL", "").strip()
    if not url:
        sys.exit("SCANNER_DATA_URL secret is missing")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = json.load(urllib.request.urlopen(req, timeout=30))
    info = {r["symbol"]: r for r in data["all"]}
    rows = []
    for r in holdings.itertuples():
        i = info.get(r.trading_symbol)
        if i and i.get("price"):
            rows.append(dict(sym=r.trading_symbol, value=r.quantity * i["price"], mine=(i["price"] / r.average_price - 1) * 100,
                             r6=i.get("ret6m"), off=i.get("from_high"), tag=i.get("tag") or ""))
    d = pd.DataFrame(rows)
    skipped = len(holdings) - len(d)
    weak = d[d.tag == "weak"].sort_values("value", ascending=False).head(15) if len(d) else d
    strong = d[d.tag.isin(["short", "long"])].sort_values("value", ascending=False) if len(d) else d
    owned = set(holdings.trading_symbol)
    ideas = lambda k: [x for x in data[k] if x["symbol"] not in owned][:5]
    f = lambda x: "–" if x is None or x != x else f"{x:+.0f}%"
    wl = lambda r: f"{r.sym}  {inr(r.value)}  | you {r.mine:+.0f}%  | 6M {f(r.r6)}  | {f(r.off)} from high"
    il = lambda x: f"{x['symbol']}  {x['ret3m']:+.0f}% 3M  ({x.get('note') or x['industry']})"
    tg = (f"🔎 Weekly portfolio review · {now:%d %b}\nScanner data: {data['generated']}\n\n"
          f"⚠️ Holdings looking weak ({len(weak)}):\n" + ("\n".join(wl(r) for r in weak.itertuples()) or "None") +
          f"\n\n✅ Holdings on strong lists ({len(strong)}):\n" + ("\n".join(f"{r.sym} ({r.tag}-term)" for r in strong.itertuples()) or "None") +
          "\n\n💡 Short-term ideas you don't own:\n" + "\n".join(il(x) for x in ideas("short")) +
          "\n\n💡 Long-term ideas you don't own:\n" + "\n".join(il(x) for x in ideas("long")) +
          (f"\n\n{skipped} holdings not in Nifty 500 (not scanned)." if skipped else "") +
          "\n\nScreens, not advice. Weak = below 200-day average, down over 6M, 25%+ off its high.")
    esc = lambda t: t.replace("&", "&amp;").replace("<", "&lt;").replace("\n", "<br>")
    send(f"<div style='font-family:Arial'>{esc(tg)}</div>", f"Weekly review: {len(weak)} weak, {len(strong)} strong holdings", tg)
    print("Review sent.")


def main():
    now = dt.datetime.now(IST)
    holdings = get_holdings()
    if MODE == "review":
        run_review(holdings, now)
        return
    syms = list(holdings.trading_symbol)
    period = {"hourly": "7d", "alert": "7d", "weekly": "1mo", "monthly": "3mo"}[MODE]
    cl = closes(syms, period)
    prices = prices_from(cl)
    if not prices:
        sys.exit("No prices fetched")
    if MODE in ("hourly", "alert") and not os.environ.get("FORCE"):
        if max(p[2] for p in prices.values()) != now.date():
            print("Market closed today, skipping.")
            return
    df = build(holdings, prices)
    if MODE == "alert":
        run_alert(df, now)
        return
    missing = sorted(set(syms) - set(prices))
    dp = df.day_pnl.sum()
    arrow = "▲" if dp >= 0 else "▼"
    is_close = now.hour >= 16
    hr = None
    if MODE == "hourly" and now.hour >= 10:
        try:
            hr = hour_change(df, now)
        except Exception as e:
            print("Last-hour calc failed:", type(e).__name__)
    if MODE == "hourly":
        heading, label = ("Closing report", "Today") if is_close else ("Portfolio update", "Today")
        subject = (f"Closing report {arrow} {inr(dp, True)} · {now:%d %b}" if is_close
                   else f"Portfolio {arrow} {inr(dp, True)} today · {now:%d %b %H:%M}")
    elif MODE == "weekly":
        heading, label = "Weekly report", "This week"
        subject = f"Weekly report {arrow} {inr(dp, True)} · week to {now:%d %b}"
    else:
        last = (now.date().replace(day=1) - dt.timedelta(days=1))
        heading, label = f"Monthly report ({last:%B %Y})", last.strftime("%B")
        subject = f"Monthly report {last:%b %Y} {arrow} {inr(dp, True)}"
    unpriced = holdings[holdings.trading_symbol.isin(missing)]
    # Hourly: Telegram every run, full email only at the close (or on a manual test run)
    want_email = MODE != "hourly" or is_close or bool(os.environ.get("FORCE"))
    html = html_report(df, missing, now, heading, label, unpriced, hr) if want_email else None
    send(html, subject, tg_summary(df, now, heading, label, unpriced, hr), want_email)
    print("Sent.")


if __name__ == "__main__":
    main()
