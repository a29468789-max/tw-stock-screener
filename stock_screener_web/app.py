import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="台股波段決策輔助", layout="wide")


def generate_market_snapshot(n=120, seed=42):
    rng = np.random.default_rng(seed)
    symbols = [str(1101 + i) for i in range(n)]
    names = [f"股票{i:03d}" for i in range(n)]

    trend = np.clip(rng.normal(0, 6, n), -20, 20)
    conf = np.clip(rng.normal(0.66, 0.12, n), 0, 1)
    rev = np.clip(rng.normal(0.33, 0.15, n), 0, 1)

    state = []
    action = []
    for t, c, r in zip(trend, conf, rev):
        if t >= 8 and c >= 0.70 and r <= 0.40:
            s = "強多"
            a = "Long"
        elif 3 <= t <= 7:
            s = "多"
            a = "Long"
        elif -2 <= t <= 2:
            s = "盤整"
            a = "Watch"
        elif -7 <= t <= -3:
            s = "空"
            a = "Short"
        elif t <= -8 and c >= 0.70 and r <= 0.40:
            s = "強空"
            a = "Short"
        else:
            s = "盤整"
            a = "Watch"
        state.append(s)
        action.append(a)

    summary_pool = [
        "均線多頭排列", "均線空頭排列", "箱體突破帶量", "箱體跌破帶量", "RSI上穿30", "RSI下穿70",
        "MACD翻正", "MACD翻負", "三紅K", "三黑K", "量能不足", "假突破風險"
    ]

    summary = ["、".join(rng.choice(summary_pool, 2, replace=False)) for _ in range(n)]
    regime = ["順勢" if rng.random() > 0.28 else "逆勢" for _ in range(n)]
    risk = ["低" if x < 0.25 else "中" if x < 0.5 else "高" for x in rev]

    return pd.DataFrame({
        "代碼": symbols,
        "名稱": names,
        "狀態": state,
        "TrendScore": np.round(trend, 2),
        "Confidence": np.round(conf, 2),
        "ReversalRisk": np.round(rev, 2),
        "建議": action,
        "策略摘要": summary,
        "順逆勢": regime,
        "風險": risk,
    })


def build_single_report(row):
    t = row["TrendScore"]
    c = row["Confidence"]
    r = row["ReversalRisk"]
    action = row["建議"]

    if action == "Long":
        entry = "突破前高/箱體上緣後回測不破，或站回 MA20"
        sl = "前低 or ATR*1.5 or MA20下方（取較緊且合理）"
        tp = "前壓區 + RR>=1.5，或 2*ATR"
        inv = "跌回箱體、跌破MA20、量能不足"
    elif action == "Short":
        entry = "跌破支撐/箱體下緣後回抽不過，或跌破 MA20"
        sl = "前高 or ATR*1.5 or MA20上方"
        tp = "前支撐區 + RR>=1.5，或 2*ATR"
        inv = "站回箱體、站回MA20、放量反彈"
    else:
        entry = "等待結構完成（突破/跌破後再確認）"
        sl = "N/A"
        tp = "N/A"
        inv = "N/A"

    top5 = [
        {"策略": "型態面", "貢獻": round(abs(t)*0.28, 2), "理由": "日K型態與趨勢一致", "無效": inv},
        {"策略": "技術面", "貢獻": round(abs(t)*0.28, 2), "理由": "均線/支撐壓力給出方向", "無效": inv},
        {"策略": "指標精選", "貢獻": round(abs(t)*0.22, 2), "理由": "RSI/MACD/KD動能配合", "無效": inv},
        {"策略": "盤中修正", "貢獻": round(abs(t)*0.10, 2), "理由": "今日日K與量能進度修正", "無效": "假突破/假跌破"},
        {"策略": "籌碼/基本面", "貢獻": round(abs(t)*0.10, 2), "理由": "作為底色偏好不主導翻轉", "無效": "資料缺失時降信心"},
    ]

    return {
        "狀態": row["狀態"],
        "TrendScore": t,
        "Confidence": c,
        "ReversalRisk": r,
        "建議": action,
        "進場參考": entry,
        "停損": sl,
        "停利": tp,
        "Top5": pd.DataFrame(top5)
    }


st.title("台股全市場多空波段決策輔助（Web MVP）")
st.caption("日K主導，盤中以今日日K即時重算（MVP 先以模擬資料示範）")

col1, col2, col3 = st.columns(3)
with col1:
    n = st.slider("掃描檔數", 50, 500, 120, 10)
with col2:
    topn = st.slider("排行榜 TopN", 5, 30, 10, 1)
with col3:
    seed = st.number_input("隨機種子", 1, 9999, 42)

market = generate_market_snapshot(n=n, seed=int(seed))

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("強多 Top")
    strong_long = market[market["狀態"] == "強多"].sort_values(["TrendScore", "Confidence"], ascending=[False, False]).head(topn)
    st.dataframe(strong_long, use_container_width=True, hide_index=True)
with c2:
    st.subheader("強空 Top")
    strong_short = market[market["狀態"] == "強空"].sort_values(["TrendScore", "Confidence"], ascending=[True, False]).head(topn)
    st.dataframe(strong_short, use_container_width=True, hide_index=True)
with c3:
    st.subheader("盤整待突破 Top")
    range_top = market[market["狀態"] == "盤整"].sort_values(["Confidence"], ascending=[False]).head(topn)
    st.dataframe(range_top, use_container_width=True, hide_index=True)

st.divider()
st.subheader("單檔決策報告")
selected = st.selectbox("選擇股票", market["代碼"].tolist(), index=0)
row = market[market["代碼"] == selected].iloc[0]
report = build_single_report(row)

m1, m2, m3, m4 = st.columns(4)
m1.metric("狀態", report["狀態"])
m2.metric("TrendScore", report["TrendScore"])
m3.metric("Confidence", report["Confidence"])
m4.metric("ReversalRisk", report["ReversalRisk"])

st.write(f"**建議動作：** {report['建議']}")
st.write(f"**進場參考：** {report['進場參考']}")
st.write(f"**停損：** {report['停損']}")
st.write(f"**停利：** {report['停利']}")

st.write("**觸發策略 Top5（示意）**")
st.dataframe(report["Top5"], use_container_width=True, hide_index=True)

st.divider()
st.subheader("全市場查詢")
kw = st.text_input("輸入代碼或名稱")
if kw:
    q = market[market["代碼"].str.contains(kw) | market["名稱"].str.contains(kw, na=False)]
    st.dataframe(q, use_container_width=True, hide_index=True)
else:
    st.dataframe(market, use_container_width=True, hide_index=True)

st.caption("若要接真實行情：替換 generate_market_snapshot() 並將今日日K動態插入日線序列後重算指標。")
