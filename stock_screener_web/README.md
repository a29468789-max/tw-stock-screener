# 台股日K主導盤中決策輔助（Web MVP）

## 啟動

```bash
cd stock_screener_web
pip install -r requirements.txt
streamlit run app.py
```

瀏覽器開啟：`http://localhost:8501`

## 功能
- 全市場排行榜（強多 / 強空 / 盤整待突破）
- 單檔決策報告（分數、風險、信心、進出場參考）
- 盤中更新模擬（每次重新整理會重算）

## 備註
目前是可跑 MVP（mock 資料）。
若要接真實台股行情，請把 `generate_market_snapshot()` 換成你的行情來源。
