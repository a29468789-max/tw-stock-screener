# 台股日K主導盤中決策輔助（Web 即時版）

## 啟動

```bash
cd stock_screener_web
pip install -r requirements.txt
streamlit run app.py
```

瀏覽器：`http://localhost:8501`

## 模式
- **真實台股即時**（預設）
  - 歷史日K：`twstock.Stock(...).fetch_from(...)`
  - 盤中即時：`twstock.realtime.get(...)`
  - 用即時 O/H/L/C/V 組今日日K，重算日線指標與分數
- **Mock示範**
  - 只用假資料示範畫面

## 功能
- 全市場排行榜：強多 / 強空 / 盤整待突破
- 單檔決策報告：狀態、分數、風險、信心、進出場參考、無效條件
- 全市場查詢：代碼/名稱

## 注意
- 免費資料源在盤後或尖峰時段可能出現缺漏。
- 真實模式建議掃描 50~120 檔並每 10~20 秒刷新一次。
- 目前為「決策輔助」：不自動下單。
