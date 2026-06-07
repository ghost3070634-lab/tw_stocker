import pandas as pd
import numpy as np

# ========= ZenTrend 原版函數 ========= #
def calc_zentrend_full(df):
    """
    輸入 df 欄位: date, open, high, low, close, volume
    """
    g = df.copy()

    # Heikin-Ashi 計算，欄位名全改英文
    ha_close = (g['open'] + g['high'] + g['low'] + g['close']) / 4
    ha_open = (g['open'] + g['close']) / 2
    for i in range(1, len(g)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2

    ha_high = pd.concat([g['high'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([g['low'], ha_open, ha_close], axis=1).min(axis=1)

    sPctH, sPctL = 0.0, 0.0
    sExH, sExL = 0.0, 0.0
    sShH, sShL = 0.0, 0.0
    dir_val = 1
    pU, pL = 0.0, 0.0

    res_dir, res_upper, res_lower = [], [], []

    for i in range(len(g)):
        _o, _h, _l, _c = ha_open.iloc[i], ha_high.iloc[i], ha_low.iloc[i], ha_close.iloc[i]
        count = i + 1

        if _o > 0:
            sPctH += (_h - _o) / _o * 100
            sPctL += (_o - _l) / _o * 100

        pHigh = _o * (1 + sPctH / count / 100) if _o > 0 else 0
        pLow = _o * (1 - sPctL / count / 100) if _o > 0 else 0

        exH = (_h - pHigh) / pHigh * 100 if _h > pHigh else 0
        shH = (pHigh - _h) / pHigh * 100 if _h < pHigh else 0
        exL = (pLow - _l) / pLow * 100 if _l < pLow and pLow > 0 else 0
        shL = (_l - pLow) / pLow * 100 if _l > pLow and pLow > 0 else 0

        sExH += exH; sShH += shH; sExL += exL; sShL += shL

        adjH_Up = pHigh * (1 + sExH / count / 100)
        adjH_Dn = pHigh * (1 - sShH / count / 100)
        adjL_Dn = pLow * (1 - sExL / count / 100)
        adjL_Up = pLow * (1 + sShL / count / 100)

        upper = (pHigh + adjH_Up + adjH_Dn) / 3
        lower = (pLow + adjL_Dn + adjL_Up) / 3

        if i > 0:
            if dir_val >= 0 and _c > pU:
                dir_val = -1
            elif dir_val < 0 and _c < pL:
                dir_val = 1

        pU, pL = upper, lower
        res_dir.append(dir_val)
        res_upper.append(upper)
        res_lower.append(lower)

    g['ZenDir'] = res_dir
    g['ZenUpper'] = res_upper
    g['ZenLower'] = res_lower
    return g

# ========= 指標計算函數 ========= #
def calc_indicators(df):
    # 確保 index 是 datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # 日線 EMA
    df['EMA34'] = df['close'].ewm(span=34, adjust=False).mean()
    
    # 34日標準差，classify_stock_state 會用到
    df['STD34'] = df['close'].rolling(34).std()
    
    # 3週高低點
    df['HIGH_3W'] = df['high'].rolling(15).max() # 3週=15個交易日
    df['LOW_3W'] = df['low'].rolling(15).min()
    df['HIGH_3W_昨天'] = df['HIGH_3W'].shift(1)
    
    # 週線資料：先把日線轉週線
    df_w = df['close'].resample('W-FRI').last().to_frame('close')
    df_w['EMA5_W'] = df_w['close'].ewm(span=5, adjust=False).mean()
    df_w['EMA21_W'] = df_w['close'].ewm(span=21, adjust=False).mean()
    df_w['EMA89_W'] = df_w['close'].ewm(span=89, adjust=False).mean()
    
    # 8週漲幅，指定 fill_method=None 避免 warning
    df_w['RET_8W'] = df_w['close'].pct_change(8, fill_method=None)
    
    # 週線 merge 回日線：用 reindex + ffill 對齊
    df = df.join(df_w[['EMA5_W','EMA21_W','EMA89_W','RET_8W']])
    df[['EMA5_W','EMA21_W','EMA89_W','RET_8W']] = df[['EMA5_W','EMA21_W','EMA89_W','RET_8W']].ffill()
    
    # 成交量 5日均
    df['VOL_MA5'] = df['volume'].rolling(5).mean()
    
    return df

# ========= 狀態分類函數 ========= #
def classify_stock_state(df, ob_ratio=0.12, std_times=2.0, n_days=5):
    """輸入的 df 已經算好 EMA34 等指標"""
    g = df.copy()

    if len(g) < 35:
        g['狀態'] = '資料不足'
        return g

    ob = g['close'] / g['EMA34'] - 1
    cond_行進中 = (ob > ob_ratio) | (g['close'] > g['EMA34'] + std_times * g['STD34'])

    low_n = g['close'].rolling(n_days).min().shift(1)
    ema_n = g['EMA34'].rolling(n_days).min().shift(1)

    cond_在高檔 = g['close'] > g['EMA34']
    cond_未破線 = g['close'] > g['EMA34'] * 0.95
    cond_曾拉回 = low_n < ema_n
    cond_休息中 = cond_在高檔 & cond_未破線 & cond_曾拉回

    cond_刪除 = g['EMA34'] < g['EMA34'].shift(5)
    cond_創高 = g['close'] >= g['close'].shift(1).rolling(15).max()

    g['狀態'] = '觀察'
    g.loc[cond_創高, '狀態'] = '創3週新高'
    g.loc[cond_刪除, '狀態'] = '刪除'
    g.loc[cond_休息中, '狀態'] = '休息中'
    g.loc[cond_行進中, '狀態'] = '行進中'

    return g
def find_pivots(high, low, depth):
    pivots_list = []
    for i in range(depth, len(high) - depth):
        if high[i] == high[i-depth:i+depth+1].max():
            pivots_list.append({'index': i, 'price': high[i], 'type': 'high'})
        if low[i] == low[i-depth:i+depth+1].min():
            pivots_list.append({'index': i, 'price': low[i], 'type': 'low'})
    return sorted(pivots_list, key=lambda x: x['index'])

def calc_indicators(df):
    # 確保 index 是 datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # 日線 EMA
    df['EMA34'] = df['close'].ewm(span=34, adjust=False).mean()
    
    # 3週高低點
    df['HIGH_3W'] = df['high'].rolling(15).max()  # 3週=15個交易日
    df['LOW_3W'] = df['low'].rolling(15).min()
    df['HIGH_3W_昨天'] = df['HIGH_3W'].shift(1)
    
    # 週線資料：先把日線轉週線
    df_w = df['close'].resample('W-FRI').last().to_frame('close')
    df_w['EMA5_W'] = df_w['close'].ewm(span=5, adjust=False).mean()
    df_w['EMA21_W'] = df_w['close'].ewm(span=21, adjust=False).mean()
    df_w['EMA89_W'] = df_w['close'].ewm(span=89, adjust=False).mean()
    
    # 8週漲幅
    df_w['RET_8W'] = df_w['close'].pct_change(8)
    
    # 週線 merge 回日線：用 reindex + ffill 對齊
    df = df.join(df_w[['EMA5_W','EMA21_W','EMA89_W','RET_8W']])
    df[['EMA5_W','EMA21_W','EMA89_W','RET_8W']] = df[['EMA5_W','EMA21_W','EMA89_W','RET_8W']].ffill()
    
    # 成交量 5日均
    df['VOL_MA5'] = df['volume'].rolling(5).mean()
    
    return df
