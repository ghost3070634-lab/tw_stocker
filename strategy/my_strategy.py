import pandas as pd
import numpy as np
from utils.indicators import calc_zentrend_full, calc_indicators, classify_stock_state


class MyStrategy:
    def __init__(self):
        self.name = "ZenTrend_Rest_Strategy"

    def find_pivots(self, high, low, depth):
        """你原本的 find_pivots，邏輯不動"""
        pivots_list = []
        for i in range(depth, len(high) - depth):
            if high[i] == high[i-depth:i+depth+1].max():
                pivots_list.append({'index': i, 'price': high[i], 'type': 'high'})
            if low[i] == low[i-depth:i+depth+1].min():
                pivots_list.append({'index': i, 'price': low[i], 'type': 'low'})
        return sorted(pivots_list, key=lambda x: x['index'])

    def generate_signals(self, df):
        """
        輸入: 單一股票的 df，欄位: date, open, high, low, close, volume
        輸出: df 多了 buy_signal, take_profit_price, stop_loss_price, entry_price
        邏輯: 完全照你的 run_backtest 進場條件
        """
        if len(df) < 100:
            df['buy_signal'] = False
            df['take_profit_price'] = None
            df['stop_loss_price'] = None
            df['entry_price'] = None
            return df

        # 1. 算指標，你原本的流程
        df = calc_zentrend_full(df)
        df = calc_indicators(df)
        df = classify_stock_state(df, ob_ratio=0.12, std_times=2.0, n_days=5)

        # 2. 你的第1層篩選條件
        VOL_5D = 10000
        c1 = df['EMA5_W'] > df['EMA21_W']
        c2 = df['EMA21_W'] > df['EMA89_W']
        c3 = df['VOL_MA5'] > VOL_5D
        c4 = df['RET_8W'] > 0.05
        c5 = df['close'] > 20
        c6 = df['close'] > df['EMA34']
        c7 = df['EMA34'] > df['EMA34'].shift(1)
        c8 = df['close'].shift(1) >= df['HIGH_3W_昨天']
        df['第1層通過'] = c1 & c2 & c3 & c4 & c5 & c6 & c7 & c8

        # 3. 你的進場邏輯：休息中 + ZenDir 翻正
        # 昨天是休息中 + ZenDir<0，今天 ZenDir>=0
        cond_昨天休息中 = df['狀態'].shift(1) == '休息中'
        cond_昨天ZenDir負 = df['ZenDir'].shift(1) < 0
        cond_今天ZenDir正 = df['ZenDir'] >= 0
        cond_第1層過 = df['第1層通過'] == True

        df['buy_signal'] = cond_昨天休息中 & cond_昨天ZenDir負 & cond_今天ZenDir正 & cond_第1層過

        # 4. 你的出場價計算：用前高前低 + FIB_TARGET
        df['take_profit_price'] = None
        df['stop_loss_price'] = None
        df['entry_price'] = df['close'] # 進場價用收盤價

        # 只對有買進訊號的日期算止盈止損
        for idx in df[df['buy_signal'] == True].index:
            past_data = df.loc[:idx].tail(50) # 取前50天算樞紐
            if len(past_data) < 20:
                continue

            pivots = self.find_pivots(past_data['high'].values, past_data['low'].values, 5)
            if len(pivots) < 2:
                continue

            highs = [p['price'] for p in pivots if p['type'] == 'high']
            lows = [p['price'] for p in pivots if p['type'] == 'low']
            if not highs or not lows:
                continue

            前高 = highs[-1]
            前低 = lows[-1]
            FIB_TARGET = 1.236

            df.loc[idx, 'take_profit_price'] = 前低 + (前高 - 前低) * FIB_TARGET
            df.loc[idx, 'stop_loss_price'] = 前低

        return df
