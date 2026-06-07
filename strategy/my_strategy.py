import pandas as pd
import my_strategy as ms # 你原本那支複雜邏輯的檔案
import config

class MyStrategy:
    def __init__(self):
        self.name = "ZenTrend_休息中翻正"

    def generate_signals(self, df_single_stock):
        """
        輸入: 單一股票的 df，欄位: date, open, high, low, close, volume, 代號, 證券名稱
        輸出: 同一個 df，但多了 buy_signal, take_profit_price, stop_loss_price
        邏輯完全照你 backtest.py 搬過來，不改
        """
        df = df_single_stock.copy()
        if len(df) < 100:
            df['buy_signal'] = False
            df['take_profit_price'] = None
            df['stop_loss_price'] = None
            return df

        # 1. 照你的流程算指標
        df = ms.calc_indicators(df)
        df = ms.calc_zentrend_full(df)
        
        # 2. 第一層篩選
        c1 = df['EMA5_W'] > df['EMA21_W']
        c2 = df['EMA21_W'] > df['EMA89_W']
        c3 = df['VOL_MA5'] > 10000
        c4 = df['RET_8W'] > 0.05
        c5 = df['close'] > 20
        c6 = df['close'] > df['EMA34']
        c7 = df['EMA34'] > df['EMA34'].shift(1)
        c8 = df['close'].shift(1) >= df['HIGH_3W_昨天']
        df['第1層通過'] = c1 & c2 & c3 & c4 & c5 & c6 & c7 & c8

        # 3. 分類狀態
        df = ms.classify_stock_state(df, 0.12, 2.0, 5)

        # 4. 照你的進場邏輯：昨天休息中 + ZenDir 翻正
        cond_昨天休息中 = df['狀態'].shift(1) == '休息中'
        cond_Zen翻正 = (df['ZenDir'] == 1) & (df['ZenDir'].shift(1) == -1)
        cond_第1層 = df['第1層通過']
        
        df['buy_signal'] = cond_昨天休息中 & cond_Zen翻正 & cond_第1層

        # 5. 找前高前低，算止盈止損價
        df['take_profit_price'] = None
        df['stop_loss_price'] = None
        
        buy_dates = df[df['buy_signal'] == True].index
        for buy_date in buy_dates:
            # 找過去50天算樞紐點
            past_data = df.loc[:buy_date].tail(50)
            if len(past_data) < 20: continue
            
            pivots = ms.find_pivots(past_data['high'].values, past_data['low'].values, 5)
            highs = [p['price'] for p in pivots if p['type'] == 'high']
            lows = [p['price'] for p in pivots if p['type'] == 'low']
            
            if len(highs) > 0 and len(lows) > 0:
                前高 = highs[-1]
                前低 = lows[-1]
                # 照你的邏輯：Fib 1.236 停利，跌破前低停損
                target = 前低 + (前高 - 前低) * 1.236
                df.loc[buy_date, 'take_profit_price'] = target
                df.loc[buy_date, 'stop_loss_price'] = 前低
            else:
                # 找不到樞紐點就用 config 的固定 %
                df.loc[buy_date, 'take_profit_price'] = df.loc[buy_date, 'close'] * (1 + config.TAKE_PROFIT_PCT)
                df.loc[buy_date, 'stop_loss_price'] = df.loc[buy_date, 'close'] * (1 - config.STOP_LOSS_PCT)

        return df
