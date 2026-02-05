#1. Win % - 74% on 1:1 Risk/Reward
#2. I have to work on it

import os
import time
import math
import joblib
import logging
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

import tensorflow as tf
from keras.layers import Bidirectional
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping

############################ Global variables ############################################
SYMBOL           = "XAUUSDm"           # change if your broker uses different symbol
TIMEFRAME        = mt5.TIMEFRAME_M5
LOOKBACK         = 60                  #60*5 = 300 minutes (5 hours) using this make the prediction
PREDICT_HORIZON  = 1                   #next 5 minute direction prediction
BATCH_SIZE       = 64                  #number of samples processed before updating weights.
EPOCHS           = 30                  #30 means the model will see the training data 30 times
TEST_SIZE        = 0.2                 #20% data for testing
RANDOM_STATE     = 42
MODEL_DIR        = "models"
os.makedirs(MODEL_DIR, exist_ok=True)  #create model directory if not exists

########################## Risk Factors ##################################################
LOT            = 0.01
SL_PIPS        = 100      # stop loss in pips (for XAUUSD 1 pip typically = 0.01 on many brokers — verify)
TP_PIPS        = 100
MAX_POS        = 1        # max concurrent positions
RISK_PER_TRADE = 0.01     # fraction of account balance — (used for position sizing if implemented)

############################ Account Information #########################################
MT5_LOGIN    = 274242894
MT5_PASSWORD = "Mritunjay@76519"
MT5_SERVER   = "Exness-MT5Trial6"


############################# Print settings ##############################################
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

############################ MT5 Connector (OOP) ##########################################
class MT5Connector:
    def __init__(self, login=None, password=None, server=None):
        self.login    = login
        self.password = password
        self.server   = server
        self._init_mt5()

    def _init_mt5(self):
        if mt5.initialize():
            logging.info("MT5 initialized (using terminal credentials).")
        else:
            logging.info("MT5 initialize failed; trying login if details provided.")
            if self.login and self.password and self.server:
                ok = mt5.initialize(login=self.login, password=self.password, server=self.server)
                if not ok:
                    raise RuntimeError("MT5 initialize/login failed.")
            else:
                raise RuntimeError("MT5 initialize failed and no login provided.")

    def shutdown(self):
        mt5.shutdown()

    def fetch_ohlc(self, symbol, timeframe, n_bars=2000):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
        if rates is None:
            raise RuntimeError(f"Failed to fetch rates for {symbol}")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def account_info(self):
        return mt5.account_info()

    def send_order(self, symbol, volume, order_type, price=None, sl=None, tp=None, deviation=20):
        if order_type == mt5.ORDER_TYPE_BUY:
            price = price or mt5.symbol_info_tick(symbol).ask
        else:
            price = price or mt5.symbol_info_tick(symbol).bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": int(order_type),
            "price": float(price),
            "deviation": deviation,
            "magic": 123456,
            "comment": "RNN_auto",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }
        if sl: request["sl"] = float(sl)
        if tp: request["tp"] = float(tp)

        result = mt5.order_send(request)
        return result

    def close_position(self, ticket):
        return mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL,
            "position": int(ticket),
            "type": mt5.ORDER_TYPE_SELL if mt5.positions_get(ticket=ticket)[0].type == 0 else mt5.ORDER_TYPE_BUY,
            "volume": mt5.positions_get(ticket=ticket)[0].volume,
            "symbol": mt5.positions_get(ticket=ticket)[0].symbol,
            "price": mt5.symbol_info_tick(mt5.positions_get(ticket=ticket)[0].symbol).bid
        })

############################### Feature Engineering(5m candle) ##########################################
class FeatureEngineer:
    def __init__(self):
        pass

    def add_indicators(self, df):
        df = df.copy()
        df['return'] = df['close'].pct_change().fillna(0)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['std_10'] = df['close'].rolling(10).std()
        rolling20 = df['close'].rolling(20)
        df['bb_mid'] = rolling20.mean()
        df['bb_std'] = rolling20.std()
        df['bb_up'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_down'] = df['bb_mid'] - 2 * df['bb_std']
        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.ewm(span=14).mean()
        roll_down = down.ewm(span=14).mean()
        rs = roll_up / (roll_down + 1e-9)
        df['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))
        # Fill na
        df.bfill(inplace=True)
        df.ffill(inplace=True)
        return df

################################## Most Important in Neural Network ##########################################
    def create_sequences(self, df, feature_cols, lookback=60, horizon=1): #horizon → how far ahead you want to predict and lookback 60 candles
        data = df[feature_cols].values
        closes = df['close'].values
        X, y = [], []
        for i in range(lookback, len(df) - horizon + 1):
            seq          = data[i - lookback:i]
            future_close = closes[i + horizon - 1]
            cur_close    = closes[i - 1]
            label = 1 if future_close > cur_close else 0
            X.append(seq)   #last 60 candles X shape: (n_samples, lookback, n_features)
            y.append(label) #1->up 0->down   y shape: (n_samples,) → labels (0 or 1).
        
        X = np.array(X)
        y = np.array(y)
        print(f"Sequence shapes -> X: {X.shape}, y: {y.shape}")
        ##############show some samples
        for i in range(min(3, len(X))):  # show first 3 sequences
            print(f"\nX[{i}] ->\n{X[i]}")
            print(f"y[{i}] -> {y[i]}")

        return X, y

####################################### Models(LSTM/GRU/RNN) ##########################################
@dataclass
class ModelConfig:
    lookback:   int            # number of past candles (sequence length)
    n_features: int            # number of input features (open, high, low, close, volume, etc.)
    units:      int   = 64     # number of neurons in hidden layers (default 64)
    dropout:    float = 0.2    # dropout rate to prevent overfitting (default 20%)
    lr:         float = 1e-3   # learning rate for optimizer (default 0.001)


class RnnClassifier:
    def __init__(self, cfg: ModelConfig, rnn_type='lstm', bidirectional=True, stacked=True):
        self.cfg           = cfg
        self.rnn_type      = rnn_type.lower()
        self.bidirectional = bidirectional
        self.stacked       = stacked
        self.model         = self._build()

    def _build(self):
        model = Sequential()
        units = self.cfg.units

        ######################## First RNN layer
        rnn_layer = None
        if self.rnn_type == 'lstm':
            rnn_layer = LSTM(units, return_sequences=self.stacked, input_shape=(self.cfg.lookback, self.cfg.n_features))
        elif self.rnn_type == 'gru':
            rnn_layer = GRU(units, return_sequences=self.stacked, input_shape=(self.cfg.lookback, self.cfg.n_features))
        elif self.rnn_type == 'rnn':
            rnn_layer = SimpleRNN(units, return_sequences=self.stacked, input_shape=(self.cfg.lookback, self.cfg.n_features))
        else:
            raise ValueError("rnn_type must be one of 'lstm','gru','rnn'")

        if self.bidirectional:
            model.add(Bidirectional(rnn_layer))
        else:
            model.add(rnn_layer)

        model.add(BatchNormalization())
        if self.cfg.dropout > 0:
            model.add(Dropout(self.cfg.dropout))

        ####################### Second stacked layer if stacked=True
        if self.stacked:
            rnn_layer2 = None
            if self.rnn_type == 'lstm':
                rnn_layer2 = LSTM(units, return_sequences=False)
            elif self.rnn_type == 'gru':
                rnn_layer2 = GRU(units, return_sequences=False)
            elif self.rnn_type == 'rnn':
                rnn_layer2 = SimpleRNN(units, return_sequences=False)
            
            if self.bidirectional:
                model.add(Bidirectional(rnn_layer2))
            else:
                model.add(rnn_layer2)

            model.add(BatchNormalization())
            if self.cfg.dropout > 0:
                model.add(Dropout(self.cfg.dropout))

        ################ Dense layers(Modify as needed)
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(self.cfg.dropout))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))  # binary classification

        opt = tf.keras.optimizers.Adam(self.cfg.lr)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def summary(self):
        self.model.summary()

    def train(self, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE, model_path=None):
        callbacks = []
        callbacks.append(EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True))
        if model_path:
            callbacks.append(ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss'))
        hist = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=2)
        return hist

    def predict_proba(self, X):
        return self.model.predict(X).ravel()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path):
        self.model.save(path)

    @classmethod
    def load(cls, path, cfg, rnn_type='lstm'):
        obj = cls(cfg, rnn_type)
        obj.model = tf.keras.models.load_model(path)
        return obj

################################ Live Trading ##########################################
class LiveTrader:
    def __init__(self, mt5conn: MT5Connector, model: RnnClassifier, scaler, pca, feature_cols, lookback=LOOKBACK):
        self.mt5          = mt5conn
        self.model        = model
        self.scaler       = scaler
        self.pca          = pca
        self.feature_cols = feature_cols
        self.lookback     = lookback

    def get_latest_df(self, bars=500):
        df = self.mt5.fetch_ohlc(SYMBOL, TIMEFRAME, bars)
        fe = FeatureEngineer()
        df = fe.add_indicators(df)
        return df

    def run_once(self):
        df = self.get_latest_df(bars=self.lookback + 10)
        fe = FeatureEngineer()
        df = fe.add_indicators(df)
        ################################################ Build a single sequence
        seq_df = df.tail(self.lookback)
        X = seq_df[self.feature_cols].values.astype(np.float32)
        ########################################### Optional scaling per feature (if scaler is fitted)
        if self.scaler and hasattr(self, 'pca'):
            # Flatten -> scale -> PCA -> reshape
            X_flat = X.reshape(-1, X.shape[1])
            X_scaled = self.scaler.transform(X_flat)
            X_pca = self.pca.transform(X_scaled)
            X = X_pca.reshape(1, self.lookback, X_pca.shape[1])
        elif self.scaler:
            X_flat = X.reshape(-1, X.shape[1])
            X_scaled = self.scaler.transform(X_flat)
            X = X_scaled.reshape(1, self.lookback, X.shape[1])
        else:
            X = np.expand_dims(X, 0)

        print(f"Live sequence shape: {X.shape}")  # Debug live input
        ############################### Predict ##########################################
        # No global transform to avoid mismatch — assume model uses raw features or the scaler is simple StandardScaler used on training flattened features
        proba = self.model.predict_proba(X)[0]
        signal = 1 if proba >= 0.5 else 0
        logging.info(f"Pred prob {proba:.4f} -> signal {signal}")



        ########################################### Check current positions
        positions = mt5.positions_get(symbol=SYMBOL)
        if positions and len(positions) >= MAX_POS:
            logging.info("Max positions open. Skipping new entry.")
            return
        # Place order with SL/TP
        tick = mt5.symbol_info_tick(SYMBOL)
        if signal == 1:
            # buy
            price = tick.ask
            sl = price - SL_PIPS * 0.01
            tp = price + TP_PIPS * 0.01
            res = self.mt5.send_order(SYMBOL, LOT, mt5.ORDER_TYPE_BUY, price=price, sl=sl, tp=tp)
            logging.info(f"BUY order result: {res}")
        else:
            price = tick.bid
            sl = price + SL_PIPS * 0.01
            tp = price - TP_PIPS * 0.01
            res = self.mt5.send_order(SYMBOL, LOT, mt5.ORDER_TYPE_SELL, price=price, sl=sl, tp=tp)
            logging.info(f"SELL order result: {res}")

    def run_loop(self, interval_seconds=30):
        try:
            while True:
                self.run_once()
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logging.info("Stopping live trader.")


############################################## Backtester ##########################################
class SimpleBacktester:
    def __init__(self, df, feature_cols, model: RnnClassifier, scaler=None, lookback=LOOKBACK, horizon=PREDICT_HORIZON):
        self.df           = df.reset_index(drop=True).copy()
        self.feature_cols = feature_cols
        self.model        = model
        self.scaler       = scaler
        self.lookback     = lookback
        self.horizon      = horizon

    def run(self, X, y, threshold=0.5, save_path=None, plot=True):
        preds_proba = self.model.predict_proba(X)
        preds       = (preds_proba >= threshold).astype(int)

        #########Map each sample index back to dataframe index:
        start_idx   = self.lookback
        trades      = []

        for i, pred in enumerate(preds):
            idx         = start_idx + i  # this corresponds to the "current" bar used to generate signal
            entry_price = self.df.loc[idx, 'close']
            pip_value   = 0.01  
            sl          = SL_PIPS * pip_value
            tp          = TP_PIPS * pip_value
            if pred == 1:  # long
                exit_price = entry_price + tp
                stop_price = entry_price - sl
            else:  # short
                exit_price = entry_price - tp
                stop_price = entry_price + sl

            # Simulate next horizon candles to see whether SL or TP hit first
            window    = self.df.loc[idx+1: idx + self.horizon + 60]  # look ahead window, safety margin
            hit       = None
            hit_price = None
            for _, r in window.iterrows():
                high = r['high']
                low  = r['low']
                if pred == 1:
                    if low <= stop_price:
                        hit = 'SL'
                        hit_price = stop_price
                        break
                    if high >= exit_price:
                        hit = 'TP'
                        hit_price = exit_price
                        break
                else:
                    if high >= stop_price:
                        hit = 'SL'
                        hit_price = stop_price
                        break
                    if low <= exit_price:
                        hit = 'TP'
                        hit_price = exit_price
                        break
            # If neither hit in window, mark as 'NoHit' and use close after horizon
            if hit is None:
                final_idx = min(idx + self.horizon, len(self.df)-1)
                hit       = 'Close'
                hit_price = self.df.loc[final_idx, 'close']

            pnl = (hit_price - entry_price) if pred == 1 else (entry_price - hit_price)
            time_col = 'time'
            if 'time' not in self.df.columns:
                if 'datetime' in self.df.columns:
                    time_col = 'datetime'
                else:
                    # fallback to index
                    self.df = self.df.reset_index()
                    time_col = 'index'

            trades.append({
                'idx': idx,
                'time': self.df.loc[idx, time_col],  # safe reference
                'pred': pred,
                'entry': entry_price,
                'exit': hit_price,
                'type': hit,
                'pnl': pnl
            })

        trades_df  = pd.DataFrame(trades)
        total_pips = trades_df['pnl'].sum() / 0.01  # convert to pip units (if pip_value=0.01)
        wins       = trades_df[trades_df['pnl'] > 0].shape[0]
        losses     = trades_df[trades_df['pnl'] <= 0].shape[0]
        ret = {
            'trades':     trades_df,
            'total_pips': total_pips,
            'wins':       wins,
            'losses':     losses,
            'win_rate':   wins / max(1, wins + losses)
        }

        # Save trades to CSV if path provided
        if save_path:
            trades_df.to_csv(save_path, index=False)
            print(f"Trades saved to {save_path}")

        # Plot equity curve if enabled
        if plot:
            import matplotlib.pyplot as plt
            trades_df['cum_pnl'] = trades_df['pnl'].cumsum()
            plt.figure(figsize=(12,6))
            plt.plot(trades_df['time'], trades_df['cum_pnl'], label="Equity Curve")
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel("Time")
            plt.ylabel("Cumulative PnL")
            plt.title("Backtest Equity Curve")
            plt.legend()
            plt.grid(True)
            plt.show()

        return ret


################################### Main Training and Backtest ##########################################
###################### Main Training and Backtest (Fixed) ##############################
def main_train_and_backtest(csv_path="C:/Users/Mritunjay Maddhesiya/OneDrive/Desktop/MT5/Data/XAUUSDm_5m.csv"):

    ####################load data from CSV instead of MT5 for training
    logging.info("Loading data from CSV...")
    df = pd.read_csv(csv_path, parse_dates=['time'])
    logging.info(f"Data loaded: {df.shape[0]} rows.")


    # mt5c = MT5Connector(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    # logging.info("Fetching data...")
    # df = mt5c.fetch_ohlc(SYMBOL, TIMEFRAME, n_bars=50000)
    
    fe = FeatureEngineer()
    df = fe.add_indicators(df)

    # Features for model
    feature_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'log_return', 'sma_5', 'sma_20', 'ema_10', 'std_10', 'bb_mid', 'bb_std', 'bb_up', 'bb_down', 'rsi_14']
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    ################################## Build sequences##########################################
    X, y = fe.create_sequences(df, feature_cols, lookback=LOOKBACK, horizon=PREDICT_HORIZON)
    print(f"Training sequences: X {X.shape}, y {y.shape}")

    n_samples, lb, n_features = X.shape

    ################################# Scale features ##########################################
    X_flat = X.reshape(-1, n_features)
    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)

    ################################# PCA ######################################################
    pca_components = 0.95
    pca = PCA(n_components=pca_components)
    X_flat_scaled = pca.fit_transform(X_flat_scaled)
    n_features = X_flat_scaled.shape[1]
    X_scaled = X_flat_scaled.reshape(n_samples, lb, n_features)
    logging.info(f"PCA applied: reduced to {n_features} features.")

    ################################# Train/Test split #########################################
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=TEST_SIZE, shuffle=False)
    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    ################################# Build model ##############################################
    cfg = ModelConfig(lookback=LOOKBACK, n_features=n_features, units=128, dropout=0.2, lr=1e-3)
    rnn_type = 'lstm'
    model_path = os.path.join(MODEL_DIR, f"{SYMBOL}_{rnn_type}.keras")  ########## Model save here (I have to train)
    model = RnnClassifier(cfg, rnn_type=rnn_type, bidirectional=True, stacked=True)
    model.summary()

    ################################# Train ####################################################
    logging.info("Training model...")
    model.train(X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE, model_path=model_path)

    # Save scaler and PCA
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_{SYMBOL}.pkl"))
    joblib.dump(pca, os.path.join(MODEL_DIR, f"pca_{SYMBOL}.pkl"))
    logging.info(f"Model, scaler, and PCA saved to {MODEL_DIR}")

    ##################################### Evaluate ################################################
    y_pred = model.predict(X_test)  # Converts probs -> 0/1
    logging.info("Classification report (test):\n" + classification_report(y_test, y_pred))
    logging.info("Accuracy: %.4f" % accuracy_score(y_test, y_pred))
    
    #mt5c.shutdown()
    return model, scaler, pca, feature_cols, df



###################### Run the Main Function ###############################
if __name__ == "__main__":

    ################ Code for Backtesting##########################
    model, scaler, pca, feature_cols, df = main_train_and_backtest("C:/Users/Mritunjay Maddhesiya/OneDrive/Desktop/MT5/Data/XAUUSDm_5m.csv")
    backtest = SimpleBacktester(df, feature_cols, model, scaler)
    X, y     = FeatureEngineer().create_sequences(df, feature_cols, lookback=LOOKBACK, horizon=PREDICT_HORIZON)

    ####### Apply same scaler+pca
    n_samples, lb, n_features = X.shape
    X_flat   = X.reshape(-1, n_features)
    X_scaled = scaler.transform(X_flat)
    X_pca    = pca.transform(X_scaled)
    X_final  = X_pca.reshape(n_samples, lb, X_pca.shape[1])

    results = backtest.run(X_final, y)
    print("Backtest Results:")
    print(results['trades'].head())
    print(f"Total Pips: {results['total_pips']}, Win Rate: {results['win_rate']:.2%}")

    ####################### Code for live trading ##########################
    # model, scaler, pca, feature_cols, df = main_train_and_backtest()
    # mt5c     = MT5Connector(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    # trader   = LiveTrader(mt5c, model, scaler, pca, feature_cols)    
    # trader.run_loop(interval_seconds=30)



















