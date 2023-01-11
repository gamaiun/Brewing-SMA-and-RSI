#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
plt.style.use("seaborn")
import talib as ta
from talib import abstract
import seaborn as sns


class FinancialInstrument():
    def __init__(self):
        self.df = None
        
    class DATA: 
        def __init__(self, ticker, main= FinancialInstrument):
            self.main = main
            self.ticker = ticker
            self.start = datetime.datetime(2006,6,1) 
            self.end = datetime.datetime.today()
            self.data = None

        def get_data(self):
            ''' retrieves (from yahoo finance) and prepares the data
            '''
            raw = yf.download(self.ticker, self.start, self.end)
            raw.rename(columns = {"Close":"price", "Volume":"volume", "Open":"open", "High":"high", 
                                 "Adj Close":"close", "Low":"low"}, inplace = True)
            raw["returns"] = (raw.price - raw.price.shift(1)) / raw.price.shift(1)
            raw["log_returns"] = np.log(raw.price/raw.price.shift(1))
            self.data = raw
            self.main.df = raw
            return self.data
                 
        def plot_prices(self):
            ''' creates a price chart
            '''
            plt.figure(figsize=(16,6))
            self.data.price.plot(figsize = (17,8), color = "black", label = "Close price")
            plt.title("Price Chart: {}".format(self.ticker), fontsize = 15)
            plt.grid()
            plt.legend();
            
        def change_ticker(self, ticker = None):
            '''sets a new ticker
            '''
            if ticker is not None:
                self.ticker = ticker
                self.get_data()

############## Simple MA backtester : class SMA ##############################

    class SMA_Backtester():
        def __init__(self, main = FinancialInstrument):
            self.main = main
            self.df =  self.main.df
            self.results = None

        def set_parameters(self, SMA_S = None, SMA_L = None):
            ''' Updates SMA parameters and the prepared dataset.
            '''
            if SMA_S is not None:
                self.SMA_S = SMA_S
                self.df["SMA_S"] = self.df["price"].rolling(self.SMA_S).mean()
            if SMA_L is not None:
                self.SMA_L = SMA_L
                self.df["SMA_L"] = self.df["price"].rolling(self.SMA_L).mean()

        def test_strategy(self, plot = None):
            ''' Backtests the SMA-based trading strategy.
            '''
            self.data = self.df.copy().dropna()
            self.data["SMA_position"] = np.where(self.data["SMA_S"] > self.data["SMA_L"], 1, -1)
            self.data["SMA_strategy"] = self.data["SMA_position"].shift(1) * self.data["returns"]
            self.data.dropna(inplace=True)
            self.data["creturns"] = self.data["returns"].cumsum().apply(np.exp)
            self.data["SMA_cstrategy"] = self.data["SMA_strategy"].cumsum().apply(np.exp)
            self.results = self.data
            self.main.df = self.data
            
            perf = self.data["SMA_cstrategy"].iloc[-1] # absolute performance of the strategy
            outperf = perf - self.data["creturns"].iloc[-1] # out-/underperformance of strategy
            if self.results is None:
                print("Run test_strategy() first.")
            elif plot == True:
                title = " | SMA_S = {} | SMA_L = {}, PERF = {}, OUTPERF = {}".format( self.SMA_S, self.SMA_L, round(perf, 6), round(outperf, 6))
                self.results[["creturns", "SMA_cstrategy"]].plot(title=title, figsize=(17, 8), color = ["black", "blue"])
            else:
                return round(perf, 6), round(outperf, 6)

        def plot_results(self):
            ''' Plots the performance of the trading strategy and compares to "buy and hold".
            '''
            if self.results is None:
                print("Run test_strategy() first.")
            elif plot == True:
                title = " | SMA_S = {} | SMA_L = {}".format( self.SMA_S, self.SMA_L)
                self.results[["creturns", "SMA_cstrategy"]].plot(title=title, figsize=(17, 8), color = ["black","blue"])
            else:
                pass

        def plot_field(self):
            ''' Plots the heatmap of performance of the trading strategy".
            '''
            if self.results is None:
                print("Run optimize_parameters() first.")
            else:
                sns.set(rc={'figure.figsize':(22,22)})
                df_wide = self.results_overview.pivot_table( index='SMA_L', columns='SMA_S', values='SMA_performance')
                sns.heatmap(df_wide);


        def optimize_parameters(self, SMA_S_range, SMA_L_range):
            ''' Finds the optimal strategy (global maximum) given the SMA parameter ranges.

            Parameters
            ----------
            SMA_S_range, SMA_L_range: tuple
                tuples of the form (start, end, step size)
            '''
            combinations = list(product(range(*SMA_S_range), range(*SMA_L_range)))

            # test all combinations
            results = []
            for comb in combinations:
                self.set_parameters(comb[0], comb[1])
                results.append(self.test_strategy()[0])

            best_perf = np.max(results) # best performance
            opt = combinations[np.argmax(results)] # optimal parameters

            # run/set the optimal strategy
            self.set_parameters(opt[0], opt[1])
            self.test_strategy()

            # create a df with many results
            many_results =  pd.DataFrame(data = combinations, columns = ["SMA_S", "SMA_L"])
            many_results["SMA_performance"] = results
            self.results_overview = many_results

            return opt, best_perf

    class RSI_tester():

        ''' Class for the vectorized backtesting of SMA-based trading strategies.
        '''

        def __init__(self,   main = FinancialInstrument):
            self.main = main
            self.df =  self.main.df            
            self.rsi = None
            self.results = None 
            #self.df["RSI"] = ta.RSI(self.df["price"], self.rsi)


        def set_parameters(self, rsi = None):
            ''' Updates RSI parameters and the prepared dataset.
            '''
            if rsi is not None:
                self.rsi = rsi
                self.df["RSI"] = abstract.RSI(self.df, self.rsi) 

        def test_strategy(self, plot = None):
            ''' Backtests the RSI-based trading strategy.
            '''
            data = self.df.copy().dropna()
            data["RSI_position"] = np.where(data["RSI"] < 20, 1, np.nan)
            data["RSI_position"] = np.where(data["RSI"] > 90, -1, data["RSI_position"])
            data["RSI_position"] = data["RSI_position"].fillna(0)

            data["RSI_strategy"] = data["RSI_position"].shift(1) * data["returns"]
            data.dropna(inplace=True)
            data["creturns"] = data["returns"].cumsum().apply(np.exp)
            data["RSI_cstrategy"] = data["RSI_strategy"].cumsum().apply(np.exp)
            self.results = data
            self.main.df  = data
           # return self.main.df
            perf = data["RSI_cstrategy"].iloc[-1] # absolute performance of the strategy
            outperf = perf - data["creturns"].iloc[-1] # out-/underperformance of strategy
            if plot == True:
                title = "RSI {} | PERF:{} | OUTPERF:{}".format(self.rsi, round(perf, 6), round(outperf, 6))
                self.results[["creturns", "RSI_cstrategy"]].plot(title=title, figsize=(17, 8), color = ["black", "red"])
            else: 
                return round(perf, 6), round(outperf, 6)


        def optimize_parameters(self, a,b):
            ''' Finds the optimal strategy (global maximum) given the RSI parameter ranges.
            '''  
            combinations = list(range(a,b))

            # test all combinations
            results = []
            self.data = self.df.copy().dropna()
            
            
            for comb in range(a,b):
                self.set_parameters(comb)
                results.append(self.test_strategy()[0])


            best_perf = np.max(results) # best performance
            opt = combinations[np.argmax(results)] # optimal parameters

            # run/set the optimal strategy
            self.set_parameters(opt)
            self.test_strategy()

            # create a df with many results
            many_results =  pd.DataFrame(data = combinations, columns = ["RSI"])
            many_results["RSI_performance"] = results
            self.results_overview = many_results

            return round(opt,6), round(best_perf,6) 

    class SMA_RSI_tester():

            ''' Class for the vectorized backtesting of SMA-based trading strategies.
            '''

            def __init__(self,  main = FinancialInstrument):
                self.main = main
                self.data =  self.main.df 
                self.results = None 
                
            def test_strategy(self, plot = None):
                ''' Backtests the RSI-based trading strategy.
                '''
                data = self.data.copy().dropna()
                data["SMA_RSI_position"] = np.where((data["RSI_position"] == 1) & data["SMA_position"]==1, 1, np.nan)
                data["SMA_RSI_position"] = np.where((data["RSI_position"] == -1) & data["SMA_position"]==-1, -1, data["SMA_RSI_position"])
                data["SMA_RSI_position"] = data["SMA_RSI_position"].fillna(0)

                data["SMA_RSI_strategy"] = data["SMA_RSI_position"].shift(1) * data["returns"]
                data.dropna(inplace=True)
                data["creturns"] = data["returns"].cumsum().apply(np.exp)
                data["SMA_RSI_cstrategy"] = data["SMA_RSI_strategy"].cumsum().apply(np.exp)
                self.results = data
                self.main.df  = data
                
                perf = data["SMA_RSI_cstrategy"].iloc[-1] # absolute performance of the strategy
                outperf = perf - data["creturns"].iloc[-1] # out-/underperformance of strategy
                if plot == True:
                   # title = "perf {} | RSI{} | outperf = {}".format(round(perf, 6), self.rsi, round(outperf, 6))
                    self.results[["creturns", "SMA_RSI_cstrategy"]].plot(title="Placeholder",  figsize=(17, 8), color = ["black", "green"])

                return round(perf, 6), round(outperf, 6)

