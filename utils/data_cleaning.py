#!/usr/bin/env python
from abc import ABC, abstractmethod, ABCMeta
import pickle
import pandas as pd
import numpy as np
from urllib.request import urlopen
import urllib.request
import datetime
from getting_market_data import AbstractGetPreprocMarketData
from inherited_classes.inh_getting_market_data import MoscowExchangeData

def main(year = '2021'):
    file_name = rf'files\{year}.csv'
    df = data_prerapation(file_name)
    df_ticker = tickers(df)
    nes_tickers = list(creating_result_df(df_ticker, df).ticker.unique())
    lst_market_prices = list(creating_dict_market_prices(nes_tickers, year).keys())
    file1 = creating_dict_df(creating_result_df(df_ticker, df))
    file2 = creating_dict_market_prices(nes_tickers, year)
    file3 = creating_result_df(df_ticker, df)
    file3.to_csv(rf'files\df_{year}.csv', index=False)
    with open(rf'files\dict_{year}.pickle', 'wb') as f:
        pickle.dump(file1, f)
    with open(rf'files\dfss_{year}.pickle', 'wb') as f:
        pickle.dump(file2, f)
    with open(rf'files\df_{year}.pickle', 'wb') as f:
        pickle.dump(file3, f)

def data_prerapation(file_name, columns_=['datetime', 'ticker', 'deals', 'price', 'user']): # Считывание файла и создание датафрейма
    df = pd.read_csv(file_name, index_col=0)
    df.columns = columns_
    df.datetime = pd.to_datetime(df.datetime)
    return df

def tickers(df): # Обработка тикеров
    tickers = df.ticker
    tickers_res = [str.strip(ticker) for ticker in tickers]
    df.ticker = tickers_res
    return df.ticker

def creating_result_df(col, df_in): # Удаление ненужные тикеры
    tickers_to_del = []
    for ticker in col:
        for sign in ticker:
            if str.isdigit(sign) or sign == '-' or sign == '_':
                tickers_to_del.append(ticker)
                continue
    tickers_to_save = set(col) - set(tickers_to_del)
    df = df_in.query('ticker in @tickers_to_save')
    return (df)

def creating_dict_market_prices(tickers, year): #Создание словаря с рыночными ценами

    dict_urls = {'foreign_shares': 'https://iss.moex.com/iss/history/engines/stock/markets/foreignshares/securities/',
                'russian_shares': 'http://iss.moex.com/iss/history/engines/stock/markets/shares/boards/tqbr/securities/'}
    start_date = rf'{year}-09-15'
    finish_date = rf'{year}-12-25'
    outer_data = MoscowExchangeData(dict_urls, tickers, start_date, finish_date)
    dict_market_prices_1 = outer_data.main()
    return(dict_market_prices_1)

def creating_dict_df(df): #Создание словаря датафрейма
    tickers = df.ticker.unique()
    dict_df = dict()
    for ticker in tickers:
        dict_df[ticker] = df[df.ticker == ticker]
    return(dict_df)


if __name__ == "__main__":
    main()
