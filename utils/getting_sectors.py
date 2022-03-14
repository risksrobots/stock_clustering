# !git clone https://github.com/lit26/finvizfinance.git
# !pip install finvizfinance

from finvizfinance.quote import finvizfinance
from tqdm import tqdm
import pandas as pd
import json


def getting_sect_per_ticker(tickers):
    dict_tick_sect = dict()
    for ticker in tqdm(tickers):
        try:
            stock = finvizfinance(ticker)
            dict_tick_sect[ticker] = stock.ticker_fundament()['Sector']
        except:
            print(ticker)
            continue
    return dict_tick_sect


def generating_datasets():
    with open('../config/config.json', 'r') as file:
        config = json.load(file)

    df = pd.read_csv('../'+config['ticker_companies'], sep=';')
    tickers = df['Ticker'].values.tolist()

    dict_tick_sect = getting_sect_per_ticker(tickers)
    df_sectors = pd.DataFrame(data={'ticker': dict_tick_sect.keys(), 'sector': dict_tick_sect.values()})
    df_sectors['sector'].nunique(), df_sectors['sector'].unique()
    df_sectors.to_csv('../'+config['tickers_sectors_path'])


if __name__ == "__main__":
    generating_datasets()
