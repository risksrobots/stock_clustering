import pandas as pd
import json


def main():
    path = '../'
    with open(path+'config/config.json', 'r') as file:
        config = json.load(file)

    df = pd.read_csv(path+config['ticker_data_close'], index_col=0)

    df_na = df.isna().sum()
    stocks_to_drop = df_na[df_na > 0].index.tolist()
    df = df.drop(stocks_to_drop, axis=1)

    df = df.pct_change()[1:]
    df = df.T

    df_sectors = pd.read_csv(path + config['tickers_sectors_path'], index_col=0)
    dict_tick_sect = dict(zip(df_sectors['ticker'].values.tolist(),
                              df_sectors['sector'].values.tolist()))

    set_tickers_from_sect = set(df_sectors['ticker'].values.tolist())
    set_tickers_from_close = set(df.index.tolist())
    tickers_to_save = list(set_tickers_from_sect & set_tickers_from_close)
    df = df.loc[sorted(tickers_to_save)]

    df['sector'] = df.index.map(dict_tick_sect)

    df.to_csv(path+config['ticker_data_preprocessed'])
    return df

if __name__ == "__main__":
    main()
