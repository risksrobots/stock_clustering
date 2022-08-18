import pandas as pd
import pandas_datareader.data as web
from tqdm import tqdm
import json


def loading_dataset(tickers, start_date, end_date, meaning):

    df = pd.DataFrame([])
    for tick in tqdm(tickers):
        try:
            panel_data = web.DataReader(tick, 'yahoo', start_date, end_date)

        except Exception as e:
            print(e)
            print('Ticker', tick, 'failed')
        temp = pd.DataFrame(data={tick: panel_data[meaning]}, index=panel_data.index)
        df = pd.concat([df, temp], axis=1)

    return df

def generating_datasets(path_df_tickers, start_date, end_date, path_to_save, meaning_dfs = ['Close']):
    tickers_df = pd.read_csv(path_df_tickers, sep=';')
    tickers = tickers_df['Ticker'].values.tolist()

    df_sectors = pd.read_csv(path + config['tickers_sectors_path'], index_col=0)
    set_tickers_from_sect = set(df_sectors['ticker'].values.tolist())
    tickers = list(set(tickers) & set_tickers_from_sect)

    for meaning in meaning_dfs:
        df = loading_dataset(tickers, start_date, end_date, meaning)


        set_tickers_from_market = set(df.columns.tolist())
        tickers_to_save = list(set_tickers_from_sect & set_tickers_from_market)
        df = df[tickers_to_save]
        df.to_csv(path_to_save.format(meaning))

if __name__ == "__main__":
    path = '../'
    with open(path + 'config/config.json', 'r') as file:
        config = json.load(file)

    start_date = config['start_date']
    end_date = config['end_date']
    path_df_tickers = path+config['ticker_companies']
    path_to_save = path+'data/ticker_data_{}.csv'
    generating_datasets(path_df_tickers, start_date, end_date, path_to_save)

    meaning='Close'
    tickers = ['^GSPC']
    df = loading_dataset(tickers, start_date, end_date, meaning)
    df.to_csv(path_to_save.format('SP500'))

