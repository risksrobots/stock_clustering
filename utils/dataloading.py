import pandas as pd
import pandas_datareader.data as web
from tqdm import tqdm

start_date = '2018-01-01'
end_date = '2022-02-01'
path_df_sect = 'data/tickers_sectors.csv'
path_to_save = 'data/ticker_data_{}.csv'

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

def generating_datasets(path_df_sect, start_date, end_date, path_to_save, meaning_dfs = ['Close', 'Volume']):
    tickers_sectors = pd.read_csv(path_df_sect, encoding='windows-1251', sep=';')
    tickers = tickers_sectors['Тикер'].values

    for meaning in meaning_dfs:
        df = loading_dataset(tickers, start_date, end_date, meaning)
        df.to_csv(path_to_save.format(meaning))

if __name__ == "__main__":
	#generating_datasets(path_df_sect, start_date, end_date, path_to_save, meaning_dfs = ['Close', 'Volume'])
    meaning='Close'
    tickers = ['^GSPC']
    df = loading_dataset(tickers, start_date, end_date, meaning)
    df.to_csv(path_to_save.format('SP500'))

