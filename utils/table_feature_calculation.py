from feature_functions import find_max_recovery, find_max_drawdown
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

path = '../'

with open(path+'config/config.json', 'r') as file:
    config = json.load(file)

def loading_data(config):
    df = pd.read_csv(path + config['ticker_data_preprocessed'], index_col=0)

    df_index = pd.read_csv(path + config['ticker_data_sp500'], index_col=0)
    df_index = df_index.pct_change()[1:].T
    df_index.index = ['market']

    df_no_sector = df.drop(['sector'], axis=1)
    df_with_market = pd.concat([df_no_sector, df_index], join='inner')
    df_with_market = df_with_market.fillna(0)
    df = df[df_with_market.columns]
    df_no_sector = df_no_sector[df_with_market.columns]

    return df, df_no_sector, df_with_market

def feature_engineering():

    df, df_no_sector, df_with_market = loading_data(config)
    table_features = pd.DataFrame(index=df_no_sector.index)

    table_features['mean_return'] = df_no_sector.T.mean()
    table_features['std_return'] = df_no_sector.T.std()
    table_features['median_return'] = df_no_sector.T.median()
    table_features['share_positive_return'] = (df_no_sector.T > 0).sum() / df_no_sector.shape[1]

    features_names = ['max_drawdown', 'rec_period', 'beta', 'alpha',
                     'sharp', 'VaR', 'CVaR', 'CAPM', 'coef_var', 'IR']
    dict_features = {name:[] for name in features_names}

    riskless_return = config['riskless_rate'] / 252
    index = df_with_market.loc['market'].T.values
    r_market = np.mean(index)

    for ticker in tqdm(df_no_sector.index):
        price = df_no_sector.loc[ticker].T.values
        price_cumprod = (df_no_sector.loc[ticker] + 1).cumprod()

        max_rec_per = find_max_recovery(price_cumprod)[0]
        max_drawdown = find_max_drawdown(price_cumprod)[0]

        covar = np.cov(price, index)[0, 1]
        std = table_features.loc[ticker, 'std_return']
        var = std ** 2
        var_market = np.var(index)
        mean_return = table_features.loc[ticker, 'mean_return']

        beta = covar / var_market
        alpha = mean_return - beta * r_market
        sharp = (mean_return - riskless_return) / std
        VaR = np.quantile(price, 0.05)
        CVaR = price[price < VaR].mean()
        CAPM = riskless_return + beta * (r_market - riskless_return)
        coef_variation = var / mean_return
        IR = (mean_return - r_market) / np.std(price - index)

        feature_meanings = [max_drawdown, max_rec_per, beta, alpha, sharp,
                            VaR, CVaR, CAPM, coef_variation, IR]
        dict_feature_meanings = dict(zip(features_names, feature_meanings))
        for name, meaning in dict_feature_meanings.items():
            dict_features[name].append(meaning)

    for name, column in dict_features.items():
        table_features[name] = column

    table_features.to_csv(path+config['features_path'])

if __name__ == "__main__":
    feature_engineering()