from autoencoders import Conv1dAutoEncoder, LSTMAutoEncoder, TickerDataModule, MLPAutoEncoder

from typing import Any
import pandas as pd
import numpy as np
from math import floor
import torch
import json

from tqdm import tqdm

with open('config/config.json', 'r') as file:
    config = json.load(file)

df = pd.read_csv(config['ticker_data_preprocessed'], index_col=[0])
df.drop(columns=['sector'], axis=1, inplace=True)

model_mlp = MLPAutoEncoder.load_from_checkpoint(config['nn_mlp_checkpoint'], in_features=100, latent_features=18)
model_lstm = LSTMAutoEncoder.load_from_checkpoint(config['nn_lstm_checkpoint'], seq_len=100, n_features=1,  embedding_dim=18)
model_cae = Conv1dAutoEncoder.load_from_checkpoint(config['nn_conv_checkpoint'], in_channels=1, n_latent_features=18, seq_len=100)
model_mlp.eval();
model_lstm.eval();
model_cae.eval();

mlp_encoded = np.zeros((df.shape[0], 18))
lstm_encoded = np.zeros((df.shape[0], 18))
cae_encoded = np.zeros((df.shape[0], 18))

for i, name_ticker in tqdm(enumerate(np.unique(df.index))):
    ts_name = df[df.index == name_ticker].values
    ts_name = ts_name.flatten()
    seq_len = ts_name.shape[0]
    fl_1 = floor(seq_len / 100)
    sample_1 = ts_name[:100 * fl_1].reshape(fl_1, 1, 100)
    fl_2 = floor(seq_len / 100)
    sample_2 = ts_name[:100 * fl_2].reshape(fl_2, 1, 100)
    
    mlp_sample = model_mlp.predict_step(torch.tensor(sample_1).float()).detach().numpy()
    cae_sample = model_cae.predict_step(torch.tensor(sample_1).float()).squeeze().detach().numpy()
    lstm_sample = model_lstm.predict_step(torch.tensor(sample_2).float()).detach().numpy()
    
    mlp_emb = mlp_sample.mean(axis=0)
    cae_emb = cae_sample.mean(axis=0)
    lstm_emb = lstm_sample.mean(axis=0)
    
    mlp_encoded[i, :] = mlp_emb
    cae_encoded[i, :] = cae_emb
    lstm_encoded[i, :] = lstm_emb


df_mlp = pd.DataFrame(mlp_encoded, index=np.unique(df.index))
df_cae = pd.DataFrame(cae_encoded, index=np.unique(df.index))
df_lstm = pd.DataFrame(lstm_encoded, index=np.unique(df.index))

df_mlp.to_csv(config['nn_mlp_data'])
df_cae.to_csv(config['nn_conv_data'])
df_lstm.to_csv(config['nn_lstm_data'])