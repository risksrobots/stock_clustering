# stock_clustering

## Description
The final project for Machine Learning course at Skoltech. We used S&P 500 stock prices from dates 2018-2021 to cluster them alternatively to economical sectors, and performed portfolio optimisation task in terms of risk minimisation.

The repository provides data and code to repeat our calculations, as well as results in a corresponding folder.

### Contents

- autoencoders folder contains autoencoder neural networks (LSTM, Convolutional, Multilinear) which process percent changes of original data
- config folder has the only .json file, which has project parameters like random state (for repeatability), and paths to data files
- data folder has raw and preprocessed data in form of .csv files
- finvizfinance is a python package that project requires and it is stored at github and not at PyPi
- results contains several .csv files with metrics of best models, params, and all the stuff
- utils stores support scripts, i.e. to preprocess data, deserialize neural networks

- install_reqs.sh and preprocess_data.sh are bash scripts for user convenience
- pipeline.ipynb contains the main code of clustering models training and saving the results
- portfolio.ipynb performs portfolio optimisation

## Requirements
  - pandas==1.3.5
  - pandas-datareader==0.10.0
  - finvizfinance==0.12.2 *
  - scipy==1.7.1
  - numpy==1.19.1
  - matplotlib==3.3.1
  - seaborn==0.11.2
  - plotly==5.3.1
  - sklearn==1.0.2
  - pytorch-lightning==1.5.10
  - torch==1.10.2
* finvizfinance library is to be cloned before `pip install`ed

All the requirements are listed in `requirements.txt`

## Results

1. clustering results are shown in jupyter notebooks and results folder
2. tl; dr: clusters do not correspond to economical sectors *at all*, in terms of clustering itself K-Means is the best model in 7/8 experiments

## Instructions
1. Run `install_reqs.sh` to install all the required dependencies
2. Run `preprocess_data.sh` to preprocess the data
3. Then run `nn_test.ipynb` notebook to deserialize neural network feature encoders
4. After that goes the main part: `pipeline.ipynb` does the main stuff and clusters the data
5. As a final step `portfolio.ipynb` calculates financial portfolio by clusters provided
