# stock_clustering

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

## Description of the project

## Results

## Instructions
1. Run `install_reqs.sh` to install all the required dependencies
2. Run `preprocess_data.sh` to preprocess the data
3. Then run `nn_test.ipynb` notebook to deserialize neural network feature encoders
4. After that goes the main part: `pipeline.ipynb` does the main stuff and clusters the data
5. As a final step `portfolio.ipynb` calculates financial portfolio by clusters provided


repo contents
