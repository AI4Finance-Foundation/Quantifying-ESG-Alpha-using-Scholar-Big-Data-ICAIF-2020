# Quantifying ESG Alpha in Scholar Big Data: An Automated Machine Learning Approach

ESG (Environmental, social, governance) factors are widely known as the three primary factors in measuring the sustainability and societal impacts of an investment in a company or business. This repoitory proposes a quantitative approach to measuring the ESG premium in stock trading using ESG scholar data.


## Data

### Alternative Data & Preprocessing

The alternative data we use is from the [Microsoft Academic Graph database](https://www.microsoft.com/en-us/research/project/microsoft-academic-graph/), which is an open resource database with records of publications, including papers, journals, conferences, books etc. It provides the demographics of the publications like public date, citations, authors and affiliated institutes. It includes ESG publication records dating back to 1970s - long enough to study the relationship between ESG publications and companies' stock prices.

* Applied topic model on papers and implemented semi-supervised learning to filter out ESG papers;
* Match ESG papers with companys' stock tickers if one or more authors are employed by the company;
* Performed feature engineering on ESG scholar data, generating a scholar feature space with 48 features.

### Financial Data

10 financial indicators are selected as representatives of company's fundamental conditions. The financial indicators alone could fit a baseline predictive model of stock prices, which would be a baseline model. The financial data we use is pulled from [Compustat database via Wharton Research Data Services](https://wrds-web.wharton.upenn.edu/wrds/ds/compd/fundq).


## Predictive Model

### Ensemble

* Linear Regression
* Lasso Regression
* Ridge Regression
* Random Forest Regression
* Support Vector Regression
* LSTM

### Rolling window backtesting

* Step 1: In each rolling window we train ML models on 48-month training window and validate models on 12-month validation window using MAE.
* Step 2: We choose the best model which has the lowest MAE in validation window as predictive model. Then we predict the stock prices of the backtesting window.
* Step 3: We select top 20% performance stocks provided by the predictive model and form our ESG portfolio by assign then equal weights.

![rolling_window](https://github.com/chenqian0168/Quantifying-ESG-Alpha-in-Scholar-Big-Data-An-Automated-Machine-Learning-Approach/blob/master/pictures/rolling_window.png)

## Performance

* Equal Weights: This is the portfolio of all stocks in the stock pool with equal weights;
* Financial Indicators Only: This is the portfolio built by predictive model trained using financial indicators only;
* Scholar Alpha: This is the portfolio built by predictive model trained using ESG scholar features and financial features.

Below is the rolling annualized Sharpe ratio of ESG alpha vs. Financial indicators only portfolio.
![rolling_sharpe](https://github.com/chenqian0168/Quantifying-ESG-Alpha-in-Scholar-Big-Data-An-Automated-Machine-Learning-Approach/blob/master/pictures/rolling_sharpe.png)

Below shows the logarithmic cumulative return of ESG scholar alpha.
![cummulative_return](https://github.com/chenqian0168/Quantifying-ESG-Alpha-in-Scholar-Big-Data-An-Automated-Machine-Learning-Approach/blob/master/pictures/cumulative_return.png)



This repository refers to the codes for ICAIF-2020 paper.
