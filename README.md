# Quantifying ESG Alpha in Scholar Big Data: An Automated Machine Learning Approach

ESG (Environmental, social, governance) factors are widely known as the three primary factors in measuring the sustainability and societal impacts of an investment in a company or business. This repoitory proposes a quantitative approach to measuring the ESG premium in stock trading using ESG scholar data.


## Data

### Alternative Data & Preprocessing

The alternative data we use is from the [Microsoft Academic Graph database](https://www.microsoft.com/en-us/research/project/microsoft-academic-graph/), which is an open resource database with records of publications, including papers, journals, conferences, books etc. It provides the demographics of the publications like public date, citations, authors and affiliated institutes. It includes ESG publication records dating back to 1970s - long enough to study the relationship between ESG publications and companies' stock prices.

* Applied topic model on papers and implemented semi-supervised learning to filter out ESG papers;
* Match ESG papers with company's stock ticker if one or more authors are employed by the company;
* Performed feature engineering on ESG scholar data, generating a scholar feature space with 48 features.

### Financial Data

10 financial indicators are selected as representatives of company's fundamental conditions. The financial indicators alone could fit a baseline predictive model of stock prices, which would be a baseline model. The financial data we use is pulled from [Compustat database via Wharton Research Data Services](https://wrds-web.wharton.upenn.edu/wrds/ds/compd/fundq).


## Predictive Model

### Ensemble
![1](https://github.com/chenqian0168/Quantifying-ESG-Alpha-in-Scholar-Big-Data-An-Automated-Machine-Learning-Approach/blob/master/pictures/cumulative_return.png)
This repository refers to the codes for ICAIF-2020 paper.
