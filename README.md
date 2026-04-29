# Benchmarking Linear Regression - Predicting Uber Fare Prices

Regression task to predict fare prices for Uber based on data provided by [Uber](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset). 

## The Challenge
Based on historical data provided by Uber, we would like to accurately predict fare prices for future rides. While this is a regression task, we aim to compare various available techniques (simple Linear Regression, RIDGE, LASSO, XGBoost, and Neural Nets) to identify the best model for this.

## Exploratory Data Analysis
EDA reveals some outliers in the data; as such, we impose certain filters on the dataframe to restrict the effect of these points on the prediction:
  
  - Fare amounts are limited between 0 and 200 dollars,
  - Passenger counts are limited between 0 and 6 people,
  - Longitudes and latitudes are restricted between (-74.5, -72.9) and (40.5, 41.8) degrees, respectively.

The distance covered by the taxi can be calculated using the Haversine formula:
```math
D = r_E \, \times\,  2\arcsin\bigg( \sqrt{\sin\bigg(\frac{lat_2-lat_1}{2}\bigg)^2 + \sin\bigg(\frac{lon_2-lon_1}{2} \bigg)^2 \cos(lat_1)\cos(lat_2)} \bigg).
```
Furthermore, to capture variations of fare prices based on time, we recast the hourly and monthly information in trigonometric functions to reflect their periodicity. 

## Approaches
Five models are explored and compared:


| Model | R2 Score |
|---|---|
| Linear Regression | 0.681272 |
| LASSO | 0.665242 |
| RIDGE | 0.681273 |
| XGBoost Regression | 0.783068 |
| Two-layer Neural Net | 0.791399 |

> Note: The neural net predicts a negative value for the fare in one case, as can be seen in the scatterplot comparing the predicted and true values.

As such, the neural net seems to predict fare prices most accurately, followed closely by XGBoost. 

> Note: An interesting next step would be to further study the fare prices based on the physical location of origin and destination, since fare prices depend on these features as well. While I don't do this yet, I think I will look into this soon. 


## Key Techniques

- `ColumnTransformer` preprocessing pipelines with `OneHotEncoder` and `StandardScaler`.
- Feature engineering to capture `datetime` data and distance. 
- XGBoost regression (`XGBRegressor`).
- Neural nets for regression.

## How to Run
1. Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow xgboost kagglehub
```
2. Run `RegressionAnalysis_Benchmarking.ipynb` top to bottom.

## Tech Stack

Python · pandas · scikit-learn · TensorFlow · XGBoost · matplotlib · seaborn
