

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


# 1. Read CSV and initial exploration
sp = pd.read_csv('sp_2016_20220915.csv')

# 1a. Print shape
print("Rows:", sp.shape[0], "Columns:", sp.shape[1])

# 1b. Print column names
print("\nColumn names:", sp.columns.tolist())

# 1c. Convert 'date' to datetime
sp['date'] = pd.to_datetime(sp['date'])
print("\nData types after conversion:")
print(sp.dtypes)

# 1d. Unique symbols
unique_symbols = sp['symbol'].nunique()
print("\nUnique symbols:", unique_symbols)

# 1e. Days per symbol analysis
days_by_symbol = pd.DataFrame(sp['symbol'].value_counts())
max_days = days_by_symbol['count'].max()
full_period_stocks = (days_by_symbol['count'] == max_days).sum()
print("\nMax days:", max_days, "\nStocks with full period:", full_period_stocks)

# 2. Create filtered dataframe and calculate metrics
sp_ = sp[['symbol', 'sector', 'industry', 'date', 'marketCap', 'ret', 'spy_ret', 'Close', 'spy_Close']]
sp_['total_mkt_cap'] = sp_.groupby('date')['marketCap'].transform('sum')

# 2b. Market-cap weight
sp_['mkt_cap_weight'] = sp_['marketCap'] / sp_['total_mkt_cap']

# 2c. Portfolio return calculation
sp_['mkt_cap*ret'] = sp_['mkt_cap_weight'] * sp_['ret']
sp_['mcapwtd_ret'] = sp_.groupby('date')['mkt_cap*ret'].transform('sum')

# 3. Create returns dataframe
df0 = sp_[['date', 'spy_ret', 'mcapwtd_ret']].drop_duplicates().reset_index(drop=True)
print("\nShape check (sp_, df0):", sp_.shape, df0.shape)

# 3b. Plot daily returns
plt.figure(figsize=(12,6))
plt.plot(df0['date'], df0['spy_ret'], label='SPY Returns')
plt.plot(df0['date'], df0['mcapwtd_ret'], label='Market-Cap Weighted Returns')
plt.title('Daily Returns Comparison')
plt.legend()
plt.show()

# 3c. Scatter plot
plt.figure(figsize=(8,8))
sns.regplot(x='spy_ret', y='mcapwtd_ret', data=df0)
plt.title('SPY vs Market-Cap Weighted Returns')
plt.show()

# 3d. Correlation and stats
print("\nReturn statistics:")
print(df0[['spy_ret', 'mcapwtd_ret']].describe())
corr = df0[['spy_ret', 'mcapwtd_ret']].corr().iloc[0,1]
print("\nCorrelation:", corr)

# 4. Lagged returns calculation
sp_['_ret_'] = abs(sp_['ret'])
sp_['prev_ret'] = sp_.groupby('symbol')['ret'].shift(1)
sp_['next_ret'] = sp_.groupby('symbol')['ret'].shift(-1)
sp_['_prev_ret_'] = abs(sp_['prev_ret'])
sp_['_next_ret_'] = abs(sp_['next_ret'])
sp_['prev_spy_ret'] = sp_.groupby('symbol')['spy_ret'].shift(1)
sp_['next_spy_ret'] = sp_.groupby('symbol')['spy_ret'].shift(-1)
sp_ = sp_.dropna()

# 5a. Regression model
model = smf.ols('next_ret ~ ret + prev_ret', data=sp_).fit()
print("\nRegression 5a results:")
print(model.summary())

# 5b. Filtered regression
_sp_5 = sp_[(abs(sp_['ret']) < 0.05) & (abs(sp_['prev_ret']) < 0.05)]
model_filtered = smf.ols('next_ret ~ ret + prev_ret', data=_sp_5).fit()
print("\nFiltered regression results:")
print(model_filtered.summary())

# 6. Absolute returns regression
model_abs = smf.ols('_next_ret_ ~ _ret_ + _prev_ret_', data=sp_).fit()
print("\nAbsolute returns regression:")
print(model_abs.summary())

# 7. SPY regression
model_spy = smf.ols('ret ~ spy_ret', data=sp_).fit()
print("\nSPY exposure regression:")
print(model_spy.summary())

# 8. Next-day SPY regression
model_next_spy = smf.ols('next_ret ~ spy_ret', data=sp_).fit()
print("\nNext-day SPY regression:")
print(model_next_spy.summary())