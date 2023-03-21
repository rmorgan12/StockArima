# Importing Packages
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
import statsmodels.api
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.arima.model import ARIMA
# -----------------------------------------------------------

# Import Data
data = pd.read_csv("DOMO.csv")
closing_data = data['Close']
# -----------------------------------------------------------

# Dickey Fuller Test For Stationary vs. Non-Stationary

result = adfuller(closing_data)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')

for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# We Now Know Non-Stationary becuse p-value > .05
# -----------------------------------------------------------

# Differencing to check data for stationary
closing_diff = pd.Series(closing_data).diff().dropna()

result = adfuller(closing_diff)
print('Diff ADF Statistic: %f' % result[0])
print('Diff p-value: %f' % result[1])
print('Diff Critical Values:')

for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# p value less than .05 -> Stationary
# -----------------------------------------------------------
# plotting original vs 1st differnce

plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})
# Original
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(closing_data)
ax1.set_title('Original Data')
ax1.axes.xaxis.set_visible(False)

# 1st Difference
ax2.plot(closing_diff)
ax2.set_title('1st Difference Data')
ax2.axes.xaxis.set_visible(False)

plt.show()
# -----------------------------------------------------------
# Searching for P and Q

plot_pacf(closing_diff, lags=60)
plot_acf(closing_diff, lags=60)
plt.show()

# -----------------------------------------------------------

# Parameters
p = 1
d = 1
q = 1

# -----------------------------------------------------------

# ARIMA MODEL

arima_model = ARIMA(closing_data, order=(p, d, q))
model_fit = arima_model.fit()
print(model_fit.summary())


predicted_data = model_fit.predict(
    start=170, end=len(closing_data), type='levels')

plt.plot(closing_data, label='Original data', color='blue')
plt.plot(predicted_data, label='Predictions',
         linestyle='dashdot', color='red')
plt.title("Domo Stock Closing Prices")
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.legend()
plt.show()


# -----------------------------------------------------------

# Error Metrics to Test Model

historical_data = closing_data[170:].tolist()
forecasted_data = predicted_data.tolist()


# MAPE

APE = []
for i in range(len(forecasted_data)-1):
    num = (historical_data[i] - forecasted_data[i])/historical_data[i]
    num = abs(num)
    APE.append(num)
MAPE = sum(APE)/len(APE)

print(f'''
MAPE = {round(MAPE, 2)}
MAPE % = {round(MAPE*100, 2)} %
''')

# MAE

total = 0
count = 0
for i in range(len(forecasted_data)-1):
    val = abs(historical_data[i] - forecasted_data[i])
    total += val
    count += 1

mae = total / count
print("M.A.E. = ", mae)

# MSE and RMSE

total = 0
count = 0
for i in range(len(forecasted_data)-1):
    num = (closing_data[i] - forecasted_data[i]) ** 2
    total += num
    count += 1
mse = total / count
print("M.S.E. = ", mse)
rmse = sqrt(mse)
print("R.M.S.E = ", rmse)
