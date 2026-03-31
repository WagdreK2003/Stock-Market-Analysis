import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# STEP 1: LOAD DATA
# =============================

folder_path = r"D:/DA/PROJECTS/Stock Market Analysis + Prediction/Data"

all_data = []

for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(folder_path, file))
        df['Symbol'] = file.replace(".csv", "")
        all_data.append(df)

final_df = pd.concat(all_data)

final_df['Date'] = pd.to_datetime(final_df['Date'])
final_df = final_df.sort_values('Date')

print(final_df.head())

# =============================
# STEP 2: PIVOT
# =============================

pivot_df = final_df.pivot(index='Date', columns='Symbol', values='Close')

# =============================
# STEP 3: PRICE COMPARISON
# =============================

pivot_df.plot(figsize=(12,6))
plt.title("Stock Price Comparison")
plt.show()

# =============================
# STEP 4: RETURNS
# =============================

returns = pivot_df.pct_change()

returns.plot(figsize=(12,6))
plt.title("Daily Returns")
plt.show()

# =============================
# STEP 5: CORRELATION
# =============================

plt.figure(figsize=(8,6))
sns.heatmap(returns.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# =============================
# STEP 6: MOVING AVERAGE + SIGNALS
# =============================

ma_10 = pivot_df.rolling(10).mean()
ma_20 = pivot_df.rolling(20).mean()

for stock in pivot_df.columns:
    
    plt.figure(figsize=(12,6))
    
    plt.plot(pivot_df[stock], label='Price')
    plt.plot(ma_10[stock], label='MA10')
    plt.plot(ma_20[stock], label='MA20')
    
    plt.title(f"{stock} Moving Average")
    plt.legend()
    plt.show()
    
    # Signal
    signal = (ma_10[stock] > ma_20[stock]).astype(int)
    
    plt.figure(figsize=(12,6))
    
    plt.plot(pivot_df[stock], label='Price')
    
    plt.plot(pivot_df[stock][signal == 1], '^', label='Buy')
    plt.plot(pivot_df[stock][signal == 0], 'v', label='Sell')
    
    plt.title(f"{stock} Buy/Sell Signals")
    plt.legend()
    plt.show()

# =============================
# STEP 7: VOLATILITY
# =============================

volatility = returns.std()
volatility.plot(kind='bar')
plt.title("Volatility")
plt.show()

# =============================
# STEP 8: RISK vs RETURN
# =============================

avg_return = returns.mean()

plt.figure()

plt.scatter(volatility, avg_return)

for stock in volatility.index:
    plt.text(volatility[stock], avg_return[stock], stock)

plt.xlabel("Risk")
plt.ylabel("Return")
plt.title("Risk vs Return")

plt.show()

# =============================
# STEP 9: MACHINE LEARNING (PREDICTION)
# =============================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 🔥 Select ONE stock (best practice)
stock = 'AAPL'

ml_df = pd.DataFrame()
ml_df['Price'] = pivot_df[stock]
ml_df['MA10'] = pivot_df[stock].rolling(10).mean()
ml_df['MA20'] = pivot_df[stock].rolling(20).mean()
ml_df['Returns'] = pivot_df[stock].pct_change()

# Remove nulls
ml_df = ml_df.dropna()

# Features & Target
X = ml_df[['MA10', 'MA20', 'Returns']]
y = ml_df['Price']

# Train-test split (NO shuffle)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# =============================
# VISUALIZATION
# =============================

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.plot(y_test.values, label='Actual Price')
plt.plot(predictions, label='Predicted Price')

plt.title(f"{stock} Price Prediction")
plt.legend()

plt.show()

# =============================
# EVALUATION
# =============================

mse = mean_squared_error(y_test, predictions)
print(f"MSE for {stock}:", mse)