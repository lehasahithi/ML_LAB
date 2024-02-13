import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean, variance

excel_file = pd.ExcelFile('C:\\Users\\lehas\\OneDrive\\Documents\\B-TECH_AIE_D\\sem_04\\ML\\Lab Session1 Data.xlsx')

df = excel_file.parse('IRCTC Stock Price')
price_mean = mean(df['Price'])
price_variance = variance(df['Price'])

print(f"Mean of Price: {price_mean}")
print(f"Variance of Price: {price_variance}")
df['Date'] = pd.to_datetime(df['Date'])


wednesday_data = df[df['Date'].dt.day_name() == 'Wednesday']['Price']
wednesday_mean = mean(wednesday_data)

print(f"Sample mean for Wednesdays: {wednesday_mean}")


april_data = df[df['Date'].dt.month == 4]['Price']
april_mean = mean(april_data)

print(f"Sample mean for April: {april_mean}")


loss_probability = len(df[df['Price'] < df['Price'].shift(1)]) / len(df['Price'])
print(f"Probability of making a loss: {loss_probability}")


wednesday_profit_probability = len(wednesday_data[wednesday_data > wednesday_data.shift(1)]) / len(wednesday_data)
print(f"Probability of making a profit on Wednesday: {wednesday_profit_probability}")


conditional_profit_probability = len(wednesday_data[wednesday_data > wednesday_data.shift(1)]) / len(wednesday_data)
print(f"Conditional probability of making profit on Wednesday: {conditional_profit_probability}")

plt.scatter(df['Date'].dt.dayofweek, df['Price'])
plt.xlabel('Day of the Week')
plt.ylabel('Price')
plt.title('Scatter Plot of Price against Day of the Week')
plt.show()

