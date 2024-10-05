import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('Real_Estate.csv')

# Explore the dataset
# print(data.head())  # Displays the first few rows of the dataset
# print(data.info())  # Gives you a summary of the dataset, including column names and data types
# print(data.describe())  # Shows statistical summaries of numerical columns


# print(data.isnull().sum()) #check for missing values

# time to train the model
# choose the columns that are relevant for predicting prices
# Select relevant columns (assuming these are available in your dataset)

#for findin features print (data.columns)
features=[ 'House age', 'Distance to the nearest MRT station',
       'Number of convenience stores', 'Latitude', 'Longitude',
       'House price of unit area']

X = data[features]  # Example features
# y = data['price']  # Target variable (price)
y_unit_price = data['House price of unit area']  # Unit price (price per square meter)

# Assuming we have a total area (if not in the dataset, add a fixed value)
# Here, we'll assume a fixed area of 100 square meters for simplicity.
# In reality, replace this with the actual area feature if it's available in your dataset.
total_area = 100  # You can also use a column like data['area'] if available


"""
# handle missing data if available
# Option 1: Drop rows with missing values
X = X.dropna()

# Option 2: Fill missing values (e.g., with the mean of the column)
X.fillna(X.mean(), inplace=True)


""" 


#normalization so that they have same scale 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Split data into 80%(0.8) training and 20%(0.2) testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_unit_price, test_size=0.2, random_state=42)



# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)





# Predict on the test set
y_pred_unit_price= model.predict(X_test)



# Calculate the Mean Squared Error (MSE) for unit price
mse = mean_squared_error(y_test, y_pred_unit_price)
print(f'Mean Squared Error for Unit Price: {mse}')

# Calculate R-squared for unit price
r2 = model.score(X_test, y_test)
print(f'R-squared for Unit Price: {r2}')

# Now, to predict the **total price**, multiply the predicted unit price by the total area
y_pred_total_price = y_pred_unit_price * total_area

# Example output for first 5 predictions
print("Predicted Total Prices for Test Set (first 5 predictions):")
for i in range(5):
    print(f"Predicted unit price: {y_pred_unit_price[i]:.2f}, Predicted total price: {y_pred_total_price[i]:.2f}")


#if model works well then save it

# Save the model
joblib.dump(model, 'house_price_model.pkl')

# Save the scaler for normalization later
np.save('mean.npy', scaler.mean_)
np.save('std.npy', scaler.scale_)