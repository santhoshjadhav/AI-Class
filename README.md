import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Create a simple dataset
data = {
    'square_footage': [1500, 2500, 1800, 2200, 3000, 4000,
                       3500, 2000, 2700, 2300],
    'bedrooms': [3, 4, 3, 4, 5, 6,
                 5, 3, 4, 4],
    'age': [10, 20, 15, 30, 25, 40,
            35, 20, 5, 10],
    'price': [300000, 500000, 400000, 600000, 700000, 800000,
              750000, 450000, 480000, 550000]
}
df = pd.DataFrame(data)
print("Dataset:\n", df)

# Step 2: Define features (X) and target (y)
X = df[['square_footage', 'bedrooms', 'age']]
y = df['price']

# Step 3: Split into training and test data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# Step 4: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict using the test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Step 7: Visualize the actual vs predicted prices
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.grid(True)
plt.show()

