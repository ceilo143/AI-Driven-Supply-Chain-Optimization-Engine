import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns



def generate_data():
    print("Generating synthetic supply chain data...")
    np.random.seed(42)
    rows = 1000

    data = {
        'day_of_week': np.random.randint(1, 8, rows),
        'promotion_active': np.random.choice([0, 1], rows, p=[0.7, 0.3]),
        'previous_day_sales': np.random.randint(20, 100, rows),
        'current_stock': np.random.randint(10, 200, rows),
        'is_weekend': np.random.choice([0, 1], rows)
    }

    df = pd.DataFrame(data)


    df['units_sold'] = (df['previous_day_sales'] * 0.6) + \
                       (df['promotion_active'] * 15) + \
                       (df['is_weekend'] * 10) + \
                       np.random.normal(0, 5, rows)

    df.to_csv('inventory_data.csv', index=False)
    print("File 'inventory_data.csv' created successfully!\n")



generate_data()


df = pd.read_csv('inventory_data.csv')


X = df[['day_of_week', 'promotion_active', 'previous_day_sales', 'current_stock', 'is_weekend']]
y = df['units_sold']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("--- Model Performance ---")
print(f"Mean Absolute Error: {mae:.2f} units")
print(f"R-Squared Score: {r2:.2%}")
print("--------------------------\n")


plt.figure(figsize=(10, 6))
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).sort_values().plot(kind='barh', color='teal')
plt.title('Business Insights: What Drives Our Sales?')
plt.xlabel('Importance Score (0 to 1)')
plt.ylabel('Factors Analyzed')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

print("Displaying Feature Importance chart...")
plt.show()


new_scenario = pd.DataFrame([[6, 1, 50, 100, 1]], columns=X.columns)

prediction = model.predict(new_scenario)
print(f"--- Real-World Application ---")
print(f"Scenario: Weekend with active Promotion.")
print(f"AI Forecast: We need to prepare {prediction[0]:.0f} units for tomorrow.")
print(f"------------------------------")
