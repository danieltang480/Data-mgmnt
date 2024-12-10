import mysql.connector
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

file_path = 'Carbon_(CO2)_Emissions_by_Country.csv'  
data = pd.read_csv(file_path)

data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y').dt.strftime('%Y-%m-%d')

print(f"Number of rows in the data: {len(data)}")

# Database
db = mysql.connector.connect(
    host="localhost",
    user="root",  
    password="Password123!",  
    database="carbon_emissions"  
)

cursor = db.cursor()

cursor.execute("DROP TABLE IF EXISTS emissions_data")

#SQL table creation
create_table_query = """
CREATE TABLE emissions_data (
    Country VARCHAR(255),
    Region VARCHAR(255),
    Date DATE,
    `Kilotons of Co2` FLOAT,
    `Metric Tons Per Capita` FLOAT
);
"""
cursor.execute(create_table_query)

# CSV files -> database
rows_inserted = 0
for index, row in data.iterrows():
    print(f"Inserting row {index}: {row}")
    insert_query = """
    INSERT INTO emissions_data (Country, Region, Date, `Kilotons of Co2`, `Metric Tons Per Capita`)
    VALUES (%s, %s, %s, %s, %s);
    """
    cursor.execute(insert_query, (row['Country'], row['Region'], row['Date'], row['Kilotons of Co2'], row['Metric Tons Per Capita']))
    rows_inserted += 1

db.commit()

print(f"{rows_inserted} rows inserted into the database.")
print(data.isnull().sum())
data = data.drop_duplicates()
print(data.describe())

scaler = StandardScaler()
data[['Kilotons of Co2', 'Metric Tons Per Capita']] = scaler.fit_transform(data[['Kilotons of Co2', 'Metric Tons Per Capita']])

X = data[['Metric Tons Per Capita']]
y = data['Kilotons of Co2']

poly = PolynomialFeatures(degree=2)  
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

ridge = Ridge(alpha=1.0)  
ridge.fit(X_train, y_train)

y_pred = ridge.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))
print("Intercept:", ridge.intercept_)
print("Coefficients:", ridge.coef_)

# Showing countries that are high in carbon emissions to emphasize how much carbon is being released into the world
plt.figure()  
country_data = data[data['Country'] == 'China']
sns.boxplot(country_data['Kilotons of Co2'])
plt.figure() 
country_data2 = data[data['Country'] == 'United States']
sns.boxplot(country_data2['Kilotons of Co2'])
plt.figure()  
country_data = data[data['Country'] == 'Germany']
sns.boxplot(country_data['Kilotons of Co2'])
plt.figure()  
country_data = data[data['Country'] == 'Japan']
sns.boxplot(country_data['Kilotons of Co2'])
plt.show()

cursor.close()
db.close()
