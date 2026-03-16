import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("customer_churn.csv")

# Display the first 5 rows of the DataFrame
df.head()
# Display column names and data types
df.info()
# Get summary statistics of numerical columns
df.describe()
df = df.drop(columns=["Unnamed: 0"])
# Use df.describe() to confirm the column was removed (code provided)
df.describe()

# Expected shape of DataFrame is (3333,11) after dropping column. 
# Ensure the results are stored in the df variable
print(f"Shape: {df.shape}. Expected is (3333, 11)")

# Select all features and set target variable
features = df.copy()
features = features.drop(columns=['Churn'])
target_variable = df['Churn']


# One-hot encoding for 'ContractRenewal' feature (provided; do not change)
features = pd.get_dummies(features,columns=['ContractRenewal'],dtype=int)
# See results with one-hot encoding (Notice last 2 columns)
features.head()

# Expected shape of features DataFrame is (3333,11) after one-hot encoding. 
print(f"features shape: {features.shape}. Expected is (3333, 11)")
# Expected shape of target_variable DataFrame is (3333,).
print(f"target_variable shape: {target_variable.shape}. Expected is (3333,)")

# Assume "x" is features and "y" is target_variables
x = features 
y = target_variable

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)

print(x_train.shape) # Expected (2333,11)
print(x_test.shape) # Expected (1000,11)
print(y_train.shape) # Expected (2333,)
print(y_test.shape) # Expected (1000,)

# Summary statistics for churned vs. non-churned customers
churned = df[df['Churn'] == 1]
non_churned = df[df['Churn'] == 0]

# Print average tenure
print("Average tenure (churned customers):",churned['tenure'].mean())
print("Average tenure (non-churned customers):",non_churned['tenure'].mean())

# Bar chart for contract renewal vs churn
churn_counts = df.groupby(['ContractRenewal','churn']).size().unstack()

# Chart options provided
churn_counts.plot(kind='bar', stacked=True)
plt.title('Contract Renewal vs. Churn')
plt.xlabel('Contract Renewal')
plt.ylabel('Count')
plt.show()

# Histogram for tenure distribution
plt.hist(churned['AccountWeeks'],alpha=0.5, label='Churned')
plt.hist(non_churned['AccountWeeks'],alpha=0.5,label='Non-Churned')

# Chart options provided
plt.title('Tenure Distribution by Churn Status')
plt.xlabel('Account Weeks')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Box plot for monthly charges
df.boxplot(column='MonthlyCharge',by='churn')

# Chart options provided
plt.title('Monthly Charges vs. Churn')
plt.xlabel('Churn')
plt.ylabel('Monthly Charge')
plt.suptitle('')  # Remove the default suptitle
plt.show()

# Create an instance of the Logistic Regression model
model = LogisticRegression(max_iter = 1000)

# Train the model
model.fit(x_train, y_train)
# Make predictions on the test set
y_pred = model.predict(x_test)

# Evaluate the model
# Round all values to 3 decimal places

accuracy = round(accuracy_score(y_test,y_pred),3)
precision = round(precision_score(y_test, y_pred),3)
recall = round(recall_score(y_test,y_pred),3)
f1 = round(f1_score(y_test, y_pred),3)

print(f"Accuracy: {accuracy}") # Expected: approximately 0.867
print(f"Precision: {precision}") # Expected: approximately 0.604
print(f"Recall: {recall}") # Expected: approximately 0.203
print(f"F1 Score: {f1}") # Expected: approximately 0.304

coefficients = pd.DataFrame({"Feature": x.columns, "Coefficient": model.coef_[0]})
print(coefficients.sort_values(by="Coefficient",ascending=False))