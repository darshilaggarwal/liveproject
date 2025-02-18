import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from warnings import filterwarnings
filterwarnings(action='ignore')

adv=pd.read_csv("click_advertisement.csv")
adv.head()

adv.tail()

adv.describe()

adv.info()

adv.shape

#Checking null values in dataset
adv.isna().sum()

# Define a color palette
palette = sns.color_palette("husl")  # "husl" is an example, you can choose any seaborn palette

sns.countplot(x=adv['Clicked'], palette=palette)
plt.xlabel('Click on Advertisement') 
plt.ylabel('Count') 
plt.title('Distribution of Target Variable: Click on Advertisement')
plt.show()

numeric_features = adv.select_dtypes(include=['int64', 'float64']).columns
adv[numeric_features].hist(figsize=(15, 10), bins=20, edgecolor='black',)
plt.suptitle('Distribution of Numeric Features')
plt.show()

palette = sns.color_palette("husl")  # "husl" is an example, you can choose any seaborn palette
categorical_features = adv.select_dtypes(include=['object', 'category']).columns
for feature in categorical_features:
    sns.countplot(x=feature, data=adv,palette=palette) 
    plt.title(f'Distribution of {feature}')
    plt.show()

sns.distplot(adv['Clicked'])

adv.plot(kind ='box',subplots = True, layout =(4,4),sharex = False)

adv.plot(kind ='density',subplots = True, layout =(4,4),sharex = False)

labels = ['Clicked', 'Not Clicked']  # Labels for each segment of the pie chart
clicked_count = adv['Clicked'].value_counts()  # Count of clicked (1) and not clicked (0) entries
sizes = [clicked_count[1], clicked_count[0]]  # Sizes of each segment based on the counts
colors = ['lightgreen', 'lightcoral']  # Colors for each segment of the pie chart
explode = (0.1, 0)  # Explode the first slice (Clicked) for emphasis

# Create a pie chart
plt.figure(figsize=(8, 5))  # Set the figure size
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)  # Create the pie chart
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.title('Click Distribution from DataFrame')  # Title of the pie chart
plt.show()  # Display the pie chart

from sklearn.preprocessing import LabelEncoder

# Select only numeric columns
numeric_data = adv.select_dtypes(include=['int64', 'float64'])

# Calculate the correlation matrix for numeric features
numeric_corr_matrix = numeric_data.corr()

# Plot the heatmap for numeric correlations
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Matrix Heatmap for Numeric Features')
plt.show()


from sklearn.preprocessing import LabelEncoder

# Select only numeric columns
numeric_data = adv.select_dtypes(include=['int64', 'float64'])

# Calculate the mean for numeric features
numeric_mean = numeric_data.mean()
print('Mean values for each column:') 
print(numeric_mean)

# Select only numeric columns
numeric_data = adv.select_dtypes(include=['int64', 'float64'])

# Calculate the mean for numeric features
numeric_median = numeric_data.median()
print('Median values for each column:') 
print(numeric_median)

# Select only numeric columns
numeric_data = adv.select_dtypes(include=['int64', 'float64'])

# Calculate the mean for numeric features
numeric_mode = numeric_data.mode()
print('Mode values for each column:') 
print(numeric_mode)

# Select only numeric columns
numeric_data = adv.select_dtypes(include=['int64', 'float64'])

# Calculate the mean for numeric features
numeric_std = numeric_data.std()
print('standard deviation for each column:') 
print(numeric_std)

# Select only numeric columns
numeric_data = adv.select_dtypes(include=['int64', 'float64'])

# Calculate the mean for numeric features
numeric_var = numeric_data.var()
print('Variance for each column:') 
print(numeric_var)

adv['City_code'].nunique()

adv['Ad_Topic'].nunique()

from sklearn.preprocessing import LabelEncoder
##binary variable
bi_var = [col for col in adv.columns if len(adv[col].unique()) ==2 ]
cat_col = [col for col in adv.select_dtypes(['object']).columns.tolist() if col not in bi_var]

encoder = LabelEncoder()
for i in bi_var:
    adv[i] = encoder.fit_transform(adv[i]) 
    
adv = pd.get_dummies(adv,columns= cat_col)


adv.head()

from sklearn.feature_selection import SelectKBest,chi2,RFE,SelectFromModel,mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier, plot_importance

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Split the data into features and target
X = adv.drop('Clicked', axis=1)
y = adv['Clicked']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print the classification report
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print(confusion_matrix(y_test, y_pred))

import pickle
# Save the trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(grid_search, file)

print("Model saved successfully!")


