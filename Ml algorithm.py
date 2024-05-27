# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2, SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------->
# Load the dataset

dataset_path = 'ml\WC_Train.csv'
cricket_data = pd.read_csv(dataset_path)

# Display the first few rows of the dataset
print(cricket_data.head())

# ----------------------------------------------------------------->


# Handle categorical features by encoding them
label_encoder = LabelEncoder()
for column in cricket_data.columns:
    if cricket_data[column].dtype == 'object':
        cricket_data[column] = label_encoder.fit_transform(cricket_data[column])


# ---------------------------------------------------------------->


# Split the dataset into features (X) and target variable (y)
feature_columns = ['Team A', 'Team B', 'Ground']  # Adjust feature names as needed
X = cricket_data[feature_columns]
y1 = cricket_data['Won']  # Replace with actual target variable name
y2 = cricket_data['Team A Won']  # Replace with actual target variable name



# Split the data into training and testing sets
X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, y1, y2, test_size=0.2, random_state=42)

# ----------------------------------------------------------->

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# PCA (reduce dimensionality)
pca = PCA(n_components=2)  # Adjust the number of components as needed
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Correlation Matrix Analysis
correlation_matrix = X_train.corr()

# Apply Chi-square feature selection separately for each target variable
chi2_selector_y1 = SelectKBest(chi2, k=2)  # Adjust k as needed
X_train_chi2_y1 = chi2_selector_y1.fit_transform(X_train, y1_train)
X_test_chi2_y1 = chi2_selector_y1.transform(X_test)

chi2_selector_y2 = SelectKBest(chi2, k=2)  # Adjust k as needed
X_train_chi2_y2 = chi2_selector_y2.fit_transform(X_train, y2_train)
X_test_chi2_y2 = chi2_selector_y2.transform(X_test)

# Information Gain (Mutual Information) for feature selection
info_gain_selector_y1 = SelectKBest(mutual_info_classif, k=2)  # Adjust k as needed
X_train_info_gain_y1 = info_gain_selector_y1.fit_transform(X_train, y1_train)
X_test_info_gain_y1 = info_gain_selector_y1.transform(X_test)

info_gain_selector_y2 = SelectKBest(mutual_info_classif, k=2)  # Adjust k as needed
X_train_info_gain_y2 = info_gain_selector_y2.fit_transform(X_train, y2_train)
X_test_info_gain_y2 = info_gain_selector_y2.transform(X_test)



# Regularization methods (Ridge and Lasso)
ridge_model = Ridge(alpha=1.0)  # Adjust alpha as needed
ridge_model.fit(X_train_scaled, y2_train)

lasso_model = Lasso(alpha=1.0)  # Adjust alpha as needed
lasso_model.fit(X_train_scaled, y2_train)

# Variance-based feature selection
variance_selector = VarianceThreshold(threshold=0.1)  # Adjust threshold as needed
X_train_variance = variance_selector.fit_transform(X_train_scaled)
X_test_variance = variance_selector.transform(X_test_scaled)


def evaluate_multioutput_model(model, X_test_data, y_test_data, label):
    y1_pred, y2_pred = model.predict(X_test_data).T
    accuracy_y1 = accuracy_score(y1_test, y1_pred)
    accuracy_y2 = accuracy_score(y2_test, y2_pred)
    classification_rep_y1 = classification_report(y1_test, y1_pred, zero_division=1)
    classification_rep_y2 = classification_report(y2_test, y2_pred, zero_division=1)
    
    print(f'{label} Model:')
    print(f'Accuracy Y1: {accuracy_y1:.2f}')
    print('Classification Report Y1:\n', classification_rep_y1)
    print(f'Accuracy Y2: {accuracy_y2:.2f}')
    print('Classification Report Y2:\n', classification_rep_y2)

# models-------------------------------------------------------->

# Logistic Regression
lr_model = MultiOutputClassifier(LogisticRegression(max_iter=1000))  # Increase max_iter as needed
lr_model.fit(X_train_scaled, pd.concat([y1_train, y2_train], axis=1))
evaluate_multioutput_model(lr_model, X_test_scaled, pd.concat([y1_test, y2_test], axis=1), 'Logistic Regression')


# ------------------------------------------------------------------>

# Initialize the Random Forest model
# Random Forest
rf_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
rf_model.fit(X_train, pd.concat([y1_train, y2_train], axis=1))
evaluate_multioutput_model(rf_model, X_test, pd.concat([y1_test, y2_test], axis=1), 'Random Forest')



# models-------------------------------------------------------->
# Decision Tree

dt_model = MultiOutputClassifier(DecisionTreeClassifier(random_state=42))
dt_model.fit(X_train, pd.concat([y1_train, y2_train], axis=1))
evaluate_multioutput_model(dt_model, X_test, pd.concat([y1_test, y2_test], axis=1), 'Decision Tree')


# models-------------------------------------------------------->

# Naïve Bayes
nb_model = MultiOutputClassifier(GaussianNB())
nb_model.fit(X_train, pd.concat([y1_train, y2_train], axis=1))
evaluate_multioutput_model(nb_model, X_test, pd.concat([y1_test, y2_test], axis=1), 'Naïve Bayes')


# models-------------------------------------------------------->

# Ridge Regression (Regression model)
ridge_reg_model = MultiOutputRegressor(Ridge(alpha=1.0))  # Adjust alpha as needed
ridge_reg_model.fit(X_train_scaled, pd.concat([y1_train, y2_train], axis=1))

# Lasso Regression (Regression model)
lasso_reg_model = MultiOutputRegressor(Lasso(alpha=1.0))  # Adjust alpha as needed
lasso_reg_model.fit(X_train_scaled, pd.concat([y1_train, y2_train], axis=1))

# models-------------------------------------------------------->
# k-Nearest Neighbors (kNN)
knn_model = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5))  # Adjust n_neighbors as needed
knn_model.fit(X_train_scaled, pd.concat([y1_train, y2_train], axis=1))
evaluate_multioutput_model(knn_model, X_test_scaled, pd.concat([y1_test, y2_test], axis=1), 'kNN')

# Support Vector Machine (SVM)
svm_model = MultiOutputClassifier(SVC(kernel='linear', C=1.0))  # Adjust kernel and C as needed
svm_model.fit(X_train_scaled, pd.concat([y1_train, y2_train], axis=1))
evaluate_multioutput_model(svm_model, X_test_scaled, pd.concat([y1_test, y2_test], axis=1), 'SVM')


# ------------------------------------------------------------------->
# Assuming you have a list of model names and their accuracies
model_names = ['Logistic Regression', 'Random Forest', 'Decision Tree', 'Naïve Bayes', 'kNN', 'SVM']
accuracies_target1 = [0.59, 0.68, 0.56, 0.73, 0.55, 0.58]
accuracies_target2 = [0.57, 0.57, 0.62, 0.57, 0.58, 0.57]

bar_width = 0.35  # Width of each bar
index = np.arange(len(model_names))  # The label locations

# Plotting the grouped bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(index, accuracies_target1, bar_width, label='Team A Won', color='skyblue')
bar2 = ax.bar(index + bar_width, accuracies_target2, bar_width, label='Won', color='orange')

ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Comparison for Two Target Variables')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(model_names)
ax.legend()

plt.figure(figsize=(10, 6))
plt.plot(model_names, accuracies_target1, marker='o', label='Team A Won', color='skyblue')
plt.plot(model_names, accuracies_target2, marker='o', label='Won', color='orange')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison for Two Target Variables')
plt.legend()
plt.grid(True)
plt.show()