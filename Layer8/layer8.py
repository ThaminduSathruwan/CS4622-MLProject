# %% [markdown]
# # CS4622 - Machine Learning
# # Project - Speaker, age, gender and accent recognition using wav2vec base
#
# # Layer 8

# %% [markdown]
# # Load Data

# %%
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
!pip install kaggle seaborn imbalanced-learn

# %%
# Import libraries

# %%
# Load the data
train_data = pd.read_csv('Data/train.csv')
valid_data = pd.read_csv('Data/valid.csv')
test_data = pd.read_csv('Data/test.csv')

# %%
train_data.describe()

# %%
valid_data.describe()

# %%
test_data.describe()

# %% [markdown]
# # Handling missing values

# %%
# Handle missing values
missing_train = train_data.isnull().sum()
missing_valid = valid_data.isnull().sum()
print("missing_train:")
print(missing_train)
print("\nmissing_valid:")
print(missing_valid)

# %% [markdown]
# There are missing values in `label_2`.

# %%
# Impute missing values
imputer = SimpleImputer(strategy='mean')
columns = ['label_2']
imputer.fit(train_data[columns])

train_data[columns] = imputer.transform(
    train_data[columns]).round().astype(int)
valid_data[columns] = imputer.transform(
    valid_data[columns]).round().astype(int)

# %%
# Check missing values again
labels = ["label_1", "label_2", "label_3", "label_4"]
train_data[labels].isnull().sum()

# %%
valid_data[labels].isnull().sum()

# %% [markdown]
# # Functions

# %%
# Function for plot class distribution


def plot_class_distribution(y):
    class_counts = y.value_counts()
    plt.figure(figsize=(18, 6))
    plt.title("Class distribution")
    sns.barplot(x=class_counts.index, y=class_counts.values, color='blue')
    plt.show()

# %%
# Function for roboust scaling


def robust_scale_data(X_train, X_valid, X_test):
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
    X_valid_scaled = pd.DataFrame(scaler.transform(X_valid))
    X_test_scaled = pd.DataFrame(scaler.transform(X_test))
    return X_train_scaled, X_valid_scaled, X_test_scaled

# %%
# Function for standard scaling


def standard_scale_data(X_train, X_valid, X_test):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
    X_valid_scaled = pd.DataFrame(scaler.transform(X_valid))
    X_test_scaled = pd.DataFrame(scaler.transform(X_test))
    return X_train_scaled, X_valid_scaled, X_test_scaled

# %%
# Function for Principal Component Analysis


def pca(X_train, X_valid, X_test, n_components=0.95, svd_solver='full'):
    pca = PCA(n_components=n_components, svd_solver=svd_solver)
    pca.fit(X_train)
    X_train_trf = pd.DataFrame(pca.transform(X_train))
    X_valid_trf = pd.DataFrame(pca.transform(X_valid))
    X_test_trf = pd.DataFrame(pca.transform(X_test))
    return X_train_trf, X_valid_trf, X_test_trf

# %%
# KNN Classifier Model


def knn_model(X_train, y_train, X_valid, y_valid, n_neighbors=5, weights='uniform', metric='minkowski'):
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors, weights=weights, metric=metric)
    knn.fit(X_train, y_train)
    return knn

# %%
# SVC Model


def svc_model(X_train, y_train, X_valid, y_valid, C=1.0, kernel='linear'):
    svc = SVC(C=C, kernel=kernel)
    svc.fit(X_train, y_train)
    return svc

# %%
# Support Vector Machine RandomizedSearchCV


def svc(X_train, y_train, X_valid, y_valid, param_grid, n_iter=3, cv=3):
    svc = SVC()
    svc.fit(X_train, y_train)
    svc_cv = RandomizedSearchCV(
        svc, param_grid, n_iter=n_iter, cv=cv, scoring='accuracy', random_state=42, n_jobs=-1)
    svc_cv.fit(X_train, y_train)

    print("Tuned hyperparameters: {}".format(svc_cv.best_params_))
    print("Best score: {}".format(svc_cv.best_score_))

    y_pred = svc_cv.predict(X_valid)
    print("Accuracy score {:.3f}".format(accuracy_score(y_valid, y_pred)))

    return svc_cv

# %%
# KNN hyperparameter tuning RandomizedSearchCV


def knn(X_train, y_train, X_valid, y_valid, n_iter=20, cv=5):
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19],
                  'metric': ['euclidean', 'manhattan']}

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_cv = RandomizedSearchCV(
        knn, param_grid, n_iter=n_iter, cv=cv, random_state=42, n_jobs=10)
    knn_cv.fit(X_train, y_train)

    print("Tuned hyperparameters: {}".format(knn_cv.best_params_))
    print("Best score: {}".format(knn_cv.best_score_))

    y_pred = knn_cv.predict(X_valid)
    print("Accuracy score {:.3f}".format(accuracy_score(y_valid, y_pred)))

    return knn_cv

# %%
# Random Forest hyperparameter tuning RandomizedSearchCV


def random_forest(X_train, y_train, X_valid, y_valid):
    param_grid = {'n_estimators': [100, 200, 300, 400, 500],
                  'max_depth': list(np.arange(10, 110, 10)), }

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_cv = RandomizedSearchCV(
        rf, param_grid, n_iter=3, cv=3, random_state=42, n_jobs=10)
    rf_cv.fit(X_train, y_train)

    print("Tuned hyperparameters: {}".format(rf_cv.best_params_))
    print("Best score: {}".format(rf_cv.best_score_))

    y_pred = rf_cv.predict(X_valid)
    print("Accuracy score {:.3f}".format(accuracy_score(y_valid, y_pred)))

# %%
# SMOTE Oversampling


def smote_oversampling(X_train, y_train, verbose=0):
    counter = Counter(y_train)
    if(verbose):
        print('Before SMOTE:', counter)

    oversample = SMOTE(sampling_strategy='auto', random_state=42)
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    counter = Counter(y_train)
    if(verbose):
        print('After SMOTE:', counter)
    return X_train, y_train

# %%
# Cross validation score


def cross_validation_score(model, X_train, y_train, cv=3):
    scores = cross_val_score(model, X_train, y_train, cv=cv)
    print("Cross-validation scores: {}".format(scores))
    print("Average cross-validation score: {:.3f}".format(scores.mean()))

# %%
# Create output csv file


def create_output_csv(y_pred, file_name):
    output_filename = f"Results/{file_name}.csv"
    df = pd.DataFrame(y_pred)
    df.to_csv(output_filename, index=False)

# %% [markdown]
# # `label_1`: Speaker ID


# %%
# Split X and y
X_train_label_1 = train_data.iloc[:, :-4]
y_train_label_1 = train_data['label_1']
X_valid_label_1 = valid_data.iloc[:, :-4]
y_valid_label_1 = valid_data['label_1']
X_test_label_1 = test_data.iloc[:, 1:]

# %%
# Plot class distribution
plot_class_distribution(y_train_label_1)

# %%
# Robust scaling
X_train_label_1, X_valid_label_1, X_test_label_1 = robust_scale_data(
    X_train_label_1, X_valid_label_1, X_test_label_1)

# %%
X_train_label_1.head()

# %%
# Apply PCA
X_train_trf_label_1, X_valid_trf_label_1, X_test_trf_label_1 = pca(
    X_train_label_1, X_valid_label_1, X_test_label_1)

# %%
X_train_trf_label_1.shape

# %%
param_grid = {'C': [100, 10, 1, 0.1],
              'kernel': ['rbf', 'linear'],
              }

# Support Vector Machine Best Model
best_model_1_svc = svc(X_train_trf_label_1, y_train_label_1,
                       X_valid_trf_label_1, y_valid_label_1, param_grid, 5)

# %%
# Predict using selected Model ('kernel': 'rbf', 'C': 100)

y_pred_label_1 = best_model_1_svc.predict(X_valid_trf_label_1)
score_label_1 = accuracy_score(y_valid_label_1, y_pred_label_1)
print("Accuracy score {:.3f}".format(score_label_1))

# %%
# Predict test data
y_test_pred_label_1 = best_model_1_svc.predict(X_test_trf_label_1)

# %%
# Create output csv file for label_1
create_output_csv(y_test_pred_label_1, "label_1")

# %% [markdown]
# # `label_2` : Speaker Age

# %%
# Split X and y
X_train_label_2 = train_data.iloc[:, :-4]
y_train_label_2 = train_data['label_2']
X_valid_label_2 = valid_data.iloc[:, :-4]
y_valid_label_2 = valid_data['label_2']
X_test_label_2 = test_data.iloc[:, 1:]

# %%
# Plot class distribution
plot_class_distribution(y_train_label_2)

# %%
# Robust scaling
X_train_label_2, X_valid_label_2, X_test_label_2 = robust_scale_data(
    X_train_label_2, X_valid_label_2, X_test_label_2)

# %%
X_train_label_2.head()

# %%
# SMOTE Oversampling
X_train_resampled_label_2, y_train_resampled_label_2 = smote_oversampling(
    X_train_label_2, y_train_label_2, verbose=1)

# %%
# Apply PCA
X_train_trf_label_2, X_valid_trf_label_2, X_test_trf_label_2 = pca(
    X_train_resampled_label_2, X_valid_label_2, X_test_label_2, n_components=0.96)

# %%
X_train_trf_label_2.shape

# %%
param_grid = {'C': [100, 10, 1, 0.1],
              'kernel': ['rbf']}

# Support Vector Machine Best Model
best_model_2 = svc(X_train_trf_label_2, y_train_resampled_label_2,
                   X_valid_trf_label_2, y_valid_label_2, param_grid=param_grid, n_iter=2, cv=3)

# %%
# Predict using selected Model ('kernel': 'rbf', 'C': 10)
y_pred_label_2 = best_model_2.predict(X_valid_trf_label_2)
score_label_2 = accuracy_score(y_valid_label_2, y_pred_label_2)
print("Accuracy score {:.3f}".format(score_label_2))

# %%
# Predict test data
y_test_pred_label_2 = best_model_2.predict(X_test_trf_label_2)

# %%
# Create output csv file for label_2
create_output_csv(y_test_pred_label_2, "label_2")

# %% [markdown]
# # `label_3` : Speaker Gender

# %%
# Split X and y
X_train_label_3 = train_data.iloc[:, :-4]
y_train_label_3 = train_data['label_3']
X_valid_label_3 = valid_data.iloc[:, :-4]
y_valid_label_3 = valid_data['label_3']
X_test_label_3 = test_data.iloc[:, 1:]

# %%
# Plot class distribution
plot_class_distribution(y_train_label_3)

# %%
# Robust scaling
X_train_label_3, X_valid_label_3, X_test_label_3 = robust_scale_data(
    X_train_label_3, X_valid_label_3, X_test_label_3)

# %%
# SMOTE Oversampling
X_train_resampled_label_3, y_train_resampled_label_3 = smote_oversampling(
    X_train_label_3, y_train_label_3, verbose=1)

# %%
# Apply PCA
X_train_trf_label_3, X_valid_trf_label_3, X_test_trf_label_3 = pca(
    X_train_resampled_label_3, X_valid_label_3, X_test_label_3, n_components=0.95)

# %%
X_train_trf_label_3.shape

# %%
param_grid = {'C': [100, 10, 1, 0.1],
              'kernel': ['rbf'],
              }

# Support Vector Machine Best Model
best_model_3 = svc(X_train_trf_label_3, y_train_resampled_label_3,
                   X_valid_trf_label_3, y_valid_label_3, param_grid=param_grid, n_iter=2, cv=3)

# %%
# KNN Classifier Model
knn_label_3 = knn_model(
    X_train_trf_label_3, y_train_resampled_label_3, X_valid_trf_label_3, y_valid_label_3)

# Cross validation score
cross_validation_score(knn_label_3, X_train_trf_label_3,
                       y_train_resampled_label_3)

# Accuracy
y_pred_label_3_knn = knn_label_3.predict(X_valid_trf_label_3)
score_label_3_knn = accuracy_score(y_valid_label_3, y_pred_label_3_knn)
print("Accuracy score {:.3f}".format(score_label_3_knn))

# %%
# Support Vector Model
svc_label_3 = svc_model(X_train_trf_label_3,
                        y_train_resampled_label_3, X_valid_trf_label_3, y_valid_label_3)

# Cross validation score
cross_validation_score(
    svc_label_3, X_train_trf_label_3, y_train_resampled_label_3)

# Accuracy
y_pred_label_3_svc = knn_label_3.predict(X_valid_trf_label_3)
score_label_3_svc = accuracy_score(y_valid_label_3, y_pred_label_3_svc)
print("Accuracy score {:.3f}".format(score_label_3_svc))

# %%
# Predict using selected Model ('kernel': 'rbf, 'C': 10)

y_pred_label_3 = best_model_3.predict(X_valid_trf_label_3)
score_label_3 = accuracy_score(y_valid_label_3, y_pred_label_3)
print("Accuracy score {:.3f}".format(score_label_3))

# %%
# Predict test data
y_test_pred_label_3 = best_model_3.predict(X_test_trf_label_3)

# %%
# Create output csv file for label_3
create_output_csv(y_test_pred_label_3, "label_3")

# %% [markdown]
# # `label_4` : Speaker Accent

# %%
# Split X and y
X_train_label_4 = train_data.iloc[:, :-4]
y_train_label_4 = train_data['label_4']
X_valid_label_4 = valid_data.iloc[:, :-4]
y_valid_label_4 = valid_data['label_4']
X_test_label_4 = test_data.iloc[:, 1:]

# %%
# Plot class distribution
plot_class_distribution(y_train_label_4)

# %%
# Robust scaling
X_train_label_4, X_valid_label_4, X_test_label_4 = robust_scale_data(
    X_train_label_4, X_valid_label_4, X_test_label_4)

# %%
# SMOTE Oversampling
X_train_resampled_label_4, y_train_resampled_label_4 = smote_oversampling(
    X_train_label_4, y_train_label_4, verbose=1)

# %%
# Apply PCA
X_train_trf_label_4, X_valid_trf_label_4, X_test_trf_label_4 = pca(
    X_train_resampled_label_4, X_valid_label_4, X_test_label_4, n_components=0.97)

# %%
X_train_trf_label_4.shape

# %%
param_grid = {'C': [100, 10],
              'kernel': ['rbf']}

# Support Vector Machine Best Model
best_model_4 = svc(X_train_trf_label_4, y_train_resampled_label_4,
                   X_valid_trf_label_4, y_valid_label_4, param_grid=param_grid, n_iter=2, cv=3)

# %%
# Train selected Model ('kernel': 'rbf', 'C': 100)

y_pred_label_4 = best_model_4.predict(X_valid_trf_label_4)
score_label_4 = accuracy_score(y_valid_label_4, y_pred_label_4)
print("Accuracy score {:.3f}".format(score_label_4))

# %%
# Predict test data
y_test_pred_label_4 = best_model_4.predict(X_test_trf_label_4)

# %%
# Create output csv file for label_4
create_output_csv(y_test_pred_label_4, "label_4")

# %%
output_filename = f"Results/final.csv"
final_combined_data = pd.DataFrame()
final_combined_data["ID"] = test_data["ID"]
for i in range(1, 5):
    label_name = f"label_{i}"
    final_combined_data[label_name] = pd.read_csv(
        f"Results/{label_name}.csv")
final_combined_data.to_csv(output_filename, index=False)
