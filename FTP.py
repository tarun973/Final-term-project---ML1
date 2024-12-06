import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support, mean_squared_error, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from statsmodels.stats.outliers_influence import variance_inflation_factor
from ydata_profiling import ProfileReport
from wordcloud import WordCloud
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from tabulate import tabulate
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

def log_timestamp(message):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")

log_timestamp("Starting the project execution...")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


#----------------------------------------------------------------------------------
# Reading the data
#__________________________________________________________________________________


airlines = pd.read_csv("C:/Users/tarun/Desktop/US Airline Flight Routes and Fares 1993-2024.csv", low_memory=False)


#----------------------------------------------------------------------------------
# Data Exploration
#__________________________________________________________________________________


print(airlines.head())
print(airlines.info())
print(airlines.describe())

missing_values = airlines.isnull().sum()
print(missing_values)


#----------------------------------------------------------------------------------
# EDA
#__________________________________________________________________________________


text = " ".join(str(city) for city in airlines['city2'])

wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud for Destination City", fontsize=16)
plt.show()

cleaned_airlines = airlines.drop(['tbl', 'citymarketid_1', 'citymarketid_2', 'airportid_1', 'airportid_2', 'tbl1apk',
                                  'city1','city2', 'airport_1','airport_2','carrier_lg','carrier_low','Geocoded_City1','Geocoded_City2'], axis=1)

df_cleaned = cleaned_airlines.dropna(subset=['fare'])

duplicates = df_cleaned.duplicated().sum()
print(f'Duplicates: {duplicates}')

target = 'fare'
categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_columns)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_columns)
    ],
    remainder='passthrough'
)

X = preprocessor.fit_transform(df_cleaned.drop(columns=[target]))
y = df_cleaned[target]

y_cleaned = y[~np.isnan(y).values]

X_cleaned = X[:len(y_cleaned)]

print(f"X shape: {X_cleaned.shape}")
print(f"y shape: {y_cleaned.shape}")

sns.pairplot(df_cleaned[['Year', 'quarter', 'nsmiles', 'fare']], diag_kind='kde')
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()

for feature in ['Year', 'nsmiles', 'quarter']:
    plt.figure(figsize=(8, 4))
    sns.scatterplot(x=df_cleaned[feature], y=df_cleaned['fare'], alpha=0.7)
    plt.title(f"{feature} vs Fare")
    plt.xlabel(feature)
    plt.ylabel("Fare")
    plt.show()


#----------------------------------------------------------------------------------
# PCA
#__________________________________________________________________________________


imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_cleaned)

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_imputed)

pca = PCA()
X_pca = pca.fit_transform(X_standardized)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

num_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\nNumber of features with more than 95% variance: {num_components}")

plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='b')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.axvline(x=num_components, color='g', linestyle='--', label=f'{num_components} components')
plt.xticks(np.arange(1, len(cumulative_variance) + 1))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs Number of Components')
plt.legend()
plt.grid(True)
plt.show()


#----------------------------------------------------------------------------------
#Random Forest
#__________________________________________________________________________________


X = cleaned_airlines.drop(columns=['fare'])
y = cleaned_airlines['fare']

if isinstance(X, np.ndarray):
    feature_names = cleaned_airlines.drop(columns=['fare']).columns
else:
    feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=5805)

rf = RandomForestRegressor(n_estimators=100, random_state=5805)
rf.fit(X_train, y_train)

importances = rf.feature_importances_

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df)

plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Feature Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Top 10 Feature Importances from Random Forest', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()

threshold = 0.02
selected_features_rf = importance_df[importance_df['Importance'] >= threshold]['Feature'].tolist()
eliminated_features_rf = importance_df[importance_df['Importance'] < threshold]['Feature'].tolist()

print("\nEliminated Features based on Random Forest:", eliminated_features_rf)
print("\nFinal Selected Features based on Random Forest:", selected_features_rf)

X_selected = X[selected_features_rf]


#----------------------------------------------------------------------------------
# SVD
#__________________________________________________________________________________


imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

svd = TruncatedSVD(n_components=X_scaled.shape[1] - 1, random_state=5805)
X_svd = svd.fit_transform(X_scaled)

singular_values = svd.singular_values_
explained_variance_ratio = svd.explained_variance_ratio_

cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(singular_values) + 1), singular_values, marker='o', label='Singular Values')
plt.title('Singular Values of Features')
plt.xlabel('Component Number')
plt.ylabel('Singular Value')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', label='Cumulative Variance')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Threshold')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.legend()
plt.show()

n_components_selected = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"Number of components explaining 95% variance: {n_components_selected}")

X_reduced = svd.transform(X_scaled)[:, :n_components_selected]

print("\nExplained Variance Ratio per Component:")
for i, variance in enumerate(explained_variance_ratio[:n_components_selected], start=1):
    print(f"Component {i}: {variance:.4f}")

X_reconstructed = np.dot(X_reduced, svd.components_[:n_components_selected, :])


#----------------------------------------------------------------------------------
# Covariance Matrix
#__________________________________________________________________________________


X = preprocessor.fit_transform(df_cleaned.drop(columns=[target]))

cov_matrix = pd.DataFrame(X).cov()

sns.heatmap(cov_matrix, cmap='coolwarm', annot=True)
plt.title('Covariance Matrix Heatmap')
plt.show()


#----------------------------------------------------------------------------------
# Correlation Matrix
#__________________________________________________________________________________


X = preprocessor.fit_transform(df_cleaned.drop(columns=[target]))

X_df = pd.DataFrame(X)

correlation_matrix = X_df.corr(method='pearson')

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Pearson Correlation Coefficient Matrix')
plt.show()


#----------------------------------------------------------------------------------
# Outlier/Anomaly Detection
#__________________________________________________________________________________


# z_scores = np.abs(zscore(X))
# threshold = 3
# outliers = (z_scores > threshold).any(axis=1)
#
# X_cleaned = X[~outliers]
#
# print(f"Number of outliers detected: {np.sum(outliers)}")
# print(f"Shape of data after outlier removal: {X_cleaned.shape}")
#
# sns.countplot(data=df_cleaned, x='fare')
# plt.title('Distribution of Target Variable')
# plt.show()


scaler = StandardScaler()
df_new_scaled = scaler.fit_transform(cleaned_airlines)

eps = 0.5
min_samples = 5

imputer = SimpleImputer(strategy='mean')
cleaned_imputed = imputer.fit_transform(df_new_scaled)

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
cluster_labels = dbscan.fit_predict(cleaned_imputed)

outliers = cluster_labels == -1

outlier_cleaned = cleaned_airlines[~outliers]

plt.figure(figsize=(12, 6))
sns.histplot(data=outlier_cleaned, x='fare', kde=True, bins=50)
plt.title('Distribution of Target Variable After Outlier Removal')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"Number of outliers detected: {np.sum(outliers)}")
print(f"Shape of data after outlier removal: {outlier_cleaned.shape}")

#----------------------------------------------------------------------------------
# Binning and Resampling
#__________________________________________________________________________________


X = df_cleaned.drop('fare', axis=1)
y = df_cleaned['fare']

num_bins = 5
y_binned, bins = pd.qcut(y, q=num_bins, labels=False, retbins=True)

data_binned = pd.concat([X, y_binned.rename('binned_target')], axis=1)

balanced_data = []
min_count = data_binned['binned_target'].value_counts().min()

for bin_label in range(num_bins):
    bin_data = data_binned[data_binned['binned_target'] == bin_label]
    if len(bin_data) > min_count:
        bin_data_resampled = resample(bin_data, replace=False, n_samples=min_count, random_state=5805)
    else:
        bin_data_resampled = resample(bin_data, replace=True, n_samples=min_count, random_state=5805)
    balanced_data.append(bin_data_resampled)

balanced_data = pd.concat(balanced_data)

X_resampled = balanced_data.drop(columns=['binned_target'])
y_resampled = balanced_data['binned_target']

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=5805)

print("Data balanced and ready for modeling!")


#----------------------------------------------------------------------------------
# VIF
#__________________________________________________________________________________


X_train_const = sm.add_constant(X_train)
X_train_const = pd.DataFrame(X_train_const).dropna()

vif_data = pd.DataFrame()
vif_data["Feature"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_const.values, i) for i in range(1, X_train_const.shape[1])]

print(vif_data)


#----------------------------------------------------------------------------------
# Pandas Profiling Report
#__________________________________________________________________________________


df_eda = df_cleaned
profile = ProfileReport(df_eda, title="Pandas Profiling Report", explorative=True)
profile.to_file("pandas_profiling_report.html")


#----------------------------------------------------------------------------------
# Phase 2
#__________________________________________________________________________________

#----------------------------------------------------------------------------------
# Multiple Linear Regression
#__________________________________________________________________________________


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)

X_train_clean = X_train.copy()
X_train_clean[np.isinf(X_train_clean)] = np.nan
valid_indices = ~np.isnan(X_train_clean).any(axis=1)
X_train_clean = X_train_clean[valid_indices]
y_train_clean = y_train[valid_indices]

X_train_clean_const = sm.add_constant(X_train_clean)
model = sm.OLS(y_train_clean, X_train_clean_const).fit()

X_test_clean = X_test[~np.isnan(X_test).any(axis=1)]
y_test_clean = y_test[~np.isnan(X_test).any(axis=1)]

y_train_pred = model.predict(sm.add_constant(X_train_clean))
y_test_pred = model.predict(sm.add_constant(X_test_clean))

plt.figure(figsize=(10, 6))
plt.scatter(y_train_clean, y_train_pred, label="Train Actual vs Predicted", color='blue', alpha=0.5)
plt.scatter(y_test_clean, y_test_pred, label="Test Actual vs Predicted", color='green', alpha=0.5)
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', lw=2, label="Perfect Fit Line")
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.title('Actual vs Predicted Values (Train and Test)')
plt.show()

r_squared = model.rsquared
adj_r_squared = model.rsquared_adj
aic = model.aic
bic = model.bic
mse_train = mean_squared_error(y_train_clean, y_train_pred)
mse_test = mean_squared_error(y_test_clean, y_test_pred)

metrics_table = pd.DataFrame({
    'Metric': ['R-squared', 'Adjusted R-squared', 'AIC', 'BIC', 'MSE (Train)', 'MSE (Test)'],
    'Value': [r_squared, adj_r_squared, aic, bic, mse_train, mse_test]
})
print(metrics_table)


#----------------------------------------------------------------------------------
# F-Test
#__________________________________________________________________________________


print(f"\nF-statistic: {model.fvalue}, p-value: {model.f_pvalue}")
print(model.conf_int(alpha=0.05))


#----------------------------------------------------------------------------------
# Backward Stepwise Regression and T-test analysis
#__________________________________________________________________________________


def backward_elimination(X, y, significance_level=0.0005):

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    X_with_const = sm.add_constant(X)

    feature_names = ['Intercept'] + X.columns.tolist()

    X_clean = X_with_const.values
    y_clean = y.values if isinstance(y, pd.Series) else y

    mask = ~np.isnan(X_clean).any(axis=1) & ~np.isinf(X_clean).any(axis=1) & ~np.isnan(y_clean) & ~np.isinf(y_clean)
    X_clean = X_clean[mask]
    y_clean = y_clean[mask]

    if X_clean.shape[0] == 0 or X_clean.shape[1] == 0:
        raise ValueError("No valid data available after cleaning")

    model = sm.OLS(y_clean, X_clean).fit()

    eliminated_features = []

    while True:
        p_values = model.pvalues[1:]
        if len(p_values) == 0:
            print("No predictors left after elimination.")
            break

        max_p_value = max(p_values)
        if max_p_value > significance_level:
            excluded_feature_index = np.argmax(p_values)
            excluded_feature_name = feature_names[excluded_feature_index + 1]

            X_clean = np.delete(X_clean, excluded_feature_index + 1, axis=1)
            eliminated_features.append(feature_names.pop(excluded_feature_index + 1))

            print(f"Excluding feature: {excluded_feature_name} with p-value: {max_p_value}")

            model = sm.OLS(y_clean, X_clean).fit()
        else:
            break

    remaining_features = feature_names[1:]
    print("Final Adjusted R-squared:", model.rsquared_adj)
    print("Remaining features after backward elimination:", remaining_features)
    print("Excluded features during backward elimination:", eliminated_features)

    return model, remaining_features

final_model, remaining_features = backward_elimination(X_train, y_train, significance_level=0.0005)

print(final_model.summary())





#----------------------------------------------------------------------------------
# Phase 3
#__________________________________________________________________________________


le = LabelEncoder()
y = le.fit_transform(y)
print(f"Encoded values in y: {np.unique(y)}")

if isinstance(X, np.ndarray):
    X = pd.DataFrame(X)

if isinstance(y, np.ndarray):
    y = pd.Series(y)

data = pd.concat([X, y], axis=1)
data.dropna(inplace=True)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print(f"Length of X after removing NaNs: {len(X)}")
print(f"Length of y after removing NaNs: {len(y)}")

bins = [0, 50, 100, 150]
labels = ['Low', 'Medium', 'High']
y = pd.cut(y, bins=bins, labels=labels)

print("NaNs in y after binning:", y.isnull().sum())

y = y.dropna()
X = X.loc[y.index]

print(f"Length of X after cleaning: {len(X)}")
print(f"Length of y after cleaning: {len(y)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=5805, stratify=None)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=5805)

results = []


def evaluate_model(y_true, y_pred, model_name, y_prob=None, model=None):
    cm = confusion_matrix(y_true, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm[0, 0] + cm[0, 1] != 0 else 0

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    roc_auc = None
    if y_prob is not None:
        y_true_binarized = label_binarize(y_true, classes=np.unique(y_train))
        n_classes = y_true_binarized.shape[1]

        fpr, tpr, _ = roc_curve(y_true_binarized.ravel(), y_prob.ravel())
        roc_auc = roc_auc_score(y_true_binarized, y_prob, average='micro', multi_class='ovr')

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f"Micro-average ROC Curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {model_name}")
        plt.legend()
        plt.grid(True)
        plt.show()

    print(f"\n{model_name} Metrics:")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score: {fscore:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")

    results.append({
        'Model': model_name,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1-Score': fscore,
        'Train Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy,
        'ROC AUC': roc_auc if roc_auc is not None else "N/A"
    })

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()


def display_results_table(results):
    results_df = pd.DataFrame(results)

    table = tabulate(results_df, headers='keys', tablefmt='grid', showindex=False, numalign="right")
    print(table)

display_results_table(results)


#----------------------------------------------------------------------------------
# Decision Tree
#__________________________________________________________________________________


def decision_tree_classifier():
    param_grid_pre_pruning = {
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                 'max_depth': [None, 10, 20, 30],
                 'min_samples_split': [2, 5, 10],
                 'max_features': [None, 'sqrt', 'log2'],
                 'ccp_alpha': [0.0, 0.01, 0.1]
             }

    dt_pre_pruned = GridSearchCV(DecisionTreeClassifier(random_state=5805), param_grid_pre_pruning, cv=cv, scoring='accuracy')
    dt_pre_pruned.fit(X_train, y_train)
    print(f"Best Parameters for Pre-Pruned Decision Tree: {dt_pre_pruned.best_params_}")
    y_pred_pre = dt_pre_pruned.predict(X_test)
    y_prob_pre = dt_pre_pruned.best_estimator_.predict_proba(X_test)
    evaluate_model(y_test, y_pred_pre, "Pre-Pruned Decision Tree", y_prob=y_prob_pre, model=dt_pre_pruned.best_estimator_)

    dt_full = DecisionTreeClassifier(random_state=5805)
    dt_full.fit(X_train, y_train)

    path = dt_full.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas

    trees = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=5805, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        trees.append(clf)

    accuracies = [accuracy_score(y_test, clf.predict(X_test)) for clf in trees]
    best_alpha_index = accuracies.index(max(accuracies))
    best_tree = trees[best_alpha_index]

    print(f"Best alpha for Post-Pruned Decision Tree: {ccp_alphas[best_alpha_index]}")
    y_pred_post = best_tree.predict(X_test)
    y_prob_post = best_tree.predict_proba(X_test)
    evaluate_model(y_test, y_pred_post, "Post-Pruned Decision Tree", y_prob=y_prob_post, model=best_tree)


#----------------------------------------------------------------------------------
# Logistic Regression
#__________________________________________________________________________________


def logistic_regression_classifier():
    param_grid = {
        'penalty': ['l2', 'none'],
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'saga'],
        'max_iter': [1000]
    }
    lr = GridSearchCV(LogisticRegression(random_state=5805), param_grid, cv=cv, scoring='accuracy')
    lr.fit(X_train, y_train)
    print(f"Best Parameters for Logistic Regression: {lr.best_params_}")
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)
    evaluate_model(y_test, y_pred, "Logistic Regression", y_prob=y_prob, model=lr.best_estimator_)


#----------------------------------------------------------------------------------
# KNN
#__________________________________________________________________________________


def knn_classifier():
    print("\n--- Finding Optimal K using Elbow Method ---")
    k_values = range(1, 21)
    accuracies = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    plt.figure(figsize=(8, 6))
    plt.plot(k_values, accuracies, marker='o', linestyle='--', color='b')
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Neighbors (K)")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.xticks(k_values)
    plt.show()

    optimal_k = k_values[np.argmax(accuracies)]
    print(f"Optimal K based on Elbow Method: {optimal_k}")

    print("\n--- Performing GridSearchCV for KNN ---")
    param_grid = {
        'n_neighbors': list(range(1, 21)),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2]
    }
    knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv, scoring='accuracy')
    knn.fit(X_train, y_train)
    print(f"Best Parameters for K-Nearest Neighbors: {knn.best_params_}")

    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)
    evaluate_model(y_test, y_pred, "K-Nearest Neighbors", y_prob=y_prob, model=knn.best_estimator_)


#----------------------------------------------------------------------------------
# Support Vector Machine
#__________________________________________________________________________________


def svm_classifier():
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'poly', 'rbf']
    }
    svm = GridSearchCV(SVC(probability=True, random_state=5805), param_grid, cv=cv, scoring='accuracy')
    svm.fit(X_train, y_train)
    print(f"Best Parameters for SVM: {svm.best_params_}")
    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)
    evaluate_model(y_test, y_pred, "SVM", y_prob=y_prob, model=svm.best_estimator_)


#----------------------------------------------------------------------------------
# Naive Bayes
#__________________________________________________________________________________


def naive_bayes_classifier():
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
    }
    nb = GridSearchCV(GaussianNB(), param_grid, cv=cv, scoring='accuracy')
    nb.fit(X_train, y_train)
    print(f"Best Parameters for Naive Bayes: {nb.best_params_}")
    y_pred = nb.predict(X_test)
    y_prob = nb.predict_proba(X_test)
    evaluate_model(y_test, y_pred, "Naive Bayes", y_prob=y_prob, model=nb.best_estimator_)


#----------------------------------------------------------------------------------
# Random Forest
#__________________________________________________________________________________


def random_forest_classifier():
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'max_features': [None, 'sqrt', 'log2']
    }
    rf = GridSearchCV(RandomForestClassifier(random_state=5805), param_grid_rf, cv=cv, scoring='accuracy')
    rf.fit(X_train, y_train)
    print(f"Best Parameters for Random Forest: {rf.best_params_}")
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)
    evaluate_model(y_test, y_pred_rf, "Random Forest (Bagging)", y_prob=y_prob_rf, model=rf.best_estimator_)

    param_grid_ada = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    }
    ada = GridSearchCV(AdaBoostClassifier(random_state=5805), param_grid_ada, cv=cv, scoring='accuracy')
    ada.fit(X_train, y_train)
    print(f"Best Parameters for AdaBoost: {ada.best_params_}")
    y_pred_ada = ada.predict(X_test)
    y_prob_ada = ada.predict_proba(X_test)
    evaluate_model(y_test, y_pred_ada, "AdaBoost (Boosting)", y_prob=y_prob_ada, model=ada.best_estimator_)

    param_grid_gboost = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    gboost = GridSearchCV(GradientBoostingClassifier(random_state=5805), param_grid_gboost, cv=cv, scoring='accuracy')
    gboost.fit(X_train, y_train)
    print(f"Best Parameters for Gradient Boosting: {gboost.best_params_}")
    y_pred_gboost = gboost.predict(X_test)
    y_prob_gboost = gboost.predict_proba(X_test)
    evaluate_model(y_test, y_pred_gboost, "Gradient Boosting (Boosting)", y_prob=y_prob_gboost, model=gboost.best_estimator_)

    param_grid_stack = {
        'rf__n_estimators': [50, 100],
        'dt__max_depth': [3, 5],
        'nb__var_smoothing': [1e-9, 1e-8]
    }
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=5805)),
        ('dt', DecisionTreeClassifier(max_depth=5, random_state=5805)),
        ('nb', GaussianNB())
    ]
    stack = GridSearchCV(StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=42)),
                         param_grid_stack, cv=cv, scoring='accuracy')
    stack.fit(X_train, y_train)
    print(f"Best Parameters for Stacking Classifier: {stack.best_params_}")
    y_pred_stack = stack.predict(X_test)
    y_prob_stack = stack.predict_proba(X_test)
    evaluate_model(y_test, y_pred_stack, "Stacking Classifier", y_prob=y_prob_stack, model=stack.best_estimator_)


#----------------------------------------------------------------------------------
# Neural Networks
#__________________________________________________________________________________


def neural_network_classifier():
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh', 'logistic'],
        'max_iter': [200, 500]
    }
    mlp = GridSearchCV(MLPClassifier(random_state=5805), param_grid, cv=cv, scoring='accuracy')
    mlp.fit(X_train, y_train)
    print(f"Best Parameters for Neural Network: {mlp.best_params_}")
    y_pred = mlp.predict(X_test)
    y_prob = mlp.best_estimator_.predict_proba(X_test)
    evaluate_model(y_test, y_pred, "Neural Network", y_prob=y_prob, model=mlp.best_estimator_)


decision_tree_classifier()
logistic_regression_classifier()
knn_classifier()
svm_classifier()
naive_bayes_classifier()
random_forest_classifier()
neural_network_classifier()

comparison_table = pd.DataFrame(results)
print("\nComparison of Classifiers:")
print(comparison_table)


#----------------------------------------------------------------------------------
# Phase 4
#__________________________________________________________________________________


new_data = cleaned_airlines

clustering_data = new_data.drop(columns=['nsmiles'])

numerical_data = clustering_data.select_dtypes(include=[np.number])

valid_indices = ~numerical_data.isnull().any(axis=1) & ~numerical_data.isin([np.inf, -np.inf]).any(axis=1)
numerical_data = numerical_data[valid_indices]
clustering_data = clustering_data[valid_indices]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)


#----------------------------------------------------------------------------------
# K-Means Clustering
#__________________________________________________________________________________


def kmeans_clustering(scaled_data, original_data):
    print("\n--- K-Means Clustering ---")

    silhouette_scores = []
    wcss = []

    k_values = range(2, 5)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=5805, init='k-means++')
        cluster_labels = kmeans.fit_predict(scaled_data)
        silhouette_scores.append(silhouette_score(scaled_data, cluster_labels))
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.title("Silhouette Analysis for K-Means")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid()
    plt.show()

    optimal_k = k_values[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")

    kmeans = KMeans(n_clusters=optimal_k, random_state=5805, init='k-means++')
    cluster_labels = kmeans.fit_predict(scaled_data)

    clustered_data = original_data.copy()
    clustered_data['Cluster'] = cluster_labels

    plt.figure(figsize=(8, 6))
    plt.plot(k_values, wcss, marker='o')
    plt.title("Within-Cluster Variation Plot (WCSS)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS")
    plt.grid()
    plt.show()

    return clustered_data

clustered_data_kmeans = kmeans_clustering(scaled_data, clustering_data)


#----------------------------------------------------------------------------------
# DBSCAN
#__________________________________________________________________________________


def dbscan_clustering(scaled_data, original_data):
    print("\n--- DBSCAN Clustering ---")

    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(scaled_data)
    distances, indices = neighbors_fit.kneighbors(scaled_data)
    distances = np.sort(distances[:, -1])

    plt.figure(figsize=(8, 6))
    plt.plot(distances)
    plt.title("k-Distance Graph for DBSCAN")
    plt.xlabel("Data Points")
    plt.ylabel("Distance to 5th Nearest Neighbor")
    plt.grid()
    plt.show()

    dbscan = DBSCAN(eps=1, min_samples=5, metric='euclidean')
    cluster_labels = dbscan.fit_predict(scaled_data)

    print(f"Number of clusters formed: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")

    clustered_data = original_data.copy()

    clustered_data['Cluster'] = cluster_labels

    clustered_data = clustered_data[clustered_data['Cluster'] != -1]

    clustered_data.reset_index(drop=True, inplace=True)

    return clustered_data

clustered_data_dbscan = dbscan_clustering(scaled_data, clustering_data)


#----------------------------------------------------------------------------------
# Association Rule Mining
#__________________________________________________________________________________


def association_rule_mining(input_data, min_support=0.001, min_confidence=0.05):
    print("\n--- Association Rule Mining using Apriori Algorithm ---")

    data = input_data.copy()

    data['fare_binned'] = pd.cut(
        data['fare'], bins=4, labels=['low', 'medium-low', 'medium-high', 'high']
    )
    data['passengers_binned'] = pd.cut(
        data['passengers'], bins=4, labels=['low', 'medium-low', 'medium-high', 'high']
    )

    categorical_columns = ['quarter', 'fare_binned', 'passengers_binned']
    categorical_data = data[categorical_columns]

    one_hot_encoded_data = pd.get_dummies(categorical_data, drop_first=True)
    one_hot_encoded_data = one_hot_encoded_data.applymap(lambda x: 1 if x > 0 else 0)

    frequent_itemsets = apriori(one_hot_encoded_data, min_support=min_support, use_colnames=True)
    print("\nFrequent Itemsets:")
    print(frequent_itemsets)

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    if not rules.empty:
        print("\nAssociation Rules:")

        rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage', 'conviction']] \
            .sort_values(by='confidence', ascending=False)

        print("\nAssociation Rules sorted by Confidence:")
        print(rules.to_string(index=False))

    else:
        print("No association rules found")

    return frequent_itemsets, rules

frequent_itemsets, rules = association_rule_mining(input_data=new_data, min_support=0.001, min_confidence=0.05)