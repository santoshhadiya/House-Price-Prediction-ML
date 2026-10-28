import sys
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('house_price_cleaned.csv')
features = ['City', 'Locality_Tier', 'BHK', 'Bathrooms', 'Super_Area_sqft', 'Carpet_Area_sqft',
            'Floor_No', 'Total_Floors', 'Property_Age_years', 'Parking', 'Furnishing',
            'Lift', 'Gated_Society', 'Distance_to_Metro_km', 'Distance_to_CityCenter_km',
            'Nearby_School_km', 'Nearby_Hospital_km', 'Crime_Rate_Index']
target = 'Market_Price_INR'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset: Total={len(df)}, Train={len(X_train)}, Test={len(X_test)}")

# 1. K-Means
cluster_features = ['Super_Area_sqft', 'BHK', 'Bathrooms', 'Property_Age_years', 'Price_Lakhs', 'Distance_to_CityCenter_km']
X_c = df[cluster_features]
scaler = StandardScaler()
X_c_scaled = scaler.fit_transform(X_c)

inertias = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_c_scaled)
    inertias[k] = round(float(km.inertia_), 1)

print("Inertias:", inertias)

km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = km3.fit_predict(X_c_scaled)
df_clust = df.copy()
df_clust['Cluster'] = clusters

for c in range(3):
    cdf = df_clust[df_clust['Cluster'] == c]
    print(f"Cluster {c}: Count={len(cdf)} ({len(cdf)/len(df)*100:.1f}%), Avg Price={cdf['Price_Lakhs'].mean():.2f}L, Avg Area={cdf['Super_Area_sqft'].mean():.1f} sqft, Avg BHK={cdf['BHK'].mean():.1f}")

# 2. Classification Experiment
def price_category(p):
    if p < 6000000: # < 60L
        return 0 # Budget
    elif p <= 12000000: # 60L - 1.2Cr
        return 1 # Mid-Range
    else:
        return 2 # Premium

y_cat_train = y_train.apply(price_category)
y_cat_test = y_test.apply(price_category)

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = ['City', 'Locality_Tier', 'Furnishing']

prep = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

clf = Pipeline([
    ('prep', prep),
    ('clf', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
])

t0 = time.time()
clf.fit(X_train, y_cat_train)
y_cat_pred = clf.predict(X_test)
y_cat_prob = clf.predict_proba(X_test)

acc = accuracy_score(y_cat_test, y_cat_pred)
prec = precision_score(y_cat_test, y_cat_pred, average='weighted')
rec = recall_score(y_cat_test, y_cat_pred, average='weighted')
f1 = f1_score(y_cat_test, y_cat_pred, average='weighted')
roc = roc_auc_score(y_cat_test, y_cat_prob, multi_class='ovr')
cm = confusion_matrix(y_cat_test, y_cat_pred)

print(f"Classification Experiment: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, ROC_AUC={roc:.4f}")
print("Confusion Matrix:\n", cm)
