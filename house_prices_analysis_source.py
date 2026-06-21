"""
House Prices Regression Analysis
Baseline model vs Feature-Engineered model
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 110

# ---------------------------------------------------------------
# PART 1: LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv('house_prices.csv')
print("Shape:", df.shape)
print(df.info())
print(df.describe().T)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('Id')
numeric_cols.remove('SalePrice')
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("\nNumerical features:", numeric_cols)
print("Categorical features:", categorical_cols)
print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])

# ---------------------------------------------------------------
# EDA PLOTS
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df['SalePrice'], kde=True, ax=axes[0], color='steelblue')
axes[0].set_title('SalePrice Distribution (raw)')
sns.histplot(np.log1p(df['SalePrice']), kde=True, ax=axes[1], color='darkorange')
axes[1].set_title('SalePrice Distribution (log1p)')
plt.tight_layout()
plt.savefig('eda_saleprice_dist.png')
plt.close()

corr = df[numeric_cols + ['SalePrice']].corr()
plt.figure(figsize=(11, 9))
sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
plt.title('Correlation Heatmap (numeric features)')
plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png')
plt.close()

top_corr = corr['SalePrice'].abs().sort_values(ascending=False).drop('SalePrice').head(8)
print("\nTop correlated numeric features with SalePrice:\n", top_corr)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.flat, ['GrLivArea', 'OverallQual', 'TotalBsmtSF', 'LotArea']):
    sns.scatterplot(x=df[col], y=df['SalePrice'], ax=ax, alpha=0.4)
    ax.set_title(f'SalePrice vs {col}')
plt.tight_layout()
plt.savefig('eda_scatter_relationships.png')
plt.close()

plt.figure(figsize=(6, 4))
sns.boxplot(y=df['LotArea'], color='lightcoral')
plt.title('LotArea Outlier Check (boxplot)')
plt.tight_layout()
plt.savefig('eda_outlier_boxplot.png')
plt.close()

# ---------------------------------------------------------------
# PART 2: BASELINE MODEL (NO FEATURE ENGINEERING)
# ---------------------------------------------------------------
X = df.drop(columns=['Id', 'SalePrice'])
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

baseline_preprocess = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='median'), num_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_features)
])

baseline_model = Pipeline([
    ('prep', baseline_preprocess),
    ('reg', LinearRegression())
])
baseline_model.fit(X_train, y_train)
pred_base = baseline_model.predict(X_test)


def get_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mse,
        'RMSE': np.sqrt(mse)
    }


baseline_metrics = get_metrics(y_test, pred_base)
print("\n=== BASELINE MODEL (Linear Regression, no FE) ===")
for k, v in baseline_metrics.items():
    print(f"{k}: {v:,.4f}")

# ---------------------------------------------------------------
# PART 3: FEATURE ENGINEERING
# ---------------------------------------------------------------
fe_df = df.copy()

# 3.1 Domain feature creation
fe_df['HouseAge'] = 2023 - fe_df['YearBuilt']
fe_df['RemodAge'] = 2023 - fe_df['YearRemodAdd']
fe_df['TotalSF'] = fe_df['TotalBsmtSF'] + fe_df['1stFlrSF'] + fe_df['2ndFlrSF']
fe_df['TotalBath'] = fe_df['FullBath'] + 0.5 * fe_df['HalfBath']
fe_df['TotalPorchSF'] = fe_df['WoodDeckSF'] + fe_df['OpenPorchSF']
fe_df['HasPool'] = (fe_df['PoolArea'] > 0).astype(int)
fe_df['HasFireplace'] = (fe_df['Fireplaces'] > 0).astype(int)
fe_df['HasGarage'] = (fe_df['GarageCars'] > 0).astype(int)
fe_df['IsRemodeled'] = (fe_df['YearBuilt'] != fe_df['YearRemodAdd']).astype(int)

# 3.2 Interaction features
fe_df['Qual_x_TotalSF'] = fe_df['OverallQual'] * fe_df['TotalSF']
fe_df['Qual_x_GrLivArea'] = fe_df['OverallQual'] * fe_df['GrLivArea']

# 3.3 Ordinal encode quality-scale categoricals (preserves order info, unlike OHE)
qual_map = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1}
fe_df['ExterQual_enc'] = fe_df['ExterQual'].map(qual_map)
fe_df['KitchenQual_enc'] = fe_df['KitchenQual'].map(qual_map)

# 3.4 Handle skewed numeric distributions with log1p
skewed_candidates = ['LotArea', 'TotalSF', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF']
skewness = fe_df[skewed_candidates].skew()
print("\nSkewness before transform:\n", skewness)
log_replaced = []
for col in skewed_candidates:
    if abs(fe_df[col].skew()) > 0.75:
        fe_df[col + '_log'] = np.log1p(fe_df[col])
        log_replaced.append(col)   # only this raw column gets dropped later

# Target log-transform (SalePrice is right-skewed; standard, proven technique on this dataset)
fe_df['SalePrice_log'] = np.log1p(fe_df['SalePrice'])

# Drop raw columns that were superseded by an engineered version. Columns
# with low skew keep their raw (untransformed) form so no information is lost.
drop_raw = ['YearBuilt', 'YearRemodAdd', 'FullBath', 'HalfBath',
            'WoodDeckSF', 'OpenPorchSF', 'ExterQual', 'KitchenQual'] + log_replaced
fe_df_model = fe_df.drop(columns=['Id', 'SalePrice'] + drop_raw)

y_fe = fe_df_model.pop('SalePrice_log')
X_fe = fe_df_model

X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(
    X_fe, y_fe, test_size=0.2, random_state=42
)

num_features_fe = X_fe.select_dtypes(include=[np.number]).columns.tolist()
cat_features_fe = X_fe.select_dtypes(include=['object']).columns.tolist()

# 3.5 Feature selection: drop near-zero-variance / low-correlation numeric noise
fe_preprocess = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_features_fe),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_features_fe)
])

# Same algorithm (Linear Regression) as the baseline, so the comparison
# isolates the effect of feature engineering rather than the effect of
# switching models.
fe_model = Pipeline([
    ('prep', fe_preprocess),
    ('reg', LinearRegression())
])
fe_model.fit(X_train_fe, y_train_fe)

pred_fe_log = fe_model.predict(X_test_fe)
# convert back from log space to compare against baseline on the same scale
pred_fe = np.expm1(pred_fe_log)
y_test_fe_actual = np.expm1(y_test_fe)

fe_metrics = get_metrics(y_test_fe_actual, pred_fe)
print("\n=== FEATURE-ENGINEERED MODEL (Gradient Boosting + FE) ===")
for k, v in fe_metrics.items():
    print(f"{k}: {v:,.4f}")

# ---------------------------------------------------------------
# PART 4: COMPARISON
# ---------------------------------------------------------------
comparison = pd.DataFrame({
    'Without Feature Engineering': baseline_metrics,
    'With Feature Engineering': fe_metrics
}).T
comparison['R2'] = comparison['R2'].round(4)
comparison['MAE'] = comparison['MAE'].round(2)
comparison['MSE'] = comparison['MSE'].round(2)
comparison['RMSE'] = comparison['RMSE'].round(2)
print("\n=== COMPARISON TABLE ===")
print(comparison)
comparison.to_csv('comparison_results.csv')

# prediction quality plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(y_test, pred_base, alpha=0.4, color='steelblue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_title(f"Baseline: Actual vs Predicted (R2={baseline_metrics['R2']:.3f})")
axes[0].set_xlabel('Actual SalePrice'); axes[0].set_ylabel('Predicted SalePrice')

axes[1].scatter(y_test_fe_actual, pred_fe, alpha=0.4, color='darkorange')
axes[1].plot([y_test_fe_actual.min(), y_test_fe_actual.max()],
             [y_test_fe_actual.min(), y_test_fe_actual.max()], 'r--')
axes[1].set_title(f"Feature-Engineered: Actual vs Predicted (R2={fe_metrics['R2']:.3f})")
axes[1].set_xlabel('Actual SalePrice'); axes[1].set_ylabel('Predicted SalePrice')
plt.tight_layout()
plt.savefig('comparison_actual_vs_predicted.png')
plt.close()

# Feature importance via standardized linear coefficients (explainability)
ohe_cols = fe_model.named_steps['prep'].named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(cat_features_fe)
all_feature_names = num_features_fe + list(ohe_cols)
coefs = fe_model.named_steps['reg'].coef_
fi = pd.Series(np.abs(coefs), index=all_feature_names).sort_values(ascending=False).head(15)

plt.figure(figsize=(8, 6))
fi.sort_values().plot(kind='barh', color='seagreen')
plt.title('Top 15 |Standardized Coefficients| (Feature-Engineered Linear Model)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("\nTop |coefficients| (feature importance proxy):\n", fi)

metrics_bar = comparison[['R2','MAE','MSE','RMSE']].copy()
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col in zip(axes, ['R2','MAE','MSE','RMSE']):
    metrics_bar[col].plot(kind='bar', ax=ax, color=['steelblue','darkorange'])
    ax.set_title(col)
    ax.set_xticklabels(['No FE','With FE'], rotation=0)
plt.tight_layout()
plt.savefig('metrics_comparison_bars.png')
plt.close()

print("\nDone. All plots and comparison_results.csv saved.")

# ---------------------------------------------------------------
# BONUS: Compare multiple algorithms on the engineered feature set
# + light hyperparameter tuning for the best non-linear model
# ---------------------------------------------------------------
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

algos = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=10),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=200),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

bonus_results = {}
for name, algo in algos.items():
    pipe = Pipeline([('prep', fe_preprocess), ('reg', algo)])
    pipe.fit(X_train_fe, y_train_fe)
    pred_log = pipe.predict(X_test_fe)
    pred = np.expm1(pred_log)
    bonus_results[name] = get_metrics(np.expm1(y_test_fe), pred)

bonus_df = pd.DataFrame(bonus_results).T.round(3)
print("\n=== BONUS: Algorithm comparison (all on engineered features) ===")
print(bonus_df)
bonus_df.to_csv('bonus_algorithm_comparison.csv')

# Hyperparameter tuning example: Gradient Boosting via GridSearchCV
gb_pipe = Pipeline([('prep', fe_preprocess), ('reg', GradientBoostingRegressor(random_state=42))])
param_grid = {
    'reg__n_estimators': [100, 200],
    'reg__max_depth': [2, 3],
    'reg__learning_rate': [0.03, 0.1]
}
grid = GridSearchCV(gb_pipe, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid.fit(X_train_fe, y_train_fe)
print("\nBest GB params:", grid.best_params_)
best_pred = np.expm1(grid.predict(X_test_fe))
tuned_metrics = get_metrics(np.expm1(y_test_fe), best_pred)
print("Tuned Gradient Boosting metrics:", tuned_metrics)

plt.figure(figsize=(8, 5))
bonus_df['R2'].plot(kind='bar', color='teal')
plt.title('R2 Score by Algorithm (Engineered Features)')
plt.ylabel('R2 Score')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('bonus_algorithm_r2.png')
plt.close()

print("\nAll bonus outputs saved.")
