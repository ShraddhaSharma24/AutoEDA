import pandas as pd
import numpy as np
from scipy import stats




def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
df = df.copy()
df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
return df




def handle_missing(df: pd.DataFrame, numeric_strategy='median', categorical_strategy='mode') -> pd.DataFrame:
df = df.copy()
for col in df.columns:
if pd.api.types.is_numeric_dtype(df[col]):
if numeric_strategy == 'median':
df[col].fillna(df[col].median(), inplace=True)
elif numeric_strategy == 'mean':
df[col].fillna(df[col].mean(), inplace=True)
else:
df[col].fillna(0, inplace=True)
else:
if categorical_strategy == 'mode':
try:
df[col].fillna(df[col].mode().iloc[0], inplace=True)
except Exception:
df[col].fillna('missing', inplace=True)
else:
df[col].fillna('missing', inplace=True)
return df




def remove_outliers_zscore(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
df = df.copy()
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) == 0:
return df
z = np.abs(stats.zscore(df[numeric_cols].fillna(0)))
mask = (z < z_thresh).all(axis=1)
return df[mask]




def clean_data_pipeline(df: pd.DataFrame, drop_duplicates=True, normalize_cols=True,
missing_num='median', missing_cat='mode', remove_outliers=True):
df = df.copy()
if drop_duplicates:
df = df.drop_duplicates()
if normalize_cols:
df = normalize_column_names(df)
df = handle_missing(df, numeric_strategy=missing_num, categorical_strategy=missing_cat)
if remove_outliers:
df = remove_outliers_zscore(df)
return df