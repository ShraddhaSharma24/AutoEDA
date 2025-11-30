from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np




def build_preprocessing_pipeline(df, numeric_strategy='median', categorical_strategy='most_frequent'):
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()


numeric_transformer = Pipeline(steps=[
('imputer', SimpleImputer(strategy=numeric_strategy)),
('scaler', StandardScaler())
])


categorical_transformer = Pipeline(steps=[
('imputer', SimpleImputer(strategy=categorical_strategy)),
('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
])


preprocessor = ColumnTransformer(transformers=[
('num', numeric_transformer, num_cols),
('cat', categorical_transformer, cat_cols)
], remainder='drop')


return preprocessor