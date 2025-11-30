import pandas as pd




def detect_imbalance(df: pd.DataFrame, target_col: str, threshold: float = 0.7) -> dict:
"""Detects class imbalance. threshold is the fraction of majority class to mark imbalance (e.g., 0.7 means >70% in one class)."""
if target_col not in df.columns:
raise ValueError('target_col not in dataframe')
counts = df[target_col].value_counts(dropna=False)
pct = (counts / counts.sum()).round(4).to_dict()
majority_frac = counts.max() / counts.sum()
is_imbalance = bool(majority_frac >= threshold)


recommendation = []
if is_imbalance:
recommendation = [
'Consider oversampling minority classes (SMOTE, ADASYN)',
'Consider undersampling majority class (RandomUnderSampler)',
'Use class_weight parameter in models that support it',
'Use stratified sampling for train/test splits'
]
else:
recommendation = ['Dataset class balance looks acceptable.']


return {
'class_counts': counts.to_dict(),
'class_pct': pct,
'majority_fraction': float(majority_frac),
'is_imbalanced': is_imbalance,
'recommendations': recommendation
}