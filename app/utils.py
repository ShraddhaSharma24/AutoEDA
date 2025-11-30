import pandas as pd
import os
from pathlib import Path


ALLOWED_EXTS = {'.csv','.xlsx','.xls','.parquet','.feather','.json'}




def load_dataset_from_upload(upload_file) -> pd.DataFrame:
"""Load dataset from a FastAPI UploadFile or a local path-like object."""
# upload_file can be a SpooledTemporaryFile or file-like object with .read()
# We try to infer format from filename when available
filename = getattr(upload_file, 'filename', None)
if filename:
ext = Path(filename).suffix.lower()
else:
ext = '.csv' # default fallback


if ext == '.csv':
return pd.read_csv(upload_file.file if hasattr(upload_file, 'file') else upload_file)
elif ext in ('.xlsx', '.xls'):
return pd.read_excel(upload_file.file if hasattr(upload_file, 'file') else upload_file)
elif ext == '.parquet':
return pd.read_parquet(upload_file.file if hasattr(upload_file, 'file') else upload_file)
elif ext == '.json':
return pd.read_json(upload_file.file if hasattr(upload_file, 'file') else upload_file)
else:
# try csv read as fallback
try:
return pd.read_csv(upload_file.file if hasattr(upload_file, 'file') else upload_file)
except Exception as e:
raise ValueError(f"Unsupported file type {ext} and fallback CSV failed: {e}")




def safe_mkdir(path: str):
os.makedirs(path, exist_ok=True)