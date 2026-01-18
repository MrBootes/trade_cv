"""load_file.py

Shared file reader used by all trade loaders.

Responsibilities:
- Read Excel/CSV into a pandas DataFrame
- Keep file I/O separate from validation/parsing logic
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd


def _detect_and_split_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""Detect if dataframe needs column splitting (single column with separators)."""
	if df is None or not isinstance(df, pd.DataFrame):
		return df

	if len(df.columns) != 1:
		return df

	first_col = df.columns[0]
	sample_value = str(df[first_col].iloc[0]) if len(df) > 0 else ""

	for sep in [';', '\t', '|', ',']:
		if sep in sample_value or sep in str(first_col):
			print(f"  Detected separator '{sep}' in data, re-parsing...")
			from io import StringIO
			csv_string = df.to_csv(index=False, sep=',')
			try:
				new_df = pd.read_csv(StringIO(csv_string.replace(',', sep)), sep=sep)
				if len(new_df.columns) > 1:
					return new_df
			except Exception:
				pass

	return df


def read_input_file(file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
	"""Read an input file (Excel/CSV) into a DataFrame.

	Supports:
	- .xlsx/.xls via pandas.read_excel
	- .csv via pandas.read_csv with auto separator detection
	"""
	if not file_path or not isinstance(file_path, str):
		raise ValueError("file_path must be a non-empty string")

	file_extension = os.path.splitext(file_path)[1].lower()

	if file_extension in {".xlsx", ".xls"}:
		if sheet_name is None:
			df = pd.read_excel(file_path)
		else:
			df = pd.read_excel(file_path, sheet_name=sheet_name)
		return _detect_and_split_columns(df)

	if file_extension == ".csv":
		return pd.read_csv(file_path, sep=None, engine="python")

	raise ValueError(
		f"Unsupported file format: {file_extension}. Use .xlsx, .xls, or .csv"
	)

