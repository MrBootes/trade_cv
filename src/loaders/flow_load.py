from __future__ import annotations

from typing import List, Optional

import os
import pandas as pd
from difflib import get_close_matches

from loaders.load_validate import (
 read_input_file,
 _is_missing_cell,
 _is_missing_number,
 _pad_to_length,
 convert_date_columns,
 convert_numeric_columns,
 validate_column_lengths,
)


def _column_aliases_flow() -> dict:
	return {
	 'date': [
	  'date', 'datetime', 'date_time', 'timestamp', 'dt', 'time',
	  'flow_date', 'cash_date', 'payment_date', 'transaction_date', 'txn_date',
	  'дата', 'время', 'времядаты', 'дата_время',
	 ],
	 'cash': [
	  'cash', 'money', 'amount', 'sum', 'value',
	  'cashflow', 'cash_flow', 'flow', 'net_flow',
	  'deposit', 'withdrawal', 'in', 'out', 'income', 'expense',
	  'руб', 'rur', 'rub', 'inflow', 'outflow', 'funds', 'usd', 'eur',
	  'payment', 'transaction', 'txn', '$', '€', '₽', '₽руб', 'рубль',
	  'деньги', 'сумма', 'dollar', 'euro', 'ruble', 'price',
	  'rate', 'cost', 'value', 'amount_cash', 'amountcash', 'cash_amount', 'cashamount',
	  'total_cash', 'totalcash', 'cash_total', 'cashtotal', 'cash_value', 'cashvalue',
	           'value_cash', 'valuecash', 'cash_cost', 'cashcost', 'cost_cash', 'costcash', 'rez',
	 ],
	}


def _norm_name(s: str) -> str:
	s = str(s or "").strip().lower()

	out = []
	for ch in s:
		if ch.isalnum():
			out.append(ch)
	return "".join(out)


def _find_candidates(df: pd.DataFrame, target: str, aliases: dict) -> List[str]:
	if df is None or not isinstance(df, pd.DataFrame):
		return []
	all_aliases = [target] + list(aliases.get(target, []) or [])
	wanted = [_norm_name(a) for a in all_aliases if str(a or "").strip()]
	wanted = [w for w in wanted if w]
	if not wanted:
		return []

	col_norm = {col: _norm_name(col) for col in list(df.columns)}
	res: List[str] = []

	for col, n in col_norm.items():
		if n in wanted:
			res.append(col)

	for col, n in col_norm.items():
		if col in res:
			continue
		for w in wanted:
			if len(w) < 4:
				if n.startswith(w) and n[len(w):].isdigit():
					res.append(col)
					break
				continue
			if w in n or n in w:
				res.append(col)
				break

	try:
		keys = list({v for v in col_norm.values() if v})
		for w in wanted:
			if len(w) < 5:
				continue
			matches = get_close_matches(w, keys, n=5, cutoff=0.82)
			for m in matches:
				for col, n in col_norm.items():
					if n == m and col not in res:
						res.append(col)
	except Exception:
		pass

	return res


def _choose_columns(target: str, candidates: List[str], *, interactive: bool) -> List[str]:
	if not candidates:
		return []
	seen = set()
	out: List[str] = []
	for c in candidates:
		if c in seen:
			continue
		seen.add(c)
		out.append(c)
	return out


def _validate_flow_rows(data: dict) -> None:
	_validate_flow_pairs(data, interactive=False)


def _validate_flow_pairs(data: dict, *, interactive: bool) -> None:
	dates = _pad_to_length(data.get('date', []), 0)
	cash = _pad_to_length(data.get('cash', []), 0)
	n = max(len(dates), len(cash))
	dates = _pad_to_length(dates, n)
	cash = _pad_to_length(cash, n)

	def _has_valid_date(v) -> bool:
		if _is_missing_cell(v):
			return False
		return (v is not None) and (not isinstance(v, str))

	def _has_valid_cash(v) -> bool:
		if _is_missing_number(v):
			return False
		return (v is not None) and (not isinstance(v, str))

	problems: List[str] = []
	for idx in range(n):
		dt = dates[idx]
		c = cash[idx]
		has_date = _has_valid_date(dt)
		has_cash = _has_valid_cash(c)


		if (not has_date) and (not has_cash):
			continue
		if has_date and (not has_cash):
			problems.append(f"Row {idx + 1}: date is present but cash is missing")
		elif has_cash and (not has_date):
			problems.append(f"Row {idx + 1}: cash is present but date is missing")

	if not problems:
		return

	preview = problems[:10]
	more = len(problems) - len(preview)
	msg = (
	 "FLOW data has incomplete rows (DATE without CASH or CASH without DATE).\n"
	 + "\n".join(preview)
	)
	if more > 0:
		msg += f"\n... and {more} more row(s)."

	if not interactive:
		raise ValueError(msg)

	print("\n" + msg)
	ans = input("\nUse incomplete FLOW data anyway? (y/n): ").strip().lower()
	if ans != 'y':
		raise ValueError("FLOW aborted: incomplete data was not approved")


def _validate_flow_has_any_complete_row(data: dict, *, interactive: bool) -> None:
	dates = _pad_to_length(data.get('date', []), 0)
	cash = _pad_to_length(data.get('cash', []), 0)
	n = max(len(dates), len(cash))
	dates = _pad_to_length(dates, n)
	cash = _pad_to_length(cash, n)

	def _has_valid_date(v) -> bool:
		if _is_missing_cell(v):
			return False
		return (v is not None) and (not isinstance(v, str))

	def _has_valid_cash(v) -> bool:
		if _is_missing_number(v):
			return False
		return (v is not None) and (not isinstance(v, str))

	complete = 0
	for i in range(n):
		if _has_valid_date(dates[i]) and _has_valid_cash(cash[i]):
			complete += 1
			break

	if complete > 0:
		return

	msg = "FLOW contains 0 complete rows (no row has BOTH date and cash)."
	if not interactive:
		raise ValueError(msg)
	print("\n" + msg)
	ans = input("Proceed anyway (this may save an all-empty file)? (y/n): ").strip().lower()
	if ans != 'y':
		raise ValueError("FLOW aborted: no complete rows")


def _validate_date_cash_lengths(data: dict) -> None:
	dates = data.get('date', [])
	cash = data.get('cash', [])
	len_dates = len(dates) if isinstance(dates, list) else 0
	len_cash = len(cash) if isinstance(cash, list) else 0
	if len_dates != len_cash:
		raise ValueError(
		 "ERROR: Length mismatch between 'date' and 'cash' columns. "
		 f"date rows={len_dates}, cash rows={len_cash}. "
		 "Please ensure both columns have the same number of rows."
		)


def _sort_flow_rows(data: dict) -> dict:
	cols = ['date', 'cash']
	values = {}
	for c in cols:
		v = data.get(c, [])
		values[c] = v if isinstance(v, list) else []

	if not any(values[c] for c in cols):
		return data

	df = pd.DataFrame(values)
	before = len(df)
	if before == 0:
		return data

	if 'date' in df.columns:
		df = df.sort_values(by=['date'], kind='mergesort', na_position='last')

	after = len(df)
	if after != before:
		raise ValueError(f"FLOW sort changed row count: {before} -> {after}")

	out = dict(data)
	for c in cols:
		out[c] = df[c].tolist() if c in df.columns else out.get(c, [])
	return out


def display_data(data):
	if not data:
		print("No data to display")
		return

	visible_values = [v for k, v in data.items() if not str(k).startswith('_') and v]
	max_len = max(len(v) for v in visible_values) if visible_values else 0

	print(f"\n{'='*80}")
	print(f"DATA SUMMARY - Total rows: {max_len}")
	print(f"{'='*80}")
	print("Note: FLOW loader may concatenate repeated blocks; sorting may change order.")

	for key, values in data.items():
		if str(key).startswith('_') or not values:
			continue
		non_null = [v for v in values if not _is_missing_cell(v)]
		sample = non_null[0] if non_null else None
		print(f"  {key}: {len(non_null)}/{max_len} values, sample: {sample}")

	print(f"\n✓ Data loaded and validated successfully!")


def read_trade_data(
 file_path: str,
 number_format: Optional[dict] = None,
 custom_column_names: Optional[dict] = None,
 sheet_name: Optional[str] = None,
 *,
 df: Optional[pd.DataFrame] = None,
 interactive: bool = True,
):
	try:
		input_rows = None
		concatenate_blocks = False
		if df is None:
			df = read_input_file(file_path, sheet_name=sheet_name, header_row=0)
			header_row_used = 0
			file_extension = os.path.splitext(file_path)[1].lower()
			if file_extension in ['.xlsx', '.xls']:
				if sheet_name is None:
					print(f"Successfully read Excel file: {file_path}")
				else:
					print(f"Successfully read Excel file: {file_path} (sheet: {sheet_name})")
			elif file_extension == '.csv':
				print(f"Successfully read CSV file: {file_path}")
			else:
				print(f"Successfully read file: {file_path}")

		target_columns = ['date', 'cash']
		# Header may be on the 2nd row. Only retry if 1st-row header matches nothing.
		if df is not None and file_path and not custom_column_names:
			column_aliases_probe = _column_aliases_flow()
			date_cands_1 = _find_candidates(df, 'date', column_aliases_probe)
			cash_cands_1 = _find_candidates(df, 'cash', column_aliases_probe)
			if not date_cands_1 and not cash_cands_1:
				try:
					df2 = read_input_file(file_path, sheet_name=sheet_name, header_row=1)
				except Exception:
					df2 = None
				if isinstance(df2, pd.DataFrame):
					date_cands_2 = _find_candidates(df2, 'date', column_aliases_probe)
					cash_cands_2 = _find_candidates(df2, 'cash', column_aliases_probe)
					if date_cands_2 or cash_cands_2:
						print("\nNo columns matched in row 1 header; using row 2 as header (data starts from row 3).")
						df = df2
						header_row_used = 1
		else:
			header_row_used = None
		if custom_column_names:
			print(f"\nUsing custom column names...")

			col_sel: dict[str, List[str]] = {}
			for target_col in target_columns:
				wanted = custom_column_names.get(target_col)
				if not wanted:
					col_sel[target_col] = []
					continue
				if isinstance(wanted, (list, tuple)):
					w_list = [str(x).strip() for x in wanted if str(x).strip()]
				else:
					w_list = [str(wanted).strip()]

				w_list = [c for c in w_list if c in df.columns]
				col_sel[target_col] = w_list
				if w_list:
					print(f"  {target_col}: ✓ → {w_list}")
				else:
					print(f"  {target_col}: ✗ No provided columns found in file")
		else:
			column_aliases = _column_aliases_flow()
			print(f"\nMatching columns...")
			col_sel = {}
			for target_col in target_columns:
				cands = _find_candidates(df, target_col, column_aliases)
				picked = _choose_columns(target_col, cands, interactive=interactive)
				col_sel[target_col] = picked
				if picked:
					print(f"  {target_col}: ✓ → {picked}")
				else:
					print(f"  {target_col}: ✗ not found")

			# If nothing matched at all and we're on header row 1, retry with header row 2.
			if (
				not col_sel.get('date')
				and not col_sel.get('cash')
				and header_row_used == 0
				and file_path
			):
				try:
					df2 = read_input_file(file_path, sheet_name=sheet_name, header_row=1)
				except Exception:
					df2 = None
				if isinstance(df2, pd.DataFrame):
					print("\nNo columns matched in row 1 header; retrying with row 2 as header (data starts from row 3).")
					df = df2
					col_sel = {}
					for target_col in target_columns:
						cands = _find_candidates(df, target_col, column_aliases)
						picked = _choose_columns(target_col, cands, interactive=interactive)
						col_sel[target_col] = picked
						if picked:
							print(f"  {target_col}: ✓ → {picked}")
						else:
							print(f"  {target_col}: ✗ not found")


		data_raw = {}
		input_rows = int(len(df))
		data_raw['date'] = []
		data_raw['cash'] = []

		date_cols = list(col_sel.get('date', []) or [])
		cash_cols = list(col_sel.get('cash', []) or [])

		def _suffix_key(col_name: str) -> str:
			name = str(col_name)
			if '.' in name:
				base, tail = name.rsplit('.', 1)
				if tail.isdigit():
					return tail
			return ''

		date_by_suffix = {_suffix_key(c): c for c in date_cols}
		cash_by_suffix = {_suffix_key(c): c for c in cash_cols}
		pair_suffixes = [s for s in date_by_suffix.keys() if s in cash_by_suffix]
		pair_suffixes.sort(key=lambda s: (-1 if s == '' else int(s)))
		if '' in pair_suffixes:
			pair_suffixes = [''] + [s for s in pair_suffixes if s != '']

		concatenate_blocks = len(pair_suffixes) > 1

		print(f"\nValidating column lengths...")

		if concatenate_blocks:
			for suf in pair_suffixes:
				d_col = date_by_suffix.get(suf)
				c_col = cash_by_suffix.get(suf)
				if not d_col or not c_col:
					continue
				tmp = {
					'_date': [None if pd.isna(v) or (isinstance(v, str) and not v.strip()) else v for v in df[d_col].tolist()],
					'_cash': [None if pd.isna(v) or (isinstance(v, str) and not v.strip()) else v for v in df[c_col].tolist()],
				}
				validate_column_lengths(tmp)
				tmp = convert_numeric_columns(tmp, ['_cash'], manual_format=number_format, interactive=interactive)
				tmp = convert_date_columns(tmp, ['_date'])

				dates = _pad_to_length(tmp.get('_date', []), 0)
				cash = _pad_to_length(tmp.get('_cash', []), 0)
				n = max(len(dates), len(cash))
				dates = _pad_to_length(dates, n)
				cash = _pad_to_length(cash, n)
				for i in range(n):
					dt = dates[i]
					cv = cash[i]
					if _is_missing_cell(dt) and _is_missing_number(cv):
						continue
					data_raw['date'].append(dt)
					data_raw['cash'].append(cv)

		if not data_raw['date'] and not data_raw['cash']:
			data_raw = {}
			data_raw['date'] = [None] * input_rows
			data_raw['cash'] = [None] * input_rows
			for j, col in enumerate(date_cols):
				key = f"_date__{j}"
				raw_list = df[col].tolist()
				data_raw[key] = [None if pd.isna(v) or (isinstance(v, str) and not v.strip()) else v for v in raw_list]
			for j, col in enumerate(cash_cols):
				key = f"_cash__{j}"
				raw_list = df[col].tolist()
				data_raw[key] = [None if pd.isna(v) or (isinstance(v, str) and not v.strip()) else v for v in raw_list]

			validate_column_lengths(data_raw)
			cash_keys = [k for k in data_raw.keys() if k.startswith('_cash__')]
			date_keys = [k for k in data_raw.keys() if k.startswith('_date__')]
			if cash_keys:
				data_raw = convert_numeric_columns(data_raw, cash_keys, manual_format=number_format, interactive=interactive)
			if date_keys:
				data_raw = convert_date_columns(data_raw, date_keys)

			merged_date: List[object] = [None] * input_rows
			for i in range(input_rows):
				val = None
				for k in date_keys:
					v = data_raw.get(k, [None] * input_rows)[i]
					if v is None or isinstance(v, str):
						continue
					val = v
					break
				merged_date[i] = val

			merged_cash: List[object] = [None] * input_rows
			for i in range(input_rows):
				total = 0.0
				has_any = False
				for k in cash_keys:
					v = data_raw.get(k, [None] * input_rows)[i]
					if _is_missing_number(v) or isinstance(v, str):
						continue
					try:
						total += float(v)
						has_any = True
					except Exception:
						continue
				merged_cash[i] = total if has_any else None

			data_raw['date'] = merged_date
			data_raw['cash'] = merged_cash
			for k in list(data_raw.keys()):
				if k.startswith('_date__') or k.startswith('_cash__'):
					data_raw.pop(k, None)

		_validate_date_cash_lengths(data_raw)


		_validate_flow_pairs(data_raw, interactive=interactive)

		_validate_flow_has_any_complete_row(data_raw, interactive=interactive)

		print(f"\nSorting data...")
		data = _sort_flow_rows(data_raw)

		visible_values = [v for k, v in data.items() if not str(k).startswith('_') and isinstance(v, list)]
		output_rows = max((len(v) for v in visible_values), default=0)
		if output_rows != input_rows and input_rows is not None and not concatenate_blocks:
			raise ValueError(f"FLOW loader must preserve row count: input {input_rows}, output {output_rows}")
		data['_input_rows'] = input_rows
		data['_output_rows'] = output_rows
		print(f"\n✓ Loaded {input_rows} input row(s); produced {output_rows} output row(s)")
		return data

	except FileNotFoundError:
		print(f"Error: File not found - {file_path}")
		return None
	except ValueError:
		raise
	except Exception as e:
		print(f"Error reading file: {str(e)}")
		return None

