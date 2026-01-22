from __future__ import annotations

from typing import List, Optional

import os
from difflib import get_close_matches
import pandas as pd

from loaders.load_validate import (
 read_input_file,
 _collect_bad_uno_rows,
 _is_missing_cell,
 _is_missing_number,
 _pad_to_length,
 _report_and_skip_bad_rows,
 convert_date_columns,
 convert_numeric_columns,
 find_matching_column,
 forward_fill_rows,
 normalize_marketboards_and_types,
 resolve_ambiguous_columns,
 validate_column_lengths,
 validate_required_identifiers,
)


def _get_available_marketboards() -> tuple[str, ...]:
	raw = os.environ.get("MARKETBOARDS", "").strip()
	if not raw:
		return ("MOEX",)
	parts = [p.strip() for p in raw.replace(";", ",").replace(" ", ",").split(",")]
	parts = [p for p in parts if p]
	return tuple(dict.fromkeys(parts)) or ("MOEX",)


def _canonicalize_marketboard(value) -> tuple[str | None, str | None]:
	if _is_missing_cell(value):
		return None, None
	s = str(value).strip()
	if not s:
		return None, None

	trans = str.maketrans({
	 "М": "M", "м": "M",
	 "О": "O", "о": "O",
	 "Е": "E", "е": "E",
	 "Х": "X", "х": "X",
	})
	norm = s.translate(trans)
	norm = "".join(ch for ch in norm if ch.isalnum()).upper()
	if not norm:
		return None, None

	allowed = tuple(str(b).strip().upper() for b in _get_available_marketboards())
	allowed_norm = []
	for b in allowed:
		b_norm = b.translate(trans)
		b_norm = "".join(ch for ch in b_norm if ch.isalnum()).upper()
		if b_norm:
			allowed_norm.append(b_norm)


	for b, b_norm in zip(allowed, allowed_norm):
		if norm == b_norm:
			return b, None


	match = get_close_matches(norm, allowed_norm, n=1, cutoff=0.75)
	if match:
		idx = allowed_norm.index(match[0])
		return allowed[idx], None

	return None, s


def _apply_marketboard_rules_uno(data: dict, *, interactive: bool) -> dict:
	if not isinstance(data, dict):
		return data

	list_cols = {k: v for k, v in data.items() if not str(k).startswith('_') and isinstance(v, list)}
	if not list_cols:
		return data
	lengths = {k: len(v) for k, v in list_cols.items()}
	if len(set(lengths.values())) > 1:
		raise ValueError(f"Cannot validate boards: columns have different lengths: {lengths}")
	n = next(iter(lengths.values()))

	boards = data.get('boards') if isinstance(data.get('boards'), list) else None
	original_has_board = False
	if isinstance(boards, list) and boards:
		for b in boards:
			if not _is_missing_cell(b) and str(b).strip() != "":
				original_has_board = True
				break
	if boards is None:
		boards = [None] * n

	invalid_found = set()
	keep_idx = []
	canon_boards = [None] * n
	missing_idx = []
	for i, b in enumerate(boards):
		canon, invalid = _canonicalize_marketboard(b)
		if invalid is not None:
			invalid_found.add(invalid)
			continue
		keep_idx.append(i)
		canon_boards[i] = canon
		if canon is None:
			missing_idx.append(i)


	if invalid_found:
		bad_list = ", ".join(sorted(invalid_found))
		allowed_list = ", ".join(_get_available_marketboards())
		print(
		 f"\n⚠ Unavailable marketboards detected in ONES: {bad_list}. "
		 f"Allowed: {allowed_list}. Those rows will be excluded."
		)
		out = dict(data)
		for k, v in list_cols.items():
			out[k] = [v[i] for i in keep_idx]
		out['_original_has_board'] = original_has_board
		data = out

		return _apply_marketboard_rules_uno(data, interactive=interactive)


	if missing_idx:
		allowed_list = ", ".join(_get_available_marketboards())
		if not interactive:
			raise RuntimeError(
			 f"Missing marketboard values in ONES ({len(missing_idx)} rows) and non-interactive mode. "
			 f"Allowed: {allowed_list}."
			)
		print(
		 f"\nMarketboard is missing in {len(missing_idx)} row(s) for ONES. "
		 f"Available marketboards: {allowed_list}."
		)
		while True:
			ans = input("Enter marketboard to use for missing values: ").strip()
			if not ans:
				print(f"Marketboard is required. Allowed: {allowed_list}.")
				continue
			canon, invalid = _canonicalize_marketboard(ans)
			if invalid is None and canon is not None:
				fill = canon
				break
			print(f"Invalid marketboard '{ans}'. Allowed: {allowed_list}.")


		boards_out = data.get('boards') if isinstance(data.get('boards'), list) else [None] * n
		boards_out = list(boards_out) if isinstance(boards_out, list) else [None] * n
		for i in range(n):
			canon, invalid = _canonicalize_marketboard(boards_out[i] if i < len(boards_out) else None)
			if invalid is None and canon is None:
				boards_out[i] = fill
		data = dict(data)
		data['boards'] = boards_out
	else:

		boards_out = []
		for b in boards:
			canon, _ = _canonicalize_marketboard(b)
			boards_out.append(canon)
		data = dict(data)
		data['boards'] = boards_out

	data['_original_has_board'] = original_has_board
	return data


def _column_aliases_uno() -> dict:
	return {
	 'boards': ['board', 'mboards', 'mboard', 'market_boards', 'market_board', 'marketboards', 'marketboard', 'exchange', 'exchanges', 'market', 'markets'],
	 'type': ['mtype', 'market_type', 'markettype', 'trade_type', 'tradetype', 'type', 'security_type', 'securitytype', 'sectype', 'asset_type', 'assettype', 'instrument_type', 'instrumenttype', 'asset', 'instrument', 'security'],
	 'tickers': ['ticker', 'symbol', 'symbols', 'stock', 'stocks', 'isin', 'name', 'names', 'asset_name', 'assetname', 'instrument_name', 'instrumentname', 'security_name', 'securityname', 'code', 'codes', 'id', 'ids'],
	 'date_trade': ['date', 'datetime', 'date_time', 'timestamp', 'dt', 'time', 'trade_date', 'trade_datetime', 'execution_date', 'execution_datetime', 'deal_date', 'deal_datetime'],
	 'volume_signed': ['volume', 'vol', 'qty', 'quantity', 'amount', 'size', 'signed_volume', 'volume_signed', 'directional_volume', 'direction', 'side_volume'],
	 'price_trade': ['price', 'trade_price', 'execution_price', 'deal_price', 'rate', 'cost', 'value', 'price_trade'],
	}


def _validate_uno_trade_events(data: dict) -> None:
	volume = data.get('volume_signed', [])
	dates = data.get('date_trade', [])
	prices = data.get('price_trade', [])

	n = max(len(volume), len(dates), len(prices))
	volume = _pad_to_length(volume, n)
	dates = _pad_to_length(dates, n)
	prices = _pad_to_length(prices, n)

	problems: List[str] = []
	for idx in range(n):
		v = volume[idx]
		if _is_missing_number(v):
			continue
		try:
			v_num = float(v)
		except Exception:
			problems.append(f"Row {idx + 1}: volume_signed='{v}' is not a number")
			continue
		if v_num == 0.0:
			continue

		dt = dates[idx]
		pr = prices[idx]

		has_date = (dt is not None) and (not isinstance(dt, str))
		has_price = (not _is_missing_number(pr)) and (not isinstance(pr, str))

		if has_date and has_price:
			continue

		missing = []
		if not has_date:
			missing.append("date_trade")
		if not has_price:
			missing.append("price_trade")
		problems.append(
		 f"Row {idx + 1}: volume_signed={v_num:g} requires {', '.join(missing)}"
		)

	if problems:
		preview = problems[:10]
		more = len(problems) - len(preview)
		msg = "ONES trade events validation failed.\n" + "\n".join(preview)
		if more > 0:
			msg += f"\n... and {more} more row(s)."
		raise ValueError(msg)




def _sort_uno_rows(data: dict) -> dict:
	visible_lists = [
	 v for k, v in data.items()
	 if not str(k).startswith('_') and isinstance(v, list) and len(v) > 0
	]
	if not visible_lists:
		return data
	row_count = max(len(v) for v in visible_lists)
	if row_count == 0:
		return data


	sortable_keys = [
	 k for k, v in data.items()
	 if not str(k).startswith('_') and isinstance(v, list) and len(v) == row_count
	]
	if not sortable_keys:
		return data

	values = {k: data[k] for k in sortable_keys}
	df = pd.DataFrame(values)
	before = len(df)
	if before == 0:
		return data


	sort_cols = [c for c in ['boards', 'type', 'tickers', 'date_trade'] if c in df.columns]
	if sort_cols:
		df = df.sort_values(by=sort_cols, kind='mergesort', na_position='last')

	after = len(df)
	if after != before:
		raise ValueError(f"UNO sort changed row count: {before} -> {after}")

	out = dict(data)
	for c in sortable_keys:
		out[c] = df[c].tolist()
	return out


def _drop_bad_rows_uno(data: dict, bad_indices: list) -> dict:
	if not bad_indices:
		return data

	out = {}
	for k, v in data.items():
		if isinstance(v, list):
			out[k] = [v[i] for i in range(len(v)) if i not in bad_indices]
		else:
			out[k] = v
	return out


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
		if df is None:
			df = read_input_file(file_path, sheet_name=sheet_name)
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

		target_columns = ['boards', 'type', 'tickers', 'date_trade', 'volume_signed', 'price_trade']
		optional_columns = ['commission']
		all_columns = target_columns + optional_columns
		column_aliases = _column_aliases_uno()


		column_mapping = {}
		manual = custom_column_names if isinstance(custom_column_names, dict) else {}
		if manual:
			print(f"\nApplying custom column mapping...")
			for target_col in all_columns:
				chosen = manual.get(target_col)
				if not chosen:
					continue
				if chosen in df.columns:
					column_mapping[target_col] = chosen
					print(f"  {target_col}: ✓ → '{chosen}'")
				else:
					print(f"  {target_col}: ✗ Column '{chosen}' not found in file")


		required_for_uno = ['type', 'tickers', 'date_trade', 'volume_signed', 'price_trade']
		missing_required = [c for c in required_for_uno if c not in column_mapping]
		if missing_required:
			print(f"\nMatching columns for missing fields: {', '.join(missing_required)}")
			auto_map = find_matching_column(df, missing_required, aliases=column_aliases, interactive=interactive)
			auto_map = resolve_ambiguous_columns(df, auto_map, missing_required, aliases=column_aliases, interactive=interactive)
			column_mapping.update(auto_map)


		missing_optional = [c for c in optional_columns if c not in column_mapping]
		if missing_optional:
			auto_opt = find_matching_column(df, missing_optional, aliases=column_aliases, interactive=False)

			column_mapping.update(auto_opt)

		data_raw = {}
		for target_col in target_columns:
			if target_col in column_mapping:
				actual_col = column_mapping[target_col]
				raw_list = df[actual_col].tolist()
				data_raw[target_col] = [None if pd.isna(v) or (isinstance(v, str) and not v.strip()) else v for v in raw_list]
			else:

				if target_col == 'boards':
					continue
				data_raw[target_col] = []



		for target_col in optional_columns:
			if target_col in column_mapping:
				actual_col = column_mapping[target_col]
				raw_list = df[actual_col].tolist()
				data_raw[target_col] = [None if pd.isna(v) or (isinstance(v, str) and not v.strip()) else v for v in raw_list]

		print(f"\nValidating column lengths...")
		validate_column_lengths(data_raw)


		row_count = max((len(v) for k, v in data_raw.items() if isinstance(v, list) and v and not str(k).startswith('_')), default=int(len(df)))
		present_optional = [c for c in optional_columns if c in data_raw and isinstance(data_raw[c], list)]
		for col in present_optional:
			if len(data_raw[col]) == 0:
				data_raw[col] = [None] * row_count
			elif len(data_raw[col]) != row_count:
				data_raw[col] = (data_raw[col] + [None] * row_count)[:row_count]


		data_raw = forward_fill_rows(data_raw, ['type', 'tickers'])
		data_raw = _apply_marketboard_rules_uno(data_raw, interactive=interactive)
		data_raw = normalize_marketboards_and_types(data_raw, interactive=interactive)
		validate_required_identifiers(data_raw)

		num_cols = ['volume_signed', 'price_trade'] + (["commission"] if 'commission' in data_raw else [])
		data_raw = convert_numeric_columns(data_raw, num_cols, manual_format=number_format, interactive=interactive)
		data_raw = convert_date_columns(data_raw, ['date_trade'])


		print(f"\nChecking for problematic rows...")
		bad_rows_result = _collect_bad_uno_rows(data_raw)
		if bad_rows_result and bad_rows_result.get('problems'):
			_report_and_skip_bad_rows(bad_rows_result, trade_type="UNO")

			data_raw = _drop_bad_rows_uno(data_raw, bad_rows_result['bad_row_indices'])
			print(f"✓ Skipped {len(bad_rows_result['bad_row_indices'])} problematic row(s)")

		print(f"\nSorting data...")
		data = _sort_uno_rows(data_raw)

		input_rows = int(len(df))
		visible_values = [v for k, v in data.items() if not str(k).startswith('_') and isinstance(v, list)]
		output_rows = max((len(v) for v in visible_values), default=0)

		if output_rows > input_rows:
			raise ValueError(f"UNO loader row count increased unexpectedly: input {input_rows}, output {output_rows}")
		if output_rows < input_rows:
			skipped = input_rows - output_rows
			print(f"\n⚠ Skipped {skipped} row(s) during normalization/validation")
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


def display_data(data):
	if not data:
		print("No data to display")
		return

	visible_values = [v for k, v in data.items() if not str(k).startswith('_') and v]
	max_len = max(len(v) for v in visible_values) if visible_values else 0

	print(f"\n{'='*80}")
	print(f"DATA SUMMARY - Total rows: {max_len}")
	print(f"{'='*80}")
	print("Note: UNO loader keeps event rows; only sorting may change order.")

	for key, values in data.items():
		if str(key).startswith('_') or not values:
			continue
		filled = [v for v in values if not _is_missing_cell(v)]
		sample = filled[0] if filled else None
		print(f"  {key}: rows={len(values)}/{max_len}, filled={len(filled)}/{max_len}, sample: {sample}")

	print(f"\n✓ Data loaded and validated successfully!")

