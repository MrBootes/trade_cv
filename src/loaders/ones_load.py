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
 normalize_column_name,
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
	 'price_trade': ['price', 'trade_price', 'execution_price', 'deal_price', 'rate', 'price_trade', 'unit_price', 'price_per_unit'],
	 'cost_trade': ['cost', 'value', 'amount', 'total', 'sum', 'trade_value', 'trade_amount', 'trade_total', 'cost_trade', 'total_cost', 'gross_amount'],
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
			df = read_input_file(file_path, sheet_name=sheet_name, header_row=0)
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

		target_columns = ['boards', 'type', 'tickers', 'date_trade', 'volume_signed', 'price_trade', 'cost_trade']
		optional_columns = ['commission']
		all_columns = target_columns + optional_columns
		column_aliases = _column_aliases_uno()

		def _header_has_any_matches(_df: pd.DataFrame) -> bool:
			try:
				df_norm = {normalize_column_name(c) for c in list(_df.columns)}
			except Exception:
				return False
			wanted = set()
			for t in (target_columns + optional_columns):
				wanted.add(normalize_column_name(t))
				for a in column_aliases.get(t, []):
					wanted.add(normalize_column_name(a))
			return bool(df_norm & wanted)

		# Header may be on the 2nd row. Only retry if 1st-row header matches nothing.
		if df is not None and file_path and not _header_has_any_matches(df):
			try:
				df2 = read_input_file(file_path, sheet_name=sheet_name, header_row=1)
			except Exception:
				df2 = None
			if isinstance(df2, pd.DataFrame) and _header_has_any_matches(df2):
				print("\nNo columns matched in row 1 header; using row 2 as header (data starts from row 3).")
				df = df2


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


		required_core = ['type', 'tickers', 'date_trade', 'volume_signed']
		missing_required = [c for c in required_core if c not in column_mapping]
		if missing_required:
			print(f"\nMatching columns for missing fields: {', '.join(missing_required)}")
			auto_map = find_matching_column(df, missing_required, aliases=column_aliases, interactive=interactive)
			auto_map = resolve_ambiguous_columns(df, auto_map, missing_required, aliases=column_aliases, interactive=interactive)
			column_mapping.update(auto_map)
			column_mapping = resolve_ambiguous_columns(df, column_mapping, all_columns, aliases=column_aliases, interactive=interactive)


		missing_optional = [c for c in optional_columns if c not in column_mapping]
		if missing_optional and not manual:
			auto_opt = find_matching_column(df, missing_optional, aliases=column_aliases, interactive=False)
			column_mapping.update(auto_opt)
			column_mapping = resolve_ambiguous_columns(df, column_mapping, all_columns, aliases=column_aliases, interactive=interactive)

		def _excel_col_letter(idx0: int) -> str:
			idx = int(idx0) + 1
			out = []
			while idx > 0:
				idx, rem = divmod(idx - 1, 26)
				out.append(chr(ord('A') + rem))
			return ''.join(reversed(out))

		def _col_label(col_name: str) -> str:
			try:
				idx0 = list(df.columns).index(col_name)
				return f"{_excel_col_letter(idx0)}: {col_name}"
			except Exception:
				return str(col_name)

		def _sample_values(col_name: str, *, limit: int = 5) -> list:
			try:
				return df[col_name].head(limit).tolist()
			except Exception:
				return []

		def _gather_candidates(target: str) -> list[tuple[str, str]]:
			df_cols_normalized = {normalize_column_name(col): col for col in df.columns}
			normalized_target = normalize_column_name(target)
			candidates: list[tuple[str, str]] = []

			if normalized_target in df_cols_normalized:
				candidates.append((df_cols_normalized[normalized_target], 'exact'))
			for alias in column_aliases.get(target, []):
				norm_alias = normalize_column_name(alias)
				if norm_alias in df_cols_normalized:
					col_name = df_cols_normalized[norm_alias]
					if all(col_name != c[0] for c in candidates):
						candidates.append((col_name, 'alias'))

			possible_names = [normalized_target] + [normalize_column_name(x) for x in column_aliases.get(target, [])]
			for possible in possible_names:
				if len(possible) < 5:
					continue
				matches = get_close_matches(possible, df_cols_normalized.keys(), n=3, cutoff=0.75)
				for m in matches:
					col_name = df_cols_normalized[m]
					if all(col_name != c[0] for c in candidates):
						candidates.append((col_name, 'fuzzy'))

			def _rank(item: tuple[str, str]) -> int:
				return {'exact': 0, 'alias': 1, 'fuzzy': 2}.get(item[1], 99)
			return sorted(candidates, key=_rank)

		def _choose_from_candidates(kind: str, cands: list[tuple[str, str]]) -> str | None:
			if not cands:
				return None
			if len(cands) == 1 or not interactive:
				return cands[0][0]

			print(f"\nSelect {kind.upper()} column:")
			for i, (col_name, reason) in enumerate(cands, 1):
				print(f"  {i}) {_col_label(col_name)} [{reason}]")
				samples = _sample_values(col_name, limit=5)
				for j, v in enumerate(samples, 1):
					print(f"     {j}. {v}")
			print("Enter number, or blank for default.")
			while True:
				try:
					ans = input("> ").strip().lower()
				except EOFError:
					ans = ''
				if not ans:
					return cands[0][0]
				if ans.isdigit():
					idx = int(ans)
					if 1 <= idx <= len(cands):
						return cands[idx - 1][0]
				print(f"Invalid selection. Choose 1..{len(cands)} or blank.")

		def _prompt_kind(*, have_price: bool, have_cost: bool) -> str:
			default = 'price' if have_price else 'cost'
			if not interactive:
				return default
			while True:
				try:
					ans = input(f"Use price or cost? (p/c) [default {default[0]}]: ").strip().lower()
				except EOFError:
					ans = ''
				if not ans:
					return default
				if ans in {'p', 'price'}:
					return 'price'
				if ans in {'c', 'cost'}:
					return 'cost'
				print("Please enter 'p' (price) or 'c' (cost).")

		price_cands = [(column_mapping['price_trade'], 'manual')] if 'price_trade' in column_mapping else _gather_candidates('price_trade')
		cost_cands = [(column_mapping['cost_trade'], 'manual')] if 'cost_trade' in column_mapping else _gather_candidates('cost_trade')

		if not price_cands and not cost_cands:
			raise ValueError("ONES requires either price_trade (per-unit) or cost_trade (total cost).")

		print("\n" + "=" * 80)
		if price_cands and cost_cands:
			print("Found PRICE candidates:")
			for col_name, reason in price_cands[:10]:
				print(f"  - {_col_label(col_name)} [{reason}]")
			print("Found COST candidates:")
			for col_name, reason in cost_cands[:10]:
				print(f"  - {_col_label(col_name)} [{reason}]")
			print("You will choose whether to use PRICE or COST, then pick a column.")
		elif price_cands:
			print("Found value column(s) that look like PRICE:")
			for col_name, reason in price_cands[:10]:
				print(f"  - {_col_label(col_name)} [{reason}]")
			print("Do you want to treat the selected column as per-unit PRICE or as total COST (will be divided by volume)?")
		else:
			print("Found value column(s) that look like COST:")
			for col_name, reason in cost_cands[:10]:
				print(f"  - {_col_label(col_name)} [{reason}]")
			print("Do you want to treat the selected column as per-unit PRICE or as total COST (will be divided by volume)?")

		kind = _prompt_kind(have_price=bool(price_cands), have_cost=bool(cost_cands))
		if kind == 'price' and price_cands:
			value_col = _choose_from_candidates('price', price_cands)
			chosen_kind = 'price'
		elif kind == 'cost' and cost_cands:
			value_col = _choose_from_candidates('cost', cost_cands)
			chosen_kind = 'cost'
		else:
			chosen_kind = 'price' if price_cands else 'cost'
			value_col = _choose_from_candidates(chosen_kind, price_cands if price_cands else cost_cands)

		column_mapping['_value_col'] = value_col
		column_mapping['_value_kind'] = chosen_kind

		data_raw = {}
		base_targets = ['boards', 'type', 'tickers', 'date_trade', 'volume_signed']
		for target_col in base_targets:
			if target_col in column_mapping:
				actual_col = column_mapping[target_col]
				raw_list = df[actual_col].tolist()
				data_raw[target_col] = [None if pd.isna(v) or (isinstance(v, str) and not v.strip()) else v for v in raw_list]
			else:

				if target_col == 'boards':
					continue
				data_raw[target_col] = []

		value_col = column_mapping.get('_value_col')
		data_raw['price_trade'] = [None if pd.isna(v) or (isinstance(v, str) and not v.strip()) else v for v in df[value_col].tolist()]
		data_raw['_value_kind'] = column_mapping.get('_value_kind')



		for target_col in optional_columns:
			if target_col in column_mapping:
				actual_col = column_mapping[target_col]
				raw_list = df[actual_col].tolist()
				data_raw[target_col] = [None if pd.isna(v) or (isinstance(v, str) and not v.strip()) else v for v in raw_list]

		print(f"\nValidating column lengths...")
		validate_column_lengths(data_raw)


		row_count = max((len(v) for k, v in data_raw.items() if isinstance(v, list) and v and not str(k).startswith('_')), default=int(len(df)))

		original_has_commission = bool('commission' in column_mapping)
		data_raw['_original_has_commission'] = original_has_commission

		if 'commission' not in data_raw or not isinstance(data_raw.get('commission'), list):
			data_raw['commission'] = None
		else:
			if len(data_raw['commission']) == 0:
				data_raw['commission'] = [None] * row_count
			elif len(data_raw['commission']) != row_count:
				data_raw['commission'] = (data_raw['commission'] + [None] * row_count)[:row_count]


		data_raw = forward_fill_rows(data_raw, ['type', 'tickers'])
		data_raw = _apply_marketboard_rules_uno(data_raw, interactive=interactive)
		data_raw = normalize_marketboards_and_types(data_raw, interactive=interactive)
		validate_required_identifiers(data_raw)

		num_cols = ['volume_signed', 'price_trade'] + (["commission"] if isinstance(data_raw.get('commission'), list) else [])
		data_raw = convert_numeric_columns(data_raw, num_cols, manual_format=number_format, interactive=interactive)

		def _compute_price_from_cost(cost_list: list, volume_list: list) -> list:
			import numpy as np
			cost_arr = np.asarray(cost_list, dtype=float)
			vol_arr = np.asarray(volume_list, dtype=float)
			den = np.abs(vol_arr)
			out = np.full(cost_arr.shape, np.nan, dtype=float)
			np.divide(np.abs(cost_arr), den, out=out, where=den != 0)
			res = out.tolist()
			return [None if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x) for x in res]

		# Convert COST into per-unit PRICE before any further validations.
		if data_raw.get('_value_kind') == 'cost':
			data_raw['price_trade'] = _compute_price_from_cost(data_raw.get('price_trade', []), data_raw.get('volume_signed', []))

		data_raw = convert_date_columns(data_raw, ['date_trade'])
		data_raw.pop('_value_kind', None)


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

