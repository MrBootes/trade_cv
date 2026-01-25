from __future__ import annotations

from typing import Optional

import os
from difflib import get_close_matches
import pandas as pd

from loaders.load_validate import (
 read_input_file,
 _collect_bad_inout_rows,
 _is_missing_cell,
 check_and_fill_volume,
 convert_date_columns,
 convert_numeric_columns,
 find_matching_column,
 forward_fill_rows,
 normalize_column_name,
 normalize_marketboards_and_types,
 resolve_ambiguous_columns,
 sort_data_by_board,
 validate_column_lengths,
 validate_date_price_pairing,
 validate_required_identifiers,
 validate_trade_rows,
 _report_and_skip_bad_rows,
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


def _apply_marketboard_rules_inout(data: dict, *, interactive: bool) -> dict:
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
	missing_idx = []
	canon = [None] * n
	for i, b in enumerate(boards):
		c, invalid = _canonicalize_marketboard(b)
		if invalid is not None:
			invalid_found.add(invalid)
			continue
		keep_idx.append(i)
		canon[i] = c
		if c is None:
			missing_idx.append(i)

	if invalid_found:
		bad_list = ", ".join(sorted(invalid_found))
		allowed_list = ", ".join(_get_available_marketboards())
		print(
		 f"\n⚠ Unavailable marketboards detected in INOUT: {bad_list}. "
		 f"Allowed: {allowed_list}. Those rows will be excluded."
		)
		out = dict(data)
		for k, v in list_cols.items():
			out[k] = [v[i] for i in keep_idx]
		out['_original_has_board'] = original_has_board
		data = out
		return _apply_marketboard_rules_inout(data, interactive=interactive)

	if missing_idx:
		allowed_list = ", ".join(_get_available_marketboards())
		if not interactive:
			raise RuntimeError(
			 f"Missing marketboard values in INOUT ({len(missing_idx)} rows) and non-interactive mode. "
			 f"Allowed: {allowed_list}."
			)
		print(
		 f"\nMarketboard is missing in {len(missing_idx)} row(s) for INOUT. "
		 f"Available marketboards: {allowed_list}."
		)
		while True:
			ans = input("Enter marketboard to use for missing values: ").strip()
			if not ans:
				print(f"Marketboard is required. Allowed: {allowed_list}.")
				continue
			c, invalid = _canonicalize_marketboard(ans)
			if invalid is None and c is not None:
				fill = c
				break
			print(f"Invalid marketboard '{ans}'. Allowed: {allowed_list}.")

		boards_out = data.get('boards') if isinstance(data.get('boards'), list) else [None] * n
		boards_out = list(boards_out) if isinstance(boards_out, list) else [None] * n
		for i in range(n):
			c, invalid = _canonicalize_marketboard(boards_out[i] if i < len(boards_out) else None)
			if invalid is None and c is None:
				boards_out[i] = fill
		data = dict(data)
		data['boards'] = boards_out
	else:
		boards_out = []
		for b in boards:
			c, _ = _canonicalize_marketboard(b)
			boards_out.append(c)
		data = dict(data)
		data['boards'] = boards_out

	data['_original_has_board'] = original_has_board
	return data


def _column_aliases_inout() -> dict:
	return {
	 'boards': ['board', 'mboards', 'mboard', 'market_boards', 'market_board', 'marketboards', 'marketboard', 'exchange', 'exchanges', 'market', 'markets', 'market_type', 'markettype', 'board_type', 'boardtype', 'marketboardtype', 'market_board_type'],
	 'type': ['mtype', 'market_type', 'markettype', 'trade_type', 'tradetype', 'type', 'security_type', 'securitytype', 'sectype', 'asset_type', 'assettype', 'instrument_type', 'instrumenttype', 'assetclass', 'asset_class', 'instrumentclass', 'instrument_class', 'securityclass', 'security_class', 'asset', 'instrument', 'security'],
	 'tickers': ['ticker', 'symbol', 'symbols', 'stock', 'stocks', 'isin', 'name', 'names', 'asset_name', 'assetname', 'instrument_name', 'instrumentname', 'security_name', 'securityname', 'tickersymbol', 'tickersymbols', 'stockname', 'stock_name', 'tickername', 'ticker_name', 'assetticker', 'instrumentticker', 'securityticker', 'instrument_ticker', 'security_ticker', 'asset_ticker', 'stock_ticker', 'ticker_code', 'tickercode', 'code', 'codes', 'asset_code', 'assetcode', 'instrument_code', 'instrumentcode', 'security_code', 'securitycode', 'isin_code', 'isincode', 'stock_code', 'stockcode', 'tickerid', 'ticker_id', 'securityid', 'security_id', 'instrumentid', 'instrument_id', 'id', 'ids', 'assetid', 'asset_id', 'stockid', 'stock_id', 'symbol_code', 'symbolcode', 'symbolid', 'symbol_id'],
	 'volume': ['vol', 'quantity', 'qty', 'amount', 'count', 'shares', 'volume_traded', 'volumetraded', 'trade_volume', 'tradevolume', 'number_of_shares', 'numberofshares', 'num_shares', 'numshares', 'trade_qty', 'tradeqty', 'trade_quantity', 'tradequantity'],
	 'dates_buy': ['datebuy', 'date_buy', 'datesbuy', 'dates_buy', 'buy_dates', 'buydates', 'dbuy', 'buyd', 'bdate', 'dateb', 'bday', 'dayb', 'buy_day', 'buyday', 'buy_date', 'buydate', 'opendate', 'open_date', 'purchase_date', 'purchasedate', 'buy_time', 'buytime', 'open_time', 'opentime', 'purchase_time', 'purchasetime', 'trade_open_date', 'tradeopendate', 'trade_open_time', 'tradeopentime', 'opening_date', 'openingdate', 'opening_time', 'openingtimes', 'entry_date', 'entrydate', 'entry_time', 'entrytime', 'buy_datetime', 'buydatetime', 'open_datetime', 'opendatetime', 'purchase_datetime', 'purchasedatetime', 'trade_open_datetime', 'tradeopendatetime', 'entry_datetime', 'entrydatetime'],
	 'buy_price': ['buy', 'buy_c', 'price_buy', 'pricebuy', 'pbuy', 'buyp', 'buyc', 'buy_price', 'buyprice', 'purchase_price', 'open_price', 'openprice', 'trade_entry_price', 'tradeentryprice', 'entry_price', 'entryprice'],
	 'buy_cost': ['open_cost', 'opencost', 'buy_cost', 'buycost', 'purchase_cost', 'purchasecost', 'trade_open_cost', 'tradeopencost', 'entry_cost', 'entrycost', 'trade_entry_cost', 'tradeentrycost', 'purchase', 'buy_value', 'buy_amount', 'buy_total', 'amount_buy', 'value_buy', 'total_buy', 'cost_buy'],
	 'dates_sell': ['date_sell', 'datesell', 'dates_sell', 'datessell', 'sell_dates', 'selldates', 'dsell', 'selld', 'sdate', 'sday', 'sell_day', 'sellday', 'date_close', 'dateclose', 'sell_date', 'selldate', 'close_date', 'closedate', 'sell_time', 'selltime', 'close_time', 'closetime', 'exit_time', 'exittime', 'exit_date', 'exitdate', 'trade_close_date', 'tradeclosedate', 'trade_close_time', 'tradeclosetime', 'closing_date', 'closingdate', 'closing_time', 'closingtime', 'sell_datetime', 'selldatetime', 'close_datetime', 'closedatetime', 'exit_datetime', 'exitdatetime', 'trade_close_datetime', 'tradeclosedatetime'],
	 'sell_price': ['sell', 'sell_price', 'sellprice', 'price_sell', 'pricesell', 'psell', 'sellp', 'sell_c', 'sellc', 'trade_exit_price', 'tradeexitprice', 'close_price', 'closeprice', 'exit_price', 'exitprice'],
	 'sell_cost': ['sell_cost', 'sellcost', 'close_cost', 'closecost', 'exit_cost', 'exitcost', 'trade_close_cost', 'tradeclosecost', 'trade_exit_cost', 'tradeexitcost', 'sell_value', 'sell_amount', 'sell_total', 'amount_sell', 'value_sell', 'total_sell', 'cost_sell'],
	}


def display_data(data):
	if not data:
		print("No data to display")
		return

	visible_values = [v for k, v in data.items() if not str(k).startswith('_') and v]
	max_len = max(len(v) for v in visible_values) if visible_values else 0
	print(f"\n{'='*80}")
	print(f"DATA SUMMARY - Total rows: {max_len}")
	print(f"{'='*80}")

	for key, values in data.items():
		if str(key).startswith('_') or not values:
			continue
		non_null = [v for v in values if not _is_missing_cell(v)]
		sample = non_null[0] if non_null else None
		print(f"  {key}: {len(non_null)}/{max_len} values, sample: {sample}")

	print(f"\n✓ Data loaded and validated successfully!")


def _drop_bad_rows_inout(data: dict, bad_indices: list) -> dict:
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

		target_columns = ['boards', 'type', 'tickers', 'volume', 'dates_buy', 'buy_price', 'buy_cost', 'buy_commission', 'dates_sell', 'sell_price', 'sell_cost', 'sell_commission']
		optional_columns = ['buy_commission', 'sell_commission']
		column_aliases = _column_aliases_inout()

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
			if 'buy' in manual and 'buy_price' not in manual and 'buy_cost' not in manual:
				manual = dict(manual)
				manual['buy_price'] = manual.get('buy')
			if 'sell' in manual and 'sell_price' not in manual and 'sell_cost' not in manual:
				manual = dict(manual)
				manual['sell_price'] = manual.get('sell')
		if manual:
			print(f"\nApplying custom column mapping...")
			for target_col in target_columns:
				chosen = manual.get(target_col)
				if not chosen:
					continue
				if chosen in df.columns:
					column_mapping[target_col] = chosen
					print(f"  {target_col}: ✓ → '{chosen}'")
				else:
					print(f"  {target_col}: ✗ Column '{chosen}' not found in file")


		required_core = ['type', 'tickers', 'volume', 'dates_buy', 'dates_sell']
		missing_required = [c for c in required_core if c not in column_mapping]
		if missing_required:
			print(f"\nMatching columns for missing fields: {', '.join(missing_required)}")
			auto_map = find_matching_column(df, missing_required, aliases=column_aliases, interactive=interactive)
			auto_map = resolve_ambiguous_columns(df, auto_map, missing_required, aliases=column_aliases, interactive=interactive)
			column_mapping.update(auto_map)
			column_mapping = resolve_ambiguous_columns(df, column_mapping, target_columns, aliases=column_aliases, interactive=interactive)


		missing_optional = [c for c in optional_columns if c not in column_mapping]
		if missing_optional and not manual:
			auto_opt = find_matching_column(df, missing_optional, aliases=column_aliases, interactive=False)
			column_mapping.update(auto_opt)
			column_mapping = resolve_ambiguous_columns(df, column_mapping, target_columns, aliases=column_aliases, interactive=interactive)

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
				series = df[col_name]
				vals = series.head(limit).tolist()
				return vals
			except Exception:
				return []

		def _normalize_choice(s: str) -> str:
			return str(s).strip().lower()

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

			sorted_cands = sorted(candidates, key=_rank)
			# Heuristic: keep PRICE candidates from being polluted by COST-ish column names (and vice versa)
			# due to fuzzy matching (e.g. "sell cost" showing up as a price candidate).
			price_targets = {'buy_price', 'sell_price'}
			cost_targets = {'buy_cost', 'sell_cost'}
			if target in price_targets:
				filtered = []
				for col_name, reason in sorted_cands:
					norm_col = normalize_column_name(col_name)
					if reason != 'manual' and any(x in norm_col for x in ('cost', 'total', 'amount', 'value')):
						continue
					filtered.append((col_name, reason))
				sorted_cands = filtered
			elif target in cost_targets:
				filtered = []
				for col_name, reason in sorted_cands:
					norm_col = normalize_column_name(col_name)
					if reason != 'manual' and 'price' in norm_col:
						continue
					filtered.append((col_name, reason))
				sorted_cands = filtered

			return sorted_cands

		def _choose_value_column(
			side: str,
			kind: str,
			options: list[str],
			*,
			default_col: str,
		) -> str:
			if not interactive:
				return default_col
			if not options:
				return default_col

			options_unique = []
			seen = set()
			for c in options:
				if c in seen:
					continue
				seen.add(c)
				options_unique.append(c)

			default_label = _col_label(default_col)
			print(f"\nWhich column to use for {kind.upper()}?")
			for i, col_name in enumerate(options_unique, 1):
				print(f"  {i}) {_col_label(col_name)}")
			print(f"Enter number, index, or name to choose [default {default_label}]:")

			lookup: dict[str, str] = {}
			for i, col_name in enumerate(options_unique, 1):
				lookup[str(i)] = col_name
				try:
					idx0 = list(df.columns).index(col_name)
					lookup[_normalize_choice(_excel_col_letter(idx0))] = col_name
				except Exception:
					pass
				lookup[_normalize_choice(col_name)] = col_name
				lookup[_normalize_choice(str(col_name).split(':', 1)[0])] = col_name

			while True:
				try:
					ans = input("> ").strip()
				except EOFError:
					ans = ''
				if not ans:
					return default_col
				key = _normalize_choice(ans)
				picked = lookup.get(key)
				if picked:
					return picked
				if ':' in key:
					picked = lookup.get(key.split(':', 1)[0].strip())
					if picked:
						return picked
				print("Invalid selection. Enter a number, Excel column letter, or exact column name.")

		def _prompt_kind(side: str, *, have_price: bool, have_cost: bool) -> str:
			default = 'price' if have_price else 'cost'
			if not interactive:
				return default
			while True:
				try:
					ans = input(f"Use {side} price or cost? (p/c) [default {default[0]}]: ").strip().lower()
				except EOFError:
					ans = ''
				if not ans:
					return default
				if ans in {'p', 'price'}:
					return 'price'
				if ans in {'c', 'cost'}:
					return 'cost'
				print("Please enter 'p' (price) or 'c' (cost).")

		def _select_price_or_cost_for_side(side: str) -> tuple[str, str]:
			price_target = 'buy_price' if side == 'BUY' else 'sell_price'
			cost_target = 'buy_cost' if side == 'BUY' else 'sell_cost'

			price_cands = []
			cost_cands = []
			if price_target in column_mapping:
				price_cands = [(column_mapping[price_target], 'manual')]
			else:
				price_cands = _gather_candidates(price_target)
			if cost_target in column_mapping:
				cost_cands = [(column_mapping[cost_target], 'manual')]
			else:
				cost_cands = _gather_candidates(cost_target)

			if not price_cands and not cost_cands:
				raise ValueError(
					f"TWOS requires {side} value column(s). Provide either {price_target} or {cost_target} (or both)."
				)

			print("\n" + "=" * 80)
			if price_cands and cost_cands:
				print(f"Found {side} PRICE candidates:")
				for col_name, reason in price_cands[:10]:
					print(f"  - {_col_label(col_name)} [{reason}]")
				print(f"Found {side} COST candidates:")
				for col_name, reason in cost_cands[:10]:
					print(f"  - {_col_label(col_name)} [{reason}]")
				print("You will choose whether to use PRICE or COST, then pick a column.")
			elif price_cands:
				print(f"Found {side} value column(s) that look like PRICE:")
				for col_name, reason in price_cands[:10]:
					print(f"  - {_col_label(col_name)} [{reason}]")
				print("Do you want to treat the selected column as per-unit PRICE or as total COST (will be divided by volume)?")
			else:
				print(f"Found {side} value column(s) that look like COST:")
				for col_name, reason in cost_cands[:10]:
					print(f"  - {_col_label(col_name)} [{reason}]")
				print("Do you want to treat the selected column as per-unit PRICE or as total COST (will be divided by volume)?")

			both_present = bool(price_cands) and bool(cost_cands)
			kind = _prompt_kind(side, have_price=bool(price_cands), have_cost=bool(cost_cands))

			if kind not in {'price', 'cost'}:
				kind = 'price' if price_cands else 'cost'

			# Default column is the best match inside the chosen kind.
			preferred = price_cands if kind == 'price' else cost_cands
			fallback = cost_cands if kind == 'price' else price_cands
			default_col = preferred[0][0] if preferred else (fallback[0][0] if fallback else None)
			if default_col is None:
				raise ValueError(
					f"TWOS requires {side} value column(s). Provide either {price_target} or {cost_target} (or both)."
				)

			if both_present and interactive:
				options = [c[0] for c in preferred] + [c[0] for c in fallback]
				col = _choose_value_column(side, kind, options, default_col=default_col)
				return kind, col

			return kind, default_col

		buy_kind, buy_value_col = _select_price_or_cost_for_side('BUY')
		sell_kind, sell_value_col = _select_price_or_cost_for_side('SELL')
		column_mapping['_buy_value_col'] = buy_value_col
		column_mapping['_sell_value_col'] = sell_value_col

		data = {}
		base_targets = ['boards', 'type', 'tickers', 'volume', 'dates_buy', 'dates_sell', 'buy_commission', 'sell_commission']
		for target_col in base_targets:
			if target_col in column_mapping:
				actual_col = column_mapping[target_col]
				raw_list = df[actual_col].tolist()
				data[target_col] = [None if pd.isna(v) or (isinstance(v, str) and not v.strip()) else v for v in raw_list]
			else:
				if target_col == 'boards' or target_col in optional_columns:
					continue
				data[target_col] = []

		data['buy'] = [None if pd.isna(v) or (isinstance(v, str) and not v.strip()) else v for v in df[buy_value_col].tolist()]
		data['sell'] = [None if pd.isna(v) or (isinstance(v, str) and not v.strip()) else v for v in df[sell_value_col].tolist()]
		data['_buy_value_kind'] = buy_kind
		data['_sell_value_kind'] = sell_kind

		print(f"\nValidating column lengths...")
		validate_column_lengths(data)


		row_count = max((len(v) for k, v in data.items() if isinstance(v, list) and v and not str(k).startswith('_')), default=int(len(df)))

		original_has_buy_commission = bool('buy_commission' in column_mapping)
		original_has_sell_commission = bool('sell_commission' in column_mapping)
		data['_original_has_buy_commission'] = original_has_buy_commission
		data['_original_has_sell_commission'] = original_has_sell_commission

		for col in optional_columns:
			if col not in data or not isinstance(data.get(col), list):
				data[col] = None
				continue
			if len(data[col]) == 0:
				data[col] = [None] * row_count
			elif len(data[col]) != row_count:
				data[col] = (data[col] + [None] * row_count)[:row_count]


		data = forward_fill_rows(data, ['type', 'tickers'])
		data = _apply_marketboard_rules_inout(data, interactive=interactive)
		data = normalize_marketboards_and_types(data, interactive=interactive)
		validate_required_identifiers(data)

		num_cols = ['volume', 'buy', 'sell']
		if isinstance(data.get('buy_commission'), list):
			num_cols.append('buy_commission')
		if isinstance(data.get('sell_commission'), list):
			num_cols.append('sell_commission')
		data = convert_numeric_columns(data, num_cols, manual_format=number_format, interactive=interactive)

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
		if data.get('_buy_value_kind') == 'cost':
			data['buy'] = _compute_price_from_cost(data.get('buy', []), data.get('volume', []))
		if data.get('_sell_value_kind') == 'cost':
			data['sell'] = _compute_price_from_cost(data.get('sell', []), data.get('volume', []))

		data = convert_date_columns(data, ['dates_buy', 'dates_sell'])


		print(f"\nChecking for problematic rows...")
		bad_rows_result = _collect_bad_inout_rows(data)
		if bad_rows_result and bad_rows_result.get('problems'):
			_report_and_skip_bad_rows(bad_rows_result, trade_type="INOUT")

			data = _drop_bad_rows_inout(data, bad_rows_result['bad_row_indices'])
			print(f"✓ Skipped {len(bad_rows_result['bad_row_indices'])} problematic row(s)")

		validate_date_price_pairing(data)
		data = check_and_fill_volume(data, interactive=interactive)
		validate_trade_rows(data)
		data = sort_data_by_board(data)

		input_rows = int(len(df))
		visible_values = [v for k, v in data.items() if not str(k).startswith('_') and isinstance(v, list)]
		output_rows = max((len(v) for v in visible_values), default=0)
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

