import pandas as pd
import os
import re
import sys
from typing import Optional
from difflib import get_close_matches
from dateutil import parser as date_parser



_RE_NUM_KEEP = re.compile(r'[^\d,\.\s-]+')
_RE_NON_DIGITS = re.compile(r'\D+')
_RE_WS = re.compile(r'\s+')
_RE_DATE_SEP = re.compile(r'\d+\s*[/.-]\s*\d+\s*[/.-]\s*\d+')
_RE_YMD = re.compile(r'\d{4}[-/.]\d{1,2}[-/.]\d{1,2}')
_RE_HAS_LETTERS = re.compile(r'[A-Za-zА-Яа-я]{3,}')


def _detect_and_split_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def _excel_col_letter(idx):
    idx += 1
    letters = []
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        letters.append(chr(65 + rem))
    return ''.join(reversed(letters))


def _col_label(idx, name):
    letter = _excel_col_letter(idx)
    number = idx + 1
    return f"{name} (index: {letter})"


def extract_data_from_df(
    df,
    target_columns,
    *,
    column_aliases=None,
    custom_column_names=None,
):
    if column_aliases is None:
        column_aliases = {}
    if custom_column_names is None:
        custom_column_names = {}

    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    if not isinstance(target_columns, list) or not all(isinstance(x, str) for x in target_columns):
        raise ValueError("target_columns must be list[str]")

    if custom_column_names:
        column_mapping = {}
        for target_col in target_columns:
            if target_col in custom_column_names and custom_column_names[target_col]:
                wanted = custom_column_names[target_col]
                if wanted not in df.columns:
                    raise ValueError(
                        f"Column '{wanted}' for '{target_col}' not found in file"
                    )
                column_mapping[target_col] = wanted
    else:
        interactive = sys.stdin.isatty()
        column_mapping = find_matching_column(df, target_columns, aliases=column_aliases, interactive=interactive)
        column_mapping = resolve_ambiguous_columns(
            df,
            column_mapping,
            target_columns,
            aliases=column_aliases,
            interactive=interactive,
        )

    data = {}
    for target_col in target_columns:
        if target_col in column_mapping:
            actual_col = column_mapping[target_col]
            raw_list = df[actual_col].tolist()
            data[target_col] = [
                None if pd.isna(v) or (isinstance(v, str) and not v.strip()) else v
                for v in raw_list
            ]
        else:
            data[target_col] = []

    return data, column_mapping


def parse_date(value, dayfirst=None):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp) or hasattr(value, 'year'):
        return value

    date_str = str(value).strip()
    if not date_str:
        return None

    try:
        if dayfirst is not None:
            parsed = pd.to_datetime(date_str, dayfirst=dayfirst, errors='coerce')
        else:
            parsed = pd.to_datetime(date_str, infer_datetime_format=True, errors='coerce')

        if pd.notna(parsed):
            return parsed

        if dayfirst is not None:
            parsed = date_parser.parse(date_str, dayfirst=dayfirst)
        else:
            parsed = date_parser.parse(date_str)
        return parsed

    except (ValueError, TypeError, date_parser.ParserError):
        return value


def parse_dates_fast(values, dayfirst=None):
    series = pd.Series(values)
    parsed = pd.to_datetime(series, errors='coerce', dayfirst=dayfirst)

    mask_try = series.notna() & series.astype(str).str.strip().ne('') & parsed.isna()
    if mask_try.any():
        idx = mask_try[mask_try].index
        for i in idx:
            val = series.iloc[i]
            try:
                parsed.iloc[i] = date_parser.parse(str(val), dayfirst=dayfirst)
            except Exception:
                pass

    out = []
    for v, p in zip(values, parsed.tolist()):
        if pd.isna(v):
            out.append(None)
        elif isinstance(p, pd.Timestamp):
            out.append(p)
        elif p is None or (isinstance(p, float) and pd.isna(p)):
            out.append(None)
        else:
            out.append(p)
    return out


def detect_date_format(series):
    def _extract_date_samples(values, limit=10):
        samples_local = []
        for val in values:
            if pd.notna(val):
                s = str(val).strip()
                if not s:
                    continue
                if _RE_DATE_SEP.search(s):
                    samples_local.append(s)
                else:
                    if _RE_YMD.search(s) or _RE_HAS_LETTERS.search(s):
                        samples_local.append(s)
            if len(samples_local) >= limit:
                break
        return samples_local

    def _score_parsing(samples_local, dayfirst_flag):
        if not samples_local:
            return 0.0, 0, 0
        parsed = pd.to_datetime(pd.Series(samples_local), errors='coerce', dayfirst=dayfirst_flag)
        ok_mask = parsed.notna()
        ok_count = int(ok_mask.sum())
        now = pd.Timestamp.today(tz=None)
        max_future = now + pd.Timedelta(days=365 * 3)
        plausible_mask = ok_mask & (parsed.dt.year >= 1970) & (parsed.dt.year <= 2100) & (parsed <= max_future)
        plausible_count = int(plausible_mask.sum())
        score = ok_count + 0.2 * plausible_count
        return float(score), ok_count, plausible_count

    samples = _extract_date_samples(series.head(50).tolist(), limit=10)
    if not samples:
        return {'dayfirst': None, 'confidence': 0.0, 'reason': 'no_samples'}

    dayfirst_evidence = 0
    monthfirst_evidence = 0
    for sample in samples:
        parts = re.findall(r'\d+', sample)
        if len(parts) >= 3:
            try:
                first_num = int(parts[0])
                second_num = int(parts[1])
            except ValueError:
                continue
            if first_num > 12:
                dayfirst_evidence += 1
            elif second_num > 12:
                monthfirst_evidence += 1

    if dayfirst_evidence > monthfirst_evidence and dayfirst_evidence > 0:
        return {'dayfirst': True, 'confidence': 0.95, 'reason': 'unambiguous_dayfirst'}
    if monthfirst_evidence > dayfirst_evidence and monthfirst_evidence > 0:
        return {'dayfirst': False, 'confidence': 0.95, 'reason': 'unambiguous_monthfirst'}

    score_true, ok_true, plausible_true = _score_parsing(samples, True)
    score_false, ok_false, plausible_false = _score_parsing(samples, False)

    if score_true == 0.0 and score_false == 0.0:
        return {'dayfirst': None, 'confidence': 0.0, 'reason': 'parse_failed'}

    if score_true > score_false:
        best = True
        best_score = score_true
        other_score = score_false
        reason = 'scored_dayfirst'
    elif score_false > score_true:
        best = False
        best_score = score_false
        other_score = score_true
        reason = 'scored_monthfirst'
    else:
        return {'dayfirst': None, 'confidence': 0.0, 'reason': 'tie'}

    ratio = (best_score + 1e-9) / (other_score + 1e-9)
    if ratio >= 2.0:
        confidence = 0.90
    elif ratio >= 1.3:
        confidence = 0.75
    else:
        confidence = 0.60

    if abs(ok_true - ok_false) >= 3:
        confidence = min(0.95, confidence + 0.10)

    return {'dayfirst': best, 'confidence': float(confidence), 'reason': reason}


def convert_date_columns(data, column_names):
    for col_name in column_names:
        if col_name in data and data[col_name]:
            series = pd.Series(data[col_name])
            detection = detect_date_format(series)
            dayfirst = detection.get('dayfirst')
            confidence = float(detection.get('confidence', 0.0))
            confidence_pct = confidence * 100

            if dayfirst is True:
                print(f"  Detected day-first for '{col_name}' (DD/MM/YYYY) (confidence: {confidence_pct:.1f}%)")
            elif dayfirst is False:
                print(f"  Detected month-first for '{col_name}' (MM/DD/YYYY) (confidence: {confidence_pct:.1f}%)")
            else:
                print(f"  Could not confidently detect date format for '{col_name}' (confidence: {confidence_pct:.1f}%)")

            if dayfirst is None or confidence < 0.70:
                suggested_dayfirst = True if dayfirst is None else dayfirst
                suggestion_text = "day-first (DD/MM/YYYY)" if suggested_dayfirst else "month-first (MM/DD/YYYY)"
                print(f"  ⚠ WARNING: Date format for '{col_name}' is ambiguous.")
                print(f"     Suggested: {suggestion_text}")
                answer = input(f"     Use day-first format for '{col_name}'? (y/n, Enter=use suggested): ").strip().lower()
                if answer == 'y':
                    dayfirst = True
                elif answer == 'n':
                    dayfirst = False
                else:
                    dayfirst = suggested_dayfirst
                print(f"  Using {'day-first' if dayfirst else 'month-first'} for '{col_name}'.")

            original_values = data[col_name]
            parsed = parse_dates_fast(original_values, dayfirst=dayfirst)
            data[col_name] = parsed

            total_count = len([val for val in original_values if pd.notna(val) and str(val).strip()])
            success_count = sum(1 for val in parsed if val is not None and not isinstance(val, str))
            if total_count > 0:
                success_rate = (success_count / total_count) * 100
                print(f"  Successfully parsed {success_count}/{total_count} dates ({success_rate:.1f}%)")

    return data


def normalize_column_name(col_name):
    if pd.isna(col_name):
        return ""
    return str(col_name).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _is_missing_cell(value):
    if value is None:
        return True
    if pd.isna(value):
        return True
    if isinstance(value, str):
        s = value.strip()
        if s == "" or s.lower() in {"nan", "none", "null", "n/a", "na", "-"}:
            return True
    return False


def forward_fill_rows(data, column_names):
    lengths = [len(data.get(c, [])) for c in column_names if isinstance(data.get(c, []), list)]
    if not lengths:
        return data
    n = max(lengths)
    if n == 0:
        return data

    for col in column_names:
        values = data.get(col)
        if not isinstance(values, list):
            continue
        if len(values) < n:
            values.extend([None] * (n - len(values)))
        data[col] = values

    last_seen = {col: None for col in column_names}
    for i in range(n):
        for col in column_names:
            values = data.get(col)
            if not isinstance(values, list) or i >= len(values):
                continue
            v = values[i]
            if _is_missing_cell(v):
                if last_seen[col] is not None:
                    values[i] = last_seen[col]
                else:
                    values[i] = None
            else:
                last_seen[col] = v

    return data


def normalize_marketboards_and_types(data, *, interactive: bool = True):
    import re

    def _norm_token(v) -> str:
        s = str(v or "").strip().lower()
        s = s.replace(" ", "_").replace("-", "_")
        s = re.sub(r"[^0-9a-zа-я_]+", "", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s




    def _drop_rows(data_local: dict, bad_row_idxs_0_based: list[int], *, meta_key: str) -> dict:
        if not bad_row_idxs_0_based:
            return data_local

        list_cols = [
            (k, v) for k, v in data_local.items()
            if not str(k).startswith('_') and isinstance(v, list)
        ]
        if not list_cols:
            return data_local
        n = max((len(v) for _k, v in list_cols), default=0)
        if n <= 0:
            return data_local

        bad = {i for i in bad_row_idxs_0_based if 0 <= i < n}
        if not bad:
            return data_local

        keep = [i for i in range(n) if i not in bad]
        out = dict(data_local)
        for k, v in list_cols:
            if len(v) == n:
                out[k] = [v[i] for i in keep]

        out[meta_key] = sorted([i + 1 for i in bad])
        return out


    type_values = data.get('type', [])
    if isinstance(type_values, list) and type_values:

        CANON = {
            'share': 'share',
            'bond': 'bond',
            'etf': 'etf',
            'futures': 'futures',
            'currency': 'currency',
            'index': 'index',
        }


        token_to_bucket: dict[str, str] = {}
        def _add(bucket: str, tokens: list[str]):
            for t in tokens:
                token_to_bucket[_norm_token(t)] = bucket

        _add('share', [
            'common_share', 'preferred_share', 'pref_share', 'ordinary_share',
            'depositary_receipt', 'depository_receipt', 'depositoryreceipt', 'depositaryreceipt',
            'dr', 'adr', 'gdr',
            'share', 'shares', 'stock', 'stocks', 'equity', 'commonstock', 'preferredstock',
        ])
        _add('bond', [
            'exchange_bond', 'corporate_bond', 'ofz_bond', 'ofz',
            'gov_bond', 'government_bond',
            'bond', 'bonds', 'eurobond', 'corporatebond', 'ofzbond',
        ])
        _add('etf', [
            'exchange_ppif', 'ppif', 'pif', 'bpif', 'piff', 'bpiff',
            'etf', 'fund', 'funds', 'mutual_fund', 'mutualfund', 'index_fund', 'fund_etf',
        ])
        _add('futures', ['future', 'futures', 'fut', 'фьючерс', 'фьючерсы'])
        _add('currency', ['currency', 'forex', 'fx', 'валюта', 'валютный', 'curr'])
        _add('index', ['index', 'индекс', 'индексы'])


        for b in list(CANON.keys()):
            token_to_bucket.setdefault(_norm_token(b), b)

        corrected_types: list[object] = []
        skipped_type_rows_0: list[int] = []
        unknown_values: dict[str, list[int]] = {}
        corrected_count = 0


        known_tokens = list(token_to_bucket.keys())

        for idx, type_val in enumerate(type_values):
            if _is_missing_cell(type_val):

                skipped_type_rows_0.append(idx)
                unknown_values.setdefault('', []).append(idx + 1)
                corrected_types.append(type_val)
                continue

            raw = str(type_val).strip()
            token = _norm_token(type_val)
            bucket = token_to_bucket.get(token)

            if bucket is None:

                m = get_close_matches(token, known_tokens, n=1, cutoff=0.75)
                if m:
                    bucket = token_to_bucket.get(m[0])
                    corrected_count += 1

            if bucket is None or bucket not in CANON:

                skipped_type_rows_0.append(idx)
                unknown_values.setdefault(raw, []).append(idx + 1)
                corrected_types.append(raw)
            else:
                corrected_types.append(CANON[bucket])

        if skipped_type_rows_0:

            lines = []
            for val, rows in unknown_values.items():
                if val == '':
                    label = '<missing>'
                else:
                    label = val
                sample_rows = rows[:10]
                more = len(rows) - len(sample_rows)
                line = f"  {label!r} at row(s): {', '.join(str(r) for r in sample_rows)}"
                if more > 0:
                    line += f" ... +{more}"
                lines.append(line)

            print(
                "\n⚠ WARNING: Some 'type' values are not allowed and will be skipped.\n"
                "Allowed buckets are: share, bond, etf, futures, currency, index.\n"
                + "\n".join(lines)
                + f"\nSkipped rows: {len(skipped_type_rows_0)}"
            )


            data = _drop_rows(data, skipped_type_rows_0, meta_key='_skipped_invalid_type_rows')

            type_values_after = data.get('type', [])
            if isinstance(type_values_after, list):

                final_types = []
                for v in type_values_after:
                    if _is_missing_cell(v):
                        final_types.append(v)
                    else:

                        s = str(v).strip()
                        if s in set(CANON.values()):
                            final_types.append(s)
                        else:

                            t = _norm_token(s)
                            b = token_to_bucket.get(t)
                            if b is None:
                                m = get_close_matches(t, known_tokens, n=1, cutoff=0.75)
                                if m:
                                    b = token_to_bucket.get(m[0])
                            final_types.append(CANON[b] if b in CANON else s)
                data['type'] = final_types
        else:
            data['type'] = corrected_types

        if corrected_count > 0:
            print(f"✓ Corrected {corrected_count} 'type' value(s) via fuzzy matching")


        remaining = data.get('type', [])
        if isinstance(remaining, list) and len(remaining) == 0:
            raise ValueError("ERROR: All rows were skipped due to invalid 'type' values.")


    tickers_values = data.get('tickers', [])
    if isinstance(tickers_values, list) and tickers_values:
        corrected = []
        for t in tickers_values:
            if _is_missing_cell(t):
                corrected.append(t)
            else:
                corrected.append(str(t).strip().upper())
        data['tickers'] = corrected

    return data


def _is_missing_number(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _pad_to_length(values, n):
    if not isinstance(values, list):
        return [None] * n
    if len(values) < n:
        return values + [None] * (n - len(values))
    return values


def validate_column_lengths(data):
    columns_with_data = {}
    for col_name, col_data in data.items():
        if not col_name.startswith('_') and col_data:
            columns_with_data[col_name] = len(col_data)

    if not columns_with_data:
        return

    lengths = list(columns_with_data.values())
    if len(set(lengths)) > 1:
        error_msg = "ERROR: Columns have different lengths!\n\n"
        error_msg += "Column lengths:\n"
        for col_name, length in sorted(columns_with_data.items(), key=lambda x: x[1], reverse=True):
            error_msg += f"  {col_name}: {length} rows\n"
        error_msg += "\nAll columns must have the same number of rows."
        raise ValueError(error_msg)


def check_and_fill_volume(data, interactive=True):
    volume = data.get('volume', [])
    if not volume:
        return data

    missing_volume_rows = []
    for idx, vol in enumerate(volume):
        if _is_missing_number(vol) or vol == 0:
            missing_volume_rows.append(idx + 1)

    if missing_volume_rows:
        print(f"\n⚠ WARNING: Found {len(missing_volume_rows)} row(s) with missing or zero volume:")
        examples = missing_volume_rows[:5]
        for row_num in examples:
            print(f"  Row {row_num}")
        if len(missing_volume_rows) > 5:
            print(f"  ... and {len(missing_volume_rows) - 5} more row(s)")

        print(f"\nVolume is required for trade calculations.")
        if not interactive:
            fill_volume = 'n'
        else:
            try:
                fill_volume = input("Would you like to populate missing volume with 1 (representing 1 unit traded)? (y/n): ").lower()
            except EOFError:
                fill_volume = 'n'

        if fill_volume == 'y':
            for idx in range(len(volume)):
                if _is_missing_number(volume[idx]) or volume[idx] == 0:
                    volume[idx] = 1.0
            data['volume'] = volume
            print(f"✓ Filled {len(missing_volume_rows)} row(s) with volume = 1")
        else:
            print("⚠ Volume values left unchanged. Rows with missing volume may fail validation.")

    return data


def validate_and_fill_boards(data, interactive=True):
    boards_values = data.get('boards', [])
    if not boards_values:
        return data


    corrected = []
    invalid = []
    for idx, v in enumerate(boards_values):
        if _is_missing_cell(v):
            corrected.append('MOEX')
            continue
        s = str(v).strip().upper()
        if s == 'MOEX':
            corrected.append('MOEX')
        else:
            invalid.append((idx + 1, v))
            corrected.append(s)

    if invalid:
        preview = invalid[:10]
        msg = "ERROR: Only marketboard MOEX is allowed. Found other values:\n" + "\n".join(
            [f"  Row {r}: {val!r}" for r, val in preview]
        )
        if len(invalid) > len(preview):
            msg += f"\n... and {len(invalid) - len(preview)} more row(s)."

        if not interactive:
            raise ValueError(msg)

        print("\n" + msg)
        ans = input("Force ALL boards to MOEX anyway? (y/n): ").strip().lower()
        if ans != 'y':
            raise ValueError("Boards validation aborted")
        corrected = ['MOEX'] * len(boards_values)

    data['boards'] = corrected

    return data


def _collect_bad_uno_rows(data: dict) -> dict:
    problems_list = []
    bad_indices = set()

    type_vals = _pad_to_length(data.get('type', []), 0)
    ticker_vals = _pad_to_length(data.get('tickers', []), 0)
    date_vals = _pad_to_length(data.get('date_trade', []), 0)
    vol_vals = _pad_to_length(data.get('volume_signed', []), 0)
    price_vals = _pad_to_length(data.get('price_trade', []), 0)

    n = max(len(type_vals), len(ticker_vals), len(date_vals), len(vol_vals), len(price_vals))
    type_vals = _pad_to_length(type_vals, n)
    ticker_vals = _pad_to_length(ticker_vals, n)
    date_vals = _pad_to_length(date_vals, n)
    vol_vals = _pad_to_length(vol_vals, n)
    price_vals = _pad_to_length(price_vals, n)

    for idx in range(n):
        reasons = []
        typ = type_vals[idx]
        ticker = ticker_vals[idx]
        dt = date_vals[idx]
        vol = vol_vals[idx]
        price = price_vals[idx]

        has_date = not _is_missing_cell(dt)
        has_vol = not _is_missing_number(vol)
        has_price = not _is_missing_number(price)
        has_any_trade_field = has_date or has_vol or has_price


        if _is_missing_cell(typ) and _is_missing_cell(ticker):
            reasons.append("missing both type and tickers (no security identifier)")



        if not has_any_trade_field:
            reasons.append("empty row (no date_trade/volume_signed/price_trade)")


        if (not has_any_trade_field) and _is_missing_cell(typ) and _is_missing_cell(ticker):
            reasons.append("completely empty row")


        if has_any_trade_field:
            if not has_vol:
                reasons.append("missing volume_signed")
            if not has_date:
                reasons.append("missing date_trade")
            if not has_price:
                reasons.append("missing price_trade")


        if has_vol:
            try:
                vol_num = float(vol)
                if vol_num == 0.0:
                    reasons.append("volume_signed=0 (no trade)")
                else:

                    if _is_missing_cell(dt):
                        reasons.append(f"volume={vol_num:g} but date_trade is missing")
                    elif isinstance(dt, str):
                        reasons.append(f"volume={vol_num:g} but date_trade could not be parsed: '{dt}'")
                    if _is_missing_number(price):
                        reasons.append(f"volume={vol_num:g} but price_trade is missing")
                    elif isinstance(price, str):
                        reasons.append(f"volume={vol_num:g} but price_trade could not be parsed: '{price}'")
            except (ValueError, TypeError):
                reasons.append(f"volume_signed='{vol}' cannot be converted to number")

        if reasons:
            bad_indices.add(idx)
            row_data = {
                'type': typ,
                'tickers': ticker,
                'date_trade': dt,
                'volume_signed': vol,
                'price_trade': price,
            }

            for opt_col in ['boards', 'commission']:
                if opt_col in data:
                    opt_vals = _pad_to_length(data.get(opt_col, []), n)
                    row_data[opt_col] = opt_vals[idx]

            problems_list.append({
                'index': idx,
                'row_number': idx + 1,
                'reasons': reasons,
                'data': row_data,
            })

    return {
        'bad_row_indices': sorted(bad_indices),
        'problems': problems_list,
    }


def _collect_bad_inout_rows(data: dict) -> dict:
    problems_list = []
    bad_indices = set()

    type_vals = _pad_to_length(data.get('type', []), 0)
    ticker_vals = _pad_to_length(data.get('tickers', []), 0)
    vol_vals = _pad_to_length(data.get('volume', []), 0)
    buy_date_vals = _pad_to_length(data.get('dates_buy', []), 0)
    buy_vals = _pad_to_length(data.get('buy', []), 0)
    sell_date_vals = _pad_to_length(data.get('dates_sell', []), 0)
    sell_vals = _pad_to_length(data.get('sell', []), 0)

    n = max(
        len(type_vals),
        len(ticker_vals),
        len(vol_vals),
        len(buy_date_vals),
        len(buy_vals),
        len(sell_date_vals),
        len(sell_vals),
    )
    type_vals = _pad_to_length(type_vals, n)
    ticker_vals = _pad_to_length(ticker_vals, n)
    vol_vals = _pad_to_length(vol_vals, n)
    buy_date_vals = _pad_to_length(buy_date_vals, n)
    buy_vals = _pad_to_length(buy_vals, n)
    sell_date_vals = _pad_to_length(sell_date_vals, n)
    sell_vals = _pad_to_length(sell_vals, n)

    for idx in range(n):
        reasons = []
        typ = type_vals[idx]
        ticker = ticker_vals[idx]
        vol = vol_vals[idx]
        buy_dt = buy_date_vals[idx]
        buy_pr = buy_vals[idx]
        sell_dt = sell_date_vals[idx]
        sell_pr = sell_vals[idx]

        has_any_trade_field = (
            not _is_missing_number(vol)
            or (not _is_missing_cell(buy_dt))
            or (not _is_missing_number(buy_pr))
            or (not _is_missing_cell(sell_dt))
            or (not _is_missing_number(sell_pr))
        )


        if _is_missing_cell(typ) and _is_missing_cell(ticker):
            reasons.append("missing both type and tickers (no security identifier)")



        if not has_any_trade_field:
            reasons.append("empty row (no volume/buy/sell data)")
        if (not has_any_trade_field) and _is_missing_cell(typ) and _is_missing_cell(ticker):
            reasons.append("completely empty row")


        if not _is_missing_number(vol):
            try:
                vol_num = float(vol)
                if vol_num == 0.0:
                    reasons.append("volume=0 (no trade)")
            except (ValueError, TypeError):
                reasons.append(f"volume='{vol}' cannot be converted to number")


        has_buy_date = not _is_missing_cell(buy_dt)
        has_buy_pr = not _is_missing_number(buy_pr)
        if has_buy_date and not has_buy_pr:
            reasons.append(f"dates_buy='{buy_dt}' but buy price is missing")
        elif has_buy_pr and not has_buy_date:
            reasons.append(f"buy='{buy_pr}' but dates_buy is missing")


        has_sell_date = not _is_missing_cell(sell_dt)
        has_sell_pr = not _is_missing_number(sell_pr)
        if has_sell_date and not has_sell_pr:
            reasons.append(f"dates_sell='{sell_dt}' but sell price is missing")
        elif has_sell_pr and not has_sell_date:
            reasons.append(f"sell='{sell_pr}' but dates_sell is missing")

        if reasons:
            bad_indices.add(idx)
            row_data = {
                'type': typ,
                'tickers': ticker,
                'volume': vol,
                'dates_buy': buy_dt,
                'buy': buy_pr,
                'dates_sell': sell_dt,
                'sell': sell_pr,
            }

            for opt_col in ['boards', 'buy_commission', 'sell_commission']:
                if opt_col in data:
                    opt_vals = _pad_to_length(data.get(opt_col, []), n)
                    row_data[opt_col] = opt_vals[idx]

            problems_list.append({
                'index': idx,
                'row_number': idx + 1,
                'reasons': reasons,
                'data': row_data,
            })

    return {
        'bad_row_indices': sorted(bad_indices),
        'problems': problems_list,
    }


def _report_and_skip_bad_rows(bad_rows_result: dict, *, trade_type: str = "UNO") -> None:
	if not bad_rows_result or not bad_rows_result.get('problems'):
		return

	problems = bad_rows_result['problems']
	print(f"\n{'='*80}")
	print(f"⚠ FOUND {len(problems)} PROBLEMATIC ROW(S) IN {trade_type} DATA")
	print(f"{'='*80}")

	for prob in problems[:15]:
		row_num = prob['row_number']
		reasons = prob['reasons']
		data = prob['data']

		print(f"\nRow {row_num}:")
		for reason in reasons:
			print(f"  • {reason}")
		print(f"  Data received: {data}")

	if len(problems) > 15:
		print(f"\n... and {len(problems) - 15} more problematic row(s)")

	print(f"\n{'='*80}")
	print("These rows cannot be used for trade calculations and will be SKIPPED.")
	ans = input("Do you want to continue with the remaining valid rows? (y/n): ").strip().lower()
	if ans != 'y' and ans != 'yes' and ans != '':
		raise ValueError(f"{len(problems)} problematic rows were not approved for skipping.")


def validate_required_identifiers(data):
    type_values = data.get('type', [])
    tickers_values = data.get('tickers', [])

    has_type = False
    if type_values:
        for val in type_values:
            if not _is_missing_cell(val):
                has_type = True
                break

    has_tickers = False
    if tickers_values:
        for val in tickers_values:
            if not _is_missing_cell(val):
                has_tickers = True
                break

    missing_fields = []
    if not has_type:
        missing_fields.append("type (security type)")
    if not has_tickers:
        missing_fields.append("tickers (security identifier/ISIN/symbol)")

    if missing_fields:
        error_msg = "ERROR: Unable to calculate trade results!\n\n"
        error_msg += "The following required identifier field(s) are completely missing:\n"
        for field in missing_fields:
            error_msg += f"  - {field}\n"
        error_msg += "\nThese fields are essential for identifying securities and calculating trade results.\n"
        error_msg += "Please ensure your file contains at least one valid entry for each required field."
        raise ValueError(error_msg)

    return True


def validate_date_price_pairing(data):
    dates_buy = data.get('dates_buy', [])
    buy = data.get('buy', [])
    dates_sell = data.get('dates_sell', [])
    sell = data.get('sell', [])

    n = max(len(dates_buy), len(buy), len(dates_sell), len(sell))

    dates_buy = _pad_to_length(dates_buy, n)
    buy = _pad_to_length(buy, n)
    dates_sell = _pad_to_length(dates_sell, n)
    sell = _pad_to_length(sell, n)

    problems = []
    problem_count = 0

    for idx in range(n):
        buy_date = dates_buy[idx]
        buy_price = buy[idx]
        sell_date = dates_sell[idx]
        sell_price = sell[idx]

        has_buy_date = not _is_missing_cell(buy_date)
        has_buy_price = not _is_missing_number(buy_price)

        if has_buy_date and not has_buy_price:
            problem_count += 1
            if problem_count <= 3:
                problems.append(f"Row {idx + 1}: dates_buy='{buy_date}' but buy price is missing")

        if has_buy_price and not has_buy_date:
            problem_count += 1
            if problem_count <= 3:
                problems.append(f"Row {idx + 1}: buy='{buy_price}' but dates_buy is missing")

        has_sell_date = not _is_missing_cell(sell_date)
        has_sell_price = not _is_missing_number(sell_price)

        if has_sell_date and not has_sell_price:
            problem_count += 1
            if problem_count <= 3:
                problems.append(f"Row {idx + 1}: dates_sell='{sell_date}' but sell price is missing")

        if has_sell_price and not has_sell_date:
            problem_count += 1
            if problem_count <= 3:
                problems.append(f"Row {idx + 1}: sell='{sell_price}' but dates_sell is missing")

    if problem_count > 0:
        error_message = f"Date-price pairing validation failed: {problem_count} issue(s) found\n"
        if problems:
            error_message += "\n".join(problems)
            if problem_count > 3:
                error_message += f"\n... and {problem_count - 3} more issue(s)"
        error_message += "\n\nEach buy_date must have a buy_price, and each sell_date must have a sell_price."
        raise ValueError(error_message)


def validate_trade_rows(data):
    volume = data.get('volume', [])
    dates_buy = data.get('dates_buy', [])
    buy = data.get('buy', [])
    dates_sell = data.get('dates_sell', [])
    sell = data.get('sell', [])

    n = max(
        len(volume) if isinstance(volume, list) else 0,
        len(dates_buy) if isinstance(dates_buy, list) else 0,
        len(buy) if isinstance(buy, list) else 0,
        len(dates_sell) if isinstance(dates_sell, list) else 0,
        len(sell) if isinstance(sell, list) else 0,
    )

    volume = _pad_to_length(volume, n)
    dates_buy = _pad_to_length(dates_buy, n)
    buy = _pad_to_length(buy, n)
    dates_sell = _pad_to_length(dates_sell, n)
    sell = _pad_to_length(sell, n)

    problems = []

    for idx in range(n):
        vol = volume[idx]
        if _is_missing_number(vol):
            continue
        try:
            vol_num = float(vol)
        except Exception:
            problems.append(f"Row {idx + 1}: volume='{vol}' is not a number.")
            continue
        if vol_num == 0.0:
            continue

        open_date = dates_buy[idx]
        buy_price = buy[idx]
        close_date = dates_sell[idx]
        sell_price = sell[idx]

        has_open = (open_date is not None) and (not isinstance(open_date, str))
        has_close = (close_date is not None) and (not isinstance(close_date, str))
        has_buy = not _is_missing_number(buy_price) and not isinstance(buy_price, str)
        has_sell = not _is_missing_number(sell_price) and not isinstance(sell_price, str)

        has_open_pair = has_open and has_buy
        has_close_pair = has_close and has_sell

        if has_open_pair or has_close_pair:
            continue

        missing_parts = []
        if has_open and not has_buy:
            missing_parts.append("buy_price (buy) is missing/invalid")
        if not has_open and has_buy:
            missing_parts.append("open_date (dates_buy) is missing/invalid")
        if has_close and not has_sell:
            missing_parts.append("sell_price (sell) is missing/invalid")
        if not has_close and has_sell:
            missing_parts.append("close_date (dates_sell) is missing/invalid")

        if not missing_parts:
            missing_parts.append("neither (dates_buy+buy) nor (dates_sell+sell) is complete")

        problems.append(
            f"Row {idx + 1}: volume={vol_num:g} but cannot calculate trade result because "
            + "; ".join(missing_parts)
        )

    if problems:
        preview = problems[:10]
        more = len(problems) - len(preview)
        msg = "Cannot calculate trade result for the provided data.\n" + "\n".join(preview)
        if more > 0:
            msg += f"\n... and {more} more row(s)."
        raise ValueError(msg)

    return True


def sort_data_by_board(data):
    boards = data.get('boards')
    if not boards:
        return data

    length = len(boards)
    sortable_keys = [k for k, v in data.items() if not str(k).startswith('_') and isinstance(v, list) and len(v) == length]
    order = sorted(range(length), key=lambda i: (str(boards[i]).lower() if boards[i] is not None else ''))

    if order == list(range(length)):
        return data

    for key in sortable_keys:
        data[key] = [data[key][i] for i in order]

    return data


def analyze_sample_format(sample_str):
    cleaned = re.sub(r'[^\d,.\s-]', '', sample_str).strip()

    if not cleaned or not re.search(r'\d', cleaned):
        return ('.', '', 0.0)

    comma_count = cleaned.count(',')
    dot_count = cleaned.count('.')
    space_count = len(re.findall(r'\s+', cleaned))

    decimal_sep = '.'
    thousand_sep = ''
    confidence = 0.5

    if comma_count > 0 and dot_count > 0:
        last_comma = cleaned.rfind(',')
        last_dot = cleaned.rfind('.')
        if last_comma > last_dot:
            decimal_sep = ','
            thousand_sep = '.'
        else:
            decimal_sep = '.'
            thousand_sep = ','
        confidence = 0.95
    elif space_count > 0:
        thousand_sep = ' '
        if comma_count == 1:
            decimal_sep = ','
            confidence = 0.90
        elif dot_count == 1:
            decimal_sep = '.'
            confidence = 0.90
        else:
            confidence = 0.70
    elif comma_count == 1 and dot_count == 0:
        comma_pos = cleaned.rfind(',')
        digits_after = len(cleaned) - comma_pos - 1
        if digits_after >= 2 and digits_after <= 5:
            decimal_sep = ','
            confidence = 0.85
        elif digits_after == 3:
            decimal_sep = ','
            confidence = 0.60
        else:
            decimal_sep = ','
            confidence = 0.70
    elif dot_count == 1 and comma_count == 0:
        dot_pos = cleaned.rfind('.')
        digits_after = len(cleaned) - dot_pos - 1
        if digits_after >= 2 and digits_after <= 5:
            decimal_sep = '.'
            confidence = 0.85
        elif digits_after == 3:
            decimal_sep = '.'
            confidence = 0.60
        else:
            decimal_sep = '.'
            confidence = 0.70
    elif comma_count > 1:
        thousand_sep = ','
        decimal_sep = '.'
        confidence = 0.95
    elif dot_count > 1:
        thousand_sep = '.'
        decimal_sep = ','
        confidence = 0.95
    else:
        confidence = 0.80

    return (decimal_sep, thousand_sep, confidence)


def detect_number_format(series):
    samples = []
    for val in series.head(50):
        if pd.notna(val):
            val_str = str(val).strip()
            if re.search(r'\d', val_str):
                samples.append(val_str)
        if len(samples) >= 10:
            break

    if not samples:
        return {'decimal_sep': '.', 'thousand_sep': '', 'confidence': 0.0}

    format_votes = []
    for sample in samples:
        result = analyze_sample_format(sample)
        format_votes.append(result)

    format_counts = {}
    for dec_sep, thou_sep, conf in format_votes:
        key = (dec_sep, thou_sep)
        if key not in format_counts:
            format_counts[key] = {'count': 0, 'total_confidence': 0.0}
        format_counts[key]['count'] += 1
        format_counts[key]['total_confidence'] += conf

    best_format = None
    best_score = 0.0
    for (dec_sep, thou_sep), fmt_data in format_counts.items():
        avg_conf = fmt_data['total_confidence'] / fmt_data['count']
        score = fmt_data['count'] * avg_conf
        if score > best_score:
            best_score = score
            best_format = (dec_sep, thou_sep, avg_conf)

    if best_format:
        decimal_sep, thousand_sep, confidence = best_format
    else:
        decimal_sep, thousand_sep, confidence = ('.', '', 0.5)

    return {'decimal_sep': decimal_sep, 'thousand_sep': thousand_sep, 'confidence': confidence}


def clean_number(value, decimal_sep='.', thousand_sep=''):
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return value

    value_str = str(value).strip()
    value_str = re.sub(r'[^\d,.\s-]', '', value_str)

    decimal_places = 0
    if decimal_sep in value_str:
        decimal_part = value_str.split(decimal_sep)[-1]
        decimal_part = re.sub(r'[^\d]', '', decimal_part)
        decimal_places = len(decimal_part)

    if thousand_sep:
        value_str = value_str.replace(thousand_sep, '')

    if decimal_sep != '.':
        value_str = value_str.replace(decimal_sep, '.')

    value_str = value_str.replace(' ', '')

    try:
        num = float(value_str)
        if decimal_places > 0:
            return float(f"{num:.{decimal_places}f}")
        return num
    except (ValueError, AttributeError):
        return value


def convert_numeric_series_fast(values, decimal_sep='.', thousand_sep=''):
    series = pd.Series(values)
    s = series.astype(str)
    mask_valid = series.notna() & s.str.strip().ne('') & (s.str.lower() != 'nan')
    cleaned = s.where(mask_valid, '')
    cleaned = cleaned.str.replace(_RE_NUM_KEEP, '', regex=True)

    if decimal_sep and decimal_sep in [',', '.']:
        split = cleaned.str.split(decimal_sep, n=1, expand=True)
        if split.shape[1] == 2:
            dec_part = split[1].fillna('')
            dec_digits = dec_part.str.replace(_RE_NON_DIGITS, '', regex=True)
            dec_places = dec_digits.str.len().fillna(0).astype(int)
        else:
            dec_places = pd.Series([0] * len(cleaned))
    else:
        dec_places = pd.Series([0] * len(cleaned))

    if thousand_sep:
        if thousand_sep.isspace():
            cleaned = cleaned.str.replace(_RE_WS, '', regex=True)
        else:
            cleaned = cleaned.str.replace(re.escape(thousand_sep), '', regex=True)

    if decimal_sep and decimal_sep != '.':
        cleaned = cleaned.str.replace(re.escape(decimal_sep), '.', regex=True)

    cleaned = cleaned.str.replace(_RE_WS, '', regex=True)
    numeric = pd.to_numeric(cleaned, errors='coerce')

    out_values = []
    out_decimals = []
    num_list = numeric.tolist()
    dec_list = dec_places.tolist()
    for original, num, dec in zip(values, num_list, dec_list):
        if pd.isna(num):
            out_values.append(None if pd.isna(original) else original)
            out_decimals.append(0)
        else:
            out_values.append(float(num))
            out_decimals.append(int(dec))

    return out_values, out_decimals


def convert_numeric_columns(data, column_names, manual_format=None, interactive=True):
    decimal_places_meta = data.get('_decimal_places', {})

    for col_name in column_names:
        if col_name in data and data[col_name]:

            try:
                if all(_is_missing_number(v) or (isinstance(v, str) and not v.strip()) for v in data[col_name]):
                    continue
            except Exception:
                pass
            if manual_format:
                format_info = {
                    'decimal_sep': manual_format.get('decimal_sep', '.'),
                    'thousand_sep': manual_format.get('thousand_sep', ''),
                    'confidence': 1.0
                }
                print(f"  Using manual format for '{col_name}': decimal='{format_info['decimal_sep']}', thousand='{format_info['thousand_sep'] or 'none'}'")
            else:
                series = pd.Series(data[col_name])
                format_info = detect_number_format(series)
                confidence_pct = format_info['confidence'] * 100
                print(f"  Detected format for '{col_name}': decimal='{format_info['decimal_sep']}', thousand='{format_info['thousand_sep'] or 'none'}' (confidence: {confidence_pct:.1f}%)")

                if format_info['confidence'] < 0.70:
                    print(f"  ⚠ WARNING: Low confidence ({confidence_pct:.1f}%) in format detection for '{col_name}'.")
                    print(f"     Auto-detected: decimal='{format_info['decimal_sep']}', thousand='{format_info['thousand_sep'] or 'none'}'")
                    if not interactive:
                        use_manual = 'n'
                    else:
                        try:
                            use_manual = input(f"     Do you want to manually specify format for '{col_name}'? (y/n): ").lower()
                        except EOFError:
                            use_manual = 'n'
                    if use_manual == 'y':
                        try:
                            dec_sep = input(f"       Enter decimal separator for '{col_name}' (. or ,): ").strip()
                            thou_sep = input(f"       Enter thousand separator for '{col_name}' (space, comma, dot, or Enter for none): ").strip()
                        except EOFError:
                            dec_sep = ''
                            thou_sep = ''
                        if dec_sep:
                            format_info['decimal_sep'] = dec_sep
                        if thou_sep or thou_sep == '':
                            format_info['thousand_sep'] = thou_sep
                        print(f"  Using manual format for '{col_name}': decimal='{format_info['decimal_sep']}', thousand='{format_info['thousand_sep'] or 'none'}'")

            converted, dec_places = convert_numeric_series_fast(
                data[col_name],
                decimal_sep=format_info['decimal_sep'],
                thousand_sep=format_info['thousand_sep'],
            )
            data[col_name] = converted
            decimal_places_meta[col_name] = dec_places

    data['_decimal_places'] = decimal_places_meta
    return data


def resolve_ambiguous_columns(df, column_mapping, target_columns, aliases=None, interactive=True):
    if aliases is None:
        aliases = {}

    reverse_mapping = {}
    for target, file_col in column_mapping.items():
        if file_col not in reverse_mapping:
            reverse_mapping[file_col] = []
        reverse_mapping[file_col].append(target)

    ambiguous = {file_col: targets for file_col, targets in reverse_mapping.items() if len(targets) > 1}

    if not ambiguous:
        return column_mapping

    if not interactive:


        for file_col, target_list in ambiguous.items():
            tset = set(target_list)
            if {'dates_buy', 'dates_sell'}.issubset(tset):
                continue

            keep = target_list[0]
            for target in target_list[1:]:
                if target in column_mapping and column_mapping[target] == file_col:
                    del column_mapping[target]
            column_mapping[keep] = file_col
        return column_mapping

    print(f"\n⚠ WARNING: Found {len(ambiguous)} column(s) that could match multiple target columns!")
    print("="*80)

    for file_col, target_list in ambiguous.items():
        idx = list(df.columns).index(file_col)
        label = _col_label(idx, file_col)
        print(f"\nFile column: {label}")
        print(f"Could be used for: {', '.join(target_list)}")

        print(f"\nSample values from {label}:")
        sample_values = df[file_col].head(5).tolist()
        for s_idx, val in enumerate(sample_values, 1):
            print(f"  {s_idx}. {val}")

        option_text = ", ".join([f"{i+1}={t}" for i, t in enumerate(target_list)])
        print(f"\nPlease specify which target column(s) should use '{file_col}':")
        print(f"Options: {option_text}")
        print("Enter target names or numbers separated by commas, 'auto' (first option), or 'skip' to skip.")

        while True:
            try:
                user_input = input("> ").strip()
            except EOFError:
                user_input = 'auto'

            if not user_input:
                print("  ✗ No input provided. Please enter a choice, 'auto', or 'skip'.")
                continue

            lower_input = user_input.lower()
            if lower_input in {'n', 'no'}:
                lower_input = 'auto'
            if lower_input == 'skip':
                for target in target_list:
                    if target in column_mapping and column_mapping[target] == file_col:
                        del column_mapping[target]
                print(f"  Skipped '{file_col}' - no targets will use this column")
                break

            if lower_input == 'auto':
                chosen = target_list[:1]
                print(f"  Auto-selected: {', '.join(chosen)}")
            else:
                selected_raw = [p.strip() for p in user_input.split(',') if p.strip()]
                chosen = []
                for entry in selected_raw:
                    if entry.isdigit():
                        idx_num = int(entry) - 1
                        if 0 <= idx_num < len(target_list):
                            chosen.append(target_list[idx_num])
                        else:
                            print(f"  ✗ Number '{entry}' is out of range (1-{len(target_list)})")
                    else:
                        norm = normalize_column_name(entry)
                        match = None
                        for target in target_list:
                            if norm == normalize_column_name(target):
                                match = target
                                break
                        if match:
                            chosen.append(match)
                        else:
                            print(f"  ✗ Unknown target '{entry}'")

            if not chosen:
                print(f"  Options: {option_text}")
                print("  Please enter a choice, 'auto', or 'skip'.")
                continue

            for target in target_list:
                if target in column_mapping and column_mapping[target] == file_col:
                    del column_mapping[target]

            for target in chosen:
                column_mapping[target] = file_col
                print(f"  ✓ Mapped '{file_col}' to target '{target}'")
            break

    print("="*80)
    return column_mapping


def find_matching_column(df, target_columns, aliases=None, interactive=True):
    if aliases is None:
        aliases = {}

    df_cols_normalized = {normalize_column_name(col): col for col in df.columns}
    column_mapping = {}
    df_cols_list = list(df.columns)

    def _pick_best_candidate(cands):

        rank = {'exact': 0, 'alias': 1, 'explicit_alias': 1, 'fuzzy': 2}
        return sorted(cands, key=lambda x: rank.get(x[1], 99))[0]

    def _is_date_like(name: str) -> bool:
        name = (name or '').lower()
        return (
            'date' in name
            or 'time' in name
            or name in {'dt'}
            or 'timestamp' in name
        )

    def _is_price_like(name: str) -> bool:
        name = (name or '').lower()
        return any(k in name for k in ['price', 'cost', 'value', 'rate'])

    for target in target_columns:
        normalized_target = normalize_column_name(target)
        candidates = []


        if normalized_target in df_cols_normalized:
            candidates.append((df_cols_normalized[normalized_target], "exact"))


        for alias in aliases.get(target, []):
            normalized_alias = normalize_column_name(alias)
            if normalized_alias in df_cols_normalized:
                col_name = df_cols_normalized[normalized_alias]
                if all(col_name != c[0] for c in candidates):
                    candidates.append((col_name, "alias"))


        all_possible_names = [normalized_target] + aliases.get(target, [])
        all_normalized_possible = [normalize_column_name(name) for name in all_possible_names]
        for possible_name in all_normalized_possible:


            if len(possible_name) < 5:
                continue

            matches = get_close_matches(possible_name, df_cols_normalized.keys(), n=3, cutoff=0.75)
            for match in matches:

                if _is_price_like(normalized_target) and _is_date_like(match):
                    continue
                if _is_date_like(normalized_target) and _is_price_like(match):
                    continue

                col_name = df_cols_normalized[match]
                if all(col_name != c[0] for c in candidates):
                    candidates.append((col_name, "fuzzy"))

        if not candidates:
            continue

        if len(candidates) == 1:
            selected_col, reason = candidates[0]
            column_mapping[target] = selected_col
            if interactive:
                idx = df_cols_list.index(selected_col)
                print(f"  ✓ Matched '{target}' to { _col_label(idx, selected_col) } ({reason})")
            continue



        if not interactive:
            selected_col, _reason = _pick_best_candidate(candidates)
            column_mapping[target] = selected_col
            continue

        print(f"\nMultiple columns could match target '{target}':")
        for i, (col_name, reason) in enumerate(candidates, 1):
            col_idx = df_cols_list.index(col_name)
            label = _col_label(col_idx, col_name)
            samples = df[col_name].head(5).tolist()
            print(f"  {i}) {label} [{reason}]")
            for j, val in enumerate(samples, 1):
                print(f"     {j}. {val}")

        print("Select the column number/name/label to use, or type 'skip' to leave unmapped.")
        try:
            user_input = input("> ").strip()
        except EOFError:
            user_input = ''

        if not user_input:
            selected = candidates[0][0]
            sel_idx = df_cols_list.index(selected)
            print(f"  Using default: {_col_label(sel_idx, selected)}")
        elif user_input.lower() == 'skip':
            print(f"  Skipped mapping for target '{target}'")
            continue
        else:
            selected = None
            if user_input.isdigit():
                choice = int(user_input)
                if 1 <= choice <= len(candidates):
                    selected = candidates[choice - 1][0]
            if selected is None:
                norm_input = normalize_column_name(user_input)
                for col_name, _ in candidates:
                    label_letter = _excel_col_letter(df_cols_list.index(col_name)).lower()
                    if norm_input == normalize_column_name(col_name) or norm_input == label_letter:
                        selected = col_name
                        break
            if selected is None:
                selected = candidates[0][0]
                sel_idx = df_cols_list.index(selected)
                print(f"  ✗ Invalid selection '{user_input}', using default: {_col_label(sel_idx, selected)}")

        column_mapping[target] = selected
        sel_idx = df_cols_list.index(selected)
        print(f"  ✓ Mapped '{target}' to { _col_label(sel_idx, selected) }")

    return column_mapping


def read_trade_data(file_path, number_format=None, custom_column_names=None, sheet_name=None, *, interactive=True):
    df = read_input_file(file_path, sheet_name=sheet_name)
    return load_inout_from_df(
        df,
        number_format=number_format,
        custom_column_names=custom_column_names,
        interactive=interactive,
    )


def _column_aliases_inout() -> dict:
    return {
        'boards': ['market', 'markets', 'mkt', 'mrkt', 'mrt', 'board', 'mboards', 'mboard', 'market_boards', 'market_board', 'marketboards', 'marketboard', 'exchange', 'exchanges', 'market', 'markets', 'market_type', 'markettype', 'board_type', 'boardtype', 'marketboardtype', 'market_board_type'],
        'type': ['mtype', 'market_type', 'markettype', 'trade_type', 'tradetype', 'type', 'security_type', 'securitytype', 'sectype', 'asset_type', 'assettype', 'instrument_type', 'instrumenttype', 'assetclass', 'asset_class', 'instrumentclass', 'instrument_class', 'securityclass', 'security_class', 'asset', 'instrument', 'security'],
        'tickers': ['ticker', 'symbol', 'symbols', 'stock', 'stocks', 'isin', 'name', 'names', 'asset_name', 'assetname', 'instrument_name', 'instrumentname', 'security_name', 'securityname', 'tickersymbol', 'tickersymbols', 'stockname', 'stock_name', 'tickername', 'ticker_name', 'assetticker', 'instrumentticker', 'securityticker', 'instrument_ticker', 'security_ticker', 'asset_ticker', 'stock_ticker', 'ticker_code', 'tickercode', 'code', 'codes', 'asset_code', 'assetcode', 'instrument_code', 'instrumentcode', 'security_code', 'securitycode', 'isin_code', 'isincode', 'stock_code', 'stockcode', 'tickerid', 'ticker_id', 'securityid', 'security_id', 'instrumentid', 'instrument_id', 'id', 'ids', 'assetid', 'asset_id', 'stockid', 'stock_id', 'symbol_code', 'symbolcode', 'symbolid', 'symbol_id'],
        'volume': ['vol', 'volume', 'quantity', 'qty', 'amount', 'count', 'shares', 'volume_traded', 'volumetraded', 'trade_volume', 'tradevolume', 'number_of_shares', 'numberofshares', 'num_shares', 'numshares', 'trade_qty', 'tradeqty', 'trade_quantity', 'tradequantity'],
        'dates_buy': ['datebuy', 'date_buy', 'datesbuy', 'dates_buy', 'buy_dates', 'buydates', 'dbuy', 'buyd', 'bdate', 'dateb', 'bday', 'dayb', 'buy_day', 'buyday', 'buy_date', 'buydate', 'opendate', 'open_date', 'purchase_date', 'purchasedate', 'buy_time', 'buytime', 'open_time', 'opentime', 'purchase_time', 'purchasetime', 'trade_open_date', 'tradeopendate', 'trade_open_time', 'tradeopentime', 'opening_date', 'openingdate', 'opening_time', 'openingtimes', 'entry_date', 'entrydate', 'entry_time', 'entrytime', 'buy_datetime', 'buydatetime', 'open_datetime', 'opendatetime', 'purchase_datetime', 'purchasedatetime', 'trade_open_datetime', 'tradeopendatetime', 'entry_datetime', 'entrydatetime'],
        'buy': ['buy', 'buy_c', 'price_buy', 'pricebuy', 'pbuy', 'buyp', 'buyc', 'buy_price', 'buyprice', 'price', 'cost', 'purchase_price', 'open_price', 'openprice', 'purchase', 'open_cost', 'opencost', 'buy_cost', 'buycost', 'purchase_cost', 'purchasecost', 'trade_open_cost', 'tradeopencost', 'entry_cost', 'entrycost', 'trade_entry_cost', 'tradeentrycost', 'trade_entry_price', 'tradeentryprice', 'entry_price', 'entryprice'],
        'buy_price': ['buy', 'buy_c', 'price_buy', 'pricebuy', 'pbuy', 'buyp', 'buyc', 'buy_price', 'buyprice', 'purchase_price', 'open_price', 'openprice', 'trade_entry_price', 'tradeentryprice', 'entry_price', 'entryprice'],
        'buy_cost': ['purchase', 'open_cost', 'opencost', 'buy_cost', 'buycost', 'purchase_cost', 'purchasecost', 'trade_open_cost', 'tradeopencost', 'entry_cost', 'entrycost', 'trade_entry_cost', 'tradeentrycost', 'buy_value', 'buy_amount', 'buy_total', 'amount_buy', 'value_buy', 'total_buy', 'cost_buy'],
        'dates_sell': ['date_sell', 'datesell', 'dates_sell', 'datessell', 'sell_dates', 'selldates', 'dsell', 'selld', 'sdate', 'sday', 'sell_day', 'sellday', 'date_close', 'dateclose', 'sell_date', 'selldate', 'close_date', 'closedate', 'sell_time', 'selltime', 'close_time', 'closetime', 'exit_time', 'exittime', 'exit_date', 'exitdate', 'trade_close_date', 'tradeclosedate', 'trade_close_time', 'tradeclosetime', 'closing_date', 'closingdate', 'closing_time', 'closingtime', 'sell_datetime', 'selldatetime', 'close_datetime', 'closedatetime', 'exit_datetime', 'exitdatetime', 'trade_close_datetime', 'tradeclosedatetime'],
        'sell': ['sell', 'sell_price', 'sellprice', 'price_sell', 'pricesell', 'psell', 'sellp', 'sell_c', 'sellc', 'sell_cost', 'sellcost', 'price', 'cost', 'close_cost', 'closecost', 'exit_cost', 'exitcost', 'trade_close_cost', 'tradeclosecost', 'trade_exit_cost', 'tradeexitcost', 'trade_exit_price', 'tradeexitprice', 'close_price', 'closeprice', 'exit_price', 'exitprice'],
        'sell_price': ['sell', 'sell_price', 'sellprice', 'price_sell', 'pricesell', 'psell', 'sellp', 'sell_c', 'sellc', 'trade_exit_price', 'tradeexitprice', 'close_price', 'closeprice', 'exit_price', 'exitprice'],
        'sell_cost': ['sell_cost', 'sellcost', 'close_cost', 'closecost', 'exit_cost', 'exitcost', 'trade_close_cost', 'tradeclosecost', 'trade_exit_cost', 'tradeexitcost', 'sell_value', 'sell_amount', 'sell_total', 'amount_sell', 'value_sell', 'total_sell', 'cost_sell'],
        'buy_commission': ['buy_commission', 'commission_buy', 'buy_fee', 'fee_buy', 'broker_fee_buy', 'broker_commission_buy', 'commissionb', 'feeb', 'com_buy', 'fee_buy_total'],
        'sell_commission': ['sell_commission', 'commission_sell', 'sell_fee', 'fee_sell', 'broker_fee_sell', 'broker_commission_sell', 'commissions', 'fees', 'com_sell', 'fee_sell_total'],
    }


def _column_aliases_uno() -> dict:
    return {
        'boards': ['market', 'markets', 'mkt', 'mrkt', 'mrt', 'board', 'mboards', 'mboard', 'market_boards', 'market_board', 'marketboards', 'marketboard', 'exchange', 'exchanges', 'market', 'markets'],
        'type': ['mtype', 'market_type', 'markettype', 'trade_type', 'tradetype', 'type', 'security_type', 'securitytype', 'sectype', 'asset_type', 'assettype', 'instrument_type', 'instrumenttype', 'asset', 'instrument', 'security'],
        'tickers': ['ticker', 'symbol', 'symbols', 'stock', 'stocks', 'isin', 'name', 'names', 'asset_name', 'assetname', 'instrument_name', 'instrumentname', 'security_name', 'securityname', 'code', 'codes', 'id', 'ids'],
        'date_trade': ['date', 'datetime', 'date_time', 'timestamp', 'dt', 'time', 'trade_date', 'trade_datetime', 'execution_date', 'execution_datetime', 'deal_date', 'deal_datetime'],
        'volume_signed': ['volume', 'vol', 'qty', 'quantity', 'amount', 'size', 'signed_volume', 'volume_signed', 'directional_volume', 'direction', 'side_volume'],
        'price_trade': ['price', 'trade_price', 'execution_price', 'deal_price', 'rate', 'price_trade', 'unit_price', 'price_per_unit'],
        'cost_trade': ['cost', 'value', 'amount', 'total', 'sum', 'trade_value', 'trade_amount', 'trade_total', 'cost_trade', 'total_cost', 'gross_amount'],
        'commission': ['commission', 'fee', 'fees', 'broker_fee', 'broker_commission', 'trade_fee', 'trade_commission', 'commission_trade', 'fee_trade'],
    }


def load_inout_from_df(df, number_format=None, custom_column_names=None, *, interactive=True):
    target_columns = ['boards', 'type', 'tickers', 'volume', 'dates_buy', 'buy', 'buy_commission', 'dates_sell', 'sell', 'sell_commission']
    column_aliases = _column_aliases_inout()

    print(f"\nMatching columns...")
    data, _mapping = extract_data_from_df(
        df,
        target_columns,
        column_aliases=column_aliases,
        custom_column_names=custom_column_names,
    )

    print(f"\nValidating column lengths...")
    validate_column_lengths(data)



    if 'boards' not in _mapping:
        data.pop('boards', None)
    for opt in ['buy_commission', 'sell_commission']:
        if opt not in _mapping:
            data.pop(opt, None)

    ff_cols = ['type', 'tickers']
    if 'boards' in data:
        ff_cols = ['boards'] + ff_cols
    data = forward_fill_rows(data, ff_cols)
    data = normalize_marketboards_and_types(data, interactive=interactive)
    validate_required_identifiers(data)
    data = validate_and_fill_boards(data, interactive=interactive)

    numeric_columns = ['volume', 'buy', 'sell']
    if 'buy_commission' in data:
        numeric_columns.append('buy_commission')
    if 'sell_commission' in data:
        numeric_columns.append('sell_commission')
    data = convert_numeric_columns(data, numeric_columns, manual_format=number_format, interactive=interactive)

    date_columns = ['dates_buy', 'dates_sell']
    data = convert_date_columns(data, date_columns)

    validate_date_price_pairing(data)
    data = check_and_fill_volume(data, interactive=interactive)
    validate_trade_rows(data)
    data = sort_data_by_board(data)

    print(f"\n✓ Loaded {len(df)} rows")
    return data


def identify_trade_type(df, custom_column_names=None, interactive=False):
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    def _score(schema_targets, schema_aliases, required, unique_bonus_key=None):
        mapping = find_matching_column(df, schema_targets, aliases=schema_aliases, interactive=interactive)
        mapping = resolve_ambiguous_columns(df, mapping, schema_targets, aliases=schema_aliases, interactive=interactive)

        have_required = sum(1 for c in required if c in mapping)
        have_total = sum(1 for c in schema_targets if c in mapping)
        bonus = 0
        if unique_bonus_key and unique_bonus_key in mapping:
            bonus += 2
        return {
            'mapping': mapping,
            'have_required': have_required,
            'have_total': have_total,
            'bonus': bonus,
            'score': have_required * 10 + have_total * 2 + bonus,
        }

    uno_targets = ['boards', 'type', 'tickers', 'date_trade', 'volume_signed', 'price_trade', 'cost_trade', 'commission']
    uno_required = ['type', 'tickers', 'date_trade', 'volume_signed']
    uno_res = _score(
        uno_targets,
        _column_aliases_uno(),
        uno_required,
        unique_bonus_key='volume_signed',
    )

    inout_targets = ['boards', 'type', 'tickers', 'volume', 'dates_buy', 'buy', 'buy_price', 'buy_cost', 'buy_commission', 'dates_sell', 'sell', 'sell_price', 'sell_cost', 'sell_commission']
    inout_required = ['type', 'tickers', 'dates_buy', 'dates_sell']
    inout_res = _score(
        inout_targets,
        _column_aliases_inout(),
        inout_required,
        unique_bonus_key='dates_buy',
    )


    if custom_column_names:
        keys = set(custom_column_names.keys())
        if 'volume_signed' in keys or 'date_trade' in keys or 'price_trade' in keys or 'cost_trade' in keys:
            return 'ONES'
        if 'dates_buy' in keys or 'dates_sell' in keys or 'buy' in keys or 'sell' in keys or 'buy_price' in keys or 'buy_cost' in keys or 'sell_price' in keys or 'sell_cost' in keys:
            return 'TWOS'


    uno_ok = uno_res['have_required'] >= 3
    inout_ok = inout_res['have_required'] >= 3
    if not uno_ok and not inout_ok:
        raise ValueError(
            "Could not identify trade type (ONES vs TWOS): not enough recognizable columns. "
            "Please choose type explicitly or provide custom_column_names."
        )


    def _has_negative_numbers(col_name: str) -> bool:
        if not col_name or col_name not in df.columns:
            return False
        s = pd.to_numeric(df[col_name], errors='coerce').dropna()
        if s.empty:
            return False
        return bool((s < 0).any())

    uno_mapping = uno_res.get('mapping', {})
    inout_mapping = inout_res.get('mapping', {})

    if 'volume_signed' in uno_mapping and _has_negative_numbers(uno_mapping['volume_signed']):
        uno_res['score'] += 8
    if 'volume' in inout_mapping and _has_negative_numbers(inout_mapping['volume']):
        inout_res['score'] -= 8


    if inout_mapping.get('dates_buy') and inout_mapping.get('dates_sell') and inout_mapping['dates_buy'] == inout_mapping['dates_sell']:
        inout_res['score'] -= 4
    if inout_mapping.get('buy') and inout_mapping.get('sell') and inout_mapping['buy'] == inout_mapping['sell']:
        inout_res['score'] -= 4

    if uno_res['score'] > inout_res['score']:
        return 'ONES'
    if inout_res['score'] > uno_res['score']:
        return 'TWOS'


    if 'volume_signed' in uno_res['mapping']:
        return 'ONES'
    return 'TWOS'
