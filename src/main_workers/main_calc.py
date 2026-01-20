from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta
from pathlib import Path
import time
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm, trange

try:
    from . import starter
    from . import calc_visual
except ImportError as e:
    # This module uses package-relative imports. Running it as a file
    # (`python src/main_workers/main_calc.py`) is not supported; run as a module
    # or via the console script.
    if __name__ == "__main__":
        raise SystemExit(
            "This module must be run as a package. Use one of:\n"
            "  - trade-cv\n"
            "  - python src/main.py\n"
            "  - python -m main_workers.main_calc\n"
            "  - python -m main_workers\n"
        )
    raise


def generate_lookup_days():
    global date_format, START_DATE, END_DATE

    while True:
        raw = input("Choose the date record format (d - days; w - weeks, m - months): ").strip().lower()
        if raw.isdigit() and int(raw) in [0, 1, 2]:
            date_format = int(raw)
            break
        if raw in ['d', 'w', 'm']:
            date_format = {'d': 0, 'w': 1, 'm': 2}[raw]
            break
        print("Invalid input. Please enter 'd', 'w', 'm', or their corresponding numbers 0, 1, 2.")

    while True:
        try:
            start_raw = input("Enter the start record date (YYYY-MM-DD): ")
            START_DATE = pd.to_datetime(start_raw, format="%Y-%m-%d")
            break
        except ValueError:
            print("Invalid date format. Please enter the date in YYYY-MM-DD format.")
    while True:
        try:
            end_raw = input("Enter the end record date (YYYY-MM-DD): ")
            END_DATE = pd.to_datetime(end_raw, format="%Y-%m-%d")
            if END_DATE >= START_DATE:
                break
            else:
                print("End date must be on or after start date. Please re-enter.")
        except ValueError:
            print("Invalid date format. Please enter the date in YYYY-MM-DD format.")

    if date_format == 1:

        first = START_DATE
        while first.weekday() != 6 and first < END_DATE:
            first += timedelta(days=1)

        last = END_DATE
        while last.weekday() != 6 and last > START_DATE:
            last -= timedelta(days=1)

        if first > last:
            return []

        out = []
        d = first
        while d <= last:
            out.append(d)
            d += timedelta(days=7)

        out.extend([START_DATE, END_DATE])
        out = sorted({dt.replace(hour=0, minute=0, second=0, microsecond=0) for dt in out})
        return out

    if date_format == 2:

        start_ts = pd.Timestamp(START_DATE.date())
        end_ts = pd.Timestamp(END_DATE.date())

        first = start_ts + pd.offsets.MonthEnd(0)

        last = end_ts + pd.offsets.MonthEnd(0)
        if last.date() > END_DATE.date():
            last = end_ts + pd.offsets.MonthEnd(-1)

        if first > last:
            return []

        months = []
        cur = first
        while cur <= last:
            months.append(cur.to_pydatetime())
            cur = cur + pd.offsets.MonthEnd(1)

        months.extend([START_DATE, END_DATE])
        months = sorted({dt.replace(hour=0, minute=0, second=0, microsecond=0) for dt in months})
        return months

    delta = END_DATE - START_DATE
    return [START_DATE + timedelta(days=i) for i in range(delta.days + 1)]


def fetch_data(ticker, date, endDivider, first_date=-1, multi_return=False):
    global date_format

    if first_date == -1:
        first_date = date
    elif first_date == -2:
        if date_format == 1:
            first_date = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=10)
            first_date = first_date.strftime("%Y-%m-%d")
        elif date_format == 2:
            first_date = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=30)
            first_date = first_date.strftime("%Y-%m-%d")
    else:
        first_date = first_date = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=(first_date - 1))
        first_date = first_date.strftime("%Y-%m-%d")

    for _ in range(5):
        try:
            url = f'http://iss.moex.com/iss/engines/{endDivider[0]}/markets/{endDivider[1]}/securities/{ticker}/candles.json?from={first_date}&till={date}&interval=24'
            response = (requests.get(url, timeout=10))
            response.raise_for_status()
            json_data = response.json()


            if 'candles' in json_data and 'data' in json_data['candles'] and json_data['candles']['data']:
                data = [
                    {k: r[i] for i, k in enumerate(json_data['candles']['columns'])}
                    for r in json_data['candles']['data']
                ]
                frame = pd.DataFrame(data)
                frame_list = frame['close'].tolist()
                if frame_list and len(frame_list) > 0:
                    if not multi_return:
                        return frame_list[-1]
                    elif multi_return:
                        close_dates = frame['begin'].tolist()
                        return frame_list, close_dates
                else:
                    if not multi_return:
                        return None
                    elif multi_return:
                        return None, None
            else:
                if not multi_return:
                    return None
                elif multi_return:
                    return None, None
        except Exception as e:
            time.sleep(1)
    print(f"Unable to fetch data for '{ticker}' ({date}): MOEX API service error.")
    if not multi_return:
        return None
    elif multi_return:
        return None, None

def getPriceTo(ticker, date, endDivider, multi_return=1, isdelta=False):
    global date_format, sequent_none_count, START_DATE, END_DATE

    if date_format == 2 and sequent_none_count >= 3:
        if not isdelta:
            return None
        else:
            return [None for _k in range(multi_return)]
    elif date_format == 1 and sequent_none_count >= 7:
        if not isdelta:
            return None
        else:
            none_return_count = 0
            current_date = pd.to_datetime(date) - timedelta(days=multi_return - 1)
            for di in range(multi_return):
                if current_date.weekday() == 6 or current_date == END_DATE:
                    none_return_count += 1
                current_date += timedelta(days=1)
            multi_return = none_return_count
            sequent_none_count += 2.5
            return [None for _k in range(multi_return)]
    elif date_format == 0 and sequent_none_count >= 12:
        if not isdelta:
            return None
        else:
            return [None for _k in range(multi_return)]


    if date is None:
        if not isdelta:
            return None
        else:
            return [None for _k in range(multi_return)]
    if isinstance(date, np.datetime64):
        if np.isnat(date):
            if not isdelta:
                return None
            else:
                return [None for _k in range(multi_return)]
        date = pd.to_datetime(date).strftime("%Y-%m-%d")
    elif hasattr(date, 'strftime'):
        date = date.strftime("%Y-%m-%d")
    else:
        date = str(date)

    if not isdelta or date_format == 2:
        close_price = fetch_data(ticker, date, endDivider, first_date=-1, multi_return=False)
        if (close_price is not None) and (close_price != 0):
            sequent_none_count = 0
            return close_price
        else:
            close_price = fetch_data(ticker, date, endDivider, first_date=30, multi_return=False)
            if close_price is not None and close_price != 0:
                sequent_none_count = 1
                return close_price
            else:
                sequent_none_count += 1
                return None

    elif isdelta:
        close_price, close_dates = fetch_data(ticker, date, endDivider, first_date=multi_return, multi_return=True)
        if (close_price is not None):
            if len(close_price) < multi_return:
                for k in range(multi_return):
                    current_date = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=multi_return - k - 1)
                    if len(close_dates) <= k:
                        toinsert = True
                    elif len(close_dates) > k and current_date != pd.to_datetime(close_dates[k]):
                        toinsert = True
                    else:
                        toinsert = False
                    if toinsert:
                        try:
                            current_date_temp = current_date
                            delta_prev = 0
                            isinserted = False
                            while current_date_temp >= pd.to_datetime(close_dates[0]):
                                if close_price[k-delta_prev-1] is not None and close_price[k-delta_prev-1] != 0:
                                    close_dates.insert(k, current_date.strftime("%Y-%m-%d"))
                                    close_price.insert(k, close_price[k-delta_prev-1])
                                    isinserted = True
                                    break
                                current_date_temp -= timedelta(days=1)
                                delta_prev += 1
                            if not isinserted:
                                close_price_temp = fetch_data(ticker, datetime.strftime(datetime.strptime(date, "%Y-%m-%d") - timedelta(days=multi_return - k), "%Y-%m-%d"), endDivider, first_date=10, multi_return=False)
                                if close_price_temp is not None and close_price_temp != 0:
                                    close_dates.insert(k, current_date.strftime("%Y-%m-%d"))
                                    close_price.insert(k, close_price_temp)
                                else:
                                    close_dates.insert(k, current_date.strftime("%Y-%m-%d"))
                                    close_price.insert(k, 0)
                        except:
                            close_price_temp = fetch_data(ticker, datetime.strftime(datetime.strptime(date, "%Y-%m-%d") - timedelta(days=multi_return - k), "%Y-%m-%d"), endDivider, first_date=10, multi_return=False)
                            if close_price_temp is not None and close_price_temp != 0:
                                close_dates.insert(k, current_date.strftime("%Y-%m-%d"))
                                close_price.insert(k, close_price_temp)
                            else:
                                close_dates.insert(k, current_date.strftime("%Y-%m-%d"))
                                close_price.insert(k, 0)
            if date_format == 1:
                weekly_price = []
                weekly_dates = []
                for di in range(len(close_dates)):
                    weekly_date = close_dates[di]
                    if pd.to_datetime(weekly_date).weekday() == 6 or pd.to_datetime(weekly_date) == END_DATE or pd.to_datetime(weekly_date) == START_DATE:
                        weekly_price.append(close_price[close_dates.index(weekly_date)])
                        weekly_dates.append(weekly_date)
                close_price = weekly_price
            sequent_none_count = 0
            return close_price
        else:
            if date_format == 1:
                none_return_count = 0
                current_date = pd.to_datetime(date) - timedelta(days=multi_return - 1)
                for di in range(multi_return):
                    if current_date.weekday() == 6 or current_date == END_DATE:
                        none_return_count += 1
                    current_date += timedelta(days=1)
                multi_return = none_return_count
            sequent_none_count += 2.5
            return [None for _k in range(multi_return)]


def dates_traded_finer(start_date, end_date):
    global date_format

    if isinstance(start_date, np.datetime64):
        if np.isnat(start_date):
            return [], []
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, np.datetime64):
        if np.isnat(end_date):
            return [], []
        end_date = pd.to_datetime(end_date)

    new_dates = []
    new_delta = []
    if date_format == 0:
        current_date = start_date - timedelta(days=1)
    elif date_format == 1:
        current_date = start_date - timedelta(days=1)
        while current_date.weekday() != 6:
            current_date -= timedelta(days=1)
    elif date_format == 2:
        if start_date.month != 1:
            current_date = start_date.replace(day=1) - timedelta(days=1)
        else:
            current_date = start_date.replace(month=1, day=31)

    if date_format != 2:
        day_delta = min((end_date - start_date).days + 10, 500)

    if end_date - current_date >= timedelta(days=day_delta):
        while current_date <= end_date:
            if end_date - current_date >= timedelta(days=day_delta):
                current_date += timedelta(days=day_delta)
                new_dates.append(current_date)
                new_delta.append(day_delta)
            else:
                new_dates.append(end_date)
                new_delta.append((end_date - current_date).days)
                break
    else:
        if current_date < start_date:
            current_date = start_date - timedelta(days=1)
        new_dates.append(end_date)
        new_delta.append((end_date - current_date).days)
    return new_dates, new_delta


def process_market_type(mtype):

    if not isinstance(mtype, str):
        return (None, None)

    market = mtype.strip().lower()

    if market in ["share", "stock", "etf", "shares", "stocks", "etfs"]:
        return ("stock", "shares")
    elif market in ["bond", "bonds"]:
        return ("stock", "bonds")
    elif market in ["futures", "future"]:
        return ("futures", "forts")
    elif market == "currency":
        return ("currency", "selt")
    elif market == "index":
        return ("stock", "index")
    else:
        return (None, None)


def calc_main(loaded_data, start_key=None, end_key=None):
    [trade_data, cash_flow_data] = loaded_data
    data_list = trade_data

    if start_key is None:
        start_key = 0
    if end_key is None:
        end_key = len(data_list['tickers'])

    lookup_days = generate_lookup_days()
    lookup_array = np.array(lookup_days, dtype='datetime64[D]')

    global date_format, START_DATE, END_DATE, endDivider, dataList


    endDivider = []
    dataList = []


    if len(lookup_days) == 0:
        return [], [], []

    def _to_day64(dt: datetime) -> np.datetime64:
        return np.datetime64(dt.date(), 'D')

    range_start = START_DATE
    range_end_excl = END_DATE + timedelta(days=1)




    open_missing_buy_is_short = True
    open_missing_sell_is_long = True

    tickers = []
    types = []
    dates_traded = []
    ticker_volume = []
    ticker_abs_trade = []
    ticker_abs_krater = []
    skip_first = False

    for idx in tqdm(range(start_key, end_key), desc="Processing trades", unit="trade"):
        i = idx + start_key

        ticker = data_list['tickers'][i]
        if (not ticker in tickers) or (data_list['type'][i] != types[-1]):
            tickers.append(ticker)
            market_type = data_list['type'][i]
            types.append(market_type)
            if skip_first:
                dates_traded[-1] = np.where(ticker_mask, lookup_array, np.datetime64("NaT"))
            else:
                skip_first = True
            dates_traded.append([])
            ticker_mask = np.full(len(lookup_days), False, dtype=bool)
            ticker_volume.append(np.zeros(len(lookup_days)))
            ticker_abs_trade.append(np.zeros(len(lookup_days)))
            ticker_abs_krater.append(np.zeros(len(lookup_days)))
        date_buy = data_list['dates_buy'][i]
        date_sell = data_list['dates_sell'][i]


        if isinstance(date_buy, datetime) is False and hasattr(date_buy, 'year') and hasattr(date_buy, 'month') and hasattr(date_buy, 'day') and not isinstance(date_buy, str) and date_buy is not None:

            date_buy = datetime.combine(date_buy, datetime.min.time())
        if isinstance(date_sell, datetime) is False and hasattr(date_sell, 'year') and hasattr(date_sell, 'month') and hasattr(date_sell, 'day') and not isinstance(date_sell, str) and date_sell is not None:
            date_sell = datetime.combine(date_sell, datetime.min.time())

        vol = data_list.get('volume', [0])[i]
        try:
            vol = float(vol)
        except Exception:
            vol = 0.0
        if vol == 0.0:
            continue

        buy_cost = data_list.get('buy', [None])[i]
        sell_cost = data_list.get('sell', [None])[i]
        if isinstance(buy_cost, float) and np.isnan(buy_cost):
            buy_cost = None
        if isinstance(sell_cost, float) and np.isnan(sell_cost):
            sell_cost = None






        if date_buy is None and date_sell is None:
            continue

        if date_buy is None:


            open_dt = date_sell
            close_dt = range_end_excl
            close_cash = 0.0
            close_event_dt = None
            if open_missing_buy_is_short:
                position_sign = -1.0
                open_cash = (abs(sell_cost) * vol) if sell_cost is not None else 0.0
            else:

                position_sign = 1.0
                open_cash = -(abs(sell_cost) * vol) if sell_cost is not None else 0.0

        elif date_sell is None:


            open_dt = date_buy
            close_dt = range_end_excl
            close_cash = 0.0
            close_event_dt = None
            if open_missing_sell_is_long:
                position_sign = 1.0
                open_cash = -(abs(buy_cost) * vol) if buy_cost is not None else 0.0
            else:

                position_sign = -1.0
                open_cash = (abs(buy_cost) * vol) if buy_cost is not None else 0.0
        elif date_buy <= date_sell:

            open_dt = date_buy
            close_dt = date_sell
            position_sign = 1.0
            open_cash = -(abs(buy_cost) * vol) if buy_cost is not None else 0.0
            close_cash = (abs(sell_cost) * vol) if sell_cost is not None else 0.0
            close_event_dt = date_sell
        elif date_sell < date_buy:

            open_dt = date_sell
            close_dt = date_buy
            position_sign = -1.0
            open_cash = (abs(sell_cost) * vol) if sell_cost is not None else 0.0
            close_cash = -(abs(buy_cost) * vol) if buy_cost is not None else 0.0
            close_event_dt = date_buy


        if open_dt is None:
            continue


        effective_open = max(open_dt, range_start)
        effective_close = min(close_dt, range_end_excl)
        if (effective_open >= effective_close):
            continue

        open_day = _to_day64(effective_open)
        close_day = _to_day64(effective_close)


        hold_mask = (lookup_array >= open_day) & (lookup_array < close_day)
        if hold_mask.any():
            ticker_mask = ticker_mask | hold_mask
            ticker_volume[-1] = ticker_volume[-1] + (position_sign * vol * hold_mask)


        if open_cash != 0.0:
            step_open_day = _to_day64(max(open_dt, range_start))
            if step_open_day > lookup_array[0]:
                ticker_abs_trade[-1] = ticker_abs_trade[-1] + (open_cash * (lookup_array >= step_open_day))
            else:
                ticker_abs_krater[-1] = ticker_abs_krater[-1] + (position_sign * vol * (lookup_array >= step_open_day))

        if close_cash != 0.0 and close_event_dt is not None:
            step_close_day = _to_day64(max(close_event_dt, range_start))
            ticker_abs_trade[-1] = ticker_abs_trade[-1] + (close_cash * (lookup_array >= step_close_day))
    dates_traded[-1] = np.where(ticker_mask, lookup_array, np.datetime64("NaT"))


    global sequent_none_count
    sequent_none_count = 0

    tickers_profit = np.zeros((len(tickers), len(lookup_days)), dtype=float)
    ticker_volume_prices = np.zeros((len(tickers), len(lookup_days)), dtype=float)

    nptypes = np.array(types)
    ticker_volume = np.array(ticker_volume)
    ticker_abs_trade = np.array(ticker_abs_trade)

    fetch_data = tickers, types, dates_traded, lookup_days, ticker_volume, ticker_abs_trade, ticker_abs_krater
    with ThreadPoolExecutor(max_workers=10) as executor:
        tasks = [
            (tickers[i], executor.submit(fetch_prices, i, *fetch_data))
            for i in range(len(tickers))
        ]

        valid_futures = [fut for _, fut in tasks if fut is not None]
        if valid_futures:
            wait(valid_futures, return_when=ALL_COMPLETED)

        ordered_outputs = []
        for ti in range(len(tasks)):
            _, fut = tasks[ti]
            tickers_profit[ti], ticker_volume_prices[ti], ticker_abs_trade[ti] = fut.result()

    tickers_profit = tickers_profit - tickers_profit[:, 0][:, None]
    ticker_abs_trade = ticker_abs_trade - (ticker_abs_trade * (ticker_volume == 0))[:, 0][:, None]

    start_value = get_starting_cash()
    if start_value is None or start_value == "" or start_value == 0:
        end_value = get_starting_cash(end_value=True)
        if end_value is None or end_value == "" or end_value == 0:
            end_value = None
            start_value = 0
        else:
            start_value = None
    else:
        end_value = None

    if cash_flow_data is not None:
        result_received = cash_calc(lookup_days, cash_flow_data, start_value, end_value, tickers_profit)
        cash_profit_array, total_profit, total_value, start_money = result_received
    else:
        cash_profit_array = np.zeros(len(lookup_days), dtype=float)
        total_profit = np.array(tickers_profit).sum(axis=0)
        if start_value is not None:
            start_money = start_value - total_profit[0]
        elif end_value is not None:
            start_money = end_value - total_profit[-1]
        else:
            start_money = 0.0 - total_profit[0]
        total_value = np.full(len(lookup_days), start_money) + total_profit


    not_future_mask = (nptypes[:, None] != 'futures') & (nptypes[:, None] != 'future')
    plus_ticker_value = (ticker_abs_trade * ((ticker_volume > 0) & not_future_mask)).sum(axis=0)
    total_credit = (ticker_abs_trade * ((ticker_volume < 0) & not_future_mask)).sum(axis=0)
    total_cash = np.full(len(total_profit), start_money)
    total_cash = total_cash + plus_ticker_value + cash_profit_array + (ticker_abs_trade * (ticker_volume == 0)).sum(axis=0)

    total_credit = total_credit - (total_cash * (total_cash < 0))
    total_cash = total_cash * (total_cash >= 0)

    types_unique = np.unique(nptypes).tolist()
    types_profits = np.full(len(types_unique), dtype=object, fill_value=None)
    types_volume_prices = np.full(len(types_unique), dtype=object, fill_value=None)
    for ti in range(len(types_unique)):
        types_profits[ti] = (np.array(tickers_profit)[nptypes == types_unique[ti]].sum(axis=0))
        types_volume_prices[ti] = (np.array(ticker_volume_prices)[nptypes == types_unique[ti]].sum(axis=0))


    result = [
        tickers_profit,
        lookup_days,
        tickers,
        types,
        ticker_volume_prices,
        cash_profit_array,
        total_profit,
        total_cash,
        total_credit,
        total_value,
        types_unique,
        types_profits,
        types_volume_prices,
        start_money
    ]

    return result


def fetch_prices(i, tickers, types, dates_traded, lookup_days, ticker_volume, ticker_abs_trade, ticker_abs_krater):
    global date_format


    if not isinstance(dates_traded[i], np.ndarray) or len(dates_traded[i]) != len(lookup_days):
        return 0.0, np.zeros(len(lookup_days), dtype=float)

    endDivider = process_market_type(types[i])

    flow_dates = [[]]
    for di in range(len(dates_traded[i])):
        date = dates_traded[i][di]
        accept_date = isinstance(date, np.datetime64) and not np.isnat(date) and date is not None
        if accept_date:
            if len(flow_dates[-1]) > 0:
                if accept_date and accept_prev:
                    flow_dates[-1].append(date)
                elif accept_date and not accept_prev:
                    if (len(flow_dates[-1]) < 5 and date_format != 0) or (len(flow_dates[-1]) < 3 and date_format == 0):
                        flow_dates.pop(-1)
                    flow_dates.append([date])
            else:
                flow_dates[-1].append(date)
        accept_prev = accept_date
    if (len(flow_dates[-1]) < 5 and date_format != 0) or (len(flow_dates[-1]) < 3 and date_format == 0):
        flow_dates.pop(-1)

    if len(flow_dates) == 0 or (len(flow_dates) == 1 and len(flow_dates[0]) == 0) or (date_format == 2):
        flow_dates_use = False
    else:
        flow_dates_use = True

    flow_traded = []
    delta_days = []
    if (date_format == 0 or date_format == 1) and flow_dates_use:
        for fd in flow_dates:
            flow_traded.append([])
            delta_days.append([])
            flow_traded[-1], delta_days[-1] = dates_traded_finer(fd[0], fd[-1])
    elif date_format == 2 and flow_dates_use:
        flow_traded = flow_dates


    flow_index = 0


    with ThreadPoolExecutor(max_workers=10) as executor:
        tasks = []
        j = -1
        while j < len(dates_traded[i]) - 1:
            j += 1
            date = dates_traded[i][j]
            if date is None:
                tasks.append((date, None))
                continue
            if isinstance(date, np.datetime64) and np.isnat(date):
                tasks.append((date, None))
                continue

            if flow_dates_use and (flow_index < len(flow_dates)):
                if date in flow_dates[flow_index]:
                    for fi in range(len(flow_traded[flow_index])):
                        fdate = flow_traded[flow_index][fi]
                        fdelta = delta_days[flow_index][fi]
                        tasks.append((fdate, executor.submit(getPriceTo, tickers[i], fdate, endDivider, multi_return=fdelta, isdelta=True)))
                    j += len(flow_dates[flow_index]) - 1
                    flow_index += 1
                else:
                    tasks.append((date, executor.submit(getPriceTo, tickers[i], date, endDivider)))
            else:
                tasks.append((date, executor.submit(getPriceTo, tickers[i], date, endDivider)))


        valid_futures = [fut for _, fut in tasks if fut is not None]
        if valid_futures:
            wait(valid_futures, return_when=ALL_COMPLETED)


        ordered_outputs = []
        for _, fut in tasks:
            if fut is None or fut == 0:
                idx = 0
                has_price = False
                while len(ordered_outputs) - idx > 1:
                    idx += 1
                    if ordered_outputs[-idx] is not None and not np.isnan(ordered_outputs[-idx]):
                        ordered_outputs.append(ordered_outputs[-idx])
                        has_price = True
                        break
                if not has_price:
                    ordered_outputs.append(0)
            else:
                price = fut.result()
                # IMPORTANT: MOEX can return no candle for a date (None).


                if price is None or price == 0:
                    idx = 0
                    has_price = False
                    while len(ordered_outputs) - idx > 1:
                        idx += 1
                        if ordered_outputs[-idx] is not None and not np.isnan(ordered_outputs[-idx]):
                            ordered_outputs.append(ordered_outputs[-idx])
                            has_price = True
                            break
                    if not has_price:
                        ordered_outputs.append(np.nan)
                else:
                    if not isinstance(price, list):
                        ordered_outputs.append(price)
                    elif isinstance(price, list):
                        for out_price in price:
                            if out_price is None or out_price == 0:
                                idx = 0
                                has_price = False
                                while len(ordered_outputs) - idx > 1:
                                    idx += 1
                                    if ordered_outputs[-idx] is not None and not np.isnan(ordered_outputs[-idx]):
                                        ordered_outputs.append(ordered_outputs[-idx])
                                        has_price = True
                                        break
                                if not has_price:
                                    ordered_outputs.append(0)
                            else:
                                ordered_outputs.append(out_price)

    prices = np.array(ordered_outputs, dtype=float)

    ticker_abs_trade[i] = ticker_abs_trade[i] - (ticker_abs_krater[i] * prices[0])

    volume = ticker_volume[i]
    cash = ticker_abs_trade[i]


    profit = cash.astype(float).copy()
    known_price = ~np.isnan(prices)

    if known_price.any():
        profit[known_price] = profit[known_price] + (volume[known_price] * prices[known_price])


    missing_price_while_held = np.isnan(prices) & (volume != 0)
    if missing_price_while_held.any():
        profit[missing_price_while_held] = -np.abs(cash[missing_price_while_held])

    tickers_profit = profit
    ticker_volume_prices = volume * prices

    return tickers_profit, ticker_volume_prices, ticker_abs_trade[i]


def cash_calc(lookup_days, cash_flow_data, start_value=None, end_value=None, tickers_profit=None):
    cash_dates_array = np.array(cash_flow_data.get('date', []), dtype='datetime64[D]')
    cash_flow_array = np.array(cash_flow_data.get('cash', []), dtype=float)

    cash_addr_mask = np.array(lookup_days, dtype='datetime64[D]')[:, None] >= cash_dates_array[None, :]
    cash_profit_array = np.sum(cash_flow_array[None, :] * cash_addr_mask.astype(float), axis=1)

    if tickers_profit is not None:
        total_ticker_profit = np.array(tickers_profit).sum(axis=0)
        total_profit = cash_profit_array + total_ticker_profit
    else:
        total_profit = cash_profit_array

    if start_value is not None:
        start_money = start_value - total_profit[0]
    elif end_value is not None:
        start_money = end_value - total_profit[-1]
    else:
        start_money = 0.0 - total_profit[0]

    total_value = total_profit + start_money

    return cash_profit_array, total_profit, total_value, start_money

def get_starting_cash(end_value=False):
    while True:
        start_value = input(f"Enter the {'starting' if not end_value else 'ending'} portfolio value: ")
        try:
            start_value = float(start_value)
            return start_value
        except Exception:
            if start_value == "":
                return 0.0
            print(f"Invalid input. Please enter a numeric value for {'starting' if not end_value else 'ending'} portfolio value.")



def calc_trade_data(boards_lists):

    for bi in range(len(boards_lists)):
        board, results = boards_lists[bi]

        ok, reason = calc_visual.validate_results(results)
        if not ok:
            print(f"BOARD: {board} - Visuals unavailable: {reason}")
            continue

        specs = calc_visual.available_visuals(results)
        if not specs:
            print(f"BOARD: {board} - No visuals available (all relevant data is None).")
            continue

        board_label = str(board).upper() if board is not None else "(NO BOARD)"
        print("\n" + "=" * 70)
        print(f"BOARD: {board_label} - Available visuals:")
        for i, s in enumerate(specs, start=1):
            print(f"  {i}. {s.label}")
        print("  a. Open ALL")
        print("  n. Next board")
        print("  q. Quit visuals")

        while True:
            choice = input("Select visual(s) to open: ").strip().lower()
            if choice in ("q", "quit", "exit"):
                return
            if choice in ("n", "next", "skip"):
                break
            if choice in ("a", "all"):
                try:
                    calc_visual.open_visuals_dashboard(results, specs, board)
                except Exception as e:
                    print(f"Failed to open ALL visuals dashboard: {e}")
                continue

            parts = [p.strip() for p in choice.split(",") if p.strip()]
            if not parts:
                continue

            selected = []
            valid = True
            for p in parts:
                if not p.isdigit():
                    valid = False
                    break
                selected.append(int(p))
            if not valid:
                print("Invalid selection. Use numbers (e.g. 1 or 1,3) or 'a'/'n'/'q'.")
                continue

            for idx in selected:
                if idx < 1 or idx > len(specs):
                    print(f"Invalid number: {idx}")
                    continue
                s = specs[idx - 1]
                try:
                    calc_visual.open_visual(results, s, board)
                except Exception as e:
                    print(f"Failed to open '{s.label}': {e}")



def router_main(loaded_data):
    trade_data, flow_data = loaded_data

    boards_lists = []

    if trade_data and (trade_data is not None):
        boards = np.unique(trade_data['boards'])
        for bi in range(len(boards)):
            start_key = trade_data['boards'].index(boards[bi])
            try:
                end_key = trade_data['boards'].index(boards[bi + 1])
            except Exception:
                end_key = len(trade_data['boards'])

            result = calc_main((trade_data, flow_data), start_key, end_key)
            boards_lists.append((boards[bi], result))

    elif flow_data and (flow_data is not None) and (not trade_data or trade_data is None):
        lookup_days = generate_lookup_days()
        start_value = get_starting_cash()
        if start_value is None or start_value == "" or start_value == 0:
            end_value = get_starting_cash(end_value=True)
            start_value = None
        else:
            end_value = None

        result_received = cash_calc(lookup_days, flow_data, start_value=start_value, end_value=end_value)
        cash_profit_array, total_profit, total_value, start_money = result_received

        total_cash = np.full(len(total_profit), start_money) + cash_profit_array
        total_credit = np.zeros(len(lookup_days), dtype=float) - (total_cash * (total_cash < 0))
        total_cash = total_cash * (total_cash >= 0)


        result = [
            None,
            lookup_days,
            None,
            None,
            None,
            cash_profit_array,
            total_profit,
            total_cash,
            total_credit,
            total_value,
            None,
            None,
            None,
            start_money
        ]

        boards_lists = [(None, result)]

    return boards_lists



def main() -> None:
    global date_format, START_DATE, END_DATE

    date_format = 0
    START_DATE = None
    END_DATE = None
    loaded_data = starter.main()
    boards_lists = router_main(loaded_data)
    while True:
        calc_trade_data(boards_lists)



if __name__ == "__main__":
    main()
