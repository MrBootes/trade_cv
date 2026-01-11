import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def search_moex(query, fit=None, search_failures=0):
    """
    Function to search for a ticker or secid on MOEX by name or ISIN code.
    
    :param query: Name, ISIN code, or any other identifier.
    :return: Best matching security (SECID and Type) 
            or None if not found.
    """

    query = query.lower().strip()

    if fit is None:
        fit = query

    base_url = "https://iss.moex.com/iss/securities.json"
    params = {"q": query}
    
    for connects in range(7):
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            securities = data.get("securities", {}).get("data", [])
            columns = data.get("securities", {}).get("columns", [])

            if not securities:
                print("No securities found.")
                return None

            result = []
            for security in securities:
                sec_info = dict(zip(columns, security))
                if fit in (
                    str(sec_info.get("name", "")).lower(), 
                    str(sec_info.get("shortname", "")).lower(), 
                    str(sec_info.get("isin", "")).lower(), 
                    str(sec_info.get("regnumber", "")).lower(), 
                    str(sec_info.get("secid", "")).lower()
                ):
                    result = {
                        "secid": sec_info.get("secid"),
                        "name": sec_info.get("name"),
                        "short_name": sec_info.get("shortname"),
                        "isin": sec_info.get("isin"),
                        "type": sec_info.get("type"),
                        "group": sec_info.get("group"),
                        "regnumber": sec_info.get("regnumber")
                    }
                    return result

            if search_failures >= 2:
                return None

            # Try to find the most accurate alignment with fuzzy/partial matching
            partial_matches = []
            for security in securities:
                sec_info = dict(zip(columns, security))
                name_lower = str(sec_info.get("name", "")).lower()
                shortname_lower = str(sec_info.get("shortname", "")).lower()
                isin_lower = str(sec_info.get("isin", "")).lower()
                regnumber_lower = str(sec_info.get("regnumber", "")).lower()
                secid_lower = str(sec_info.get("secid", "")).lower()

                # Calculate match scores for each field
                fields = [name_lower, shortname_lower, isin_lower, regnumber_lower, secid_lower]
                best_score = 0

                for field in fields:
                    if not field:
                        continue
                    
                    # Exact substring match (highest priority)
                    if fit in field:
                        # Prefer exact matches; penalize based on extra length
                        length_penalty = len(field) / (len(fit) + 1)
                        score = 1.0 / length_penalty
                    elif field in fit:
                        score = len(field) / len(fit)
                    else:
                        # No substring match; use character overlap with length difference penalty
                        overlap = sum(1 for c in fit if c in field)
                        char_score = overlap / max(len(fit), 1)
                        
                        len_diff = abs(len(fit) - len(field))
                        len_penalty = 1.0 - (len_diff / max(len(fit), len(field)))
                        len_penalty = max(0, len_penalty)
                        
                        score = char_score * len_penalty

                    if score > best_score:
                        best_score = score

                if best_score > 0:
                    partial_matches.append({
                        "score": best_score,
                        "data": {
                            "secid": sec_info.get("secid"),
                            "name": sec_info.get("name"),
                            "short_name": sec_info.get("shortname"),
                            "isin": sec_info.get("isin"),
                            "type": sec_info.get("type"),
                            "group": sec_info.get("group"),
                            "regnumber": sec_info.get("regnumber")
                        }
                    })

            # Return top matches sorted by score
            if partial_matches:
                partial_matches.sort(key=lambda x: x["score"], reverse=True)

            best_match = partial_matches[0]
            new_query = best_match["data"]["secid"]
            new_query = f"{new_query.lower().strip()[:-1]}{query[-1]}"
            search_failures += 1
            return search_moex(new_query, fit=query, search_failures=search_failures)

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            #return None
        except Exception as e:
            print(f"An error occurred: {e}")
            #return None
    return None


def search_router(data):

    out_data = []

    with ThreadPoolExecutor(max_workers=20) as executor:
        progress = tqdm(total=len(data), desc="Searching")
        results = []

        for query, result in zip(data, executor.map(search_moex, data)):
            if (isinstance(result, list) or isinstance(result, dict)) and result:
                out_data.append([query, [result['secid'], result['type']]])
            else:
                out_data.append([query, None])
            progress.update(1)

        progress.close()

    return out_data
    

### {f, g, h, j, k, m, n, q, u, v, x, z}
### {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
#fl_map = ('f', 'g', 'h', 'j', 'k', 'm', 'n', 'q', 'u', 'v', 'x', 'z')

# Use case example
if __name__ == "__main__":
    data_app = []

    while True:
        user_query = input("Enter company name, ISIN code, or part of the instrument name (or press Enter to finish): ").strip()
        if user_query == "":
            break
        data_app.append(user_query)

    print("Securities found:")

    results = search_router(data_app)
    for result in results:
        if result[1] is not None:
            print(f"{result[0]}, {result[1]}")
        else:
            print(f"{result[0]}: No results found.")

