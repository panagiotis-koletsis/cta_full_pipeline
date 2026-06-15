from collections import Counter
import json
import math
from collections import defaultdict
import pandas as pd
import os
import time

def cell_index_matching(cell_header_dict, col_data, tab_dom, dom_dict):

    attr = "nope"

    for i in range(len(col_data)):
        cell = col_data.iloc[i]
        #print("cell", cell)

        entry = cell_header_dict.get(str(cell))
        if entry is not None:
            if len(entry) == 1:
                attr, count = next(iter(entry.items()))
                #attr = next(iter(entry))
                if attr not in dom_dict[tab_dom] or count < 5:
                    attr = "nope"
            else: 
                sum_counts = sum(entry.values())
                max_count = max(entry.values())
                if max_count/sum_counts > 0.9:
                    max_header = max(entry, key=entry.get)
                    if max_header in dom_dict[tab_dom] and max_count > 5:
                        attr = max_header
                    else:
                        attr = "nope"

    return attr





MIN_COUNT = 5

def flatten(x):
    for item in x:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

def header_cell_dict(GT_URL_VAL,GT_URL_TRAIN, TABLES_URL, ROUND):
    start_time = time.time()
    #define in which set to create the dictionary
    df_val = pd.read_csv(GT_URL_VAL)
    if GT_URL_TRAIN is not None:
        df_val1 = pd.read_csv(GT_URL_TRAIN)
        df_val = pd.concat([df_val1, df_val], ignore_index=True)

    #create an emtpy dictionary with headers as keys
    header_dict = {h: [] for h in df_val['header'].dropna().unique()}

    #populate the dictionary
    for gt_row in range(len(df_val)):
        #print(df_val.iloc[gt_row])
        table_path = os.path.join(TABLES_URL, df_val["table"][gt_row])
        table = pd.read_json(table_path,compression="gzip",lines=True)
        #print("-",table.head(5),"-")
        col_name = df_val["header"][gt_row]
        col_num = df_val["col"][gt_row]
        #print("--",table[col_num],"--")
        #print(col_name, col_num)
        col = table[col_num]
        for cell in col:
            header_dict[col_name].append(cell)
            #print("---",cell,"---")

    #add counts of each cell value for each header
    counter_dict = {
        key: dict(Counter(flatten(values)).most_common())
        for key, values in header_dict.items()
    }
    #invert in order to have cell values as keys and headers as values (with counts)
    inverted = defaultdict(lambda: defaultdict(int))

    for outer_key, inner_dict in counter_dict.items():
        for k, v in inner_dict.items():
            # Handle NaN (since NaN != NaN)
            if isinstance(k, float) and math.isnan(k):
                k = 'NaN'
            inverted[k][outer_key] += v

    # optional: convert to normal dict
    inverted = dict(inverted)

    #remove the counters that are below the threshold (MIN_COUNT)
    filtered_inverted = {}

    for cell_value, headers in inverted.items():
        # keep only headers with count >= MIN_COUNT
        filtered_headers = {
            h: c for h, c in headers.items() if c >= MIN_COUNT
        }
        
        # only keep cell_value if something remains
        if filtered_headers:
            filtered_inverted[cell_value] = filtered_headers

    inverted = filtered_inverted


    banned = {"NaN", "null", "none", "None"}
    # Dictionary comprehension
    inverted = {
        k: v for k, v in inverted.items() 
        if k not in banned and k is not None
    }


    #save the inverted dictionary as json
    with open(f"SOTABV2forSemTab2023/CTA-SCH-R{ROUND}/supplementary_files/inverted_header_dict.json", "w", encoding="utf-8") as f:
        json.dump(inverted, f, ensure_ascii=False, indent=2)

    end_time = time.time()
    print(f"Total execution time for cell index: {end_time - start_time:.2f} seconds")
    return inverted

