import json
import numpy as np
import pandas as pd
import pickle as pkl
import sys

from grapher import Grapher, store_edges


# def get_dataframe_memory_gb(df):
#     """Get DataFrame memory usage in GB"""
#     if df.empty:
#         return 0.0
#     return df.memory_usage(deep=True).sum() / (1024**3)

# def should_skip_large_dataframe(df, threshold_gb=8.0):
#     """Check if DataFrame exceeds memory threshold"""
#     memory_gb = get_dataframe_memory_gb(df)
#     return memory_gb > threshold_gb

# def estimate_merge_memory_gb(df_1, df_2, df_col):
#     item_1 = set(df_1[df_col].unique())
#     item_2 = set(df_2[df_col].unique())
#     item = item_1 & item_2
#     n = df_1[df_col].isin(item).sum()
#     m = df_2[df_col].isin(item).sum()
#     k_1 = df_1.shape[1]
#     k_2 = df_2.shape[1]

#     output_rows = n * m
#     output_cols =  (k_1 + k_2) - 1
#     bytes_per_value = 26  # int32

#     total_bytes = output_rows * output_cols * bytes_per_value
#     total_gb = total_bytes / (1024**3)
    
#     return total_gb

def estimate_merge_memory_gb(df_1, df_2, df_col):
    """
    Estimate memory usage of a pandas merge operation more accurately.
    
    Parameters:
        df_1 (pd.DataFrame): First DataFrame
        df_2 (pd.DataFrame): Second DataFrame  
        df_col (str): Column name to merge on
        
    Returns:
        float: Estimated memory usage in GB
    """
    if df_1.empty or df_2.empty:
        return 0.0
    
    # Get unique values in merge column for both DataFrames
    unique_1 = set(df_1[df_col].unique())
    unique_2 = set(df_2[df_col].unique())
    
    # Find common keys (only these will produce results)
    common_keys = unique_1 & unique_2
    
    if not common_keys:
        return 0.0  # No matching keys = empty result
    
    # Calculate actual merge result size more accurately
    total_output_rows = 0
    
    for key in common_keys:
        # Count rows in each DataFrame for this key
        count_1 = (df_1[df_col] == key).sum()
        count_2 = (df_2[df_col] == key).sum()
        
        # For each key, result size is count_1 * count_2
        total_output_rows += count_1 * count_2
        
        # Early termination if getting too large
        if total_output_rows > 10000000:  # 10M rows
            break
    
    # Calculate memory usage
    output_cols = df_1.shape[1] + df_2.shape[1] - 1  # -1 for merge column
    bytes_per_value = 2  # uint16 = 2 bytes
    
    total_bytes = total_output_rows * output_cols * bytes_per_value
    total_gb = total_bytes / (1024**3)
    
    return total_gb


def estimate_merge_memory_gb_fast(df_1, df_2, df_col):
    """
    Fast approximation of merge memory usage using sampling.
    Use this for very large DataFrames where the accurate method is too slow.
    """
    if df_1.empty or df_2.empty:
        return 0.0
    
    # Sample-based estimation for large DataFrames
    if len(df_1) > 50000 or len(df_2) > 50000:
        # Sample 1000 rows from each DataFrame
        sample_size = min(1000, len(df_1), len(df_2))
        df_1_sample = df_1.sample(n=sample_size)
        df_2_sample = df_2.sample(n=sample_size)
        
        # Get average duplicates per key in samples
        avg_dup_1 = len(df_1_sample) / len(df_1_sample[df_col].unique()) if not df_1_sample.empty else 1
        avg_dup_2 = len(df_2_sample) / len(df_2_sample[df_col].unique()) if not df_2_sample.empty else 1
        
        # Estimate overlap
        unique_1 = set(df_1[df_col].unique())
        unique_2 = set(df_2[df_col].unique())
        common_keys = len(unique_1 & unique_2)
        
        if common_keys == 0:
            return 0.0
        
        # Rough estimation
        estimated_output_rows = common_keys * avg_dup_1 * avg_dup_2
        
    else:
        # Use accurate method for smaller DataFrames
        return estimate_merge_memory_gb(df_1, df_2, df_col)
    
    # Calculate memory
    output_cols = df_1.shape[1] + df_2.shape[1] - 1
    bytes_per_value = 2  # uint16
    
    total_bytes = estimated_output_rows * output_cols * bytes_per_value
    total_gb = total_bytes / (1024**3)
    
    return total_gb

# Example:
# memory_gb = estimate_merge_memory_gb(100000, 50000, 5, avg_matches_per_row=10)
# print(f"Estimated memory: {memory_gb:.4f} GB")
# # Output: Estimated memory: 0.0084 GB



def filter_rules(rules_dict, min_conf, min_body_supp, rule_lengths):
    """
    Filter for rules with a minimum confidence, minimum body support, and
    specified rule lengths.

    Parameters.
        rules_dict (dict): rules
        min_conf (float): minimum confidence value
        min_body_supp (int): minimum body support value
        rule_lengths (list): rule lengths

    Returns:
        new_rules_dict (dict): filtered rules
    """

    new_rules_dict: dict[int, list] = dict()
    for k in rules_dict:
        new_rules_dict[k] = []
        for rule in rules_dict[k]:
            if rule["type"] == "link_star":
                cond = (
                    (rule["back_conf"] >= min_conf)
                    and (rule["forw_conf"] >= min_conf)
                    and (rule["back_body_supp"] >= min_body_supp)
                    and (rule["forw_body_supp"] >= min_body_supp)
                )
                # cond = (
                #     (max(rule["back_conf"], rule["forw_conf"]) >= min_conf)
                #     and (max(rule["back_body_supp"], rule["forw_body_supp"]) >= min_body_supp)
                # )
            else:
                cond = (
                    (rule["conf"] >= min_conf)
                    and (rule["body_supp"] >= min_body_supp)
                    and (len(rule["body_rels"]) in rule_lengths)
                )
            if cond:
                new_rules_dict[k].append(rule)

    return new_rules_dict

def get_window_edges_v2(all_data, test_query_ts, learn_edges, window=-1):
    """
    Get the edges in the data (for rule application) that occur in the specified time window.
    If window is 0, all edges before the test query timestamp are included.
    If window is -1, the edges on which the rules are learned are used.
    If window is an integer n > 0, all edges within n timestamps before the test query
    timestamp are included.

    Parameters:
        all_data (np.ndarray): complete dataset (train/valid/test)
        test_query_ts (np.ndarray): test query timestamp
        learn_edges (dict): edges on which the rules are learned
        window (int): time window used for rule application

    Returns:
        window_edges (dict): edges in the window for rule application
    """

    if window > 0:
        mask = (all_data[:, 3] < test_query_ts) & (
            all_data[:, 3] >= test_query_ts - window
        )
        window_edges = store_edges(all_data[mask])
    elif window == 0:
        mask = all_data[:, 3] < test_query_ts
        window_edges = store_edges(all_data[mask])
    elif window == -1:
        window_edges = learn_edges

    return window_edges

# Have to change this functionality
def get_window_edges(train_data, valid_data, test_data, test_query_ts, window=-1):
    """
    Get the edges in the data (for rule application) that occur in the specified time window.
    If window is 0, all edges before the test query timestamp are included.
    If window is -1, the edges on which the rules are learned are used.
    If window is an integer n > 0, all edges within n timestamps before the test query
    timestamp are included.

    Parameters:
        all_data (np.ndarray): complete dataset (train/valid/test)
        test_query_ts (np.ndarray): test query timestamp
        learn_edges (dict): edges on which the rules are learned
        window (int): time window used for rule application

    Returns:
        window_edges (dict): edges in the window for rule application
    """
    all_data = np.vstack((train_data.all_edges, valid_data.all_edges, test_data.all_edges))

    if window > 0:
        mask = (all_data[:, 3] < test_query_ts) * (
            all_data[:, 3] >= test_query_ts - window
        )
        window_edges = store_edges(all_data[mask])
    elif window == 0:
        mask = all_data[:, 3] < test_query_ts
        window_edges = store_edges(all_data[mask])
    elif window == -1:
        window_edges = train_data.edges

    return window_edges

# def store_edges(edges):
#     edges_dict = dict()
#     for i in range(len(edges)):
#         rel = edges[i][1]
#         if rel not in edges_dict:
#             edges_dict[rel] = []
#         edges_dict[rel].append(edges[i])
    
#     for rel in edges_dict:
#         edges_dict[rel] = np.array(edges_dict[rel])
    
#     return edges_dict

def match_link_star_body_relations(rule, edges, test_query_sub):

    head_rel = rule["head_rel"]
    body_rels = rule["body_rels"]
    walk_edges = []
    try:
        head_edges = edges[head_rel]
        mask = head_edges[:, 0] == test_query_sub
        head_edges = head_edges[mask]
        # walk_edges = [np.hstack((new_edges[:, 0:1], new_edges[:, 2:4]))]  # [sub, obj, ts]


        try:
            rel_edge = edges[body_rels[0]]
            mask = rel_edge[:, 2] == test_query_sub
            new_edges = rel_edge[mask]
            walk_edges.append(np.hstack((new_edges[:, 0:1], new_edges[:, 2:4])))  # [sub, obj, ts]
        except KeyError as e:
            print(f"KeyError: {e} for body relation {body_rels[0]}")
            walk_edges.append([])


        walk_edges.append(np.hstack((head_edges[:, 0:1], head_edges[:, 2:4])))  # [sub, obj, ts]

        cur_targets = np.array(list(set(walk_edges[-1][:, 1])))

        try:
            rel_edge = edges[body_rels[1]]
            mask = np.any(rel_edge[:, 0] == cur_targets[:, None], axis=0)
            new_edges = rel_edge[mask]
            walk_edges.append(np.hstack((new_edges[:, 0:1], new_edges[:, 2:4])))  # [sub, obj, ts]
        except KeyError:
            walk_edges.append([])

    except KeyError:
        return [[]]

    return walk_edges

def match_body_relations(rule, edges, test_query_sub):
    """
    Find edges that could constitute walks (starting from the test query subject)
    that match the rule.
    First, find edges whose subject match the query subject and the relation matches
    the first relation in the rule body. Then, find edges whose subjects match the
    current targets and the relation the next relation in the rule body.
    Memory-efficient implementation.

    Parameters:
        rule (dict): rule from rules_dict
        edges (dict): edges for rule application
        test_query_sub (int): test query subject

    Returns:
        walk_edges (list of np.ndarrays): edges that could constitute rule walks
    """

    rels = rule["body_rels"]
    # Match query subject and first body relation
    try:
        rel_edges = edges[rels[0]]
        mask = rel_edges[:, 0] == test_query_sub
        new_edges = rel_edges[mask]
        walk_edges = [
            np.hstack((new_edges[:, 0:1], new_edges[:, 2:4]))
        ]  # [sub, obj, ts]
        cur_targets = np.array(list(set(walk_edges[0][:, 1])))

        for i in range(1, len(rels)):
            # Match current targets and next body relation
            try:
                rel_edges = edges[rels[i]]
                mask = np.any(rel_edges[:, 0] == cur_targets[:, None], axis=0)
                new_edges = rel_edges[mask]
                walk_edges.append(
                    np.hstack((new_edges[:, 0:1], new_edges[:, 2:4]))
                )  # [sub, obj, ts]
                cur_targets = np.array(list(set(walk_edges[i][:, 1])))
            except KeyError:
                walk_edges.append([])
                break
    except KeyError:
        walk_edges = [[]]

    return walk_edges


def match_body_relations_complete(rule, edges, test_query_sub):
    """
    Find edges that could constitute walks (starting from the test query subject)
    that match the rule.
    First, find edges whose subject match the query subject and the relation matches
    the first relation in the rule body. Then, find edges whose subjects match the
    current targets and the relation the next relation in the rule body.

    Parameters:
        rule (dict): rule from rules_dict
        edges (dict): edges for rule application
        test_query_sub (int): test query subject

    Returns:
        walk_edges (list of np.ndarrays): edges that could constitute rule walks
    """

    rels = rule["body_rels"]
    # Match query subject and first body relation
    try:
        rel_edges = edges[rels[0]]
        mask = rel_edges[:, 0] == test_query_sub
        new_edges = rel_edges[mask]
        walk_edges = [new_edges]
        cur_targets = np.array(list(set(walk_edges[0][:, 2])))
        

        for i in range(1, len(rels)):
            # Match current targets and next body relation
            try:
                rel_edges = edges[rels[i]]
                mask = np.any(rel_edges[:, 0] == cur_targets[:, None], axis=0)
                new_edges = rel_edges[mask]
                walk_edges.append(new_edges)
                cur_targets = np.array(list(set(walk_edges[i][:, 2])))
            except KeyError:
                walk_edges.append([])
                break
    except KeyError:
        walk_edges = [[]]

    return walk_edges

def get_link_star_walks_v2(rule, walk_edges):
    rule_walk = pd.DataFrame(
        walk_edges[0],
        columns=["entity_" + str(0), "entity_" + str(1), "timestamp_" + str(0)],
        dtype=np.uint16,
    )

    for i in range(1, len(walk_edges)):
        tmp_df = pd.DataFrame(
            walk_edges[i],
            columns=["entity_" + str(i), "entity_" + str(i + 1), "timestamp_" + str(i)],
            dtype=np.uint16,
        )

        rule_walk = pd.merge(rule_walk, tmp_df, on=["entity_" + str(i)])

        if i == 1:
            rule_walk = rule_walk[
                rule_walk["timestamp_" + str(i - 1)] < rule_walk["timestamp_" + str(i)]
            ]
        else:
            rule_walk = rule_walk[
                rule_walk["timestamp_" + str(i - 1)] > rule_walk["timestamp_" + str(i)]
            ]
        
        del tmp_df
    
    # del rule_walk["entity_" + str(0)]
    # del rule_walk["entity_" + str(len(walk_edges) - 1)]

    return rule_walk


## Need improvement
def get_link_star_walks(rule, walk_edges):
    df_edges = []
    df = pd.DataFrame(
        walk_edges[0],
        columns=["entity_" + str(0), "entity_" + str(1), "timestamp_" + str(0)],
        dtype=np.uint16,
    )  # Change type if necessary for better memory efficiency
    # del df["entity_" + str(0)]
    df_edges.append(df)
    df = df[0:0]  # Memory efficiency

    for i in range(1, len(walk_edges)):
        df = pd.DataFrame(
            walk_edges[i],
            columns=["entity_" + str(i), "entity_" + str(i + 1), "timestamp_" + str(i)],
            dtype=np.uint16,
        )
        df_edges.append(df)
        df = df[0:0]
    
    rule_walks = df_edges[0]
    df_edges[0] = df_edges[0][0:0]
    for i in range(1, len(df_edges)):
        rule_walks = pd.merge(rule_walks, df_edges[i], on=["entity_" + str(i)])
        if i == 1:
            rule_walks = rule_walks[
                rule_walks["timestamp_" + str(i - 1)] < rule_walks["timestamp_" + str(i)]
            ]
        else:
            rule_walks = rule_walks[
                rule_walks["timestamp_" + str(i - 1)] > rule_walks["timestamp_" + str(i)]
            ]

        df_edges[i] = df_edges[i][0:0]
    
    
    
    return rule_walks


def get_walks_v2(rule, walk_edges, rules_type="cyclic", delta=0):
    rule_walks = pd.DataFrame(
        walk_edges[0],
        columns=["entity_" + str(0), "entity_" + str(1), "timestamp_" + str(0)],
        dtype=np.uint16,
    )
    if not rule["var_constraints"]:
        del rule_walks["entity_" + str(0)]
    
    for i in range(1, len(walk_edges)):
        tmp_df = pd.DataFrame(
            walk_edges[i],
            columns=["entity_" + str(i), "entity_" + str(i + 1), "timestamp_" + str(i)],
            dtype=np.uint16,
        )

        rule_walks = pd.merge(rule_walks, tmp_df, on=["entity_" + str(i)])

        if rules_type == "cyclic":
            # print(",", end="")
            rule_walks = rule_walks[
                rule_walks["timestamp_" + str(i - 1)] <= rule_walks["timestamp_" + str(i)]
            ]
        elif rules_type == "relaxed_cyclic":
            rule_walks = rule_walks[
                (rule_walks["timestamp_" + str(i - 1)]-delta) <= rule_walks["timestamp_" + str(i)]
            ]

        if not rule["var_constraints"]:
            del rule_walks["entity_" + str(i)]
        
        del tmp_df
    
    for i in range(1, len(rule["body_rels"])):
        del rule_walks["timestamp_" + str(i)]

    return rule_walks

def get_walks(rule, walk_edges, rules_type="cyclic", id2ts=None, delta=0):
    """
    Get walks for a given rule. Take the time constraints into account.
    Memory-efficient implementation.

    Parameters:
        rule (dict): rule from rules_dict
        walk_edges (list of np.ndarrays): edges from match_body_relations

    Returns:
        rule_walks (pd.DataFrame): all walks matching the rule
    """
    # print(delta)
    df_edges = []
    df = pd.DataFrame(
        walk_edges[0],
        columns=["entity_" + str(0), "entity_" + str(1), "timestamp_" + str(0)],
        dtype=np.uint16,
    )  # Change type if necessary for better memory efficiency
    if rules_type == "relaxed_cyclic":
        df["timestamp_tmp_0"] = df["timestamp_0"]
        df["timestamp_"+ str(0)] = df["timestamp_" + str(0)].map(id2ts)
        df["timestamp_" + str(0)] = pd.to_datetime(df["timestamp_" + str(0)])
    if not rule["var_constraints"]:
        del df["entity_" + str(0)]
    df_edges.append(df)
    df = df[0:0]  # Memory efficiency

    for i in range(1, len(walk_edges)):
        df = pd.DataFrame(
            walk_edges[i],
            columns=["entity_" + str(i), "entity_" + str(i + 1), "timestamp_" + str(i)],
            dtype=np.uint16,
        )  # Change type if necessary
        if rules_type == "relaxed_cyclic":
            df["timestamp_" + str(i)] = df["timestamp_" + str(i)].map(id2ts)
            df["timestamp_" + str(i)] = pd.to_datetime(df["timestamp_" + str(i)])

        df_edges.append(df)
        df = df[0:0]

    rule_walks = df_edges[0]
    df_edges[0] = df_edges[0][0:0]
    for i in range(1, len(df_edges)):
        rule_walks = pd.merge(rule_walks, df_edges[i], on=["entity_" + str(i)])
        if rules_type == "cyclic":
            rule_walks = rule_walks[
                rule_walks["timestamp_" + str(i - 1)] <= rule_walks["timestamp_" + str(i)]
            ]
        elif rules_type == "relaxed_cyclic":

            rule_walks = rule_walks[
                (rule_walks["timestamp_" + str(i - 1)]-pd.to_timedelta(delta, unit='d')) <= rule_walks["timestamp_" + str(i)]
            ]


        if not rule["var_constraints"]:
            del rule_walks["entity_" + str(i)]
        df_edges[i] = df_edges[i][0:0]

    for i in range(1, len(rule["body_rels"])):
        del rule_walks["timestamp_" + str(i)]
    
    if rules_type == "relaxed_cyclic":
        rule_walks["timestamp_0"] = rule_walks["timestamp_tmp_0"]
        del rule_walks["timestamp_tmp_0"]

    return rule_walks


def get_walks_complete(rule, walk_edges):
    """
    Get complete walks for a given rule. Take the time constraints into account.

    Parameters:
        rule (dict): rule from rules_dict
        walk_edges (list of np.ndarrays): edges from match_body_relations

    Returns:
        rule_walks (pd.DataFrame): all walks matching the rule
    """

    df_edges = []
    df = pd.DataFrame(
        walk_edges[0],
        columns=[
            "entity_" + str(0),
            "relation_" + str(0),
            "entity_" + str(1),
            "timestamp_" + str(0),
        ],
        dtype=np.uint16,
    )  # Change type if necessary for better memory efficiency
    df_edges.append(df)

    for i in range(1, len(walk_edges)):
        df = pd.DataFrame(
            walk_edges[i],
            columns=[
                "entity_" + str(i),
                "relation_" + str(i),
                "entity_" + str(i + 1),
                "timestamp_" + str(i),
            ],
            dtype=np.uint16,
        )  # Change type if necessary
        df_edges.append(df)

    rule_walks = df_edges[0]
    for i in range(1, len(df_edges)):
        rule_walks = pd.merge(rule_walks, df_edges[i], on=["entity_" + str(i)])
        rule_walks = rule_walks[
            rule_walks["timestamp_" + str(i - 1)] <= rule_walks["timestamp_" + str(i)]
        ]

    return rule_walks


def check_var_constraints(var_constraints, rule_walks):
    """
    Check variable constraints of the rule.

    Parameters:
        var_constraints (list): variable constraints from the rule
        rule_walks (pd.DataFrame): all walks matching the rule

    Returns:
        rule_walks (pd.DataFrame): all walks matching the rule including the variable constraints
    """

    for const in var_constraints:
        for i in range(len(const) - 1):
            rule_walks = rule_walks[
                rule_walks["entity_" + str(const[i])]
                == rule_walks["entity_" + str(const[i + 1])]
            ]

    return rule_walks

def check_var_constraints_acyclic(rule_walks):
    
    rule_walks = rule_walks[rule_walks["entity_0"] != rule_walks["entity_2"]]
    rule_walks = rule_walks[rule_walks["entity_1"] != rule_walks["entity_3"]]

    return rule_walks

def get_candidates_v2(
    rule, rule_walks, test_query_ts, cands_dict, cands_dict_comb, score_func, args, dicts_idx
):
    """
    Get from the walks that follow the rule the answer candidates.
    Add the confidence of the rule that leads to these candidates.

    Parameters:
        rule (dict): rule from rules_dict
        rule_walks (pd.DataFrame): rule walks (satisfying all constraints from the rule)
        test_query_ts (int): test query timestamp
        cands_dict (dict): candidates along with the confidences of the rules that generated these candidates
        cands_dict_comb (dict): combined candidates along with the confidences of the rules that generated these candidates
        score_func (function): function for calculating the candidate score
        args (list): arguments for the scoring function
        dicts_idx (list): indices for candidate dictionaries

    Returns:
        cands_dict (dict): updated candidates
    """
    if rule["type"] == "link_star":
        max_entity = "entity_" + str(2)
    else:
        max_entity = "entity_" + str(len(rule["body_rels"]))
    
    cands = set(rule_walks[max_entity])

    for cand in cands:
        cands_walks = rule_walks[rule_walks[max_entity] == cand]
        for s in dicts_idx:
            score = score_func(rule, cands_walks, test_query_ts, *args[s]).astype(
                np.float32
            )
            try:
                cands_dict[s][cand].append(score)
            except KeyError:
                cands_dict[s][cand] = [score]

            try:
                cands_dict_comb[s][cand].append(score)
            except KeyError:
                cands_dict_comb[s][cand] = [score]

    return cands_dict, cands_dict_comb

def get_candidates(
    rule, rule_walks, test_query_ts, cands_dict, score_func, args, dicts_idx
):
    """
    Get from the walks that follow the rule the answer candidates.
    Add the confidence of the rule that leads to these candidates.

    Parameters:
        rule (dict): rule from rules_dict
        rule_walks (pd.DataFrame): rule walks (satisfying all constraints from the rule)
        test_query_ts (int): test query timestamp
        cands_dict (dict): candidates along with the confidences of the rules that generated these candidates
        score_func (function): function for calculating the candidate score
        args (list): arguments for the scoring function
        dicts_idx (list): indices for candidate dictionaries

    Returns:
        cands_dict (dict): updated candidates
    """
    if rule["type"] == "link_star":
        max_entity = "entity_" + str(2)
    else:
        max_entity = "entity_" + str(len(rule["body_rels"]))
    
    cands = set(rule_walks[max_entity])

    for cand in cands:
        cands_walks = rule_walks[rule_walks[max_entity] == cand]
        for s in dicts_idx:
            score = score_func(rule, cands_walks, test_query_ts, *args[s]).astype(
                np.float32
            )
            try:
                cands_dict[s][cand].append(score)
            except KeyError:
                cands_dict[s][cand] = [score]

    return cands_dict


def save_candidates(
    rules_file:str, dir_path, all_candidates, rule_lengths, window, score_func_str
):
    # pass
    all_candidates = {int(k): v for k, v in all_candidates.items()}
    for k in all_candidates:
        all_candidates[k] = {int(cand): float(v) for cand, v in all_candidates[k].items()}
    filename = "{0}_cands_r{1}_w{2}_{3}.pkl".format(
        ".".join(rules_file.split('.')[:-1]), rule_lengths, window, score_func_str
    )
    filename = filename.replace(" ", "")
    # with open(dir_path + filename, "w", encoding="utf-8") as fout:
    #     json.dump(all_candidates, fout)

    with open(dir_path + filename, "wb") as fout:
        pkl.dump(all_candidates, fout)


def verbalize_walk(walk, data):
    """
    Verbalize walk from rule application.

    Parameters:
        walk (pandas.core.series.Series): walk that matches the rule body from get_walks
        data (grapher.Grapher): graph data

    Returns:
        walk_str (str): verbalized walk
    """

    l = len(walk) // 3
    walk = walk.values.tolist()

    walk_str = data.id2entity[walk[0]] + "\t"
    for j in range(l):
        walk_str += data.id2relation[walk[3 * j + 1]] + "\t"
        walk_str += data.id2entity[walk[3 * j + 2]] + "\t"
        walk_str += data.id2ts[walk[3 * j + 3]] + "\t"

    return walk_str[:-1]

# def match_and_get_walks_v(rule, edges, test_query_sub, rules_type="cyclic", delta=0):
#     rels = rule["body_rels"]

#     if not rels:
#         return pd.DataFrame()
    
#     try:
#         rel_edges = edges[rels[0]]
#         mask = rel_edges[:, 0] == test_query_sub
#         first_edges = rel_edges[mask]

#         if len(first_edges) == 0:
#             return pd.DataFrame()
        
#         rule_walks = pd.DataFrame(
#             np.hstack((first_edges[:, 0:1], first_edges[:, 2:4])),  # [sub, obj, ts]
#             columns=["entity_0", "entity_1", "timestamp_0"],
#             dtype=np.uint16,
#         )

#         if not rule["var_constraints"]:
#             del rule_walks["entity_" + str(0)]
        
#         del first_edges
#         rel_edges = None
        
#     except KeyError:
#         return pd.DataFrame()
    
#     for i in range(1, len(rels)):
        
def match_and_get_link_star_walks_v2(rule, edges, test_query_sub, max_memory_gb=4):
    rels = rule["body_rels"]

    if not rels:
        return pd.DataFrame(), False
    
    try:
        rel_edges = edges[rule["head_rel"]]
        mask = rel_edges[:, 0] == test_query_sub
        head_edges = rel_edges[mask]

        if len(head_edges) == 0:
            return pd.DataFrame(), False
        
        rule_walks = pd.DataFrame(
            np.hstack((head_edges[:, 0:1], head_edges[:, 2:4])),  # [sub, obj, ts]
            columns=["entity_1", "entity_2", "timestamp_1"],
            dtype=np.uint16,
        )

        del head_edges
        rel_edges = None
    
    except KeyError:
        return pd.DataFrame(), False

    try:
        if rule_walks.empty:
            return pd.DataFrame(), False

        body_edges_1 = edges[rels[0]]
        mask = body_edges_1[:, 2] == test_query_sub
        new_edges_1 = body_edges_1[mask]

        next_df = pd.DataFrame(
            np.hstack((new_edges_1[:, 0:1], new_edges_1[:, 2:4])),  # [sub, obj, ts]
            columns=["entity_0", "entity_1", "timestamp_0"],
            dtype=np.uint16,
        )

        if max_memory_gb:
                est_size = estimate_merge_memory_gb(rule_walks, next_df, f"entity_{1}")
                if est_size > max_memory_gb:
                    print(est_size)
                    return pd.DataFrame(), True

        rule_walks_tmp = pd.merge(rule_walks, next_df, on=["entity_1"], how='inner')
        next_df = next_df[0:0]
        new_edges_1 = None
        body_edges_1 = None

        if rule_walks_tmp.empty:
            return pd.DataFrame(), False
        
        rule_walks_tmp = rule_walks_tmp[
            rule_walks_tmp["timestamp_0"] < rule_walks_tmp["timestamp_1"]
        ]
        rule_walks_tmp = rule_walks_tmp[
            rule_walks_tmp["entity_0"] != rule_walks_tmp["entity_2"]
        ]
        
        if rule_walks_tmp.empty:
            return pd.DataFrame(), False
        
        
        entity_col = f"entity_{2}"
        if entity_col not in rule_walks_tmp.columns:
            return pd.DataFrame(), False
        
        cur_targets = np.array(list(set(rule_walks_tmp[entity_col])))
        del rule_walks_tmp

        rel_edges = edges[rels[1]]
        mask = np.any(rel_edges[:, 0] == cur_targets[:, None], axis=0)
        new_edges_2 = rel_edges[mask]
        if len(new_edges_2) == 0:
            return pd.DataFrame(), False
        
        next_df = pd.DataFrame(
            np.hstack((new_edges_2[:, 0:1], new_edges_2[:, 2:4])),  # [sub, obj, ts]
            columns=["entity_2", "entity_3", "timestamp_0"],
            dtype=np.uint16,
        )
        # est_size = estimate_merge_memory_gb(rule_walks, next_df, "entity_2")

        if max_memory_gb:
                est_size = estimate_merge_memory_gb(rule_walks, next_df, f"entity_{2}")
                if est_size > max_memory_gb:
                    print(est_size)
                    return pd.DataFrame(), True

        rule_walks = pd.merge(rule_walks, next_df, on=["entity_2"], how='inner')
        # breakpoint()
        next_df = next_df[0:0]
        new_edges_2 = None
        rel_edges = None
        if rule_walks.empty:
            return pd.DataFrame(), False
    
        rule_walks = rule_walks[
            rule_walks["timestamp_1"] > rule_walks["timestamp_0"]
        ]

        rule_walks = rule_walks[
            rule_walks["entity_1"] != rule_walks["entity_3"]
        ]

    except KeyError:
        return pd.DataFrame(), False
    
    return rule_walks, False


def match_and_get_link_star_walks(rule, edges, test_query_sub):
    rels = rule["body_rels"]

    if not rels:
        return pd.DataFrame()
    
    try:
        rel_edges = edges[rule["head_rel"]]
        mask = rel_edges[:, 0] == test_query_sub
        head_edges = rel_edges[mask]

        if len(head_edges) == 0:
            return pd.DataFrame()
        
        rule_walks = pd.DataFrame(
            np.hstack((head_edges[:, 0:1], head_edges[:, 2:4])),  # [sub, obj, ts]
            columns=["entity_1", "entity_2", "timestamp_1"],
            dtype=np.uint16,
        )

        del head_edges
        rel_edges = None
    
    except KeyError:
        return pd.DataFrame()

    try:
        if rule_walks.empty:
            return pd.DataFrame()

        body_edges_1 = edges[rels[0]]
        mask = body_edges_1[:, 2] == test_query_sub
        new_edges_1 = body_edges_1[mask]

        next_df = pd.DataFrame(
            np.hstack((new_edges_1[:, 0:1], new_edges_1[:, 2:4])),  # [sub, obj, ts]
            columns=["entity_0", "entity_1", "timestamp_0"],
            dtype=np.uint16,
        )

        rule_walks = pd.merge(rule_walks, next_df, on=["entity_1"], how='inner')
        del next_df
        del new_edges_1
        del body_edges_1

        if rule_walks.empty:
            return pd.DataFrame()
        
        rule_walks = rule_walks[
            rule_walks["timestamp_0"] < rule_walks["timestamp_1"]
        ]
        
        if rule_walks.empty:
            return pd.DataFrame()
            
        entity_col = f"entity_{2}"
        if entity_col not in rule_walks.columns:
            return pd.DataFrame()
        
        cur_targets = np.array(list(set(rule_walks[entity_col])))
        rel_edges = edges[rels[1]]
        mask = np.any(rel_edges[:, 0] == cur_targets[:, None], axis=0)
        new_edges_2 = rel_edges[mask]
        if len(new_edges_2) == 0:
            return pd.DataFrame()
        
        next_df = pd.DataFrame(
            np.hstack((new_edges_2[:, 0:1], new_edges_2[:, 2:4])),  # [sub, obj, ts]
            columns=["entity_2", "entity_3", "timestamp_2"],
            dtype=np.uint16,
        )
        rule_walks = pd.merge(rule_walks, next_df, on=["entity_2"], how='inner')
        del next_df
        del new_edges_2
        del rel_edges
        if rule_walks.empty:
            return pd.DataFrame()
    
        rule_walks = rule_walks[
            rule_walks["timestamp_1"] > rule_walks["timestamp_2"]
        ]

    except KeyError:
        return pd.DataFrame()
    
    return rule_walks



def match_and_get_walks_combined(rule, edges, test_query_sub, rules_type="cyclic", delta=0, max_memory_gb=4):
    """
    Combined function that matches body relations and builds walks directly.
    This eliminates the intermediate walk_edges arrays and is more memory efficient.
    
    Parameters:
        rule (dict): rule from rules_dict
        edges (dict): edges for rule application
        test_query_sub (int): test query subject
        rules_type (str): type of rules ("cyclic" or "relaxed_cyclic")
        id2ts (dict): mapping from timestamp ID to actual timestamp (for relaxed_cyclic)
        delta (int): time delta for relaxed_cyclic rules
        
    Returns:
        rule_walks (pd.DataFrame): final walks matching the rule with time constraints applied
    """
    rels = rule["body_rels"]
    
    # Early return if no relations
    if not rels:
        return pd.DataFrame(), False
    
    # Step 1: Get first relation edges matching query subject
    try:
        rel_edges = edges[rels[0]]
        mask = rel_edges[:, 0] == test_query_sub
        first_edges = rel_edges[mask]
        # breakpoint()
        
        if len(first_edges) == 0:
            return pd.DataFrame(), False
            
        # Create initial DataFrame
        rule_walks = pd.DataFrame(
            np.hstack((first_edges[:, 0:1], first_edges[:, 2:4])),  # [sub, obj, ts]
            columns=["entity_0", "entity_1", "timestamp_0"],
            dtype=np.uint16,
        )
        
        # # Handle timestamp processing for relaxed_cyclic
        # if rules_type == "relaxed_cyclic":
        #     rule_walks["timestamp_tmp_0"] = rule_walks["timestamp_0"]
        #     rule_walks["timestamp_0"] = rule_walks["timestamp_0"].map(id2ts)
        #     rule_walks["timestamp_0"] = pd.to_datetime(rule_walks["timestamp_0"])
        
        # Remove entity_0 if no var_constraints
        if not rule["var_constraints"]:
            rule_walks = rule_walks.drop(columns=["entity_0"])
        
        del first_edges
        rel_edges = None
            
    except KeyError:
        return pd.DataFrame(), False
    
    # Step 2: Iteratively build walks for remaining relations
    for i in range(1, len(rels)):
        if rule_walks.empty:
            return pd.DataFrame(), False
            
        # Get current targets from previous step
        entity_col = f"entity_{i}"
        if entity_col not in rule_walks.columns:
            return pd.DataFrame(), False
            
        cur_targets = np.array(list(set(rule_walks[entity_col])))
        
        try:
            # Get edges for next relation
            rel_edges = edges[rels[i]]
            mask = np.any(rel_edges[:, 0] == cur_targets[:, None], axis=0)
            next_edges = rel_edges[mask]
            
            if len(next_edges) == 0:
                return pd.DataFrame(), False
            
            # Create DataFrame for next step
            next_df = pd.DataFrame(
                np.hstack((next_edges[:, 0:1], next_edges[:, 2:4])),  # [sub, obj, ts]
                columns=[f"entity_{i}", f"entity_{i+1}", f"timestamp_{i}"],
                dtype=np.uint16,
            )
            
            # Handle timestamp processing for relaxed_cyclic
            # if rules_type == "relaxed_cyclic":
            #     next_df[f"timestamp_{i}"] = next_df[f"timestamp_{i}"].map(id2ts)
            #     next_df[f"timestamp_{i}"] = pd.to_datetime(next_df[f"timestamp_{i}"])
            
            # Merge with existing walks
            if max_memory_gb:
                est_size = estimate_merge_memory_gb(rule_walks, next_df, f"entity_{i}")
                if est_size > max_memory_gb:
                    print(est_size)
                    return pd.DataFrame(), True

            rule_walks = pd.merge(rule_walks, next_df, on=[f"entity_{i}"], how='inner')
            next_df = next_df[0:0]
            next_edges = None
            rel_edges = None
            
            # Apply time constraints
            if rules_type == "cyclic":
                rule_walks = rule_walks[
                    rule_walks[f"timestamp_{i-1}"] <= rule_walks[f"timestamp_{i}"]
                ]
            elif rules_type == "relaxed_cyclic":
                rule_walks = rule_walks[
                    (rule_walks[f"timestamp_{i-1}"] - delta) <= rule_walks[f"timestamp_{i}"]
                ]
            
            # Remove intermediate entity column if no var_constraints
            if not rule["var_constraints"]:
                rule_walks = rule_walks.drop(columns=[f"entity_{i}"])
            
            # Early exit if no walks remain after time filtering
            if rule_walks.empty:
                return pd.DataFrame(), False
                
        except KeyError:
            return pd.DataFrame(), False

    # Step 3: Final cleanup
    if not rule_walks.empty:
        # Remove intermediate timestamp columns
        timestamp_cols_to_drop = [f"timestamp_{i}" for i in range(1, len(rels))]
        rule_walks = rule_walks.drop(columns=[col for col in timestamp_cols_to_drop if col in rule_walks.columns])
        
        # Handle relaxed_cyclic timestamp restoration
        if rules_type == "relaxed_cyclic" and "timestamp_tmp_0" in rule_walks.columns:
            rule_walks["timestamp_0"] = rule_walks["timestamp_tmp_0"]
            rule_walks = rule_walks.drop(columns=["timestamp_tmp_0"])
    
    return rule_walks, False


if __name__ == "__main__":
    # rule = {'type': 'relaxed_cyclic', 'head_rel': 160, 'body_rels': [385, 372, 160], 'var_constraints': [[0, 2]], 'conf': 1.0, 'rule_supp': 209, 'body_supp': 209}
    rule = {'type': 'link_star', 'head_rel': 57, 'body_rels': [409, 424], 'back_conf': 0.8782961460446247, 'forw_conf': 0.8918367346938776, 'conf': 0.8850664403692512, 'back_rule_supp': 433, 'forw_rule_supp': 437, 'back_body_supp': 493, 'forw_body_supp': 490}
    dataset = 'icews18'
    data = Grapher(dataset)
    learn_edges = store_edges(data.train_idx)
    query = np.array([  39,   57, 2372,  270])

    edges = get_window_edges_v2(data.all_idx, query[3], learn_edges, window=0)
    breakpoint()
    # edges = match_link_star_body_relations(rule, edges, query[0])
    # breakpoint()
    # walks = get_link_star_walks_v2(rule, edges)
    walks = match_and_get_link_star_walks_v2(rule, edges, query[0])
    breakpoint()
    print(walks)
    print(walks["entity_2"].unique())
    print(len(walks["entity_2"].unique()))