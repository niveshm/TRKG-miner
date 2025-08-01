import itertools
import pickle as pkl
import numpy as np
import gc  # Add garbage collection
from tqdm import tqdm

import pickle as pkl
from grapher import Grapher, store_edges
from rule_learning import rule_stats
from score_functions import score_12
import rule_application as ra


dataset = 'icews18'
rule_file_name = '2025-07-25_20:21_rules_len[1,2,3]_walks200_trans_exp_d1_acr1.pkl'
skipped_queries_file = '2025-07-25_20:21_rules_len[1,2,3]_walks200_trans_exp_d1_acr1.pklskipped_queries_score_12[0.1,0.5].pkl'
candidate_file = '2025-07-25_20:21_rules_len[1,2,3]_walks200_trans_exp_d1_acr1_cands_r[1,2,3]_w0_score_12[0.1,0.5].pkl'
# rule_file_name = "2025-06-10_16:02_rules_len[1, 2, 3]_walks200_trans_exp.pkl"
dir_path = f'./outputs/{dataset}/rules/'
rule_file_path = f'{dir_path}{rule_file_name}'
rule_lengths = [1,2,3]
top_k = 20
window = 0
# num_process = 8
rules_type = 'relaxed_cyclic'
delta = 1

with open(rule_file_path, 'rb') as f:
    rules_dict = pkl.load(f)

data = Grapher(dataset)
test_data = data.test_idx

print("Rule stats before filtering:")
rule_stats(rules_dict)


rules_dict = ra.filter_rules(
    rules_dict, min_conf=0.01, min_body_supp=2, rule_lengths=rule_lengths
)

print("Rule stats after filtering:")
rule_stats(rules_dict)

learn_edges = store_edges(data.train_idx)

score_func = score_12

args = [[0.1, 0.5]] # lambda, a

with open(f"{dir_path}{skipped_queries_file}", 'rb') as f:
    skipped_queries = pkl.load(f)

queries = []
for x in skipped_queries:
    queries += x

print(len(queries))

def apply_rules(query_idx):

    all_candidates = [dict() for _ in range(len(args))]
    no_cands_counter = 0
    curr_ts = -1
    edges = None  # Cache edges to avoid repeated computation
    try:
        for ind in tqdm(query_idx):
            
            query = test_data[ind] #[h, r, t, ts]
            cands_dict = [dict() for _ in range(len(args))]

            # Only recompute edges when timestamp changes
            if query[3] != curr_ts:
                curr_ts = query[3]
                # Clear previous edges to free memory
                if edges is not None:
                    del edges
                edges = ra.get_window_edges_v2(data.all_idx, curr_ts, learn_edges, window)
            
            if query[1] in rules_dict:
                rules = rules_dict[query[1]]
                dicts_idx = list(range(len(args)))

                for rule in rules:
                    if rule["type"] == "link_star":
                        # walk_edges = ra.match_link_star_body_relations(rule, edges, query[0])
                        rule_walks, skip_query = ra.match_and_get_link_star_walks_v2(rule, edges, query[0], max_memory_gb=None)

                        if skip_query:
                            break

                        # breakpoint()
                        # if not rule_walks.empty:
                        #     rule_walks = ra.check_var_constraints_acyclic(rule_walks)
                    else:
                        # walk_edges = ra.match_body_relations(rule, edges, query[0])
                        rule_walks, skip_query = ra.match_and_get_walks_combined(rule, edges, query[0], rules_type, delta=delta, max_memory_gb=None)
                        if skip_query:
                            break
                        # breakpoint()
                        if not rule_walks.empty and rule["var_constraints"]:
                            rule_walks = ra.check_var_constraints(
                                rule["var_constraints"], rule_walks
                            )
                    
                    if not rule_walks.empty:
                        if rule["type"] == "link_star":
                            max_entity = "entity_" + str(2)
                        else:
                            max_entity = "entity_" + str(len(rule["body_rels"]))
                        
                        rule_walks = rule_walks[["timestamp_0", max_entity]]

                        cands_dict = ra.get_candidates(
                            rule,
                            rule_walks,
                            curr_ts,
                            cands_dict,
                            score_func,
                            args,
                            dicts_idx,
                        )
                        del rule_walks
                    
                        for s in dicts_idx:
                            cands_dict[s] = {
                                x: sorted(cands_dict[s][x], reverse=True) for x in cands_dict[s].keys()
                            }
                            cands_dict[s] = dict(
                                sorted(
                                    cands_dict[s].items(),
                                    key=lambda item: item[1],
                                    reverse=True,
                                )
                            )
                            top_k_scores = [v for _, v in cands_dict[s].items()][:top_k]
                            unique_scores = list(
                                scores for scores, _ in itertools.groupby(top_k_scores)
                            )
                            if len(unique_scores) >= top_k:
                                dicts_idx.remove(s)

                        if not dicts_idx:
                            break

                        # Explicit cleanup of rule_walks DataFrame
                        
                if cands_dict[0]:
                    for s in range(len(args)):
                        # Calculate noisy-or scores
                        scores = list(
                            map(
                                lambda x: 1 - np.prod(1 - np.array(x)),
                                cands_dict[s].values(),
                            )
                        )
                        cands_scores = dict(zip(cands_dict[s].keys(), scores))
                        noisy_or_cands = dict(
                            sorted(cands_scores.items(), key=lambda x: x[1], reverse=True)
                        )
                        all_candidates[s][ind] = noisy_or_cands
                else:  # No candidates found by applying rules
                    no_cands_counter += 1
                    for s in range(len(args)):
                        all_candidates[s][ind] = dict()
            
            else:  # No rules exist for this relation
                no_cands_counter += 1
                for s in range(len(args)):
                    all_candidates[s][ind] = dict()
            
            # Clear cands_dict to free memory
            del cands_dict
            
            if (ind+1) % 100 == 0 or ind == len(query_idx) - 1:
                print(f"Processed query {ind + 1}/{len(query_idx)}")
                # Force garbage collection every 100 queries to prevent memory buildup
                gc.collect()
        
        # Final cleanup
        if edges is not None:
            del edges
    except KeyboardInterrupt:
        # return partial results
        print("KeyboardInterrupt: Returning partial results")
        return all_candidates, no_cands_counter

    return all_candidates, no_cands_counter


candidates, no_cands_counter = apply_rules(queries)

with open(f"{dir_path}{candidate_file}", "rb") as f:
    all_candidates = pkl.load(f)

    
tmp_all_candidates = dict() #for _ in range(len(args))
for s in range(len(args)):
    tmp_all_candidates.update(candidates[s])


# for s in range(len(args)):
score_func_str = score_func.__name__ + str(args[s])
score_func_str = score_func_str.replace(" ", "")
    
#     ra.save_candidates(
#         rule_file_name,
#         dir_path,
#         all_candidates[s],
#         rule_lengths,
#         window,
#         score_func_str,
#     )

tmp_all_candidates = {int(k): v for k, v in tmp_all_candidates.items()}
for k in tmp_all_candidates:
    tmp_all_candidates[k] = {int(cand): float(v) for cand, v in tmp_all_candidates[k].items()}

all_candidates.update(tmp_all_candidates)

print(f"Number of queries with no candidates: {no_cands_counter}")
filename = "{0}_skippedqueries_cands_r{1}_w{2}_{3}.pkl".format(
    ".".join(rule_file_name.split('.')[:-1]), rule_lengths, window, score_func_str
)

gc.collect()  # Explicitly call garbage collection at the end