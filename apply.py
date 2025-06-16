import itertools
import pickle as pkl
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from datetime import datetime

from data import Graph
from rule_learning import rule_stats
from score_functions import score_12
import rule_application as ra

dataset = 'icews14'
rule_file_name = '2025-06-14_22:23_rules_len[1,2,3]_walks300_trans_exp_d1_acr0.5.pkl'
# rule_file_name = "2025-06-10_16:02_rules_len[1, 2, 3]_walks200_trans_exp.pkl"
dir_path = f'./outputs/{dataset}/rules/'
rule_file_path = f'{dir_path}{rule_file_name}'
rule_lengths = [1,2,3]
top_k = 20
window = 0
num_process = 8
rules_type = 'relaxed_cyclic'
delta = 1


with open(rule_file_path, 'rb') as f:
    rules_dict = pkl.load(f)

test_data = Graph(dataset, 'test', True)
valid_data = Graph(dataset, 'valid', True)
train_data = Graph(dataset, 'train', True)

all_edges = np.vstack((train_data.all_edges, valid_data.all_edges, test_data.all_edges))

print("Rule stats before filtering:")
rule_stats(rules_dict)


rules_dict = ra.filter_rules(
    rules_dict, min_conf=0.01, min_body_supp=2, rule_lengths=rule_lengths
)

print("Rule stats after filtering:")
rule_stats(rules_dict)

score_func = score_12


args = [[0.1, 0.5]] # lambda, a

def apply_rules(i, batch_size):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(test_data.all_edges))
    idx_range = range(start_idx, end_idx)
    print(f"Processing batch {i + 1}/{num_process} with batch size {end_idx-start_idx}")

    all_candidates = [dict() for _ in range(len(args))]
    no_cands_counter = 0
    curr_ts = -1

    for ind in idx_range:
        
        query = test_data.all_edges[ind] #[h, r, t, ts]
        cands_dict = [dict() for _ in range(len(args))]

        if query[3] != curr_ts:
            curr_ts = query[3]
            edges = ra.get_window_edges(all_edges, curr_ts, train_data.edges, window)
        
        if query[1] in rules_dict:
            rules = rules_dict[query[1]]
            dicts_idx = list(range(len(args)))

            for rule in rules:
                if rule["type"] == "link_star":
                    walk_edges = ra.match_link_star_body_relations(rule, edges, query[0])
                    # breakpoint()
                else:
                    walk_edges = ra.match_body_relations(rule, edges, query[0])

                if 0 not in [len(walk) for walk in walk_edges]:
                    if rule["type"] == "link_star":
                        rule_walks = ra.get_link_star_walks(rule, walk_edges)
                        # breakpoint()
                    else:
                        rule_walks = ra.get_walks(rule, walk_edges, rules_type, train_data, delta=delta)

                        if rule["var_constraints"]:
                            rule_walks = ra.check_var_constraints(
                                rule["var_constraints"], rule_walks
                            )

                    if not rule_walks.empty:
                        cands_dict = ra.get_candidates(
                            rule,
                            rule_walks,
                            curr_ts,
                            cands_dict,
                            score_func,
                            args,
                            dicts_idx,
                        )
                    
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
        
        if (ind-start_idx+1) % 100 == 0 or ind == end_idx - 1:
            print(f"Process {i}: Processed query {ind - start_idx + 1}/{end_idx-start_idx}")
        
    return all_candidates, no_cands_counter    


batch_size = len(test_data.all_edges) // num_process
start_time = datetime.now()
output = Parallel(n_jobs=num_process)(
    delayed(apply_rules)(i, batch_size) for i in range(num_process)
)
end_time = datetime.now()
total_time = (end_time - start_time).seconds
print(f"Time taken for applying rules: {total_time} seconds")

no_cands_counter = 0
all_candidates = [dict() for _ in range(len(args))]
for s in range(len(args)):
    for i in range(num_process):
        all_candidates[s].update(output[i][0][s])
        output[i][0][s].clear()

for i in range(num_process):
    no_cands_counter += output[i][1]

print(f"Number of queries with no candidates: {no_cands_counter} out of {len(test_data.all_edges)}")

for s in range(len(args)):
    score_func_str = score_func.__name__ + str(args[s])
    score_func_str = score_func_str.replace(" ", "")
    
    ra.save_candidates(
        rule_file_name,
        dir_path,
        all_candidates[s],
        rule_lengths,
        window,
        score_func_str,
    )
