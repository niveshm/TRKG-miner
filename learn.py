

from joblib import Parallel, delayed
from data import Graph
from rule_learning import RuleLearning, rule_stats
from temporal_walk import TemporalWalk
import numpy as np

dataset = 'icews14'

# np.random.seed(12)

seed = 12
num_process = 16
num_walks = 200
rule_lengths = [1,2,3]
data = Graph(dataset, 'train')
trasition_dist = 'exp'
delta = 1 # days, used for relaxed cyclic walks
acr = 1 # cyclic to acyclic ratio


temporal_walk = TemporalWalk(data, trasition_dist, delta)
# rule_learner = RuleLearning(data, delta=delta)
relations = list(data.edges.keys())
num_rules = 0
# rule_type = 'relaxed_cyclic'

def learn_rules(i, batch_size):

    if seed:
        np.random.seed(seed)
    
    rule_learner = RuleLearning(data, delta=delta)

    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(relations))
    idx_range = range(start_idx, end_idx)
    print(f"Processing batch {i + 1}/{num_process} with batch size {end_idx-start_idx}")

    num_rules = 0

    for id in idx_range:
        rel_id = relations[id]
        for rule_length in rule_lengths:
            for _ in range(num_walks):
                success, walk = temporal_walk.sample_relaxed_cyclic_walk(rule_length+1, rel_id)
                if success:
                    rule_learner.create_rule(walk, rule_type="relaxed_cyclic")


        for _ in range(int(num_walks*acr)):
            success, walk = temporal_walk.sample_link_star(rel_id)
            if success:
                rule_learner.create_star_link_rule(walk)
        
        curr_rules = rule_learner.rule_cnt - num_rules
        print(f"Process {i}, relations {id-start_idx+1}/{end_idx-start_idx}, found rules: {curr_rules}")
        num_rules = rule_learner.rule_cnt
    
    return rule_learner.rules

# cnt = 0
# for rel_id in relations:
#     for rule_length in rule_lengths:
#         for _ in range(num_walks):
#             success, walk = temporal_walk.sample_relaxed_cyclic_walk(rule_length+1, rel_id)
#             if success:
#                 rule_learner.create_rule(walk, rule_type="relaxed_cyclic")


#     for _ in range(int(num_walks*acr)):
#         success, walk = temporal_walk.sample_link_star(rel_id)
#         if success:
#             rule_learner.create_star_link_rule(walk)
                
        

            
#     curr_rules = rule_learner.rule_cnt - num_rules
#     print(f'relations {cnt+1}/{len(relations)}, rule length {rule_length}, found rules: {curr_rules}')
#     cnt += 1
#     num_rules = rule_learner.rule_cnt
    # breakpoint()

batch_size = len(relations) // num_process
# print(batch_size)
output = Parallel(n_jobs=num_process)(
    delayed(learn_rules)(i, batch_size) for i in range(num_process)
)

all_rules = output[0]
for i in range(1, num_process):
    all_rules.update(output[i])

rule_learner = RuleLearning(data, delta=delta)
rule_learner.rules = all_rules

print(f'\nTotal rules found: {rule_learner.rule_cnt}')
rule_learner.sort_rules()
# rule_learner.save_rules(rule_lengths, num_walks, 'uni')
rule_learner.save_rules(rule_lengths, num_walks, trasition_dist, acr, delta=delta)
rule_learner.save_rules_verbalized(rule_lengths, num_walks, trasition_dist, acr, delta=delta)
rule_stats(rule_learner.rules)