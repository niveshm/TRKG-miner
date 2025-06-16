

from data import Graph
from rule_learning import RuleLearning, rule_stats
from temporal_walk import TemporalWalk
import numpy as np

dataset = 'icews14'

# np.random.seed(42)

num_walks = 300
rule_lengths = [1,2,3]
data = Graph(dataset, 'train')
trasition_dist = 'exp'
delta = 1 # days, used for relaxed cyclic walks
acr = 0.5 # cyclic to acyclic ratio


temporal_walk = TemporalWalk(data, trasition_dist, delta)
rule_learner = RuleLearning(data, delta=delta)
relations = data.edges.keys()
num_rules = 0
# rule_type = 'relaxed_cyclic'


cnt = 0
for rel_id in relations:
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
    print(f'relations {cnt+1}/{len(relations)}, rule length {rule_length}, found rules: {curr_rules}')
    cnt += 1
    num_rules = rule_learner.rule_cnt
    # breakpoint()


print(f'\nTotal rules found: {rule_learner.rule_cnt}')
rule_learner.sort_rules()
# rule_learner.save_rules(rule_lengths, num_walks, 'uni')
rule_learner.save_rules(rule_lengths, num_walks, trasition_dist, acr, delta=delta)
rule_learner.save_rules_verbalized(rule_lengths, num_walks, trasition_dist, acr, delta=delta)
rule_stats(rule_learner.rules)