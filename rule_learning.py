
from data import Edge, Graph
import random
import os
import datetime
from datetime import timedelta
import numpy as np

import pickle as pkl


class RuleLearning:
    def __init__(self, data:Graph, delta=1):
        self.data = data
        self.rules = dict()
        self. output_dir = f'./outputs/{data.database}/rules'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.found_rules = []

        self.rules = dict()
        self.rule_cnt = 0
        self.delta = delta

    def create_star_link_rule(self, walk: list[Edge]):
        if not walk:
            return
        rule = dict()
        rule['type'] = 'link_star'
        rule['head_rel'] = walk[0].relation
        rule['body_rels'] = [edge.relation for edge in walk[1:]]
        # rule['var_constraints'] = self.get_var_constraints(walk[1:], rev=False)

        if rule not in self.found_rules:
            self.found_rules.append(rule.copy())
            back_conf, forw_conf, back_rule_support, forw_rule_support, back_body_support, forw_body_support = self.estimate_link_star_confidence(rule)
            rule['back_conf'] = back_conf
            rule['forw_conf'] = forw_conf
            rule["conf"] = (back_conf + forw_conf) / 2
            rule['back_rule_supp'] = back_rule_support
            rule['forw_rule_supp'] = forw_rule_support
            rule['back_body_supp'] = back_body_support
            rule['forw_body_supp'] = forw_body_support

            if rule['back_conf'] > 0 and rule['forw_conf'] > 0:
                self.rules[rule['head_rel']] = self.rules.get(rule['head_rel'], []) + [rule]
                self.rule_cnt += 1
                # print(rule)
                # print(f"Rule created: {rule}")

            
    def estimate_link_star_confidence(self, rule, samples=500):
        edges = self.data.edges[rule['head_rel']]
        b1_rel = rule['body_rels'][0]
        b2_rel = rule['body_rels'][1]
        b1_edges = self.data.edges[b1_rel]
        b2_edges = self.data.edges[b2_rel]
        edges = self.data.convert_to_np(edges)
        b1_edges = self.data.convert_to_np(b1_edges)
        b2_edges = self.data.convert_to_np(b2_edges)

        unique_bodies_back = set()
        for _ in range(samples):
            body = b1_edges[np.random.choice(b1_edges.shape[0])]
            unique_bodies_back.add(tuple(body))
        
        unique_bodies_back = [list(body) for body in unique_bodies_back]

        back_body_support = len(unique_bodies_back)
        back_rule_support = 0
        back_conf = 0
        if len(unique_bodies_back):
            back_rule_support = self.link_star_rule_support(rule, unique_bodies_back, type='back')
            if back_rule_support:
                back_conf = back_rule_support / back_body_support
        

        unique_bodies_forw = set()
        for _ in range(samples):
            body = b2_edges[np.random.choice(b2_edges.shape[0])]
            unique_bodies_forw.add(tuple(body))
        
        unique_bodies_forw = [list(body) for body in unique_bodies_forw]

        forw_body_support = len(unique_bodies_forw)
        forw_rule_support = 0
        forw_conf = 0
        if len(unique_bodies_forw):
            forw_rule_support = self.link_star_rule_support(rule, unique_bodies_forw, type='forward')
            if forw_rule_support:
                forw_conf = forw_rule_support / forw_body_support
        

        return back_conf, forw_conf, back_rule_support, forw_rule_support, back_body_support, forw_body_support
    
    def link_star_rule_support(self, rule, bodies, type):

        support = 0
        for body in bodies:
            if type == 'back':
                edges = self.data.get_edges(body[2])
                edges = [edge for edge in edges if edge.time > body[3] and edge.relation == rule['head_rel']]
            else:
                edges = self.data.get_edges(body[0])
                edges = [edge for edge in edges if edge.time > body[3] and edge.relation == self.data.inv_relation_id[rule['head_rel']]]
            if len(edges) > 0:
                support += 1
        
        return support

    # def link_star_rule_sample(self, rule):
    #     head_rel = rule['head_rel']

    #     head = random.choice(self.data.edges[head_rel])

    #     backward_edge = self.data.convert_to_np(self.data.edges[rule['body_rels'][0]])
    #     backward_edge = backward_edge[backward_edge[:, 0] == head.head and backward_edge[:, 3] < head.time]

    #     forward_edge = self.data.convert_to_np(self.data.edges[rule['body_rels'][1]])
    #     forward_edge = forward_edge[forward_edge[:, 2] == head.tail and forward_edge[:, 3] < head.time]
    #     if backward_edge.size == 0 or forward_edge.size == 0:
    #         return None, None
    #     backward_edge = backward_edge[random.choice(backward_edge.shape[0])]
    #     forward_edge = forward_edge[random.choice(forward_edge.shape[0])]

    #     body = [backward_edge, forward_edge]

    #     return head, body
    




    # def estimate_link_star_confidence(self, rule, sample_bodies=1000):
    #     # breakpoint()
    #     self.test(rule)
    #     breakpoint()
    #     unique_bodies = set()
    #     # print(len(set(self.data.edges[rule['body_rels'][0]])), len(set(self.data.edges[rule['body_rels'][1]])))
    #     # print(len(self.data.edges[rule['head_rel']]))
    #     # return 0, 0, 0

    #     for _ in range(sample_bodies):
    #         backward_edge = random.choice(self.data.edges[rule['body_rels'][0]])
    #         forward_edge = random.choice(self.data.edges[rule['body_rels'][1]])
    #         body = [backward_edge, forward_edge]
    #         var_cnst = self.get_var_constraints(body, rev=False)
    #         if var_cnst == rule['var_constraints']:
    #             unique_bodies.add((backward_edge.head, backward_edge.time, backward_edge.tail, forward_edge.head, forward_edge.time, forward_edge.tail))

    #     # print(f"Unique bodies found: {unique_bodies}")
    #     unique_bodies = [list(body) for body in unique_bodies]
    #     body_support = len(unique_bodies)
    #     # print(f"Body support: {body_support}")
    #     if body_support:
    #         rule_support = self.link_star_rule_support(unique_bodies, rule)
    #         confidence = rule_support / body_support
    #         # print(f"Confidence: {confidence}, Rule support: {rule_support}, Body support: {body_support}")

    #         return confidence, rule_support, body_support

        
    #     return 0, 0, 0


    
    # def link_star_body(self, body_rels, var_cnst):
    #     backward_edges = self.data.edges[body_rels[0]]
    #     backward_edges += self.data.reverse_edges(self.data.edges[self.data.inv_relation_id[body_rels[0]]])
    #     backward_edges = set(backward_edges)

    #     forward_edges = self.data.edges[body_rels[1]]
    #     forward_edges += self.data.reverse_edges(self.data.edges[self.data.inv_relation_id[body_rels[1]]])
    #     forward_edges = set(forward_edges)

    #     print(f"Backward edges: {len(backward_edges)}, Forward edges: {len(forward_edges)}")

    #     if not backward_edges or not forward_edges:
    #         return set()
        
    #     unique_bodies = set()
        
    #     for be in backward_edges:
    #         for fe in forward_edges:
    #             body = [be, fe]
    #             if self.get_var_constraints(body, rev=False) == var_cnst:
    #                 unique_bodies.add((be.head, be.time, be.tail, fe.head, fe.time, fe.tail))
        
    #     return unique_bodies


        

    def create_rule(self, walk: list[Edge], rule_type='cyclic'):
        if not walk:
            return
        rule = dict()
        rule['type'] = rule_type
        rule['head_rel'] = walk[0].relation
        rule['body_rels'] = [self.data.inv_relation_id[edge.relation] for edge in walk[1:][::-1]]

        rule['var_constraints'] = self.get_var_constraints(walk[1:])

        if rule not in self.found_rules:
            self.found_rules.append(rule.copy())
            confidence, rule_support, body_support = self.estimate_confidence(rule, rule_type=rule_type)
            rule['conf'] = confidence
            rule['rule_supp'] = rule_support
            rule['body_supp'] = body_support
            # print(rule)
            if rule['conf'] > 0:
                self.rules[rule['head_rel']] = self.rules.get(rule['head_rel'], []) + [rule]
                self.rule_cnt += 1

                # print(rule)



    
    def estimate_confidence(self, rule, sample_bodies=500, rule_type='cyclic'):

        bodies = set()
        for _ in range(sample_bodies):
            success, body, body_ent = self.sample_body(rule['body_rels'], rule['var_constraints'], rule_type)
            if success:
                # bodies.append(body)
                bodies.add(tuple(body_ent))
        
        
        unique_bodies = [list(body) for body in bodies]
        body_support = len(unique_bodies)
        # print(unique_bodies)
        # print(len(bodies), body_support)

        if body_support:
            rule_support = self.rule_support(unique_bodies, rule)
            confidence = rule_support / body_support
            return confidence, rule_support, body_support
    

        return 0, 0, 0

    def rule_support(self, unique_bodies, rule):
        support = 0
        head_rel = rule['head_rel']
        for body in unique_bodies:
            edges = self.data.get_edges_in_rel(body[0], head_rel, body[-1])
            edges = [edge for edge in edges if edge.time > body[-2]]
            if len(edges) > 0:
                support += 1
        return support
    

    def sample_body(self, body_rels, var_cnst, rule_type='cyclic'):

        # breakpoint()
        start_edge = random.choice(self.data.edges[body_rels[0]])
        body = [start_edge]
        curr_node = start_edge.tail
        curr_ts = start_edge.time
        # print(body_rels)

        for ind in range(1, len(body_rels)):
            edges = self.data.get_edges_in_rel(curr_node, body_rels[ind])
            if rule_type == 'cyclic':
                edges = [edge for edge in edges if edge.time >= curr_ts]
            elif rule_type == 'relaxed_cyclic':
                # print(self.delta)
                edges = [edge for edge in edges if edge.time >= (curr_ts - timedelta(days=self.delta))]
            else:
                edges = []
            # breakpoint()
            if not edges:
                return False, body, None
            next_edge = random.choice(edges)
            body.append(next_edge)
            curr_node = next_edge.tail
            curr_ts = next_edge.time
        
        if var_cnst:
            cnst = self.get_var_constraints(body, rev=False)
            
            if cnst != var_cnst:
                # print(".", " ")
                return False, body, None
        
        body_ent = []
        for b in body:
            body_ent.append(b.head)
            body_ent.append(b.time)
        body_ent.append(body[-1].tail)
        
        return True, body, body_ent



    def get_var_constraints(self, body, rev=True):


        body_ent = [edge.head for edge in body]
        body_ent.append(body[-1].tail)

        # print(body_ent)
        # print(body_ent[::-1])
        if rev:
            body_ent = body_ent[::-1]

        constraints = []
        for ent in set(body_ent):
            cnst = [i for i, e in enumerate(body_ent) if e == ent]
            if len(cnst) > 1:
                # print(ent, cnst)
                constraints.append(cnst)
        

        return sorted(constraints)


    def sort_rules(self):

        for rel in self.rules.keys():
            self.rules[rel] = sorted(
                self.rules[rel], key=lambda x: x["conf"], reverse=True
            )
    
    def save_rules(self, rule_length, num_walks, transition_dist, acr, delta=1):
        file_name = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}_rules_len{rule_length}_walks{num_walks}_trans_{transition_dist}_d{delta}_acr{acr}.pkl'
        #  rule_type == 'relaxed_cyclic':
        # else:
        # file_name = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}_rules_len{rule_length}_walks{num_walks}_trans_{transition_dist}_type_{rule_type}.pkl'
        file_name = file_name.replace(" ", "")

        file_path = os.path.join(self.output_dir, file_name)
        with open(file_path, 'wb') as f:
            pkl.dump(self.rules, f)
        print(f"Rules saved to {file_path}")
    
    def save_rules_verbalized(self, rule_length, num_walks, transition_dist, acr, delta=1):
        file_name = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}_rules_len{rule_length}_walks{num_walks}_trans_{transition_dist}_d{delta}_acr{acr}_verbalized.txt'
        # if rule_type == 'relaxed_cyclic':
        # else:
        # file_name = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}_rules_len{rule_length}_walks{num_walks}_trans_{transition_dist}_type_{rule_type}_verbalized.txt'
        file_name = file_name.replace(" ", "")
        file_path = os.path.join(self.output_dir, file_name)

        rule_str = []

        for rel, rules in self.rules.items():
            for rule in rules:
                verbalized_rule = self.verbalize_rule(rule)
                rule_str.append(verbalized_rule)
        
        with open(file_path, 'w') as f:
            f.write("\n".join(rule_str))
        print(f"Verbalized rules saved to {file_path}")
    
            
    

    def verbalize_rule(self, rule):
        head_rel = rule['head_rel']
        body_rels = rule['body_rels']
        if rule['type'] != 'link_star':
            var_constraints = rule['var_constraints']
        else:
            var_constraints = None

        if rule['type'] == 'link_star':
            verbalized_rule = f"{rule['back_conf']:.4f} {rule['back_rule_supp']} {rule['back_body_supp']} {rule['forw_conf']} {rule['forw_rule_supp']} {rule['forw_body_supp']} : "
        else:
            verbalized_rule = f"{rule["conf"]:.4f} {rule['rule_supp']} {rule['body_supp']} : "

        # verbalized_rule += f"{self.data.get_id_relation(head_rel)} => "
        if rule['type'] == 'link_star':
            verbalized_rule += f"1 {self.data.get_id_relation(head_rel)} 2 => "
            verbalized_rule += f"0 {self.data.get_id_relation(body_rels[0])} 1,  2{self.data.get_id_relation(body_rels[1])} 3"
        else:
            verbalized_rule += f"{self.data.get_id_relation(head_rel)} => "
            verbalized_rule += " ".join([f"{i} {self.data.get_id_relation(rel)}" for i, rel in enumerate(body_rels)])
            verbalized_rule += f" {len(body_rels)}"
            verbalized_rule += " | "

        if var_constraints:
            constraints_str = " | ".join([f"v{cnst}" for cnst in var_constraints])
            verbalized_rule += f" [{constraints_str}]"

        return verbalized_rule
        

def rule_stats(rules_dict: dict):
    
    
    rule_len = dict()
    rule_cnt = 0
    for rules in rules_dict.values():
        for rule in rules:
            if rule['type'] == 'link_star':
                rule_len[-1] = rule_len.get(-1, 0) + 1
                rule_cnt += 1
            else:
                rule_cnt += 1
                rule_len[len(rule['body_rels'])] = rule_len.get(len(rule['body_rels']), 0) + 1
    
    print(f"Total rules found: {rule_cnt}")
    print(f"Total relations with rules: {len(rules_dict)}")
    
    print("Rule lengths:")
    for length, count in sorted(rule_len.items()):
        print(f"Length {length}: {count} rules")


if __name__ == "__main__":
    graph = Graph('icews14', 'train')
    rl = RuleLearning(graph)

    with open('sample_walk_link_star.pkl', 'rb') as f:
        walk = pkl.load(f)
    
    walk = walk
    # edge = set()
    # for w in walk:
    #     edge.add((w.head, w.time, w.tail))
    
    # edge.add((walk[0].tail, walk[0].time, walk[0].head))

    # breakpoint()

    print(walk)

    print("-"*20)
    
    # breakpoint()
    rl.create_star_link_rule(walk)
    print(rl.rules)
    # if rl.rule_cnt > 0:
    #     print(f"Rule created: {rl.rules[walk[0].relation]}")
    #     rl.save_rules_verbalized([3], 100, 'uni')

