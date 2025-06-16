# import random
import pickle as pkl
from typing import Literal
import numpy as np
from datetime import timedelta

from data import Edge, Graph

class TemporalWalk:
    def __init__(self, data: Graph, transition_dist: Literal['uni', 'exp'] = 'uni', delta=1):
        self.data = data
        self.transition_dist = transition_dist
        self.delta = delta

        # rels = sorted(list(data.edges.keys()))
        # for rel in self.data.edges.keys():
        #     print(
        #         f"Number of edges for relation {rel}: {len(data.edges[rel])}"
        #     )
    
    def sample_start_edge(self, rel_id):
        edges = self.data.edges.get(rel_id, [])
        # print(f"Number of edges for relation {rel_id}: {len(edges)}")
        if not edges:
            return None
        return edges[np.random.choice(len(edges))]

    def sample_next_edge(self, edges, curr_ts=None, relaxed=False):
        if not edges:
            return None

        ## uniform sampling
        if self.transition_dist == 'uni':
            return edges[np.random.choice(len(edges))]
        ## exponential sampling
        elif self.transition_dist == 'exp':

            tss = [edge.time for edge in edges]
            tss = np.array(tss) - curr_ts
            if not relaxed:
                prob = np.exp(list(map(lambda x: x.days, list(tss))))
            else:
                prob = np.exp(list(map(lambda x: -1*abs(x.days), list(tss))))
            
            prob = prob / np.sum(prob)
            next_idx = np.random.choice(len(edges), p=prob)
            next_edge = edges[next_idx]
            return next_edge





    def sample_cyclic_walk(self, length, rel_id):
        walk = []

        start_edge = self.sample_start_edge(rel_id)
        if not start_edge:
            return False, None

        walk.append(start_edge)
        curr_node = start_edge.tail
        curr_ts = start_edge.time
        

        for ind in range(1, length):
            if ind == 1:
                next_edges = self.data.get_edges(curr_node)
                # breakpoint()
                next_edges = [edge for edge in next_edges if edge.time < curr_ts]
                # if len(next_edges) == 0:
                #     return False, None
            # elif ind == length - 2:
            #     next_edges = self.data.get_edges(curr_node, remove_inv=True, ord=walk[-1])
            #     next_edges = [edge for edge in next_edges if (edge.tail == walk[0].head and edge.time <= curr_ts)]
            else:
                next_edges = self.data.get_edges(curr_node, remove_inv=True, ord=walk[-1])
                next_edges = [edge for edge in next_edges if edge.time <= curr_ts]
            
            if ind == length - 1:
                next_edges = [edge for edge in next_edges if edge.tail == start_edge.head]
            
            if not next_edges:
                return False, None


            next_edge = self.sample_next_edge(next_edges, curr_ts)
            if not next_edge:
                return False, None
            walk.append(next_edge)
            curr_node = next_edge.tail
            curr_ts = next_edge.time
        

        return True, walk


    def sample_relaxed_cyclic_walk(self, length, rel_id):
        walk = []
        # print(self.delta)

        start_edge = self.sample_start_edge(rel_id)
        if not start_edge:
            return False, None
        
        walk.append(start_edge)
        curr_node = start_edge.tail
        curr_ts = start_edge.time

        for ind in range(1, length):
            if ind == 1:
                next_edges = self.data.get_edges(curr_node)
                next_edges = [edge for edge in next_edges if edge.time < curr_ts]
            else:
                next_edges = self.data.get_edges(curr_node, remove_inv=True, ord=walk[-1])
                next_edges = [edge for edge in next_edges if edge.time <= (curr_ts+timedelta(days=self.delta))]
            
            if ind == length - 1:
                next_edges = [edge for edge in next_edges if edge.tail == start_edge.head]
            
            if not next_edges:
                return False, None

            next_edge = self.sample_next_edge(next_edges, curr_ts, relaxed=True)

            if not next_edge:
                return False, None
            
            walk.append(next_edge)
            curr_node = next_edge.tail
            curr_ts = next_edge.time
        
        return True, walk
    
    def sample_link_star(self, rel_id):
        walk = []
        start_edge = self.sample_start_edge(rel_id)
        if not start_edge:
            return False, None
        
        # breakpoint()
        backward_edges = self.data.get_edges(start_edge.head)
        backward_edges = self.data.reverse_edges(backward_edges)
        backward_edges = [edge for edge in backward_edges if edge.tail == start_edge.head and edge.time < start_edge.time and edge.head != start_edge.tail]
        
        forward_edges = self.data.get_edges(start_edge.tail)
        forward_edges = [edge for edge in forward_edges if edge.head == start_edge.tail and edge.time < start_edge.time and edge.tail != start_edge.head]

        if not backward_edges or not forward_edges:
            return False, None
        
        walk.append(start_edge)
        walk.append(backward_edges[np.random.choice(len(backward_edges))])
        walk.append(forward_edges[np.random.choice(len(forward_edges))])

        return True, walk
        
        


if __name__ == '__main__':
    graph = Graph('icews14', 'train')
    tw = TemporalWalk(graph, 'exp')
    rel_id = np.random.choice(list(graph.edges.keys()))
    
    for _ in range(1, 5):
        success, walk = tw.sample_link_star(rel_id)
        if success:
            # breakpoint()
            # save walk in pkl file
            with open('sample_walk_link_star.pkl', 'wb') as f:
                pkl.dump(walk, f)
            
            exit()


            # print(f"Walk for relation {rel_id}:")
            # print(f"Length: {len(walk)}")
            # for edge in walk:
            #     print(f"Head: {graph.get_id_entity(edge.head)}, Relation: {graph.get_id_relation(edge.relation)}, Tail: {graph.get_id_entity(edge.tail)}, Time: {edge.time}")
            # print("*****************")
        else:
            print(f"No valid cyclic walk found for relation {rel_id}.")