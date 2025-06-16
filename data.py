from datetime import datetime
import sys
import numpy as np

# datetime.strptime(date.today(), '%Y-%m-%d')


class Node:
    def __init__(self, id):
        self.id = id
        self.edges = list()

class Edge:
    def __init__(self, head, relation, tail, time):
        self.head = head
        self.relation = relation
        self.tail = tail
        self.time = datetime.strptime(time, '%Y-%m-%d')
        self.ts = time

    def __repr__(self):
        return f"Edge({self.head}, {self.relation}, {self.tail}, {self.time})"
    
    def copy(self):
        return Edge(self.head, self.relation, self.tail, self.ts)




class Graph:
    def __init__(self, database, type='train', save_np_format=False):
        self.database = database

        with open(f'./data/{database}/entity2id.json', 'r') as f:
            entity2id = eval(f.read())
        
        with open(f'./data/{database}/relation2id.json', 'r') as f:
            relation2id_old = eval(f.read())
        
        with open(f'./data/{database}/ts2id.json', 'r') as f:
            self.ts2id = eval(f.read())
        
        with open(f'./data/{database}/{type}.txt', 'r') as f:
            data = [line.strip().split('\t') for line in f.readlines()]

        self.relation2id = relation2id_old.copy()
        counter = len(relation2id_old)
        for relation in relation2id_old:
            self.relation2id["_" + relation] = counter  # Inverse relation
            counter += 1

        self.inv_relation_id = dict()
        num_relations = len(relation2id_old)
        for i in range(num_relations):
            self.inv_relation_id[i] = i + num_relations
        for i in range(num_relations, num_relations * 2):
            self.inv_relation_id[i] = i % num_relations

        self.entity2id: dict = entity2id
        self.id2entity = {v: k for k, v in entity2id.items()}
        # self.relation2id: dict = relation2id
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        self.id2ts = {v: k for k, v in self.ts2id.items()}
        # self.data = data
        self.save_np_format = save_np_format

        if save_np_format:
            self.edges: dict[int, list[np.ndarray]] = {}
            self.all_edges = []
            self.create_graph(data)
        else:

            self.graph: dict[int, Node] = {}
            self.edges: dict[int, list[Edge]] = {}
            self.create_graph(data)


    def create_graph(self, data):
        if not self.save_np_format:
            for head, rel, tail, time in data:
                head_id = self.entity2id[head]
                tail_id = self.entity2id[tail]
                rel_id = self.relation2id[rel]

                if head_id not in self.graph:
                    self.graph[head_id] = Node(head_id)
                if tail_id not in self.graph:
                    self.graph[tail_id] = Node(tail_id)

                edge = Edge(head=head_id, relation=rel_id, tail=tail_id, time=time)
                inv_edge = Edge(head=tail_id, relation=self.inv_relation_id[rel_id], tail=head_id, time=time)
                self.edges[rel_id] = self.edges.get(rel_id, []) + [edge]
                self.edges[self.inv_relation_id[rel_id]] = self.edges.get(self.inv_relation_id[rel_id], []) + [inv_edge]
                self.graph[head_id].edges.append(edge)
                self.graph[tail_id].edges.append(inv_edge)

            
            self.edges = dict(
                sorted(
                    self.edges.items(),
                    key=lambda item: item[0]
                )
            )

        else:
            for head, rel, tail, time in data:
                head_id = self.entity2id[head]
                tail_id = self.entity2id[tail]
                rel_id = self.relation2id[rel]
                time = self.ts2id[time]

                edge = np.array([head_id, rel_id, tail_id, time], dtype=object)
                inv_edge = np.array([tail_id, self.inv_relation_id[rel_id], head_id, time], dtype=object)
                self.all_edges.append(edge)
                self.all_edges.append(inv_edge)
                
                self.edges[rel_id] = self.edges.get(rel_id, []) + [edge]
                self.edges[self.inv_relation_id[rel_id]] = self.edges.get(self.inv_relation_id[rel_id], []) + [inv_edge]
            
            self.all_edges = np.array(self.all_edges, dtype=object)
            for k, v in self.edges.items():
                self.edges[k] = np.array(v, dtype=object)
            
            self.edges = dict(
                sorted(
                    self.edges.items(),
                    key=lambda item: item[0]
                )
            )
            
            
     


    def get_entity_id(self, entity):
        return self.entity2id.get(entity)

    def get_id_entity(self, entity_id):
        return self.id2entity.get(entity_id)

    def get_relation_id(self, relation):
        return self.relation2id.get(relation)
    
    def get_id_relation(self, relation_id):
        return self.id2relation.get(relation_id)

    def get_all_edges(self) -> list[Edge]:
        all_edges = []
        for edges in self.graph.values():
            all_edges.extend(edges.edges)
        return all_edges
    
    def get_edges(self, node_id, remove_inv=False, ord=None) -> list[Edge]:
        node = self.graph.get(node_id, None)
        if node is None:
            return []
        edges = node.edges
        if remove_inv:
            # print(len(edges))
            edges = [edge for edge in edges if not (edge.tail == ord.head and edge.relation == self.inv_relation_id[ord.relation] and edge.head == ord.tail and edge.time == ord.time)]

            
        
        return edges

    def get_edges_in_rel(self, node_id, rel_id, tail_id=None) -> list[Edge]:
        edges = self.graph.get(node_id, None)
        if edges is None:
            return []
        edges = edges.edges
        if tail_id is not None:
            edges = [edge for edge in edges if edge.tail == tail_id and edge.relation == rel_id]
        else:
            edges = [edge for edge in edges if edge.relation == rel_id]
        
        return edges
    
    def get_quad_date(self, quad):
        head, relation, tail, time = quad
        return (self.get_id_entity(head), 
                self.get_id_relation(relation), 
                self.get_id_entity(tail), 
                time)


    def reverse_edges(self, edges: list[Edge]):
        reversed_edges = []
        for edge in edges:
            ed = edge.copy()
            ed.head, ed.tail = ed.tail, ed.head
            ed.relation = self.inv_relation_id[ed.relation]
            reversed_edges.append(ed)
        
        return reversed_edges

    def convert_to_np(self, edges: list[Edge]) -> np.ndarray:
        return np.array([[edge.head, edge.relation, edge.tail, edge.time] for edge in edges], dtype=object)


if __name__ == '__main__':


    graph = Graph('icews14', 'train', save_np_format=True)


    # ed = dict()

    # for edges in graph.edges.values():
    #     for edge in edges:
    #         ed[edge.__repr__()] = ed.get(edge.__repr__(), 0) + 1
    
    # for k, v in ed.items():
    #     if v > 1:
    #         print(k, v)
    #         break

    # print(sys.getsizeof(graph))
    # print(graph.graph[0].edges)
    # for edges in graph.graph[0].edges:
    #     print(graph.get_quad_date((edges.head, edges.relation, edges.tail, edges.ts)))

    # print(graph.get_entity_id('Abdullah_G\u00fcl'))
    # print(graph.get_id_entity(0))
    # print(graph.get_relation_id('Appeal_for_military_protection_or_peacekeeping'))
    # print(graph.get_id_relation(0))
    # print(len(graph.graph))