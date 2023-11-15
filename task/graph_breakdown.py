import networkx as nx
from typing import List, Any

def decompose_dag_into_chains(dag:nx.DiGraph, start_node, end_node, with_src_sink=True)-> List[List[Any]]:
    chains = []

    def dfs(node, path):
        if node in end_node:
            # remove the start and end nodes
            # chains.append(path[1:-1])
            chains.append(path)
            return
        for successor in dag.successors(node):
            dfs(successor, path + [successor])

    dfs(start_node, [start_node])
    return chains

def sort_chains_by_ddl_flops(chains, flops_dict, ddl_dict):
    chains.sort(key=lambda x: (-ddl_dict[id(chains[-1])], sum([flops_dict[n] for n in x])), reverse=True)

def test_decompose_dag_into_chains():
    # Example usage
    # Create a DAG
    dag = nx.DiGraph()
    dag.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (4, 6), (5, 7), (6, 7)])

    # Define the start and end nodes
    start_node = 1
    end_node = 7

    # Decompose the DAG into chains from start_node to end_node
    chains = decompose_dag_into_chains(dag, start_node, end_node)

    # Print the chains
    for chain in chains:
        print(chain)

if __name__ == '__main__':
    test_decompose_dag_into_chains()