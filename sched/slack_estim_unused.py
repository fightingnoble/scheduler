"""Independent unused slack helpers; original behavior is preserved."""

from __future__ import annotations

from typing import Any, Dict

from task.graph_breakdown import decompose_dag_into_chains


def build_score_dict_ref_flops(task_dict:Dict[str, TaskBase], nodes:Any, score_dict):
    for node_n in nodes: 
        assert node_n in task_dict
        _task = task_dict[node_n]
        score_dict[node_n] = _task.flops


def get_chains_info(task_graph, start_nodes, end_nodes):
    chains = []
    for start_node in start_nodes:
        chains += decompose_dag_into_chains(task_graph, start_node, end_nodes)
    # zip the chains with its e2e latency
    chains = [(chain, task_graph.nodes[chain[-1]]['ddl']) for chain in chains]
    return chains
