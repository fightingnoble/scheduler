"""Unused chain ordering helper; its historical failure is preserved."""


def sort_chains_by_ddl_flops(chains, flops_dict, ddl_dict):
    chains.sort(key=lambda x: (-ddl_dict[id(chains[-1])], sum([flops_dict[n] for n in x])), reverse=True)
