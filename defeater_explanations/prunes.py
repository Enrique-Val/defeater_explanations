from defeater_explanations.utils import *

def decomposition_prune(relevant_set, S_split, relevant_sets):
    i = len(relevant_set)
    for k in range(i, len(S_split)):
        tmp = []
        for l in S_split[k]:
            if not set(relevant_set).issubset(set(l)):
                tmp.append(l)
        S_split[k] = tmp
    return S_split, relevant_sets


def conditional_independence_prune(bn, supp_vars, hyp_vars, ev_vars, irrelevant_set, S_split, irrelevant_sets, depth):
    # Delete from the network the nodes that are conditionally independent from the hypothesis variables (target) given the evidence
    dsep_nodes = []
    for i in supp_vars:
        if i not in irrelevant_set and bn.isIndependent(i, hyp_vars, list(ev_vars.keys()) + list(irrelevant_set)):
            dsep_nodes.append(i)

    if len(dsep_nodes) == 0:
        return S_split, irrelevant_sets
    irrels = powerset(dsep_nodes, depth)
    bigger = irrels[-1]
    irrels.pop(0)
    irrels.pop()

    for i in irrels:
        try:
            S_split[len(i) - 1].remove(i)
        except ValueError:
            pass  # do nothing!
    if bigger not in irrelevant_sets:
        irrelevant_sets.append(bigger)
    return S_split, irrelevant_sets