from defeater_explanations.map_independence import *
from defeater_explanations.utils import *



def hill_climbing(bn, ev_vars, hyp_vars, depth=np.inf):
    # Check which are the supplementary variables
    variables = bn.names()
    supp_vars = []
    for var in variables:
        if var not in list(ev_vars.keys()) and var not in hyp_vars:
            supp_vars.append(var)

    bn_pr = bn
    tmp = supp_vars
    supp_vars = []
    dsep_by_ev = []

    for i in tmp:
        if bn_pr.isIndependent(i, hyp_vars, list(ev_vars.keys())):
            dsep_by_ev.append(i)
        else:
            supp_vars.append(i)

    if len(supp_vars) == 0:
        return dsep_by_ev

    irrelevant_singletons = dict()

    posterior = map_query(bn, ev_vars=ev_vars, hyp_vars=hyp_vars)
    y = posterior.argmax()[0][0]

    # For singletons
    for i in supp_vars:
        # print(ev_vars, hyp_vars, [i])
        irrel, jsd = map_independence(bn_pr, set_R=[i], ev_vars=ev_vars, hyp_vars=hyp_vars, hyp_posterior=posterior,
                                      return_jsd=True)
        if irrel:
            irrelevant_singletons[i] = jsd
    if len(irrelevant_singletons.keys()) == 0:
        return dsep_by_ev

    irrelevant_singletons = {k: v for k, v in sorted(irrelevant_singletons.items(), key=lambda item: item[1])}
    current_irrelevant_set = None
    if len(dsep_by_ev) > 0:
        current_irrelevant_set = dsep_by_ev
    else:
        current_irrelevant_set = [next(iter(irrelevant_singletons))]
        del irrelevant_singletons[next(iter(irrelevant_singletons))]
    # print(irrelevant_singletons)

    current_depth = 1

    # For higher order sets
    while current_depth < depth and len(irrelevant_singletons) > 0:
        new_irrelevant_singletons = irrelevant_singletons.copy()
        for i in irrelevant_singletons.keys():
            current_copy = current_irrelevant_set.copy()
            current_copy.append(i)
            new_irrelevant_singletons.pop(i)
            irrel = map_independence(bn_pr, set_R=current_copy, ev_vars=ev_vars, hyp_vars=hyp_vars,
                                     hyp_posterior=posterior)
            if irrel:
                current_irrelevant_set = current_copy

                # prune
                dsep_nodes = []
                added = False
                for i in new_irrelevant_singletons.keys():
                    if bn.isIndependent(i, hyp_vars, list(ev_vars.keys()) + current_copy):
                        dsep_nodes.append(i)
                        added = True
                for k in dsep_nodes:
                    new_irrelevant_singletons.pop(k)
                current_irrelevant_set = current_irrelevant_set + dsep_nodes
                while added == True:
                    dsep_nodes = []
                    added = False
                    for i in new_irrelevant_singletons.keys():
                        if bn.isIndependent(i, hyp_vars, list(ev_vars.keys()) + current_copy):
                            dsep_nodes.append(i)
                            added = True
                    for k in dsep_nodes:
                        new_irrelevant_singletons.pop(k)
                    current_irrelevant_set = current_irrelevant_set + dsep_nodes

                break

        irrelevant_singletons = new_irrelevant_singletons
        current_depth = current_depth + 1

    return current_irrelevant_set


def hill_climbing_mb(bn, ev_vars, hyp_vars, depth=np.inf):
    # Check which are the supplementary variables
    variables = bn.names()
    supp_vars = []
    for var in variables:
        if var not in list(ev_vars.keys()) and var not in hyp_vars:
            supp_vars.append(var)

    # Delete the ones conditionally independent
    bn_pr = bn
    tmp = supp_vars
    supp_vars = []
    dsep_by_ev = []

    for i in tmp:
        if bn_pr.isIndependent(i, hyp_vars, list(ev_vars.keys())):
            dsep_by_ev.append(i)
        else:
            supp_vars.append(i)

    if len(supp_vars) == 0:
        return dsep_by_ev

    # Put first the vars in the Markov Blanket
    mb = set()
    for i in hyp_vars:
        mb = mb.union(gum.MarkovBlanket(bn_pr, i).nodes())
    mb_names = []
    for i in mb:
        mb_names.append(bn_pr.variable(i).name())
    tmp = supp_vars
    supp_vars = []
    for i in tmp:
        if i in mb_names:
            supp_vars.append(i)
    tmp = list_diff(tmp, supp_vars)
    supp_vars = supp_vars + tmp

    posterior = map_query(bn, ev_vars=ev_vars, hyp_vars=hyp_vars)
    y = posterior.argmax()[0][0]

    irrelevant_singletons = []
    for i in supp_vars:
        irrel = map_independence(bn_pr, set_R=[i], ev_vars=ev_vars, hyp_vars=hyp_vars, hyp_posterior=posterior)
        if irrel:
            irrelevant_singletons.append(i)
    if len(irrelevant_singletons) == 0:
        return dsep_by_ev

    current_irrelevant_set = []
    if len(dsep_by_ev) > 0:
        current_irrelevant_set = dsep_by_ev
    current_depth = 1

    new_irrelevant_singletons = irrelevant_singletons
    while current_depth < depth and len(irrelevant_singletons) > 0:
        irrelevant_singletons = new_irrelevant_singletons
        for i in irrelevant_singletons:
            current_copy = current_irrelevant_set.copy()
            current_copy.append(i)
            # print(current_copy)
            irrel = map_independence(bn_pr, set_R=current_copy, ev_vars=ev_vars, hyp_vars=hyp_vars,
                                     hyp_posterior=posterior, return_jsd=True)
            new_irrelevant_singletons.remove(i)
            if irrel:
                current_irrelevant_set.append(i)
                dsep_nodes = []
                added = False
                for i in new_irrelevant_singletons:
                    if bn.isIndependent(i, hyp_vars, list(ev_vars.keys()) + current_irrelevant_set):
                        dsep_nodes.append(i)
                        added = True
                new_irrelevant_singletons = list_diff(new_irrelevant_singletons, dsep_nodes)
                current_irrelevant_set = current_irrelevant_set + dsep_nodes
                while added:
                    dsep_nodes = []
                    added = False
                    for i in new_irrelevant_singletons:
                        if bn.isIndependent(i, hyp_vars, list(ev_vars.keys()) + current_irrelevant_set):
                            dsep_nodes.append(i)
                            added = True
                    new_irrelevant_singletons = list_diff(new_irrelevant_singletons, dsep_nodes)
                    current_irrelevant_set = current_irrelevant_set + dsep_nodes

                current_depth = current_depth + 1

                break

    return current_irrelevant_set