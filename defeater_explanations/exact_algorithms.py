from defeater_explanations.map_independence import *
from defeater_explanations.prunes import *


def check_every_r_bn(bn, ev_vars, hyp_vars, depth=np.inf):
    # Check which are the supplementary variables
    variables = bn.names()
    supp_vars = []
    for var in variables:
        if var not in list(ev_vars.keys()) and var not in hyp_vars:
            supp_vars.append(var)

    # Delete the ones conditionally independent
    # bn_pr, dsep_by_ev = prune_network(bn, list(ev_vars.keys()), hyp_vars, supp_vars = supp_vars)
    dsep_by_ev = []
    bn_pr = bn
    # print(dsep_by_ev)

    # tmp = supp_vars
    # supp_vars = []

    # for i in tmp :
    #    if i not in dsep_by_ev :
    #        supp_vars.append(i)

    S = powerset(supp_vars, depth=depth)
    S.pop(0)
    # Divide by length
    S_split = []
    size = 0
    for i in S:
        if len(i) != size:
            size = size + 1
            S_split.append([])
        S_split[-1].append(i)

    # Variables to store relevant/irrelevant sets
    relevant_sets = []
    irrelevant_sets = []

    # Get the posterior and the argmax from the original MAP-query
    posterior = map_query(bn, ev_vars=ev_vars, hyp_vars=hyp_vars)
    y = posterior.argmax()[0][0]

    for i in range(0, len(S_split)):
        for j in S_split[i]:
            # If relevant
            # print(list(j))
            if map_dependence(bn_pr, set_R=list(j), ev_vars=ev_vars, hyp_vars=hyp_vars, hyp_vars_assign=y):
                relevant_sets.append(j)
            # If irrelevant
            else:
                irrelevant_sets.append(j)
    return relevant_sets, irrelevant_sets


# %%
def get_c_exp(bn, ev_vars, hyp_vars, hyp_vars_assign=None):
    if hyp_vars_assign is None:
        posterior = map_query(bn, ev_vars=ev_vars, hyp_vars=hyp_vars)
        hyp_vars_assign = posterior.argmax()[0][0]

    # Alternative. First compute P(H|e) and then P(e) and compute the product
    ie = gum.ShaferShenoyInference(bn)
    ie.addJointTarget(set(hyp_vars))
    ie.setEvidence(ev_vars)
    ie.makeInference()
    post = ie.jointPosterior(set(hyp_vars))

    # Compute P(e)
    ie = gum.ShaferShenoyInference(bn)
    ie.setEvidence(ev_vars)
    ie.makeInference()
    p_e = ie.evidenceProbability()
    if False and p_e < 0.0000001:
        print("Hace chapuza")
        p_e = 0.0000001

    post = post * p_e
    inst = gum.Instantiation(post)
    inst.fromdict(hyp_vars_assign)
    P_eh = post.get(inst)
    return post / P_eh


def check_every_r_silja(bn, ev_vars, hyp_vars, depth=np.inf):
    # Check which are the supplementary variables
    variables = bn.names()
    supp_vars = []
    for var in variables:
        if var not in list(ev_vars.keys()) and var not in hyp_vars:
            supp_vars.append(var)

    # Variables to store relevant/irrelevant sets
    relevant_sets = []
    irrelevant_sets = []

    # Delete the ones conditionally independent. d-separation is giving trouble right now. We will delete in the future
    dsep_by_ev = []  # = prune_network(bn, list(ev_vars.keys()), hyp_vars, supp_vars = supp_vars)
    bn_pr = bn

    supp_vars_og = supp_vars
    tmp = supp_vars
    supp_vars = []

    for i in tmp:
        if bn_pr.isIndependent(i, hyp_vars, list(ev_vars.keys())):
            dsep_by_ev.append(i)
        else:
            supp_vars.append(i)

    if len(supp_vars) == 0:
        return [], dsep_by_ev

    # supp vars contains variables that are not dsep at this point
    # tmp contains all supplementary nodes

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

    # Supp_vars still contain variables that are NOT d-separated

    dsep_by_ev = tuple(dsep_by_ev)
    irrelevant_singletons = []
    if len(dsep_by_ev) > 0:
        irrelevant_sets.append(dsep_by_ev)
        for i in dsep_by_ev:
            irrelevant_singletons.append(i)

    # Get the posterior and the argmax from the original MAP-query
    posterior = map_query(bn_pr, ev_vars=ev_vars, hyp_vars=hyp_vars)
    y = posterior.argmax()[0][0]

    h_star = list()
    # Value assignment y, but in a different format
    posterior_reorder = {k: posterior.argmax()[0][0][k] for k in hyp_vars}
    for key in posterior_reorder.keys():
        h_star.append(bn_pr.variableFromName(key).labels()[posterior_reorder[key]])
    h_star = tuple(h_star)

    c_exp = get_c_exp(bn_pr, ev_vars, hyp_vars, hyp_vars_assign=y)
    inst_c = gum.Instantiation(c_exp)

    to_prop = ev_vars.copy()
    to_prop.update(y)
    ie_max = propagate_evidence(bn_pr, to_prop)

    relevant_singletons = []
    for j in supp_vars:
        # If relevant
        if ie_max.posterior(j).argmin()[1] == 0:
            relevant_sets.append((j,))
            relevant_singletons.append(j)
    tmp = list_diff(supp_vars, relevant_singletons)

    omega_h = omega(hyp_vars, bn=bn_pr)
    for h_i in omega_h:
        if len(tmp) == 0:
            break
        if h_i == h_star:
            continue
        # print(h_i)
        # print(y)
        to_prop = ev_vars.copy()
        to_prop.update(dict(zip(hyp_vars, list(h_i))))
        ie = propagate_evidence(bn_pr, to_prop)

        inst_c.fromdict(to_prop)
        c_expon = c_exp.get(inst_c)
        c_i = None
        if c_expon == 0:
            c_i = -np.inf
        else:
            c_i = math.log(c_expon)

        for j in tmp:
            post_max = ie_max.posterior(j)
            post = None
            try:
                post = ie.posterior(j)
            except:
                continue
            inst_h = gum.Instantiation(post)
            while not inst_h.end() and (post.get(inst_h) / post_max.get(inst_h) <= 0 or not math.log(
                    post.get(inst_h) / post_max.get(inst_h)) + c_i > 0):
                inst_h.inc()
            if not inst_h.end():
                relevant_sets.append((j,))
                relevant_singletons.append(j)
        # print(S_split[0])
        # print(relevant_singletons)
        tmp = list_diff(tmp, relevant_singletons)

    irrelevant_singletons = irrelevant_singletons + tmp
    for j in irrelevant_singletons:
        irrelevant_sets.append((j,))
        # S_split, irrelevant_sets = conditional_independence_prune(bn_pr,supp_vars,hyp_vars,ev_vars,j, S_split, irrelevant_sets)

    if len(irrelevant_singletons) == 0 or depth == 1:
        return relevant_sets, irrelevant_sets

    S = powerset(irrelevant_singletons, depth=depth)
    S.pop(0)
    # Divide by length
    S_split = []
    size = 0
    for i in S:
        if len(i) != size:
            size = size + 1
            S_split.append([])
        S_split[-1].append(i)

    for j in irrelevant_singletons:
        S_split, irrelevant_sets = conditional_independence_prune(bn_pr, supp_vars, hyp_vars, ev_vars, (j,), S_split,
                                                                  irrelevant_sets, depth=depth)

    irrels = powerset(dsep_by_ev, depth)
    irrels.pop(0)
    for i in irrels:
        try:
            S_split[len(i) - 1].remove(i)
        except ValueError:
            pass  # do nothing!

    for i in range(1, len(S_split)):
        tmp = S_split[i]
        for j in tmp:
            # If relevant
            # print(list(j))
            if map_dependence(bn, set_R=list(j), ev_vars=ev_vars, hyp_vars=hyp_vars, hyp_vars_assign=y):
                relevant_sets.append(j)
                # Apply prune
                S_split, relevant_sets = decomposition_prune(j, S_split, relevant_sets)
            # If irrelevant
            else:
                irrelevant_sets.append(j)
                S_split, irrelevant_sets = conditional_independence_prune(bn_pr, supp_vars, hyp_vars, ev_vars, j,
                                                                          S_split, irrelevant_sets, depth)

    # Simplify irrelevant sets
    irrelevant_sets = sorted(irrelevant_sets, key=len)
    new_irrel_sets = []
    for i in range(0, len(irrelevant_sets)):
        subset_flag = False
        for j in range(len(irrelevant_sets) - 1, i, -1):
            if set(irrelevant_sets[i]).issubset(set(irrelevant_sets[j])):
                subset_flag = True
                break
        if not subset_flag:
            new_irrel_sets.append(irrelevant_sets[i])
    irrelevant_sets = new_irrel_sets

    return relevant_sets, irrelevant_sets


def check_every_r_mixed(bn, ev_vars, hyp_vars, depth=np.inf):
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
        return [], dsep_by_ev

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

    irrelevant_singletons = []
    irrelevant_sets = []
    relevant_sets = []
    if len(dsep_by_ev) > 0:
        irrelevant_singletons = dsep_by_ev

    posterior = map_query(bn, ev_vars=ev_vars, hyp_vars=hyp_vars)
    y = posterior.argmax()[0][0]

    min_jsd = np.inf
    most_irrel = None
    for i in supp_vars:
        # print(ev_vars, hyp_vars, [i])
        irrel = map_independence(bn_pr, set_R=[i], ev_vars=ev_vars, hyp_vars=hyp_vars)
        if irrel:
            irrelevant_singletons.append(i)
            irrelevant_sets.append((i,))
        else:
            relevant_sets.append((i,))

    if len(dsep_by_ev) > 0:
        irrelevant_sets.append(tuple(dsep_by_ev))

    if len(irrelevant_singletons) == 0 or depth == 1:
        return relevant_sets, irrelevant_sets

    S = powerset(irrelevant_singletons, depth=depth)
    S.pop(0)
    # Divide by length
    S_split = []
    size = 0
    for i in S:
        if len(i) != size:
            size = size + 1
            S_split.append([])
        S_split[-1].append(i)

    for j in irrelevant_singletons:
        S_split, irrelevant_sets = conditional_independence_prune(bn_pr, supp_vars, hyp_vars, ev_vars, (j,), S_split,
                                                                  irrelevant_sets, depth=depth)

    irrels = powerset(dsep_by_ev, depth)
    irrels.pop(0)
    for i in irrels:
        try:
            S_split[len(i) - 1].remove(i)
        except ValueError:
            pass  # do nothing!

    for i in range(1, len(S_split)):
        tmp = S_split[i]
        for j in tmp:
            # If relevant
            # print(list(j))
            if map_dependence(bn, set_R=list(j), ev_vars=ev_vars, hyp_vars=hyp_vars, hyp_vars_assign=y):
                relevant_sets.append(j)
                # Apply prune
                S_split, relevant_sets = decomposition_prune(j, S_split, relevant_sets)
            # If irrelevant
            else:
                irrelevant_sets.append(j)
                S_split, irrelevant_sets = conditional_independence_prune(bn_pr, supp_vars, hyp_vars, ev_vars, j,
                                                                          S_split, irrelevant_sets, depth)

    # Simplify irrelevant sets
    irrelevant_sets = sorted(irrelevant_sets, key=len)
    new_irrel_sets = []
    for i in range(0, len(irrelevant_sets)):
        subset_flag = False
        for j in range(len(irrelevant_sets) - 1, i, -1):
            if set(irrelevant_sets[i]).issubset(set(irrelevant_sets[j])):
                subset_flag = True
                break
        if not subset_flag:
            new_irrel_sets.append(irrelevant_sets[i])
    irrelevant_sets = new_irrel_sets

    return relevant_sets, irrelevant_sets
