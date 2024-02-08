from defeater_explanations.utils import *


# False if R is irrelevant/MAP-independent
def map_dependence(bn, set_R, ev_vars, hyp_vars, hyp_vars_assign=None, hyp_posterior=None, return_jsd=False):
    # Check which are the supplementary (missing) variables
    variables = bn.names()
    supp_vars = []
    for var in variables:
        if var not in list(ev_vars.keys()) and var not in hyp_vars:
            supp_vars.append(var)
    # Check if R in unobserved
    # print(supp_vars)
    for R in set_R:
        if R not in supp_vars:
            raise Exception("The variable", R, "is in the set R but is not a supplementary node")
    y = hyp_vars_assign
    posterior = hyp_posterior
    if y is None:
        # Perform a MAP-query and get the argmax from the posterior.
        # Check if a posterior for P(H|e) was provided
        if hyp_posterior is None:
            posterior = map_query(bn, ev_vars=ev_vars, hyp_vars=hyp_vars)
            y = posterior.argmax()[0][0]
        else:
            y = posterior.argmax()[0][0]
    # Obtain domain of R
    omega_R = omega(set_R, bn=bn)
    # For each value assignment r in omega(R)
    jsd = 0
    for value_assignment_r in omega_R:
        # Fill in values
        ev_vars_alt = ev_vars.copy()
        for i, value in enumerate(value_assignment_r):
            ev_vars_alt[set_R[i]] = value
        # print(instance)
        # print(instance_alt)
        # Inference with evidence and r
        posterior_alt = None
        try:
            posterior_alt = map_query(bn, ev_vars=ev_vars_alt, hyp_vars=hyp_vars)
        except:
            continue
        y_alt = posterior_alt.argmax()[0][0]
        # Check if we need to compute the jsd divergence between P(H|e) and P(H|e,r)
        if return_jsd:
            jsd = max(jsd, JSD(posterior, posterior_alt))
        # Comparar con prediccion de instance
        # print(value_assignment_r)
        # print(y, " == ", y_alt)
        # print(posterior.argmax()[1], "--", posterior_alt.argmax()[1])
        if y != y_alt:
            if return_jsd:
                return True, jsd
            else:
                return True
    if return_jsd:
        return False, jsd
    else:
        return False


def map_independence(bn, set_R, ev_vars, hyp_vars, hyp_vars_assign=None, hyp_posterior=None, return_jsd=False):
    if return_jsd:
        mapd, jsd = map_dependence(bn, set_R, ev_vars, hyp_vars, hyp_vars_assign=hyp_vars_assign,
                                   hyp_posterior=hyp_posterior, return_jsd=True)
        return (not mapd, jsd)
    else:
        return not map_dependence(bn, set_R, ev_vars, hyp_vars, hyp_vars_assign=hyp_vars_assign,
                                  hyp_posterior=hyp_posterior, return_jsd=False)
