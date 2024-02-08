import numpy as np
import pandas as pd
import pyAgrum as gum
import math
import scipy

from itertools import chain, combinations, product


def equal_sets(set1, set2):
    if len(set1) != len(set2):
        return False
    for i in set1:
        if i not in set2:
            return False
    return True


def list_diff(list1, list2):
    diff = []
    for i in list1:
        if i not in list2:
            diff.append(i)
    return diff


def prune_network(bn, ev_vars, hyp_vars, supp_vars=None):
    # IMPORTANT: Hard copy of the bn
    bn_pr = gum.BayesNet(bn)
    # Find supplementary nodes of a network if not given

    if supp_vars is None:
        supp_vars = []
        for i in bn_pr.names():
            if not i in ev_vars and not i in hyp_vars:
                supp_vars.append(i)
    # Delete from the network the nodes that are conditionally independent from the hypothesis variables (target) given the evidence
    dsep_nodes = []
    for i in supp_vars:
        if bn_pr.isIndependent(i, hyp_vars, ev_vars):
            dsep_nodes.append(i)
    for i in dsep_nodes:
        bn_pr.erase(i)
    return bn_pr, dsep_nodes


def map_query(bn, ev_vars, hyp_vars, alg="jt"):
    return propagate_evidence(bn, ev_vars, joint_targets=[hyp_vars], alg=alg).jointPosterior(set(hyp_vars))


def propagate_evidence(bn, ev_vars, joint_targets=[], alg="jt"):
    ie = gum.ShaferShenoyInference(bn)
    if alg == "gs":
        ie = gum.GibbsSampling(bn)
        ie.setMaxTime(3)
        for i in joint_targets:
            if len(i) > 1:
                ie = gum.ShaferShenoyInference(bn)
                break
    for i in joint_targets:
        # Ignore if the target is not joint
        ie.addJointTarget(set(i))
    ie.setEvidence(ev_vars)
    ie.makeInference()
    return ie


def omega(variables, data=None, bn=None):
    domains = []
    if bn is not None:
        for variable in variables:
            domains.append(sorted(bn.variableFromName(variable).labels()))
    else:
        for variable in variables:
            domains.append(sorted(data[variable].unique()))
    return [p for p in product(*domains)]


def prepare_mapi(instances, variables):
    # Type checking
    if isinstance(instances, pd.core.frame.DataFrame):
        instances = instances.to_numpy()
    # Array for each instance
    evidence_set = []
    # For each instance:
    for instance in instances:
        # Check which are the evidence variables
        evidence = dict()
        for i, value in enumerate(instance):
            if isinstance(value, str) or not math.isnan(value):
                evidence[variables[i]] = value
        evidence_set.append(evidence)
    return evidence_set


def powerset(iterable, depth=np.inf):
    s = list(iterable)
    tmp = list(chain.from_iterable(combinations(s, r) for r in range(min(len(s), depth) + 1)))
    return tmp


# JSD divergence
def JSD(potential_1, potential_2):
    p = potential_1.toarray().ravel()
    q = potential_2.toarray().ravel()

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    return divergence
