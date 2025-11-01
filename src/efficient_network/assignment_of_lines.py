# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:10:10 2023

@author: Lai, Chia-Tso
"""

import numpy as np
import pandas as pd
import networkx as nx
from dwave.system.samplers import LeapHybridCQMSampler
from dimod import Binary, ConstrainedQuadraticModel
from dimod.binary import quicksum
from copy import deepcopy


# Make the adjacent matrix of a graph
def adjacent_matrix(graph, n):
    adjacent = np.zeros((n, n))
    for edge in graph:
        adjacent[edge[0], edge[1]] = 1
        adjacent[edge[1], edge[0]] = 1

    return adjacent


# Assign lines by equalizing the passenger load of each line while forbidding adding extra connections
def train_line_equal_load(Wn, nx_graph, token):  # Wn stores the weights of the stations
    n = len(Wn)
    cqm = ConstrainedQuadraticModel()

    A = adjacent_matrix(list(nx_graph.edges), n)
    connections = np.array(
        [np.sum(A[i, :]) for i in range(n)]
    )  # number of connections on each station
    wn = Wn / connections  # the reduced load of each station

    trans_points = [i for i in range(n) if np.sum(A[i, :]) >= 3]
    non_trans = [i for i in range(n) if np.sum(A[i, :]) <= 2]
    nl = len(trans_points) + 1  # number of lines

    terminal = [i for i in range(n) if np.sum(A[i, :]) in [1, 3]]  # terminal stations

    x = np.array([Binary(f"x{i}{j}") for i in range(nl) for j in range(n)]).reshape(
        nl, n
    )

    # minimize the difference between sums of any two lines
    objective = 0
    for k in range(nl):
        for s in range(k + 1, nl):
            objective += (
                quicksum([x[k][j] * wn[j] for j in range(n)])
                - quicksum([x[s][j] * wn[j] for j in range(n)])
            ) ** 2
    cqm.set_objective(objective)

    # need to form a path (branch roads and crossroads are also included here)
    for i in range(nl):
        term1 = 0.5 * np.dot(x[i, :], np.dot(A, x[i, :].transpose()))
        term2 = quicksum([x[i, k] for k in range(n)])
        cqm.add_constraint(term2 - term1 == 1)

    # each line cannot have any station with more than 2 connections
    for i in range(nl):
        for j in range(n):
            cqm.add_constraint(
                quicksum(x[i, j] * A[j, :] * x[i, :]) <= 2
            )  # xij*Aj is the adjacent of "chosen" node

    # Each line needs to contain at least 2 terminal stations (3-connection stations are both terminal and intermediate)
    for i in range(nl):
        cqm.add_constraint(quicksum([x[i, k] for k in terminal]) >= 2)

    # each connected pair of stations need to be on at least one same line
    for j in range(n):
        for k in range(j + 1, n):
            if A[j, k] == 1:
                cqm.add_constraint(
                    quicksum([x[i, j] * x[i, k] for i in range(nl)]) >= 1
                )

    # every station needs to be covered
    for j in range(n):
        cqm.add_constraint(quicksum([x[k, j] for k in range(nl)]) >= 1)
    # transfer points need to have at exactly 2 lines going through
    for j in trans_points:
        cqm.add_constraint(quicksum([x[k, j] for k in range(nl)]) == 2)
    # non_transfer points need to have exactly 1 line going through
    for j in non_trans:
        cqm.add_constraint(quicksum([x[k, j] for k in range(nl)]) == 1)

    cqm_sampler = LeapHybridCQMSampler(token=token)
    sampleset = cqm_sampler.sample_cqm(cqm, label="stations_division")
    feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)

    data = pd.DataFrame(feasible_sampleset)
    best_index = np.argmin(feasible_sampleset.record.energy)
    best_dict = dict(data.iloc[best_index, :])

    lines = []
    for i in range(nl):
        line = []
        for j in range(n):
            if best_dict[f"x{i}{j}"] == 1:
                line.append(j)
        lines.append(line)

    return lines


# Assign lines by minimizing the difference in length of different lines
def train_line_equal_length(n, nx_graph, token):
    cqm = ConstrainedQuadraticModel()

    A = adjacent_matrix(list(nx_graph.edges), n)

    terminal = [i for i in range(n) if np.sum(A[i, :]) % 2 == 1]  # terminal stations
    nt = len(terminal)

    # make the distance matrix of terminals
    length = [
        {k: nx.shortest_path_length(nx_graph, t, k, weight="weight") for k in terminal}
        for t in terminal
    ]
    d = np.zeros((nt, nt))
    for i in range(nt):
        for j in range(nt):
            d[i][j] = length[i][terminal[j]]

    # upper traingle of d
    w = np.array([d[i][j] for i in range(nt) for j in range(i + 1, nt)])

    x = [
        0 if (i > j) or (i == j) else Binary(f"x{i}{j}")
        for i in range(nt)
        for j in range(nt)
    ]
    x = np.array(x).reshape(nt, nt)
    x = [x[j][i] if i > j else x[i][j] for i in range(nt) for j in range(nt)]
    x = np.array(x).reshape(nt, nt)

    upper_x = np.array([x[i, j] for i in range(nt) for j in range(i + 1, nt)])

    wx = w * upper_x

    # minimize the length difference between each line
    term1 = quicksum(
        [(wx[k] - wx[s]) ** 2 for k in range(len(wx)) for s in range(k + 1, len(wx))]
    )
    # minus the chosen-not chosen term
    term2 = (
        quicksum([wx[k] ** 2 for k in range(len(wx))]) * nt * (0.5 * nt - 1)
    )  # nt*(0.5*nt-1) is the num of non-picked pair
    # minimize the overhead in connections as well
    term3 = quicksum(wx)
    cqm.set_objective(term1 - term2 + term3)

    # each terminal can only be picked once
    for i in range(nt):
        cqm.add_constraint(quicksum([x[i, j] for j in range(nt)]) == 1)

    cqm_sampler = LeapHybridCQMSampler(token=token)
    sampleset = cqm_sampler.sample_cqm(cqm, label="line_assignment")
    feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)

    data = pd.DataFrame(feasible_sampleset)
    best_index = np.argmin(feasible_sampleset.record.energy)
    best_dict = dict(data.iloc[best_index, :])

    lines = []
    for i in range(nt):
        for j in range(i + 1, nt):
            if best_dict[f"x{i}{j}"] == 1:
                lines.append(
                    nx.shortest_path(
                        nx_graph, terminal[i], terminal[j], weight="weight"
                    )
                )

    # connections between odd degree(except for 1) nodes might be redundant lines
    refined = deepcopy(lines)
    odd = [i for i in range(n) if (np.sum(A[i, :]) % 2 == 1) and (np.sum(A[i, :]) != 1)]
    for line in refined:
        if (line[0] in odd) and (line[-1] in odd):
            refined.remove(line)

    refinedset = [set(line) for line in refined]
    k = set().union(*refinedset)
    if k.issubset(set(range(n))):
        lines = refined

    return lines
