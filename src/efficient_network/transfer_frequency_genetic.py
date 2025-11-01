# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:03:55 2023

@author: Lai,Chia-Tso
"""

import numpy as np
import networkx as nx
import random
from copy import deepcopy


def adjacent_matrix(graph, n):
    adjacent = np.zeros((n, n))
    for edge in graph:
        adjacent[edge[0], edge[1]] = 1
        adjacent[edge[1], edge[0]] = 1

    return adjacent


def generate_population(n, nx_graph, popu_size):
    # create a list of nodes with an odd number of connections
    terminal = [
        i
        for i in range(n)
        if np.sum(adjacent_matrix(list(nx_graph.edges), n)[i, :]) % 2 == 1
    ]
    # shuffle the nodes randomly to form different lines
    population = [random.sample(terminal, len(terminal)) for i in range(popu_size)]
    return population


# transfer time from a source to all the nodes in the graph
def single_source_transfer_freq(nx_graph, chromo, start):
    n = len(nx_graph.nodes)
    lines = [
        set(nx.shortest_path(nx_graph, chromo[2 * i], chromo[2 * i + 1]))
        for i in range(int(len(chromo) / 2))
    ]
    nl = len(lines)

    stations = []
    for i in range(nl):
        stations += list(lines[i])

    if set(stations) != set(
        range(n)
    ):  # if the lines do not cover all the stations then it's invalid
        return "not every station is covered"

    else:
        # adjacet matrix of lines
        ad = np.zeros((nl, nl))
        for i in range(nl):
            for j in range(nl):
                if len(lines[i].intersection(lines[j])) != 0:
                    ad[i, j] = 1
                else:
                    ad[i, j] = 0
        for i in range(nl):
            ad[i, i] = 0

        start_line = set(
            [i for i in range(nl) if start in lines[i]]
        )  # lines that go through start station

        reachable = set().union(
            *[lines[i] for i in list(start_line)]
        )  # reachable stations from the start
        rounds = [reachable]
        current = [i for i in list(start_line)]  # current lines
        explored_lines = list(start_line)

        while not set(range(n)).issubset(reachable):
            neighbors = [set(np.where(ad[i, :] == 1)[0]) for i in current]
            neighbors = list(set().union(*neighbors) - set(explored_lines))
            reachable = reachable.union(*[lines[k] for k in neighbors])
            rounds.append(reachable)
            current = neighbors
            explored_lines += neighbors
            if reachable == set(range(n)):
                break

        transfers = [
            rounds[0] if i == 0 else rounds[i] - rounds[i - 1]
            for i in range(len(rounds))
        ]

        d = np.zeros(n)
        for index, item in enumerate(transfers):
            for station in list(item):
                d[station] = index
        return d


# Average transfer times for any pair of nodes in the graph
def avg_transfer_freq(nx_graph, chromo):
    n = len(nx_graph.nodes)

    lines = [
        set(nx.shortest_path(nx_graph, chromo[2 * i], chromo[2 * i + 1]))
        for i in range(int(len(chromo) / 2))
    ]
    nl = len(lines)

    stations = []
    for i in range(nl):
        stations += list(lines[i])

    if set(stations) != set(
        range(n)
    ):  # if the lines do not cover all the stations then it's invalid
        return 100000

    else:
        distance = np.array(
            [
                list(single_source_transfer_freq(nx_graph, chromo, start))
                for start in range(n)
            ]
        )

        summ = np.sum(distance[np.triu_indices(n)])
        norm = n * (n - 1) / 2
        avg_freq = summ / norm

        return avg_freq


# evaluate the transfer time and the increase in overall length
def fitness(nx_graph, W, chromo):
    n = len(nx_graph.nodes)

    lines = [
        nx.shortest_path(nx_graph, chromo[2 * i], chromo[2 * i + 1])
        for i in range(int(len(chromo) / 2))
    ]
    nl = len(lines)

    stations = []
    for i in range(nl):
        stations += lines[i]

    if set(stations) != set(
        range(n)
    ):  # if the lines do not cover all the stations then it's invalid
        fitness = 10000000

    else:
        fitness = avg_transfer_freq(nx_graph, chromo)

    # calculate increased overall length of the network
    lines_length = [
        nx.shortest_path_length(
            nx_graph, chromo[2 * i], chromo[2 * i + 1], weight="weight"
        )
        for i in range(int(len(chromo) / 2))
    ]
    plus = np.sum(lines_length) - np.sum([W[u][v] for u, v in list(nx_graph.edges)])

    return fitness + plus / 10


def selection(popu, nx_graph, W, selection_rate):
    sample_size = round(len(popu) * selection_rate)
    fitness_score = [fitness(nx_graph, W, chromo) for chromo in popu]
    best_chromo = np.argsort(fitness_score)[:sample_size]  # take out the smaller ones
    parents = [popu[i] for i in best_chromo]
    return parents


def crossover(parents, popu_size):
    offspring = parents
    for i in range(popu_size - len(parents)):
        par = random.sample(range(len(parents)), 1)[0]
        selected = parents[par]
        indices = random.sample(range(len(parents[0])), 2)
        cross_par = deepcopy(selected)
        cross_par[indices[0]], cross_par[indices[1]] = (
            cross_par[indices[1]],
            cross_par[indices[0]],
        )
        offspring.append(cross_par)
    return offspring


# the standard deviation of the top 30% of the population
def top_std(popu, nx_graph, W):
    fitness_score = [fitness(nx_graph, W, chromo) for chromo in popu]
    return np.std(np.sort(fitness_score)[: round(len(popu) * 0.3)])


def train_line_genetic(
    nx_graph, popu_size, W, selection_rate, max_iteration, cutoff_std
):
    n = len(W)
    popu = generate_population(n, nx_graph, popu_size)

    best_individual = []
    mini_fit = []
    iteration = 0
    while iteration < max_iteration:
        iteration += 1
        parents = selection(popu, nx_graph, W, selection_rate)
        offspring = crossover(parents, popu_size)
        popu = offspring

        fitness_score = [fitness(nx_graph, W, chromo) for chromo in popu]
        best = popu[np.argmin(fitness_score)]
        mini_fit.append(np.min(fitness_score))
        best_individual.append(best)

        if top_std(popu, nx_graph, W) < cutoff_std:
            break

    optimal = best_individual[np.argmin(mini_fit)]
    optimal = [
        nx.shortest_path(nx_graph, optimal[2 * i], optimal[2 * i + 1])
        for i in range(int(len(optimal) / 2))
    ]
    print("minimum cost:", np.min(mini_fit))
    print("optimal lines:", optimal)
    print("iteration:", iteration)

    return [optimal, mini_fit]
