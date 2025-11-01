import numpy as np
from dimod import Binary, ConstrainedQuadraticModel
from dimod.binary import quicksum
from dwave.system.samplers import LeapHybridCQMSampler


# Choose locations for stations based on the MWIS criteria
def choose_location(N, W_loc, D, radius, token):
    cqm = ConstrainedQuadraticModel()

    x = np.array([Binary(i) for i in range(N)])

    objective = quicksum(-W_loc * x)  # Wanna maximize the chosen weight
    cqm.set_objective(objective)

    for i in range(N):
        for j in range(i + 1, N):
            if D[i, j] < radius:
                cqm.add_constraint(x[i] * x[j] == 0)

    cqm_sampler = LeapHybridCQMSampler(token=token)
    sampleset = cqm_sampler.sample_cqm(cqm, label="choose_location")
    feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)

    best_index = np.argmin(feasible_sampleset.record.energy)
    best_sol = feasible_sampleset.record[best_index][0]

    chosen_stations = np.where(best_sol == 1)[0]

    return chosen_stations
