import dimod
import json
import numpy as np
import matplotlib.pyplot as plt

def parse_feasible(sampleset):
    print("parsing best ...")
    best = sampleset.filter(lambda row: row.is_feasible)
    print("found best")
    return best

data = {}
feasible_data = {}
energies = {}

scenarios_vector = np.arange(20000, 0, -800)

for scenarios in scenarios_vector:
    with open(f"results/results_cost_vs_scenarios_{scenarios}scenarios_0rho_0epsilon.json", "r") as f:
        data[scenarios] = dimod.SampleSet.from_serializable(json.load(f))
        feasible = parse_feasible(data[scenarios])
        feasible_data[scenarios] = feasible
        if len(feasible) > 0:
            energies[scenarios] = np.sort(feasible.record.energy)
        else:
            energies[scenarios] = []

plot_data = {scenarios: {'avg': [], 'min': [], 'max': []} for scenarios in scenarios_vector}

averages = []
minimums = []
maximums = []

for scenarios in scenarios_vector:
    sorted_energies = energies[scenarios]
    if len(sorted_energies) >= 5:
        top5 = sorted_energies[:5]
    else:
        top5 = sorted_energies

    avg_val = np.mean(top5)
    min_val = np.min(top5)
    max_val = np.max(top5)

    plot_data[scenarios]['avg'].append(avg_val)
    plot_data[scenarios]['min'].append(min_val)
    plot_data[scenarios]['max'].append(max_val)

    averages.append(plot_data[scenarios]['avg'])
    minimums.append(plot_data[scenarios]['min'])
    maximums.append(plot_data[scenarios]['max'])


plt.figure(figsize=(10, 6))
plt.plot(scenarios_vector, averages, "o-", label = "CQM Hybrid Solver", color = "blue")
plt.fill_between(scenarios_vector, np.array(minimums).flatten(), np.array(maximums).flatten(), color= "blue", alpha=0.2)

plt.xlabel("Scenarios", fontsize = 16)
plt.ylabel("Expected cost [$]", fontsize = 16)
plt.tick_params(axis='both', labelsize=16)

plt.legend(fontsize = 16)
plt.tight_layout()
plt.savefig("plot_cost_vs_scenarios_DWave.pdf")
plt.show()