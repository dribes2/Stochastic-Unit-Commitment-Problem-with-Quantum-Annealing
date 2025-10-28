from functions import *


P_min = [50, 80, 40]
P_max = [350, 200, 140]
R_down = [300, 150, 100]
R_up = [200,100,100]
C_startup = [20, 18, 5]
C_shutdown = [0.5,0.3,1.0]
b = [0.1, 0.125, 0.150]
#b = [100, 125, 150]
c = [5, 7, 6]

demand = [0, 160, 500, 400]
u_initial = [0, 0, 1]
P_initial = [0, 0, 100]



logicON_penalty = 50
logicOFF_penalty = 50
logic1_penalty = 10
logic2_penalty = 100
coupling_penalty = 100
demand_penalty = 10
ramp_penalty = 200
capacity_penalty = 200
threshold = 0.5
slope_sigmoid = 20
center_sigmoid = 0.6
alpha = 0.5

N = 3
T = 4
n = 9
#delta = [(P_max[i] - P_min[i])/(2**n - 1) for i in range(N)]
delta = [1, 1, 1]

parameters = {"N": N,
              "T": T,
              "n": n,
              "delta": delta,
              "P_min": P_min,
              "P_max": P_max,
              "C_startup": C_startup,
              "C_shutdown": C_shutdown,
              "R_up": R_up,
              "R_down": R_down,
              "b": b,
              "c": c,
              "demand": demand,
              "u_initial": u_initial,
              "P_initial": P_initial,
              "logicON_penalty": logicON_penalty,
              "logicOFF_penalty": logicOFF_penalty,
              "logic1_penalty": logic1_penalty,
              "logic2_penalty": logic2_penalty,
              "coupling_penalty": coupling_penalty,
              "demand_penalty": demand_penalty,
              "ramp_penalty": ramp_penalty,
              "capacity_penalty": capacity_penalty,
              "threshold_constraints": threshold,
              "slope_sigmoid": slope_sigmoid,
              "center_sigmoid": center_sigmoid,
              "alpha_sigmoid": {"demand": alpha, "logic1": alpha, "logic2": alpha, "coupling": alpha, "ramp": alpha, "capacity": alpha}}

constraints = {"logic1": True,
               "logic2": True,
               "demand": True,
               "coupling": True,
               "ramp": True,
               "capacity": True}

parameters["threshold_constraints"] = 0.85
parameters["slope_sigmoid"] = 14
parameters["center_sigmoid"] = 0.3

constraints["logic1"] = True
constraints["logic2"] = True
constraints["demand"] = True
constraints["ramp"] = True
constraints["capacity"] = True
constraints["coupling"] = True


dictionary_out = run_classical_algorithm(parameters, constraints, iterations = 50, num_reads=100)
plot_after_algorithm(dictionary_out, "ratios", constraints, parameters)


sa = neal.SimulatedAnnealingSampler()
bqm = build_BQM(parameters, constraints)
#bqm.normalize()

sampleset = sa.sample(bqm,
                      num_reads = 1000,
                      num_sweeps = 5000)

min_energy = find_lowest_energy_solution(sampleset, parameters, constraints, True)

print(f"Energy of lowest feasible solution: {min_energy}")

plot_feasibility_histogram(sampleset, is_solution_feasible, cost, parameters, constraints, bin_width=3.0)