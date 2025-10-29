import numpy as np
import dimod
import neal
import matplotlib.pyplot as plt
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite

def build_BQM(parameters, constraints):
    #parameters: a dictionary containing all the parameters that define the model: costs, ramping limits, lagrange multipliers, ...
    #constraints: dictionary containing 6 keys with the 6 names of the constraints, and each of them, a boolean defining whether the constraint is introduced or not.
    N = parameters["N"]
    T = parameters["T"]
    n = parameters["n"]


    bqm = dimod.BinaryQuadraticModel("BINARY")

    u = {(i, t): f"u{i}{t}" for i in range(N) for t in range(T)}
    z_on = {(i, t): f"z_on{i}{t}" for i in range(N) for t in range(T)}
    z_off = {(i, t): f"z_off{i}{t}" for i in range(N) for t in range(T)}
    P = {(i, t, k): f"P{i}{t}_{k}" for i in range(N) for t in range(T) for k in range(n)}
    
    for i in range(N):
        for t in range(1,T):
            if constraints["logic1"]: 
                bqm.add_linear_equality_constraint([(z_on[i,t], 1), (z_off[i,t], -1), (u[i,t], -1), (u[i,t-1], 1)],
                                                lagrange_multiplier = parameters["logic1_penalty"],
                                                constant = 0)
            if constraints["logic2"]:
                bqm.add_quadratic(z_on[i,t], z_off[i,t], parameters["logic2_penalty"])

            if constraints["coupling"]:
                for k in range(parameters["n"]):
                    bqm.add_linear(P[i,t,k], 2**k*parameters["delta"][i]*parameters["coupling_penalty"])
                    bqm.add_interaction(P[i,t,k],u[i,t], -1*2**k*parameters["delta"][i]*parameters["coupling_penalty"])

            if constraints["capacity"]:
                vector_capacity = []
                vector_capacity.append([u[i,t], parameters["P_min"][i]])
                for k in range(parameters["n"]):
                    vector_capacity.append([P[i,t,k], 2**k*parameters["delta"][i]])
                bqm.add_linear_inequality_constraint(vector_capacity,
                                                    lagrange_multiplier = parameters["capacity_penalty"],
                                                    label = f"capacity_{i}_{t}",
                                                    constant = 0,
                                                    lb = 0,
                                                    ub = parameters["P_max"][i],
                                                    cross_zero = False,
                                                    penalization_method = "slack")

    for t in range(1, parameters["T"]):
        vector_total_P = []
        for i in range(parameters["N"]):
            vector_total_P.append([u[i,t], parameters["P_min"][i]])
            for k in range(parameters["n"]):
                vector_total_P.append([P[i,t,k], 2**k*parameters["delta"][i]])

            vector_difference = []
            vector_difference.append((u[i,t], parameters["P_min"][i]))
            vector_difference.append((u[i,t-1], -1*parameters["P_min"][i]))

            for k in range(parameters["n"]):
                vector_difference.append((P[i,t,k], parameters["delta"][i]*2**k))
                vector_difference.append((P[i,t-1,k], -1*parameters["delta"][i]*2**k))
            
            # Constraint for ramping limits
            if constraints["ramp"]:
                bqm.add_linear_inequality_constraint(vector_difference,
                                                    lagrange_multiplier = parameters["ramp_penalty"],
                                                    label = f"ramp_up_down_{i}_{t}_",
                                                    constant = 0,
                                                    lb = -1*parameters["R_down"][i],
                                                    ub = parameters["R_up"][i],
                                                    cross_zero = False,
                                                    penalization_method = "slack")
            
            

        # Constraint for demand

        if constraints["demand"]:
            bqm.add_linear_inequality_constraint(vector_total_P,
                                                lagrange_multiplier = parameters["demand_penalty"],
                                                label = f"demand_{t}_",
                                                constant = 0,
                                                lb = parameters["demand"][t],
                                                ub = 10**4,
                                                penalization_method = "slack",
                                                cross_zero = False)
        
    # Fixing all variables corresponding to time 0 (prior to time horizon)
    for i in range(N):
        if constraints["logic1"] or constraints["ramp"]:
            bqm.fix_variable(u[i,0], parameters["u_initial"][i])
        if constraints["ramp"]:
            p = parameters["P_initial"][i]
            p = round((p - parameters["P_min"][i]*parameters["u_initial"][i])/parameters["delta"][i])
            binary_p = list(reversed([int(j) for j in format(p, f"0{n}b")]))
            for k in range(n):
                bqm.fix_variable(P[i,0,k], binary_p[k])
    # Adding all cost terms to the cost function
    for i in range(N):
        for t in range(1, T):
            bqm.add_linear(u[i,t], parameters["c"][i]) # Fixed Cost
            bqm.add_linear(z_on[i,t], parameters["C_startup"][i]) # Start-up cost
            bqm.add_linear(z_off[i,t], parameters["C_shutdown"][i]) # Shut-down cost
            bqm.add_linear(u[i,t], parameters["b"][i]*parameters["P_min"][i])
            for k in range(n):
                bqm.add_linear(P[i,t,k], parameters["b"][i]*parameters["delta"][i]*2**k) # Linear cost
    return bqm

###### FUNCTIONS TO SHOW THE VALUES OF THE VARIABLES. THEY ALL RETURN A MATRIX OF SIZE N*T ######
def show_u(dict, parameters):
    out = np.zeros((parameters["N"], parameters["T"]))
    out[:,0] = parameters["u_initial"][:]
    for i in range(parameters["N"]):
        for t in range(1, parameters["T"]):
            out[i,t] = dict[f"u{i}{t}"]
    return out

def show_z_on(dict, parameters):
    out = np.zeros((parameters["N"], parameters["T"]))
    out[:,0] = [0, 0, 0]
    for i in range(parameters["N"]):
        for t in range(1, parameters["T"]):
            out[i,t] = dict[f"z_on{i}{t}"]
    return out

def show_z_off(dict, parameters):
    out = np.zeros((parameters["N"], parameters["T"]))
    out[:,0] = [0, 0, 0]
    for i in range(parameters["N"]):
        for t in range(1, parameters["T"]):
            out[i,t] = dict[f"z_off{i}{t}"]
    return out


def show_p(dict, parameters):
    out = np.zeros((parameters["N"], parameters["T"]))
    for i in range(parameters["N"]):
        for t in range(1, parameters["T"]):
            value = 0
            for k in range(parameters["n"]):
                value += np.int16(dict[f"P{i}{t}_{k}"])*2**k*parameters["delta"][i]
            out[i,t] = value + dict[f"u{i}{t}"]*parameters["P_min"][i]

    out[:,0] = parameters["P_initial"][:]
    return out


def show_all_variables(best_solution, parameters):
    #best_solution = sampleset.first.sample

    print("z_on")
    print(show_z_on(best_solution, parameters))
    print("z_off")
    print(show_z_off(best_solution, parameters))
    print("u")
    print(show_u(best_solution, parameters))
    print("P")
    print(show_p(best_solution, parameters))
    #print(f"Total Energy (with penalties): {best_solution.energy}")
    print(f"Total Cost: {cost(best_solution, parameters)}")


def total_energy(sampleset):
    return sampleset.first.energy

def cost(best_solution, parameters):
    u = show_u(best_solution, parameters)
    z_on = show_z_on(best_solution, parameters)
    z_off = show_z_off(best_solution, parameters)
    P = show_p(best_solution, parameters)    
    out = 0
    for i in range(parameters["N"]):
        for t in range(1, parameters["T"]):
            out += u[i,t]*parameters["c"][i] + z_on[i,t]*parameters["C_startup"][i] + z_off[i,t]*parameters["C_shutdown"][i] + P[i,t]*parameters["b"][i]
    return out


###### Functions to check whether the constraints have been met or not ######

def is_demand_met(best_solution, parameters):
    P = show_p(best_solution, parameters)
    #out = [0, 0, 0]
    out = np.zeros((parameters["T"] - 1))
    for t in range(1, parameters["T"]):
        if sum([P[i,t] for i in range(parameters["N"])]) >= parameters["demand"][t]:
            out[t-1] = True
        else:
            out[t-1] = False
    return out

def is_ramp_met(best_solution, parameters):
    P = show_p(best_solution, parameters)

    #out = [[0, 0, 0],[0, 0, 0],[0, 0, 0]]
    out = np.zeros((parameters["N"], parameters["T"] - 1))
    for i in range(parameters["N"]):
        for t in range(1, parameters["T"]):
            if (P[i,t] - P[i,t-1]) <= parameters["R_up"][i] and (P[i,t] - P[i,t-1]) >= -1*parameters["R_down"][i]:
                out[i][t-1] = True
            else:
                out[i][t-1] = False
    return np.array(out)


def is_logic1_met(best_solution, parameters):
    u = show_u(best_solution, parameters)
    z_on = show_z_on(best_solution, parameters)
    z_off = show_z_off(best_solution, parameters)
    #out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    out = np.zeros((parameters["N"], parameters["T"] - 1))
    for i in range(parameters["N"]):
        for t in range(1, parameters["T"]):
            if z_on[i,t] - z_off[i,t] == u[i,t] - u[i,t-1]:
                out[i][t-1] = True
            else:
                out[i][t-1] = False
    return np.array(out)

def is_logic2_met(best_solution, parameters):
    z_on = show_z_on(best_solution, parameters)
    z_off = show_z_off(best_solution, parameters)
    #out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    out = np.zeros((parameters["N"], parameters["T"] - 1))
    for i in range(parameters["N"]):
        for t in range(1, parameters["T"]):
            if z_on[i,t]*z_off[i,t] == 0:
                out[i][t-1] = True
            else:
                out[i][t-1] = False
    return np.array(out)

def is_coupling_met(best_solution, parameters):
    u = show_u(best_solution, parameters)
    P = show_p(best_solution, parameters)
    #out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    out = np.zeros((parameters["N"], parameters["T"] - 1))
    for i in range(parameters["N"]):
        for t in range(1, parameters["T"]):
            if ((P[i,t] == 0) and (u[i,t] == 0)) or ((P[i,t] != 0) and (u[i,t] == 1)):
                out[i][t-1] = True
            else:
                out[i][t-1] = False
    return np.array(out)

def is_capacity_met(best_solution, parameters):
    u = show_u(best_solution, parameters)
    P = show_p(best_solution, parameters)
    #out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    out = np.zeros((parameters["N"], parameters["T"] - 1))
    for i in range(parameters["N"]):
        for t in range(1, parameters["T"]):
            if ((P[i,t] <= parameters["P_max"][i])):
                out[i][t-1] = True
            else:
                out[i][t-1] = False
    return np.array(out)


def is_solution_feasible(solution, parameters, constraints):
    if constraints["logic1"]:
        if np.sum(is_logic1_met(solution, parameters)) < parameters["N"] * (parameters["T"] - 1):
            return False
    if constraints["ramp"]:
        if np.sum(is_ramp_met(solution, parameters)) < parameters["N"] * (parameters["T"] - 1):
            return False
    if constraints["demand"]:
        if np.sum(is_demand_met(solution, parameters)) < parameters["T"] - 1:
            return False
    if constraints["logic2"]:
        if np.sum(is_logic2_met(solution, parameters)) < parameters["N"] * (parameters["T"] - 1):
            return False
    if constraints["coupling"]:
        if np.sum(is_coupling_met(solution, parameters)) < parameters["N"] * (parameters["T"] - 1):
            return False
    if constraints["capacity"]:
        if np.sum(is_capacity_met(solution, parameters)) < parameters["N"] * (parameters["T"] - 1):
            return False
    return True


def find_lowest_energy_solution(sampleset, parameters, constraints, print_it):
    lowest_energy = float('inf')
    best_solution = None

    for solution, energy in zip(sampleset.samples(), sampleset.data_vectors["energy"]):
        if is_solution_feasible(solution, parameters, constraints):
            cost_val = cost(solution, parameters)
            if cost_val < lowest_energy:
                lowest_energy = cost_val
                best_solution = solution

    if print_it and best_solution is not None:
        show_all_variables(best_solution, parameters)
    return lowest_energy


def sigmoid(k, x, center):
    return 1/(1 + np.exp(k*(x - center)))


def update_penalty(parameters, constraint, ratio, alpha):
    amplitude = parameters["slope_sigmoid"]
    center = parameters["center_sigmoid"]
    if constraint == "demand":
        parameters["demand_penalty"] = parameters["demand_penalty"]*(1 + alpha*sigmoid(amplitude, ratio, center))
    if constraint == "ramp":
        parameters["ramp_penalty"] = parameters["ramp_penalty"]*(1 + alpha*sigmoid(amplitude, ratio, center))
    if constraint == "coupling":
        parameters["coupling_penalty"] = parameters["coupling_penalty"]*(1 + alpha*sigmoid(amplitude, ratio, center))
    if constraint == "logic1":
        parameters["logic1_penalty"] = parameters["logic1_penalty"]*(1 + alpha*sigmoid(amplitude, ratio, center))
    if constraint == "logic2":
        parameters["logic2_penalty"] = parameters["logic2_penalty"]*(1 + alpha*sigmoid(amplitude, ratio, center))
    if constraint == "capacity":
        parameters["capacity_penalty"] = parameters["capacity_penalty"]*(1 + alpha*sigmoid(amplitude, ratio, center))
    return



def run_classical_algorithm(parameters, constraints, iterations, num_reads):

    parameters["demand_penalty"] = 1
    parameters["logic1_penalty"] = 1
    parameters["logic2_penalty"] = 1
    parameters["ramp_penalty"] = 1
    parameters["coupling_penalty"] = 1
    parameters["capacity_penalty"] = 1
    
    ratios_demand = []
    ratios_logic1 = []
    ratios_logic2 = []
    ratios_ramp = []
    ratios_coupling = []
    ratios_capacity = []

    multipliers_demand = []
    multipliers_logic1 = []
    multipliers_logic2 = []
    multipliers_ramp = []
    multipliers_coupling = []
    multipliers_capacity = []

    sa = neal.SimulatedAnnealingSampler()

    output_dict = {}

    ratios_total = []

    for k in range(iterations):
        bqm = build_BQM(parameters, constraints)
        sampleset = sa.sample(bqm,
                              num_reads = num_reads)
        
        points_demand = 0
        points_logic1 = 0
        points_logic2 = 0
        points_ramp = 0
        points_coupling = 0
        points_capacity = 0
        points_total = 0

        for solution in sampleset:
            if constraints["demand"] and np.sum(is_demand_met(solution, parameters)) == parameters["T"] - 1:
                points_demand += 1
            if constraints["ramp"] and np.sum(is_ramp_met(solution, parameters)) == (parameters["T"] - 1)*parameters["N"]:
                points_ramp += 1
            if constraints["coupling"] and np.sum(is_coupling_met(solution, parameters)) == (parameters["T"] - 1)*parameters["N"]:
                points_coupling += 1
            if constraints["logic1"] and np.sum(is_logic1_met(solution, parameters)) == (parameters["T"] - 1)*parameters["N"]:
                points_logic1 += 1
            if constraints["logic2"] and np.sum(is_logic2_met(solution, parameters)) == (parameters["T"] - 1)*parameters["N"]:
                points_logic2 += 1
            if constraints["capacity"] and np.sum(is_capacity_met(solution, parameters)) == (parameters["T"] - 1)*parameters["N"]:
                points_capacity += 1
            if is_solution_feasible(solution, parameters, constraints):
                points_total += 1

        ratios_demand.append(points_demand/num_reads)
        ratios_logic1.append(points_logic1/num_reads)
        ratios_logic2.append(points_logic2/num_reads)
        ratios_ramp.append(points_ramp/num_reads)
        ratios_coupling.append(points_coupling/num_reads)
        ratios_capacity.append(points_capacity/num_reads)
        ratios_total.append(points_total/num_reads)
        
        multipliers_demand.append(parameters["demand_penalty"])
        multipliers_logic1.append(parameters["logic1_penalty"])
        multipliers_logic2.append(parameters["logic2_penalty"])
        multipliers_coupling.append(parameters["coupling_penalty"])
        multipliers_ramp.append(parameters["ramp_penalty"])
        multipliers_capacity.append(parameters["capacity_penalty"])

        #print(f"Multipliers: Logic1: {round(parameters["logic1_penalty"], 2)}, Logic2: {round(parameters["logic2_penalty"], 2)}, Demand: {round(parameters["demand_penalty"],2)}, Ramp: {round(parameters["ramp_penalty"],2)}, Coupling: {round(parameters["coupling_penalty"],2)}, Capacity: {round(parameters["capacity_penalty"],2)}")
        
        if (points_demand/num_reads >= parameters["threshold_constraints"]
            and points_coupling/num_reads >= parameters["threshold_constraints"]
            and points_ramp/num_reads >= parameters["threshold_constraints"]
            and points_capacity/num_reads >= parameters["threshold_constraints"]
            and points_logic1/num_reads >= parameters["threshold_constraints"]
            and points_logic2/num_reads >= parameters["threshold_constraints"]
            ):
            break

        else:
            update_penalty(parameters, "logic1", points_logic1/num_reads, parameters["alpha_sigmoid"]["logic1"])
            update_penalty(parameters, "logic2", points_logic2/num_reads, parameters["alpha_sigmoid"]["logic2"])
            update_penalty(parameters, "demand", points_demand/num_reads, parameters["alpha_sigmoid"]["demand"])
            update_penalty(parameters, "ramp", points_ramp/num_reads, parameters["alpha_sigmoid"]["ramp"])
            update_penalty(parameters, "coupling", points_coupling/num_reads, parameters["alpha_sigmoid"]["coupling"])
            update_penalty(parameters, "capacity", points_capacity/num_reads, parameters["alpha_sigmoid"]["capacity"])

    output_dict["ratios_demand"] = ratios_demand
    output_dict["multipliers_demand"] = multipliers_demand
    output_dict["ratios_logic1"] = ratios_logic1
    output_dict["multipliers_logic1"] = multipliers_logic1
    output_dict["ratios_logic2"] = ratios_logic2
    output_dict["multipliers_logic2"] = multipliers_logic2
    output_dict["ratios_coupling"] = ratios_coupling
    output_dict["multipliers_coupling"] = multipliers_coupling
    output_dict["ratios_ramp"] = ratios_ramp
    output_dict["multipliers_ramp"] = multipliers_ramp
    output_dict["ratios_capacity"] = ratios_capacity
    output_dict["multipliers_capacity"] = multipliers_capacity

    output_dict["ratios_total"] = ratios_total

    return output_dict


def run_quantum_algorithm(parameters, constraints, iterations, num_reads, token, annealing_time, embedding):

    parameters["demand_penalty"] = 1
    parameters["logic1_penalty"] = 1
    parameters["logic2_penalty"] = 1
    parameters["ramp_penalty"] = 1
    parameters["coupling_penalty"] = 1
    parameters["capacity_penalty"] = 1
    
    ratios_demand = []
    ratios_logic1 = []
    ratios_logic2 = []
    ratios_ramp = []
    ratios_coupling = []
    ratios_capacity = []

    multipliers_demand = []
    multipliers_logic1 = []
    multipliers_logic2 = []
    multipliers_ramp = []
    multipliers_coupling = []
    multipliers_capacity = []

    #sampler = EmbeddingComposite(DWaveSampler(token = token))
    sampler = DWaveSampler(token = token)

    output_dict = {}

    ratios_total = []

    for k in range(iterations):
        bqm = build_BQM(parameters, constraints)
        composite = FixedEmbeddingComposite(sampler, embedding)
        #sampleset = sampler.sample(bqm, num_reads=num_reads, annealing_time = annealing_time, label = f"quantum_algorithm_iteration_{k}")
        sampleset = composite.sample(bqm, num_reads=num_reads, annealing_time = annealing_time, label = f"QA_it{k}")

        points_demand = 0
        points_logic1 = 0
        points_logic2 = 0
        points_ramp = 0
        points_coupling = 0
        points_capacity = 0
        points_total = 0

        for solution in sampleset:
            if constraints["demand"] and np.sum(is_demand_met(solution, parameters)) == parameters["T"] - 1:
                points_demand += 1
            if constraints["ramp"] and np.sum(is_ramp_met(solution, parameters)) == (parameters["T"] - 1)*parameters["N"]:
                points_ramp += 1
            if constraints["coupling"] and np.sum(is_coupling_met(solution, parameters)) == (parameters["T"] - 1)*parameters["N"]:
                points_coupling += 1
            if constraints["logic1"] and np.sum(is_logic1_met(solution, parameters)) == (parameters["T"] - 1)*parameters["N"]:
                points_logic1 += 1
            if constraints["logic2"] and np.sum(is_logic2_met(solution, parameters)) == (parameters["T"] - 1)*parameters["N"]:
                points_logic2 += 1
            if constraints["capacity"] and np.sum(is_capacity_met(solution, parameters)) == (parameters["T"] - 1)*parameters["N"]:
                points_capacity += 1
            if is_solution_feasible(solution, parameters, constraints):
                points_total += 1

        ratios_demand.append(points_demand/num_reads)
        ratios_logic1.append(points_logic1/num_reads)
        ratios_logic2.append(points_logic2/num_reads)
        ratios_ramp.append(points_ramp/num_reads)
        ratios_coupling.append(points_coupling/num_reads)
        ratios_capacity.append(points_capacity/num_reads)
        ratios_total.append(points_total/num_reads)
        
        multipliers_demand.append(parameters["demand_penalty"])
        multipliers_logic1.append(parameters["logic1_penalty"])
        multipliers_logic2.append(parameters["logic2_penalty"])
        multipliers_coupling.append(parameters["coupling_penalty"])
        multipliers_ramp.append(parameters["ramp_penalty"])
        multipliers_capacity.append(parameters["capacity_penalty"])

        #print(f"Multipliers: Logic1: {round(parameters["logic1_penalty"], 2)}, Logic2: {round(parameters["logic2_penalty"], 2)}, Demand: {round(parameters["demand_penalty"],2)}, Ramp: {round(parameters["ramp_penalty"],2)}, Coupling: {round(parameters["coupling_penalty"],2)}, Capacity: {round(parameters["capacity_penalty"],2)}")
        
        if (points_demand/num_reads >= parameters["threshold_constraints"]
            and points_coupling/num_reads >= parameters["threshold_constraints"]
            and points_ramp/num_reads >= parameters["threshold_constraints"]
            and points_capacity/num_reads >= parameters["threshold_constraints"]
            and points_logic1/num_reads >= parameters["threshold_constraints"]
            and points_logic2/num_reads >= parameters["threshold_constraints"]
            ):
            break

        else:
            update_penalty(parameters, "logic1", points_logic1/num_reads, parameters["alpha_sigmoid"]["logic1"])
            update_penalty(parameters, "logic2", points_logic2/num_reads, parameters["alpha_sigmoid"]["logic2"])
            update_penalty(parameters, "demand", points_demand/num_reads, parameters["alpha_sigmoid"]["demand"])
            update_penalty(parameters, "ramp", points_ramp/num_reads, parameters["alpha_sigmoid"]["ramp"])
            update_penalty(parameters, "coupling", points_coupling/num_reads, parameters["alpha_sigmoid"]["coupling"])
            update_penalty(parameters, "capacity", points_capacity/num_reads, parameters["alpha_sigmoid"]["capacity"])

    output_dict["ratios_demand"] = ratios_demand
    output_dict["multipliers_demand"] = multipliers_demand
    output_dict["ratios_logic1"] = ratios_logic1
    output_dict["multipliers_logic1"] = multipliers_logic1
    output_dict["ratios_logic2"] = ratios_logic2
    output_dict["multipliers_logic2"] = multipliers_logic2
    output_dict["ratios_coupling"] = ratios_coupling
    output_dict["multipliers_coupling"] = multipliers_coupling
    output_dict["ratios_ramp"] = ratios_ramp
    output_dict["multipliers_ramp"] = multipliers_ramp
    output_dict["ratios_capacity"] = ratios_capacity
    output_dict["multipliers_capacity"] = multipliers_capacity

    output_dict["ratios_total"] = ratios_total

    return output_dict


'''
def average_last_half_terms(vector):
    n = len(vector)
    if n == 0:
        return None
    second_half = vector[n // 2:]
    return sum(second_half) / len(second_half)

def calculate_y(parameters,parameter_to_check, values_vector):
    out = []
    for value in values_vector:
        parameters[parameter_to_check] = value
        res_dict = run_algorithm(parameters, iterations=40, num_reads=100)
        joint_prob_vector = res_dict["ratios_total"]
        out.append(average_last_half_terms(joint_prob_vector))
    return out
    '''

def plot_after_algorithm(dict, y_axis, constraints, parameters):
    # dict: dictionary containing the results of the algorithm. It contains values of ratios of feasibility of each constraint, values of each multiplier, at each iteration
    # y_axis: string either "ratios" or "multipliers", to plot the values of R_i or the multipliers
    '''
    if y_axis != "multipliers":
        plt.figure(figsize=(6.08, 4.75))
        plt.tight_layout()
        plt.plot(np.arange(len(dict[f"{y_axis}_total"])),dict[f"{y_axis}_total"], marker = "o",label = r"$R_{joint}$", color = "blue")
        plt.xlabel("Iteration", fontsize = 18)
        plt.ylabel(r"$R_J$", fontsize = 18)
        plt.tick_params(axis='both', labelsize=18)
        plt.savefig("R_joint.pdf", bbox_inches='tight')
        plt.show()
    
    if constraints["demand"]:
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(dict[f"{y_axis}_demand"])),dict[f"{y_axis}_demand"], marker = "o", label = r"$R_{d}$", color = "blue")
        plt.axhline(y = parameters["threshold_constraints"], ls = "--", color = "red", label = "Threshold")
        plt.legend(fontsize = 16)
        plt.xlabel("Iteration", fontsize = 16)
        plt.ylabel(r"$R_d$", fontsize = 16)
        plt.tick_params(axis='both', labelsize=16)
        plt.title("Demand")
        plt.tight_layout()
        #plt.savefig("stabilization_multipliers.pdf")
        plt.show()
    if constraints["logic1"]:
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(dict[f"{y_axis}_logic1"])),dict[f"{y_axis}_logic1"], marker = "o", label = r"$R_{l1}$",  color = "blue")
        plt.axhline(y = parameters["threshold_constraints"], ls = "--", color = "red", label = "Threshold")
        plt.legend(fontsize = 16)
        plt.xlabel("Iteration", fontsize = 16)
        plt.ylabel(r"$R_{l1}$", fontsize = 16)
        #plt.ylim([-0.1,1.1])
        plt.title("Logic1")
        plt.tick_params(axis='both', labelsize=16)
        plt.tight_layout()
        plt.show()
    if constraints["logic2"]:
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(dict[f"{y_axis}_logic2"])), dict[f"{y_axis}_logic2"], marker = "o", label = r"$R_{l2}$",  color = "blue")
        plt.axhline(y = parameters["threshold_constraints"], ls = "--", color = "red", label = "Threshold")
        plt.legend(fontsize = 16)
        plt.xlabel("Iteration", fontsize = 16)
        plt.ylabel(r"$R_{l2}$", fontsize = 16)
        #plt.ylim([-0.1,1.1])
        plt.title("Logic2")
        plt.tick_params(axis='both', labelsize=16)
        plt.tight_layout()
        plt.show()
    if constraints["ramp"]:
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(dict[f"{y_axis}_ramp"])), dict[f"{y_axis}_ramp"], marker = "o", label = r"$R_{ramp}$",  color = "blue")
        plt.axhline(y = parameters["threshold_constraints"], ls = "--", color = "red", label = "Threshold")
        plt.legend(fontsize = 16)
        plt.xlabel("Iteration", fontsize = 16)
        plt.ylabel(r"$R_{r}$", fontsize = 16)
        #plt.ylim([-0.1,1.1])
        plt.title("Ramping Limits")
        plt.tick_params(axis='both', labelsize=16)
        plt.tight_layout()
        plt.show()
    if constraints["coupling"]:
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(dict[f"{y_axis}_coupling"])), dict[f"{y_axis}_coupling"], marker = "o", label = r"$R_{coupling}$",  color = "blue")
        plt.axhline(y = parameters["threshold_constraints"], ls = "--", color = "red", label = "Threshold")
        plt.legend(fontsize = 16)
        plt.xlabel("Iteration", fontsize = 16)
        plt.ylabel(r"$R_{c}$", fontsize = 16)
        #plt.ylim([-0.1,1.1])
        plt.title("Coupling")
        plt.tick_params(axis='both', labelsize=16)
        plt.tight_layout()
        plt.show()
    if constraints["capacity"]:
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(dict[f"{y_axis}_capacity"])), dict[f"{y_axis}_capacity"], marker = "o", label = r"$R_{capacity}$",  color = "blue")
        plt.axhline(y = parameters["threshold_constraints"], ls = "--", color = "red", label = "Threshold")
        plt.legend(fontsize = 16)
        plt.xlabel("Iteration", fontsize = 16)
        plt.ylabel(r"$R_{capacity}$", fontsize = 16)
        #plt.ylim([-0.1,1.1])
        plt.title("Capacity")
        plt.tick_params(axis='both', labelsize=16)
        plt.tight_layout()
        plt.show()'''

    if constraints["logic1"] and constraints["logic2"] and constraints["coupling"]:
        #plt.figure(figsize=(10, 6))
        plt.tight_layout()
        plt.plot(np.arange(len(dict[f"{y_axis}_coupling"])), dict[f"{y_axis}_coupling"], marker = "o", label = r"$R_{coupling}$",  color = "red")
        plt.plot(np.arange(len(dict[f"{y_axis}_logic2"])), dict[f"{y_axis}_logic2"], marker = "o", label = r"$R_{logic2}$",  color = "blue")
        plt.plot(np.arange(len(dict[f"{y_axis}_logic1"])),dict[f"{y_axis}_logic1"], marker = "o", label = r"$R_{logic1}$",  color = "green")
        plt.axhline(y = parameters["threshold_constraints"], ls = "--", color = "gray", label = "Threshold")
        plt.legend(fontsize = 17, ncols = 2, columnspacing=0.5)
        plt.xlabel("Iteration", fontsize = 18)
        plt.ylabel(r"$R_{i}$", fontsize = 18)
        plt.ylim([-0.05,1.05])
        #plt.title("Binary constraints")
        plt.tick_params(axis='both', labelsize=18)

        #plt.savefig("R_binaries.pdf", bbox_inches='tight')
        plt.show()
    
    if constraints["demand"] and constraints["ramp"] and constraints["capacity"]:
        #plt.figure(figsize=(10, 6))
        plt.tight_layout()
        plt.plot(np.arange(len(dict[f"{y_axis}_capacity"])), dict[f"{y_axis}_capacity"], marker = "o", label = r"$R_{capacity}$",  color = "red")
        plt.plot(np.arange(len(dict[f"{y_axis}_ramp"])), dict[f"{y_axis}_ramp"], marker = "o", label = r"$R_{ramp}$",  color = "blue")
        plt.plot(np.arange(len(dict[f"{y_axis}_demand"])),dict[f"{y_axis}_demand"], marker = "o", label = r"$R_{demand}$", color = "green")
        plt.axhline(y = parameters["threshold_constraints"], ls = "--", color = "gray", label = "Threshold")
        plt.legend(fontsize = 17, ncols = 2, columnspacing=0.5)
        plt.xlabel("Iteration", fontsize = 18)
        plt.ylabel(r"$R_{i}$", fontsize = 18)
        plt.ylim([-0.05,1.05])
        #plt.title("Continous constraints")
        plt.tick_params(axis='both', labelsize=18)
        #plt.savefig("R_continuous.pdf", bbox_inches='tight')
        plt.show()

def plot_feasibility_histogram(sampleset, is_feasible_fn, energy_fn, parameters, constraints, bin_width):
    """
    Plot a histogram of energies from a SampleSet, colored by feasibility.

    Parameters:
    - sampleset: dimod.SampleSet
    - is_feasible_fn: function(sample_dict) -> bool
    - energy_fn: function(sample_dict) -> float
    - bin_width: float, width of histogram bins
    """
    # Separate feasible and infeasible energies
    feasible_energies = []
    infeasible_energies = []

    for solution in sampleset:
        cost = energy_fn(solution, parameters)
        if is_feasible_fn(solution, parameters, constraints):
            feasible_energies.append(cost)
        else:
            infeasible_energies.append(cost)

    # Define histogram bin range
    all_energies = feasible_energies + infeasible_energies
    if not all_energies:
        print("No samples to plot.")
        return

    min_energy = min(all_energies)
    max_energy = max(all_energies)
    bins = np.arange(min_energy, max_energy + bin_width, bin_width)

    # Plot histograms
    plt.hist(infeasible_energies, bins=bins, color='blue', alpha=1.0, label='Infeasible solutions')
    plt.hist(feasible_energies, bins=bins, color='orange', alpha=1.0, label='Feasible solutions')
    plt.axvline(x = 191.8, ls = "--", color = "red", label = "Optimal solution")
    plt.xlabel('Expected Cost [$]', fontsize = 16)
    plt.ylabel('Frequency', fontsize = 16)
    plt.yscale("log")
    #plt.xlim([190, 240])
    #plt.ylim([0, 50])
    plt.legend()
    
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    #plt.savefig("Feasibility_histogram_deterministic_BQM.pdf")
    plt.show()

