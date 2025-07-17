import dimod
from dwave.system import LeapHybridCQMSampler
import numpy as np
import json


# __________________ PARAMETERS OF UCP __________________
# The following vectors of parameters have as index "i", corresponding to each of the generators
P_min = [50, 80, 40] # Minimum power a generator must produce if its status is ON
P_max = [350, 200, 140] # Maximum power a generator can prpdice if its status is ON

R_down = [300,150,100] # Maximum power decrease a generator can have between time period t and t+1
R_shutdown = [300,150,100] # Maximum power a generator can produce at time t when it's switched OFF at time t+1
R_up = [200,100,100] # Maximum power ioncrease a generator can have between time period t and t+1
R_startup = [200,100,100] # Maximum power a generator can produce at time t when it's switched ON at that time period

C_startup = [20, 18, 5] # Cost of switching ON
C_shutdown = [0.5, 0.3, 1.0] # Cost of switching OFF
b = [0.1, 0.125, 0.150] # Variable linear cost
c = [5, 7, 6] # Fixed cost

u_initial= [0,0,1] # Status of each generator prior to time horizon
p_initial = [0,0,100] # Production of each generator prior to time horizon

# The following vectors of parameters have as index "t", corresponding to each of the time periods
# Note that demand at each time period corresponds to a random variable with multivariate normal distribution
demand = [225, 630, 400] # Average demand value at each time period (\mu in normal distribution)
sigma = np.array([25,40,28]) # Standard deviation value for each variable in the random distribution of demand


# __________________ Post processing functions __________________


def show_p(dict):
    # Returns a matrix with indices (i,t) corresponding to generator and time period respectively.
    # The matrix contains continuous power variable P corresponding to each generator at each time period.
    # Inputs:
        # dict: dictionary containing the values of all the variables in the optimization problem.
    result = np.zeros((N,T))
    for i in range(N):
        for t in range(T):
            result[i,t] = round(dict["p%s%s"%(i,t)],2) 
    return result
def show_u(dict):
    # Returns a matrix with indices (i,t) corresponding to generator and time period respectively.
    # The matrix contains the binary status variable u corresponding to each generator at each time period.
    # Inputs:
        # dict: dictionary containing the values of all the variables in the optimization problem.
    result = np.zeros((N,T))
    for i in range(N):
        for t in range(T):
            result[i,t] = dict["u%s%s"%(i,t)] 
    return result
def show_z_on(dict):
    # Returns a matrix with indices (i,t) corresponding to generator and time period respectively.
    # The matrix contains the binary z_ON variable corresponding to each generator at each time period.
    # Inputs:
        # dict: dictionary containing the values of all the variables in the optimization problem.
    result = np.zeros((N,T))
    for i in range(N):
        for t in range(T):
            result[i,t] = dict["z_on%s%s"%(i,t)] 
    return result
def show_z_off(dict):
    # Returns a matrix with indices (i,t) corresponding to generator and time period respectively.
    # The matrix contains the binary z_OFF variable corresponding to each generator at each time period.
    # Inputs:
        # dict: dictionary containing the values of all the variables in the optimization problem.
    result = np.zeros((N,T))
    for i in range(N):
        for t in range(T):
            result[i,t] = dict["z_off%s%s"%(i,t)] 
    return result

def parse_best(sampleset):
        # Returns the lowest energy feasible solution in the sampleset.
        # Inputs:
            # sampleset: sampleset given by the optimization problem after solving.
        print("Parsing best ...")
        best = sampleset.filter(lambda row: row.is_feasible).first
        print("Found best")
        return best

def save_data(sampler, runs, max_time, m, label):
    # Runs the model (Constrained Quadratic Model) and returns a serialized sampleset, ready to be saved as a json file.
    # Inputs:
        # sampler: D-Wave's CQM sampler, previously introduced with an appropriate token
        # runs: amount of times you want to solve the model. The more times the model is solved, 
            # the more chances to find the oprimal solution
        # max_time: maximum amount of time you want to spend on solving the model
        # m: Constrained Quadratic Model
        # label: label for the runs. It will be reflected in D-Wave's Leap account associated to the token used in the sampler.

    for i in range(runs):
        print("Run number:",i+1)
        if i == 0:
            first_sampleset = sampler.sample_cqm(m,label=label, time_limit = max_time)
        elif i == 1:
            sampleset = sampler.sample_cqm(m,label=label, time_limit = max_time)
            larger_sampleset = dimod.concatenate((first_sampleset,sampleset))
        else:
            sampleset = sampler.sample_cqm(m,label=label, time_limit = max_time)
            larger_sampleset = dimod.concatenate((larger_sampleset,sampleset))     
    sampleset_series = larger_sampleset.to_serializable()
    best_one = parse_best(larger_sampleset)
    print("Best solution:", best_one.energy)
    return sampleset_series

# __________________ Model building functions __________________

def transform_standard_samples(Z, mu, Cov):
    # Returns a Cholesky transformation of sampleset
    # Inputs:
        # Z: Initial sampleset with average value \mu = 0 and standard deviation \sigma = 1 for each of the variables in the distribution.
        # mu: New average values of each of the variables in the samples of Z.
        # Cov: Covariance matrix to which the samples will be transformed.
    
    L = np.linalg.cholesky(Cov) # Compute Cholesky decomposition
    X = (L @ Z.T).T + mu # Apply new covariance and mean
    return X

def build_covariance_matrix(sigma, rho):
    # Returns a covariance matrix
    # Inputs:
        # sigma: standard deviation
        # rho: vector containing the correlations between variables: rho01, rho02, rho12

    rho01 = rho[0]
    rho02 = rho[1]
    rho12 = rho[2]
    matrix =  np.array([[sigma[0]**2, rho01*sigma[0]*sigma[1], rho02*sigma[0]*sigma[2]],
                           [rho01*sigma[1]*sigma[0], sigma[1]**2, rho12*sigma[1]*sigma[2]],
                           [rho02*sigma[2]*sigma[0], rho12*sigma[2]*sigma[1], sigma[2]**2]])
    return matrix

def sample(mu, sigma, rho, Z):
    # returns the sampleset containing the demand at every tie period
    # Inputs:
        # mu: average values of each variable in multivariate normal distribution
        # sigma: standard deviation of each variable in multivariate normal distribution
        # rho: Correlation btween variables
        # Z: Initial sampleset with average value \mu = 0 and standard deviation \sigma = 1 for each of the variables in the distribution.
    Cov = build_covariance_matrix(sigma, rho)
    points = transform_standard_samples(Z, mu, Cov)
    return points

def save_model_in_json(cqm, file_name):
    # It does not return anything. It saves the model in a json file, so that it's readible just by opening it. 
    cqm_data = {
        "Objective": str(cqm.objective),
        "Constraints": {
            label: {
                "lhs": str(constraint.lhs),
                "sense": constraint.sense.name,  # Convert sense to string (e.g., '<=')
                "rhs": constraint.rhs
            }
            for label, constraint in cqm.constraints.items()
        },
        "Variables": {
            var: {
                "type": cqm.vartype(var).name,
                "lower_bound": cqm.lower_bound(var) if cqm.vartype(var).name != "BINARY" else 0,
                "upper_bound": cqm.upper_bound(var) if cqm.vartype(var).name != "BINARY" else 1,
            }
            for var in cqm.variables
        }
        }

    # Save JSON to file
    with open(file_name, "w") as f:
        json.dump(cqm_data, f, indent=4)

    print("CQM model saved to model.json")



# __________________ Building and solving model __________________

runs = 5 # Total number of times that a specific model is solved
max_time = 20 # Max time the solver has to solve the problem (in seconds)

scenarios_array = np.array([1000]) # Total number of scenarios
rho_vector = np.array([[0, 0, 0], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]]) # Different values of correlations
epsilon_vector = np.array([0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]) # Different values of reliability level (in epsilon)

for N in scenarios_array:
    Z = np.random.multivariate_normal(np.zeros(len(demand)), np.eye(len(demand)), size=N)

    for k in range(len(rho_vector)):
        rho = rho_vector[k]

        for l in range(len(epsilon_vector)):
            epsilon = epsilon_vector[l]
            print(f"Reliability level: p = {1-epsilon}")
            print(f"Rho vector: {rho}")
            print(f"{N} scenarios:")
            print("Building model ...")
            cqm = dimod.ConstrainedQuadraticModel()
            G = 3 # Number of generators
            T = 3 # Number of time periods
            mu = demand
            
            Z_transformed = sample(mu, sigma, rho, Z)

            u = {(i, t): dimod.Binary(f"u{i}{t}") for i in range(G) for t in range(T)}
            z_on = {(i, t): dimod.Binary(f"z_on{i}{t}") for i in range(G) for t in range(T)}
            z_off = {(i, t): dimod.Binary(f"z_off{i}{t}") for i in range(G) for t in range(T)}
            y = {(i): dimod.Binary(f"y{i}") for i in range(N)}
            p = {(i, t): dimod.Real(f"p{i}{t}", lower_bound = 0, upper_bound = None) for i in range(G) for t in range(T)}

            for i in range(N):
                    for t in range(T):
                            cqm.add_constraint(p[0,t] + p[1,t] + p[2,t] - Z_transformed[i,t]*y[i] >= 0)#, label = f"Demand_{i}_{t}")

            cqm.add_constraint(sum([y[i] for i in range(N)]) >= (1-epsilon)*N)
            for i in range(G):
                for t in range(T):
                    cqm.add_constraint(z_on[i,t] + z_off[i,t] <= 1)
                    cqm.add_constraint(p[i,t] - P_max[i]*u[i,t] <= 0)
                    cqm.add_constraint(p[i,t] - P_min[i]*u[i,t] >= 0)
                    if t != 0:
                        cqm.add_constraint(u[i,t] - u[i,t-1] - z_on[i,t] + z_off[i,t] == 0)
                        cqm.add_constraint(p[i,t] - p[i,t-1] - R_up[i]*u[i,t-1] - R_startup[i]*z_on[i,t] <= 0)
                        cqm.add_constraint(p[i,t-1] - p[i,t] - R_down[i]*u[i,t] - R_shutdown[i]*z_off[i,t] <= 0)
                    else:
                        cqm.add_constraint(u[i,t] - z_on[i,t] + z_off[i,t] == u_initial[i])#, label = f"Logic_{i}_{t}")                    
                        cqm.add_constraint(p[i,t] - R_startup[i]*z_on[i,t] <= p_initial[i] + R_up[i]*u_initial[i])#, label = f"Ramp_up_{i}_{t}")
                        cqm.add_constraint(-p[i,t] - R_down[i]*u[i,t] - R_shutdown[i]*z_off[i,t] <= -p_initial[i])#, label = f"Ramp_down_{i}_{t}")

            cqm.set_objective(sum([C_shutdown[i]*z_off[i,t] + C_startup[i]*z_on[i,t] + b[i]*p[i,t] + c[i]*u[i,t] for i in range(G) for t in range(T)]))
            
            save_model_in_json(cqm, f"CQM_{N}scenarios_{k}rho_{l}epsilon.json")
            



            sampler = LeapHybridCQMSampler(token = "introduce_your_token")
            sampler.properties["minimum_time_limit_s"] = 20

            serialized_sampleset = save_data(sampler = sampler, runs = runs, max_time = max_time, m = cqm, label = f"{N}scenarios_{k}rho_{l}epsilon")

            with open(f"results_{N}scenarios_{k}rho_{l}epsilon.json", "w") as f:
                json.dump(serialized_sampleset, f, indent=4)

            