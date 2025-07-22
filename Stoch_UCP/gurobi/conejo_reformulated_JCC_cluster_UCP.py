import numpy as np
import pyomo.environ as pyo
from pyomo.environ import value
from scipy.stats import multivariate_normal



seed = 42
np.random.seed(seed)


# __________________ PARAMETERS OF UCP __________________
# The following vectors of parameters have as index "i", corresponding to each of the generators
P_min_numpy = np.array([50,80,40]) # Minimum power a generator must produce if its status is ON
P_max_numpy = np.array([350,200,140]) # Maximum power a generator can prpdice if its status is ON
R_down_numpy = np.array([300,150,100]) # Maximum power decrease a generator can have between time period t and t+1
R_shutdown_numpy = np.array([300,150,100]) # Maximum power a generator can produce at time t when it's switched OFF at time t+1
R_up_numpy = np.array([200,100,100]) # Maximum power ioncrease a generator can have between time period t and t+1
R_startup_numpy = np.array([200,100,100]) # Maximum power a generator can produce at time t when it's switched ON at that time period
C_startup_numpy = np.array([20, 18, 5]) # Cost of switching ON
C_shutdown_numpy = np.array([0.5, 0.3, 1.0]) # Cost of switching OFF
b_numpy = np.array([0.1, 0.125, 0.150]) # Variable linear cost
c_numpy = np.array([5,7,6]) # Fixed cost

u_initial_numpy = np.array([0,0,1]) # Status of each generator prior to time horizon
P_initial_numpy = np.array([0,0,100]) # Production of each generator prior to time horizon


# The following vectors of parameters have as index "t", corresponding to each of the time periods
# Note that demand at each time period corresponds to a random variable with multivariate normal distribution
demand_numpy = np.array([225,630,400]) # Average demand value at each time period (\mu in normal distribution)
sigma_numpy = np.array([25,40,28]) # Standard deviation value for each variable in the random distribution of demand


def create_dict(array):
    # Returns a dictionary with all the data correspondig to the parameters.
    # Dictionaries are necessary to create parameter objects for Pyomo.
    # Inputs:
        # array: Vector containing certain parameters. The vector must have a single axis.
    return {(i): array[i] for i in range(len(array))}

def create_dict_sampleset(array):
    # Returns a dictionary with all the data correspondig to the parameters.
    # Specifically, the dataset from the multivariate normal distribution.
    # Inputs:
        # array: Vector containing certain parameters. The vector must have two axes. 
    return {(i,j): array[i,j] for i in range(len(array)) for j in range(len(array[0]))}

P_min = create_dict(P_min_numpy)
P_max = create_dict(P_max_numpy)
R_down = create_dict(R_down_numpy)
R_shutdown = create_dict(R_shutdown_numpy)
R_up = create_dict(R_up_numpy)
R_startup = create_dict(R_startup_numpy)
C_startup = create_dict(C_startup_numpy)
C_shutdown = create_dict(C_shutdown_numpy)
b = create_dict(b_numpy)
c = create_dict(c_numpy)
demand = create_dict(demand_numpy)
sigma = create_dict(sigma_numpy)
u_initial = create_dict(u_initial_numpy)
P_initial = create_dict(P_initial_numpy)


#DEFINITION OF ALL CONSTRAINTS

def logic_constraint1(model,i,t):
    # Returns a boolean stating if the logic1 constraint is met (True), or not (False)
    # Inputs:
        # model: the optimization model containing the constraint.
        # i: index corresponding to the generator.
        # t: index corresponding to the time period.
    if t != 0:
        return model.u[i,t] - model.u[i,t-1] == model.z_on[i,t] - model.z_off[i,t]
    else:
        return model.u[i,t] - model.u_initial[i] == model.z_on[i,t] - model.z_off[i,t]
    
def logic_constraint2(model,i,t):
    # Returns a boolean stating if the logic2 constraint is met (True), or not (False)
    # Inputs:
        # model: the optimization model containing the constraint.
        # i: index corresponding to the generator.
        # t: index corresponding to the time period.
    return model.z_off[i,t] + model.z_on[i,t] <= 1

#def demand_constraint(model,t):
#    total_power = sum([model.P[i,t] for i in model.G])
#    return total_power >= model.demand[t]

def lower_limit_constraint(model,i,t):
    # Returns a boolean stating if the lower limit constraint is met (True), or not (False)
    # Inputs:
        # model: the optimization model containing the constraint.
        # i: index corresponding to the generator.
        # t: index corresponding to the time period.
    return model.P_min[i]*model.u[i,t] <= model.P[i,t]

def upper_limit_constraint(model,i,t):
    # Returns a boolean stating if the lower limit constraint is met (True), or not (False)
    # Inputs:
        # model: the optimization model containing the constraint.
        # i: index corresponding to the generator.
        # t: index corresponding to the time period.
    return model.P_max[i]*model.u[i,t] >= model.P[i,t]

def ramp_up_limit_constraint(model,i,t):
    # Returns a boolean stating if the ramp up limit constraint is met (True), or not (False)
    # Inputs:
        # model: the optimization model containing the constraint.
        # i: index corresponding to the generator.
        # t: index corresponding to the time period.
    if t != 0:
        return model.P[i,t] - model.P[i,t-1] <= model.R_up[i]*model.u[i,t-1] + model.R_startup[i]*model.z_on[i,t]
    else:
        return model.P[i,t] - model.P_initial[i] <= model.R_up[i]*model.u_initial[i] + model.R_startup[i]*model.z_on[i,t]
    
def ramp_down_limit_constraint(model,i,t):
    # Returns a boolean stating if the ramp down limit constraint is met (True), or not (False)
    # Inputs:
        # model: the optimization model containing the constraint.
        # i: index corresponding to the generator.
        # t: index corresponding to the time period.
    if t != 0:
        return model.P[i,t-1] - model.P[i,t] <= model.R_down[i]*model.u[i,t] + model.R_shutdown[i]*model.z_off[i,t]
    else:
        return model.P_initial[i] - model.P[i,t] <= model.R_down[i]*model.u[i,t] + model.R_shutdown[i]*model.z_off[i,t]

def demand_constraint_from_sample(model, j, t):
    # Returns a boolean stating if the demand constraint is met (True), or not (False)
    # Inputs:
        # model: the optimization model containing the constraint.
        # j: index correspondig to the scenario in the sampleset
        # t: index corresponding to the time period.
    total_power = sum([model.P[i,t] for i in model.G])
    return total_power >= model.sampleset[j,t]*model.y[j]

def amount_of_violations_from_sample(model):
    # Returns a boolean stating if the constraint accounting for the reliability level is met (True), or not (False)
    # Inputs:
        # model: the optimization model containing the constraint.
    scenarios = value(model.scenarios)
    epsilon = value(model.epsilon)
    return sum([model.y[i] for i in range(scenarios)]) >= (1-epsilon)*scenarios

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

#DEFINITION OF OBJECTIVE FUNCTION
def cost(model):
    # Returns the total cost building the objective function
    # Inputs:
        # model: the optimization model containing the constraint.
    startup_cost = sum(model.z_on[i,j]*model.C_startup[i] for i in model.G for j in model.T)
    shutdown_cost = sum(model.z_off[i,j]*model.C_shutdown[i] for i in model.G for j in model.T)
    b_cost = sum(model.b[i]*model.P[i,j] for i in model.G for j in model.T)
    c_cost = sum(model.c[i]*model.u[i,j] for i in model.G for j in model.T)
    total = startup_cost + shutdown_cost + b_cost + c_cost
    return total

# __________________ Building and solving model __________________


# Comment one section or the other, depending on the plot you want to obtain:

#______________________ Parameters for cost_vs_scenarios plot ______________________
epsilon_vector = [0.1] # Different values of reliability level (in epsilon)
rho_matrix = np.array([[0.3, 0.4, 0.5]]) # Different values of correlations
max_scenarios = 20000 # Total number of scenarios you want to have
number_of_points = 25 # How many values of number of scenarios. Leave as 1 for just a single case.
#_____________________________________________________________________________________

#______________________ Parameters for cost_vs_reliability plot ______________________
#epsilon_vector = [0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] # Different values of reliability level (in epsilon)
#rho_matrix = np.array([[0, 0, 0], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]]) # Different values of correlations
#max_scenarios = 1000 # Total number of scenarios you want to have
#number_of_points = 1 # How many values of number of scenarios. Leave as 1 for just a single case.
#_____________________________________________________________________________________



costs = np.zeros((len(rho_matrix), len(epsilon_vector), number_of_points)) #tensor where data will be stored
gaps = np.zeros((len(rho_matrix), len(epsilon_vector), number_of_points))

Z = np.random.multivariate_normal(np.zeros(len(demand)), np.eye(len(demand)), size=max_scenarios)

for k in range(number_of_points):
    if k == 0:
        Z_reduced = Z
    else:
        Z_reduced = Z[:-(k)*int(max_scenarios/number_of_points)]
    print("length is :", len(Z_reduced))

    scenario = len(Z_reduced)

    for i in range(len(rho_matrix)):
        rho01, rho02, rho12 = rho_matrix[i]
        rho = {(0): rho01, (1): rho02, (2): rho12}
        print("rho", rho)

        for j in range(len(epsilon_vector)):

            model = pyo.ConcreteModel()
            epsilon_value = epsilon_vector[j]
            print(f"Solving for p = {1 - epsilon_value}")
            
            print(f"Building model with {scenario} scenarios ...")
            
            T = 3 # number of time periods
            rho_values = (T**2 - T)/2
            G = 3 # number of generators
            N = scenario # number of secnarios
            epsilon = epsilon_value

            model.T = pyo.RangeSet(0, T-1)
            model.G = pyo.RangeSet(0, G-1)
            model.R = pyo.RangeSet(0, rho_values-1)
            model.N = pyo.RangeSet(0,N-1)


            model.u = pyo.Var(model.G, model.T, within=pyo.Binary, initialize = 0) # Generator status variable
            model.z_off = pyo.Var(model.G, model.T, within=pyo.Binary, initialize = 0) # Switching off variable
            model.z_on = pyo.Var(model.G, model.T, within=pyo.Binary, initialize = 0) # Switching on variable
            model.P = pyo.Var(model.G, model.T, within = pyo.NonNegativeReals, initialize = 0) # Power variable

            model.y = pyo.Var(model.N, within = pyo.Binary, initialize = 0) # 

            # Definition of parameters inside the model
            model.P_min = pyo.Param(model.G, initialize = P_min) #not used since out pmin is 0
            model.P_max = pyo.Param(model.G, initialize = P_max)
            model.R_down = pyo.Param(model.G, initialize = R_down)
            model.R_shutdown = pyo.Param(model.G, initialize = R_shutdown)
            model.R_up = pyo.Param(model.G, initialize = R_up)
            model.R_startup = pyo.Param(model.G, initialize = R_startup)
            model.C_startup = pyo.Param(model.G, initialize = C_startup)
            model.C_shutdown = pyo.Param(model.G, initialize = C_shutdown)
            model.b = pyo.Param(model.G, initialize = b)
            model.c = pyo.Param(model.G, initialize = c)
            model.demand = pyo.Param(model.T, initialize = demand)
            model.sigma = pyo.Param(model.T, initialize = sigma)
            model.u_initial = pyo.Param(model.G, initialize = u_initial)
            model.P_initial = pyo.Param(model.G, initialize = P_initial)

            model.scenarios = pyo.Param(initialize = N)
            model.epsilon = pyo.Param(initialize = epsilon)
            model.rho = pyo.Param(model.R, initialize = rho)

            sigma = np.array([value(model.sigma[i]) for i in model.sigma])
            mu = np.array([value(model.demand[i]) for i in model.demand])

            sampleset_numpy = sample(mu, sigma, rho, Z_reduced)
            sampleset = create_dict_sampleset(sampleset_numpy)
            model.sampleset = pyo.Param(model.N, model.T, initialize = sampleset)



            model.logic_constraint1 = pyo.Constraint(model.G, model.T, expr = logic_constraint1)
            model.logic_constraint2 = pyo.Constraint(model.G, model.T, expr = logic_constraint2)
            #model.demand_constraint = pyo.Constraint(model.T, expr = demand_constraint)
            model.lower_limit_constraint = pyo.Constraint(model.G, model.T, expr = lower_limit_constraint)
            model.upper_limit_constraint = pyo.Constraint(model.G, model.T, expr = upper_limit_constraint)
            model.ramp_down_limit_constraint = pyo.Constraint(model.G, model.T, expr = ramp_down_limit_constraint)
            model.ramp_up_limit_constraint = pyo.Constraint(model.G, model.T, expr = ramp_up_limit_constraint)

            model.demand_constraint = pyo.Constraint(model.N, model.T, expr = demand_constraint_from_sample)
            model.amount_of_violations = pyo.Constraint(expr = amount_of_violations_from_sample)

            model.obj = pyo.Objective(rule=cost, sense=pyo.minimize)


            solver = pyo.SolverFactory("gurobi")
            max_time = 35
            solver.options['TimeLimit'] = max_time
            #solver.options['MIPGap'] = 0.02
            print("Solving ...")
            res = solver.solve(model, load_solutions = True, tee=True, report_timing = True)
            
            UB = res.Problem._list[0].upper_bound
            LB = res.Problem._list[0].lower_bound
            gap = (UB - LB)/UB

            costs[i,j,k] = value(model.obj)
            gaps[i,j,k] = gap


            print("Objective function: ", value(model.obj))

            print("u")
            print(np.reshape([value(model.u[i,j]) for i in model.G for j in model.T],(G,T)))
            print("z_on")
            print(np.reshape([value(model.z_on[i,j]) for i in model.G for j in model.T],(G,T)))
            print("z_off")
            print(np.reshape([value(model.z_off[i,j]) for i in model.G for j in model.T], (G,T)))
            print("p")
            print(np.reshape([value(model.P[i,j]) for i in model.G for j in model.T], (G,T)))



# To plot cost_vs_scenarios (choose either this, or cost_vs_reliability):

scenarios_vector = np.arange(max_scenarios, 0, -1*int(max_scenarios/number_of_points))
with open(f"results_cost_vs_scenarios.txt", "a") as file:
    file.write("Scenarios Cost Gap\n")
    for i in range(number_of_points):
        file.write(f"{scenarios_vector[i]} {costs[0,0,i]} {gaps[0,0,i]}\n")

#___________________________________________________________________________________


# To plot cost_vs_reliability (choose either this, or cost_vs_scenarios):

#with open("result_cost_vs_reliability.txt", "a") as file:
#    file.write("p No_Correlations Moderate_Correlations High_Correlations\n")
#    for i in range(len(epsilon_vector)):
#        file.write(f"{1 - epsilon_vector[i]} {costs[0,i,0]} {costs[1,i,0]} {costs[2,i,0]}\n")


#with open("gaps_cost_vs_reliability.txt", "a") as file:
#    file.write("p No_Correlations Moderate_Correlations High_Correlations\n")
#    for i in range(len(epsilon_vector)):
#        file.write(f"{1 - epsilon_vector[i]} {gaps[0,i,0]} {gaps[1,i,0]} {gaps[2,i,0]}\n")

#___________________________________________________________________________________

