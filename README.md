# Recreating Results from "Towards Quantum Stochastic Optimization for Energy Systems under Uncertainty: Joint Chance Constraints with Quantum Annealing"

The Unit Commitment Problem (UCP) is a very common problem in the field of energy systems and modeling. It consists of minimizing (maximizing) the cost (profit) of scheduling power generators meeting constraints such as demand, capacity, ramping limits, logic etc. This problem is NP-hard and thus, very demanding and time consuming to be tackled with classical methods. Here we share the code developed to tackle this problem with an alternative approach, Quantum Annealing.

This repository contains the data and code used to reproduce the results presented in the paper **"Towards Quantum Stochastic Optimization for Energy Systems under Uncertainty: Joint Chance Constraints with Quantum Annealing"** by *David Ribes* and *Tatiana Grandon*. For further inquiries, feel free to contact: [david.r.marza@ntnu.no](mailto:david.r.marza@ntnu.no).

The paper addresses two different problem formulations:

- **Stochastic Unit Commitment Problem** — located in the `Stoch_UCP` folder  
- **Deterministic Unit Commitment Problem** — located in the `Det_UCP` folder

> ⚠️ **Note**: Some results are obtained using D-Wave quantum computing services, which require a non-free access token. For convenience, we include D-Wave results in compressed `.zip` files due to storage constraints.

---

## 📁 Stoch_UCP

The mathematical formulation of the Stochastic UCP can be found in the file `Stoch_UCP_formulation.pdf`. This document outlines the objective function, decision variables, and all relevant operational and logical constraints used in the model. The main outputs obtained from this instance of the problems are:

- **`cost_vs_reliability`**: Shows the dependence of the cost function value on the reliability level \( p = 1 - ε \). Results are generated for three types of covariance structures in the scenario sample set:
  - Non-correlated
  - Moderately correlated
  - Highly correlated

- **`cost_vs_scenarios`**: Shows how the cost function value varies with the number of scenarios in the sample set.

The problem is formulated as a **Mixed-Integer Linear Program (MILP)**. Both D-Wave and Gurobi solvers are used to generate results:

### D-Wave Runs

- **Sampler used**: `LeapHybridCQMSampler`

- **To generate `cost_vs_reliability` data**:
  - Edit `Conejo_UCP_JCC_CQM_reliability.py`
  - Add your D-Wave token at **line 237**
  - This script generates a json file containing the serialized sampleset for each point in the plot (3 covariance types × 7 reliability levels = 21 points). The json file is located in Stoch_UCP/cost_vs_reliability/results/

- **To generate `cost_vs_scenarios` data**:
  - Edit `Conejo_UCP_JCC_scenarios.py`
  - Add your D-Wave token at **line 245**
  - This script generates 25 json files containing the resulting sampleset solution corresponding to different scenario counts. The json file is located in Stoch_UCP/cost_vs_scenarios/results/

### Gurobi Runs

- **Solver used**: Gurobi via the Pyomo optimization framework

- Script used for both plots:  
  `gurobi/conejo_reformulated_JCC_cluster_UCP.py`

- **Switching between plot modes**:
  - To generate `cost_vs_reliability` data:
    - Comment lines 199–202 and 333–337
    - Uncomment lines 206–209 and 344–353
  - To generate `cost_vs_scenarios` data:
    - Reverse the above (i.e., comment `reliability` lines and uncomment `scenarios` lines)

> 🔧 **Note**: This repository focuses solely on data extraction. Post-processing and plotting of results are left to the user.

---

## 📁 Det_UCP

This folder contains the implementation and formulation of the **Deterministic Unit Commitment Problem (UCP)**. It mirrors the structure of the `Stoch_UCP` folder but focuses on the **deterministic** case, where uncertainties in demand or generation are not explicitly modeled.

### 📄 Mathematical Formulation

The mathematical formulation of the deterministic UCP is detailed in the file: `Det_UCP_formulation.pdf`. This document outlines the objective function, decision variables, and all relevant operational and logical constraints used in the model.

### 🧩 Code Structure

The folder includes two main Python scripts:

- **`functions.py`** — Contains all auxiliary functions required to define, build, and solve the deterministic UCP model.  
  These include functions for:
  - Model construction (variables, constraints, objective)
  - Data initialization and parameter handling
  - Solution extraction and formatting
  - Plotting utilities

- **`det_UCP.py`** — Serves as the main execution script.  
  It imports functions from `functions.py` to:
  1. Build and solve the deterministic UCP instance  
  2. Collect and process results  
  3. Generate plots illustrating cost, commitment schedules, or other performance metrics  

Running this script will automatically execute the deterministic UCP workflow and produce the relevant output figures.