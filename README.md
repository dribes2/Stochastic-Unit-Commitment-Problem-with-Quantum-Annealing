# Recreating Results from "Towards Quantum Stochastic Optimization for Energy Systems under Uncertainty: Joint Chance Constraints with Quantum Annealing"

The Unit Commitment Problem (UCP) is a very common problem in the field of energy systems and modeling. It consists of minimizing (maximizing) the cost (profit) of scheduling power generators meeting constraints such as demand, capacity, ramping limits, logic etc. This problem is NP-hard and thus, very demanding and time consuming to be tackled with classical methods. Here we share the code developed to tackle this problem with an alternative approach, Quantum Annealing.

This repository contains the data and code used to reproduce the results presented in the paper **"Towards Quantum Stochastic Optimization for Energy Systems under Uncertainty: Joint Chance Constraints with Quantum Annealing"** by *David Ribes* and *Tatiana Grandon*. For further inquiries, feel free to contact: [david.r.marza@ntnu.no](mailto:david.r.marza@ntnu.no).

The paper addresses two different problem formulations:

- **Stochastic Unit Commitment Problem** — located in the `Stoch_UCP` folder  
- **Deterministic Unit Commitment Problem** — located in the `Det_UCP` folder

> ⚠️ **Note**: Some results are obtained using D-Wave quantum computing services, which require a non-free access token. For convenience, we include D-Wave results in compressed `.zip` files due to storage constraints.

---

## 📁 Stoch_UCP

The mathematical formulation of the Stochastic UCP can be found in the file `Stoch_UCP_formulation.pdf`. The main outputs obtained from this instance of the problems are:

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
  - This script generates a sampleset for each point in the plot (3 covariance types × 7 reliability levels = 21 points)

- **To generate `cost_vs_scenarios` data**:
  - Edit `Conejo_UCP_JCC_scenarios.py`
  - Add your D-Wave token at **line 245**
  - This script generates 25 samples corresponding to different scenario counts

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

*Section under construction or to be filled in.*