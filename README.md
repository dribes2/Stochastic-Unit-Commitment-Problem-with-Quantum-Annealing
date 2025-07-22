This repository has the goal of sharing and recreating the results from the paper "XXXXXXX" written by David Ribes and Tatiana Grandon. For extra info, send an email to david.r.marza@ntnu.no.

The paper solves two different problems: Stochastic Unit Commitment Problem and Deterministic Unit Commitment Problem, both of them solved in the folders Stoch_UCP and Det_UCP respectively.

Since D-Wave requires a non-free access token, the results of D-Wave runs are provided in the results files. They are compressed in zip files due to storage reasons.

_______________________________ Stoch_UCP _______________________________

This problem instance is shown in two main plots:
- cost_vs_reliability: dependence of the cost function value on the reliability level p = 1 - \epsilon. This is obtained for three different covariance matrices in the sampleset of scenarios: Non-correlated, Moderately correlated and Highly Correlated.

- cost_vs_scenarios: dependence of the cost function value on the amount of scenarios considered in the sampleset.

    - D-Wave runs:
        To obtain the results of cost_vs_reliability, you need to modify the file Conejo_UCP_JCC_CQM_reliability.py by adding a token in line 237. This script will provide a sampleset for each point in the plots (3 * 7 points).

        To obtain the results of cost_vs_scenarios, you need to modify the file Conejo_UCP_JCC_scenarios.py by adding a token in line 245. This script will provide a sampleset for each point in the plot (25 points).

    - Gurobi runs:
        The same python script is used for both plots, which is found in gurobi/conejo_reformulated_JCC_cluster_UCP.py

        To obtain the data corresponding to the plot cost_vs_reliability (cost_vs_scenarios), comment (uncomment) lines 199-202, 333-337 and uncomment(comment) lines 206-209, 344-353.


_______________________________ Det_UCP _______________________________


