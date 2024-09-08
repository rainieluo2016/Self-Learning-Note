# SHAP (shapley additive exPlanations)

Resource Link: https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html
Date: September 3, 2024
Genre: XgBoost
Total Pages: 1
Pages Completed: 1
Progress: 1
Completed: Yes
Resource Type: Blog
Abstract: Understand the SHAP and how it get works in XGBoost
Status: Reads
Notion Link by Yu - [url](https://www.notion.so/SHAP-shapley-additive-exPlanations-673e178d55bf434bbafd79964e858465?pvs=4)

# Overview

- concept from cooperative game theory
- understand the prediction by computing the contribution of each feature to the prediction

---

# Cooperative Game Theory

## Coalition

- a group or a subset of players who come together and cooperate to achieve a common goal or to maximize their combined value
- **Formation** - The coalition can include any number of players from the entire set, ranging from a single player to the entire group (referred to as the "grand coalition").
    - For example, if you have three players $A$, $B$, and $C$, the possible coalitions are:
        - Single-player coalitions: $\{ A\}$, $\{ B\}$, $\{ C\}$
        - Two-player coalitions: $\{ A, B \}$, $\{ A, C \}$, $\{ B, C \}$
        - The grand coalition (all players):  $\{ A, B, C \}$
        - The empty coalition (no player)
    - ordering does not matter within a coalition
- **Value of a Coalition -** Each coalition $S$ has an associated value $v(S)$ which represents the total benefit, profit, or outcome that the coalition can achieve by working together. The value function $v$  depends on the contributions of the players within that coalition.
- **Purpose**
    - understand how players’ cooperation can lead to different outcomes
    - how rewards or payoffs should be distributed among the players based on their contribution

## Shapley Value $\phi_i$

- **Goal** - measuring each player's contribution to the game
- Steps
    - **Identify Players and Coalitions -** Consider a set of players $N$, and a charactistic value function $v(S)$ which gives the value of any coalition $S  \subseteq N$
    - **Determine Marginal Contribution for player $i$ in coalition $S$**
        - suppose coalition $S' = S \backslash \{i\}$, which is coalition $S$ without player $i$
        - calculate the difference between the coalition $v(S) - v(S')$
    - Calculate Shapley Value $\phi_i$ for Player $i$
        - the weighted average of the marginal contribution across all possible coalitions
        
        ![image.png](SHAP%20(shapley%20additive%20exPlanations)%20673e178d55bf434bbafd79964e858465/image.png)
        

---

# SHAP in XGBoost

## Model itself as a cooperative game

- each feature is considered as a player the contributes to the prediction (the value of the game)
- using Tree SHAP algorithm - [example](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Basic%20SHAP%20Interaction%20Value%20Example%20in%20XGBoost.html)

## Stepwise

- **Path Tracking:** The algorithm tracks the paths taken by a given instance through the decision trees in the ensemble (XGBoost model).
- **Contribution Calculation:** For each tree, the algorithm calculates how the presence or absence of a feature influences the path taken by the instance. This involves measuring the change in the predicted value when a feature is included versus excluded.
- **Expected Value:** The calculation considers the expected contribution of a feature to the model's output, averaging over all possible permutations of feature orderings.
- **Aggregation Across Trees:** SHAP values from all individual trees in the ensemble are aggregated to produce the final SHAP value for each feature.
    
    ![image.png](SHAP%20(shapley%20additive%20exPlanations)%20673e178d55bf434bbafd79964e858465/image%201.png)


---

# **Self Comments**

## 2024.08

- Initial Use Case - understand individual feature contribute to xgboost model
    - after reading - not sure if this is the best tool?
    - marginal distribution when correlated features?
    - tree based model handle the missing by itself considering the local min - while feeding model without the target features to understand the contributor, not exactly one to one as model with this feature.

---

# **Reference Links**

Lundberg, Scott M., and Su-In Lee. "An Introduction to Explainable AI with Shapley Values." SHAP Documentation, [shap.readthedocs.io](http://shap.readthedocs.io/), [https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An introduction to explainable AI with Shapley values.html](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html).

Fadel S. "Explainable Artificial Intelligence: Interpretable Machine Learning." Statistics Canada, [https://www.statcan.gc.ca/en/data-science/network/explainable-learning](https://www.statcan.gc.ca/en/data-science/network/explainable-learning). Accessed 3 Sept. 2024.

"How is the Shapley value calculated?" ChatGPT, OpenAI, 3 Sept. 2024, [chat.openai.com](http://chat.openai.com/).

"explainer.shap_values implementation from scratch in python" ChatGPT, OpenAI, 3 Sept. 2024, [chat.openai.com](http://chat.openai.com/).

---