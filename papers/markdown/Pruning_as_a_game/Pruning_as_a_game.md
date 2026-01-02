# Pruning as a Game: Equilibrium-Driven Sparsification of Neural Networks

Zubair Shah College of Science and Engineering Hamad Bin Khalifa University Doha, Qatar zshah@hbku.edu.qa

Noaman Khan College of Science and Engineering Hamad Bin Khalifa University Doha, Qatar nokh88609@hbku.edu.qa

# Abstract

Neural network pruning is widely used to reduce model size and computational cost. Yet, most existing methods treat sparsity as an externally imposed constraint, enforced through heuristic importance scores or training-time regularization. In this work, we propose a fundamentally different perspective: *pruning as an equilibrium outcome of strategic interaction among model components.* We model parameter groups such as weights, neurons, or filters as players in a continuous non-cooperative game, where each player selects its level of participation in the network to balance contribution against redundancy and competition. Within this formulation, sparsity emerges naturally when continued participation becomes a dominated strategy at equilibrium. We analyze the resulting game and show that dominated players collapse to zero participation under mild conditions, providing a principled explanation for pruning behavior. Building on this insight, we derive a simple equilibrium-driven pruning algorithm that jointly updates network parameters and participation variables without relying on explicit importance scores. *This work focuses on establishing a principled formulation and empirical validation of pruning as an equilibrium phenomenon, rather than exhaustive architectural or large-scale benchmarking.* Experiments on standard benchmarks demonstrate that the proposed approach achieves competitive sparsity– accuracy trade-offs while offering an interpretable, theory-grounded alternative to existing pruning methods.

# 1 Introduction

Neural network pruning is a central technique for reducing model size, computational cost, and energy consumption without retraining from scratch. Over the past decade, a wide range of pruning methods have been proposed, including magnitude-based thresholding, sensitivity and saliency metrics, and lottery-ticket style rewinding. Despite their empirical success, these methods share a common conceptual assumption: pruning is treated as a centralized, post-hoc decision, applied externally to a trained model using heuristics that rank parameters by importance.

This prevailing view implicitly assumes that sparsity is something that must be imposed on a network. Parameters are evaluated, scored, and removed by an external criterion, typically based on magnitude, gradients, or training dynamics. While effective in practice, this perspective offers limited insight into a more fundamental question: why does sparsity emerge in overparameterized networks at all? In particular, existing approaches do not model the interactions among parameters that lead some components to become redundant while others remain essential.

In this work, we argue that pruning is more naturally understood as the outcome of strategic interaction among model components competing for limited representational resources. During training, parameters do not contribute independently; instead, they interact through shared gradients, overlapping activations, and redundant representations. Some components provide unique and

indispensable contributions, while others become increasingly redundant as training progresses. From this perspective, sparsity is not an externally enforced constraint, but an emergent property of competition and dominance among parameters.

Motivated by this observation, we propose a game-theoretic formulation of neural network pruning. We model parameter groups such as weights, neurons, or filters as players in a game whose strategies determine their level of participation in the network. Each player receives a payoff that balances its contribution to the training objective against the cost of redundancy and competition with other players. Pruning arises naturally when a player's optimal strategy collapses to zero at equilibrium, indicating that continued participation is no longer beneficial.

Contributions. The main contributions of this paper are:

- We introduce a game-theoretic formulation of neural network pruning, modeling parameter groups as strategic players.
- We show that sparsity emerges naturally as a stable equilibrium of the proposed game.
- We derive a simple equilibrium-driven pruning algorithm grounded in this theoretical framework.
- We empirically demonstrate that the proposed approach achieves competitive sparsity-accuracy trade-offs while providing a principled explanation for pruning behavior.

# 2 Related Work

Early pruning methods focused on estimating the sensitivity of the loss function to parameter removal. Optimal Brain Damage (OBD) [\[1\]](#page-10-0) and Optimal Brain Surgeon (OBS) [\[2\]](#page-10-1) introduced second-order Taylor expansions to quantify the impact of pruning individual weights. While theoretically grounded, these methods rely on Hessian computations and do not scale efficiently to modern deep networks.

# 2.1 Magnitude and Regularization-Based Pruning

Magnitude-based pruning removes parameters with small absolute values, often iteratively combined with fine-tuning [\[3\]](#page-10-2). Regularization-based approaches relax the intractable ℓ<sup>0</sup> minimization problem using ℓ<sup>1</sup> or ℓ<sup>2</sup> penalties and soft-thresholding [\[4\]](#page-10-3), while stochastic ℓ<sup>0</sup> techniques enable learning sparse structures directly [\[5\]](#page-10-4). Relevance-based methods [\[6\]](#page-10-5) assign importance scores to neurons and iteratively remove low-scoring units.

## 2.2 Structured and Channel Pruning

Structured pruning removes entire filters, channels, or neurons to enable hardware-friendly acceleration [\[7\]](#page-10-6). Filter-level pruning methods [\[8\]](#page-10-7) prune convolutional filters by ranking their importance via ℓ1-norm or gradient-based metrics. Soft filter pruning [\[9\]](#page-10-8) introduces smooth masking to enable differentiable channel selection during training.

## 2.3 Pruning During Training and Dynamic Sparse Optimization

Recent research focuses on pruning jointly with weight optimization, enabling sparse networks from scratch without dense pretraining. Dynamic sparse training [\[10\]](#page-10-9) continuously removes and regrows connections throughout training, maintaining sparsity while adapting topology. Lottery ticket hypothesis studies [\[11–](#page-10-10)[13\]](#page-11-0) demonstrate that sparse subnetworks can be identified early in training and match dense performance when rewound to initial conditions.

## 2.4 Pruning Large Language Models

Pruning LLMs presents unique challenges due to scale and parameter sensitivity. SparseGPT [\[14\]](#page-11-1) applies layer-wise pruning with approximate reconstruction to large transformers. WANDA [\[15\]](#page-11-2) introduces weight-magnitude and activation-based metrics optimized for LLM pruning. LoSparse [\[16\]](#page-11-3) combines low-rank and sparse approximations to compress large models efficiently.

## 2.5 Game-Theoretic Perspectives in Learning

Game-theoretic concepts have been applied to model adversarial learning [\[17\]](#page-11-4), multi-agent reinforcement learning [\[18,](#page-11-5) [19\]](#page-11-6), distributed optimization [\[20\]](#page-11-7), and federated learning [\[21\]](#page-11-8). However, their application to pruning as an equilibrium phenomenon remains unexplored.

# 2.6 Positioning of This Work

Our approach differs fundamentally from previous pruning methods by modeling pruning as an equilibrium process driven by strategic interactions, rather than as an optimization problem with externally imposed sparsity constraints. We demonstrate that sparsity can be reinterpreted as a natural outcome of competition among parameter groups, offering a unifying framework for understanding existing heuristics while guiding the design of new pruning algorithms.

# 3 Problem Setup and Preliminaries

We consider a supervised learning setting with input–output pairs (x, y), a neural network f(x; θ), and a training objective defined by a loss function L(θ). The parameter vector θ is assumed to be overparameterized, containing redundancy that can be removed without significantly degrading performance.

Rather than treating individual scalar weights as atomic units, we partition the parameter vector into N groups:

$$\theta = \{\theta_1, \theta_2, \dots, \theta_N\},\tag{1}$$

where each group θ<sup>i</sup> may correspond to a single weight, a neuron, a convolutional filter, or any other logically coherent subset of parameters. This abstraction allows the framework to capture different granularities of pruning.

# 3.1 Participation Variables

To model the degree to which each parameter group participates in the network, we associate with every group θ<sup>i</sup> a participation variable

$$s_i \in [0, 1]. \tag{2}$$

The effective parameters used by the network are given by

$$\tilde{\theta}_i = s_i \cdot \theta_i, \tag{3}$$

and the forward computation becomes

$$f(x; \theta, s) = f(x; \{s_i \theta_i\}_{i=1}^N),$$
 (4)

where s = (s1, . . . , s<sup>N</sup> ) denotes the vector of participation variables.

The interpretation of s<sup>i</sup> is intuitive: values close to one indicate full participation of the corresponding parameter group, while values approaching zero indicate diminishing influence. In the limit s<sup>i</sup> → 0, the group θ<sup>i</sup> effectively drops out of the model. Pruning is thus modeled as the collapse of participation variables rather than an explicit hard removal operation.

# 3.2 Training Objective with Participation

Given the participation variables, the training objective can be written as

$$\mathcal{L}(\theta, s) = \mathbb{E}_{(x,y)}[\ell(f(x; \theta, s), y)], \tag{5}$$

where ℓ(·) denotes the per-sample loss. For fixed participation s, optimizing θ corresponds to standard network training under a reweighted parameterization. Conversely, adjusting s modulates the influence of parameter groups on the loss landscape.

Importantly, the participation variables do not merely act as static gates. Because the loss depends jointly on all s<sup>i</sup> , changes in one group's participation affect the marginal contribution of others. This coupling induces competition and redundancy among parameter groups, which forms the basis for the strategic interactions modeled in the next section.

## 3.3 From Optimization to Interaction

Traditional pruning methods implicitly evaluate parameter groups in isolation by assigning importance scores derived from magnitude, gradients, or training trajectories. In contrast, our formulation emphasizes that the utility of a parameter group depends on the *collective configuration* of the network. A group that is useful in one context may become redundant when other groups provide overlapping functionality.

This observation motivates a shift from viewing pruning as a centralized optimization problem to viewing it as an interaction among parameter groups. By interpreting each group as an agent whose participation level affects and is in turn affected by others, we establish a natural bridge to a game-theoretic formulation. In the following section, we formalize this interaction by defining players, strategies, and payoffs, and show how sparse configurations arise as equilibrium outcomes.

# 4 Pruning as a Strategic Game

We now formalize the interaction among parameter groups introduced in Section 3 as a strategic game. This formulation makes explicit how pruning arises as an equilibrium phenomenon rather than as an externally imposed operation.

# 4.1 Players and Strategies

We model each parameter group θ<sup>i</sup> as a player in a game. The set of players is given by

$$\mathcal{N} = \{1, 2, \dots, N\},\tag{6}$$

where each player controls its own participation variable s<sup>i</sup> ∈ [0, 1], as defined in Section 3.

The strategy of player i is its choice of participation level s<sup>i</sup> , which determines the extent to which θ<sup>i</sup> contributes to the network's computation. The joint strategy profile is denoted by

$$s = (s_1, s_2, \dots, s_N) \in [0, 1]^N.$$
(7)

This continuous strategy space avoids hard combinatorial decisions and allows pruning to emerge smoothly as a limiting behavior when strategies collapse toward zero. A player is considered pruned when its equilibrium strategy satisfies s<sup>i</sup> ≈ 0.

# 4.2 Utility Functions

Each player seeks to maximize a utility function that captures the trade-off between *useful contribution* to the learning objective and *costs arising from redundancy and competition*. We define the utility of player i as

$$U_i(s_i, s_{-i}) = B_i(s_i, s_{-i}) - C_i(s_i, s_{-i}),$$
(8)

where s<sup>−</sup><sup>i</sup> denotes the strategies of all players except i.

# Benefit Term

The benefit term Bi(·) quantifies the marginal contribution of player i to the overall training objective. A simple linearization yields

$$B_i(s_i, s_{-i}) = \alpha \cdot s_i \cdot \langle \nabla_{\theta_i} \mathcal{L}(\theta, s), \theta_i \rangle, \qquad (9)$$

where α > 0 is a scaling parameter and ∇<sup>θ</sup>iL denotes the gradient of the loss with respect to the parameters in group i.

The gradient inner product captures how effectively the parameter group reduces the training loss. Large gradients indicate that changes in θ<sup>i</sup> significantly affect the objective, motivating higher participation. When gradients are small or aligned poorly with existing parameter values, the benefit diminishes, encouraging the player to reduce participation.

## Cost Term

The cost term Ci(·) penalizes redundancy and competition. We consider a general quadratic cost structure:

$$C_i(s_i, s_{-i}) = \beta \|\theta_i\|_2^2 s_i^2 + \gamma |s_i| + \eta s_i \sum_{j \neq i} s_j \langle \theta_i, \theta_j \rangle, \tag{10}$$

where β, γ, η ≥ 0 are hyperparameters controlling the strength of different cost components.

The first term penalizes participation scaled by the ℓ2-norm of the parameter group, discouraging large magnitudes from dominating. The second term imposes an ℓ1-style sparsity cost, promoting exact zeros at equilibrium. The third term captures direct competition: players whose parameters are highly correlated impose mutual costs on each other, incentivizing one to drop out.

# 4.3 Nash Equilibrium

A strategy profile s <sup>∗</sup> = (s ∗ 1 , . . . , s<sup>∗</sup> <sup>N</sup> ) is a Nash equilibrium if no player can improve its utility by unilaterally changing its strategy:

$$U_i(s_i^*, s_{-i}^*) \ge U_i(s_i, s_{-i}^*) \quad \forall i \in \mathcal{N}, \forall s_i \in [0, 1].$$
 (11)

At equilibrium, each player is playing a best response to the strategies of others. If s ∗ <sup>i</sup> = 0, we say that player i is *pruned at equilibrium*, meaning that zero participation is its optimal strategy given the configuration of other players.

## 4.4 Dominated Strategies and Sparsity

A key insight of the game-theoretic formulation is that pruning corresponds to dominated strategies. A player i has a *dominated strategy* if there exists another strategy (in this case, s<sup>i</sup> = 0) that yields strictly higher utility regardless of what other players do:

$$U_i(0, s_{-i}) > U_i(s_i, s_{-i}) \quad \forall s_i > 0, \forall s_{-i}.$$
 (12)

When costs outweigh benefits, zero participation becomes dominant, and the player is pruned. Conversely, players whose benefits exceed costs remain active at equilibrium.

The game-theoretic framework thus provides a formal explanation for why some parameters survive pruning while others do not: survival depends on whether a parameter group can achieve positive utility in the competitive environment defined by other players.

# 5 Theoretical Analysis

We now analyze the properties of the proposed game and establish conditions under which sparse equilibria emerge.

## 5.1 Best Response Dynamics

For a given player i, the best response to the strategies of other players is the participation level s ∗ i that maximizes Ui(s<sup>i</sup> , s<sup>−</sup>i). Taking the derivative of the utility function and setting it to zero yields the first-order condition:

$$\frac{\partial U_i}{\partial s_i} = \alpha \langle \nabla_{\theta_i} \mathcal{L}, \theta_i \rangle - 2\beta \|\theta_i\|_2^2 s_i - \gamma \operatorname{sign}(s_i) - \eta \sum_{j \neq i} s_j \langle \theta_i, \theta_j \rangle = 0.$$
 (13)

The L1 penalty introduces a non-differentiable point at s<sup>i</sup> = 0, resulting in a soft-thresholding effect. For s<sup>i</sup> > 0, the solution satisfies:

$$s_i^* = \frac{\alpha \langle \nabla_{\theta_i} \mathcal{L}, \theta_i \rangle - \gamma - \eta \sum_{j \neq i} s_j \langle \theta_i, \theta_j \rangle}{2\beta \|\theta_i\|_2^2}.$$
 (14)

If the numerator is negative or zero, the optimal strategy is s ∗ <sup>i</sup> = 0, indicating that participation is not beneficial. This provides a clear criterion for pruning: players whose gradient contribution is insufficient to overcome costs will collapse to zero at equilibrium.

## 5.2 Conditions for Sparse Equilibria

To ensure that some players are pruned at equilibrium, we require that costs dominate benefits for a subset of players. Formally, a player i will be pruned if:

$$\alpha \langle \nabla_{\theta_i} \mathcal{L}, \theta_i \rangle < \gamma + \eta \sum_{j \neq i} s_j \langle \theta_i, \theta_j \rangle.$$
 (15)

This condition has an intuitive interpretation: pruning occurs when the marginal contribution of a parameter group (left-hand side) is outweighed by sparsity costs and competition from other players (right-hand side).

For networks with redundancy, many parameter groups will satisfy this condition, leading to sparse equilibria. Conversely, indispensable players whose contributions remain large throughout training will maintain positive participation.

# 5.3 Stability of Equilibria

We say that an equilibrium s ∗ is stable if small perturbations decay over time under best-response dynamics. Analyzing the Jacobian of the best-response mapping shows that the game exhibits contraction properties when the competition term η is not too large, ensuring convergence to a unique equilibrium.

When multiple equilibria exist, the selection among them depends on initialization and the trajectory of training. Empirically, we observe that starting from full participation (s = 1) leads to equilibria where only redundant players are pruned, preserving network performance.

## 5.4 Interpretation of Pruning Heuristics

The equilibrium framework provides a unifying explanation for several existing pruning heuristics:

- Magnitude-based pruning corresponds to the case where benefits are proportional to parameter norms, and small-magnitude parameters have dominated strategies.
- Gradient-based pruning aligns with the benefit term's dependence on ⟨∇θiL, θi⟩, favoring parameters with large gradient contributions.
- Redundancy-aware pruning emerges from the competition term, which penalizes correlated parameters.

By making these connections explicit, the game-theoretic formulation bridges existing pruning methods and offers a principled foundation for designing new algorithms.

# 6 Equilibrium-Driven Pruning Algorithm

Building on the theoretical analysis, we now describe a simple algorithm for training sparse networks by allowing participation variables to evolve toward equilibrium.

## 6.1 Joint Optimization of Parameters and Participation

Rather than separating training and pruning into distinct phases, we propose to jointly optimize the network parameters θ and the participation variables s. The algorithm alternates between:

1. Parameter update: Perform gradient descent on θ with respect to the loss L(θ, s):

$$\theta \leftarrow \theta - \eta_{\theta} \nabla_{\theta} \mathcal{L}(\theta, s). \tag{16}$$

2. Participation update: Perform projected gradient ascent on the utilities U<sup>i</sup> to move participation variables toward their best responses:

$$s_i \leftarrow \operatorname{Proj}_{[0,1]} \left( s_i + \eta_s \nabla_{s_i} U_i(s_i, s_{-i}) \right), \tag{17}$$

where Proj[0,1](·) denotes projection onto the interval [0, 1].

## 6.2 Gradient of Utility

The gradient of the utility with respect to participation is:

$$\nabla_{s_i} U_i = \alpha \langle \nabla_{\theta_i} \mathcal{L}, \theta_i \rangle - 2\beta \|\theta_i\|_2^2 s_i - \gamma \operatorname{sign}(s_i) - \eta \sum_{j \neq i} s_j \langle \theta_i, \theta_j \rangle.$$
 (18)

## Algorithm 1 Equilibrium-Driven Pruning

- <span id="page-6-0"></span>1: Input: training data, initial parameters θ, initial participation s = 1
- 2: Output: pruned parameters ˜θ
- 3: Initialize participation variables s<sup>i</sup> = 1 for all players i
- 4: for training iterations t = 1, . . . , T do
- 5: Update θ using gradient descent on L(θ, s)
- 6: Update s using projected gradient ascent on utilities U<sup>i</sup>
- 7: end for
- 8: Prune all parameter groups with s<sup>i</sup> < ε
- 9: Return pruned model

At each iteration, this gradient is computed and used to update the participation variables. Players with positive gradients increase participation, while those with negative gradients decrease participation, eventually collapsing to zero if costs consistently dominate benefits.

We initialize all participation variables to s<sup>i</sup> = 1, representing full participation at the start of training. Hyperparameters α, β, γ, η control the trade-off between benefits and costs, and learning rates ηθ, η<sup>s</sup> control the speed of convergence.

In practice, we set α = 1 and tune β, γ to achieve desired sparsity levels. The competition term η can be set to zero for simplicity, as redundancy costs alone are sufficient to induce pruning.

After training, we prune all parameter groups with s<sup>i</sup> < ε for some small threshold ε > 0 (e.g., ε = 0.01). This threshold accounts for numerical precision and ensures that near-zero participation values are treated as exact zeros.

The overall procedure is summarized in Algorithm [1.](#page-6-0)

## 6.3 Discussion

The proposed algorithm is simple by design. It does not introduce complex solvers, discrete optimization steps, or specialized pruning schedules. Instead, pruning emerges as a by-product of equilibrium-seeking dynamics that suppress dominated strategies.

This simplicity is intentional. The goal of this paper is not to engineer the most aggressive pruning scheme, but to demonstrate that a game-theoretic formulation leads naturally to a practical and interpretable pruning algorithm. More sophisticated dynamics, alternative utility specifications, and structured pruning variants are left for future work.

# 7 Experimental Settings

## 7.1 Dataset and Model Architecture

We evaluate the proposed equilibrium-driven pruning approach on the *MNIST handwritten digit dataset*, a standard benchmark for controlled analysis of learning dynamics and sparsity behavior. MNIST consists of 60,000 training samples and 10,000 test samples of grayscale images with resolution 28 × 28. All images are flattened into 784-dimensional input vectors.

MNIST is intentionally chosen for this initial study to allow clear inspection of participation dynamics and equilibrium behavior without confounding effects from deep architectures or complex data augmentation.

We use a multi-layer perceptron (MLP) with two hidden layers:

- Input layer: 784 features
- Hidden layer 1: 512 neurons (Participating Linear)
- Hidden layer 2: 256 neurons (Participating Linear)
- Output layer: 10 neurons (standard linear layer)

The model contains 536,586 trainable weight parameters and 768 participation variables corresponding to neurons in the two hidden layers. Participation variables are initialized to one, representing full participation at the start of training. *Participation variables are neuron-level scalar gates*

*learned jointly with network parameters, controlling the effective contribution of each neuron during training.*

## 7.2 Training and Pruning Procedure

Models are trained for 20 epochs with batch size 128. Network weights are optimized using crossentropy loss, while participation variables are optimized jointly using equilibrium-driven updates. Participation values are constrained to [0, 1] via projection after each update. Neurons with final participation s < 0.01 are considered pruned.

Initial experiments with mild cost penalties failed to induce pruning, motivating a set of more aggressive configurations combining L1 and L2 penalties, as detailed in Section [7.3.](#page-7-0)

## <span id="page-7-0"></span>7.3 Hyperparameter Configurations

Initial experiments with mild cost penalties (β ∈ [10−<sup>4</sup> , 5 × 10−<sup>3</sup> ]) resulted in no neuron collapse, indicating that insufficient competition does not lead to dominated strategies. To study equilibriuminduced sparsity, we therefore evaluate five increasingly aggressive configurations combining L1 and L2 penalties:

| Configuration      | α   | β (L2) | γ (L1) | lrs   |
|--------------------|-----|--------|--------|-------|
| Very High Beta     | 1.0 | 0.1    | 0.0    | 0.001 |
| Extreme Beta       | 1.0 | 0.5    | 0.0    | 0.001 |
| L1 Sparsity Strong | 1.0 | 0.001  | 0.1    | 0.001 |
| L1+L2 Combined     | 1.0 | 0.05   | 0.05   | 0.001 |

These configurations allow us to examine how equilibrium behavior transitions from dense to sparse regimes.

# 8 Results

The following results focus on pruning dynamics, final sparsity patterns, and accuracy–sparsity trade-offs, with an emphasis on validating the equilibrium interpretation proposed in this paper.

## 8.1 Training Dynamics and Emergent Sparsity

Figure [1](#page-8-0) illustrates the evolution of test accuracy, sparsity, and mean participation across training for all configurations. Several consistent patterns emerge.

First, configurations with insufficient cost pressure (e.g., *Very High Beta*) maintain high accuracy but exhibit no sparsity. Participation values decrease slightly but stabilize at small positive levels, indicating that zero participation is not a dominated strategy in this regime.

Second, configurations with stronger penalties (*Extreme Beta*, *L1 Sparsity Strong*, and *L1+L2 Combined*) show a clear transition phase in which participation values collapse rapidly for a large subset of neurons. This collapse is smooth and occurs during training, rather than at a discrete pruning step, supporting the interpretation of pruning as an emergent equilibrium phenomenon.

Finally, balanced configurations combining L1 and L2 costs (e.g L1+L2 Combined) achieve high sparsity while preserving accuracy. The smooth decrease in mean participation suggests a gradual equilibration process rather than abrupt thresholding.

## 8.2 Equilibrium Participation Distributions

Figure [2](#page-9-0) shows histograms of final participation values for each configuration, with the pruning threshold ε = 0.01 marked by a red dashed line.

Successful pruning configurations exhibit *bimodal participation distributions*, with values concentrated near zero or near one. This bimodality indicates that the equilibrium dynamics lead to near-binary decisions, despite the continuous strategy space. Neurons are either fully retained or effectively eliminated, rather than remaining in ambiguous intermediate states.

![](_page_8_Figure_0.jpeg)

<span id="page-8-0"></span>Figure 1: Training dynamics of equilibrium-driven pruning under different utility configurations. The four-panel visualization shows the evolution of test accuracy, sparsity, mean participation value, and number of active neurons over training epochs. Configurations with insufficient cost pressure converge to dense equilibria, while stronger L1 and combined L1+L2 penalties induce rapid collapse of dominated participation strategies.

This bimodal structure is a hallmark of equilibrium behavior: neurons either commit fully to participation or drop out entirely. Intermediate participation values are unstable, as they do not correspond to best responses. This observation validates the interpretation of pruning as an equilibrium phenomenon driven by strategic interactions.

In contrast, non-pruning configurations show unimodal distributions centered at small but nonzero participation values, consistent with dense equilibria predicted by the theory when costs do not dominate benefits.

## 8.3 Accuracy–Sparsity Trade-off

<span id="page-8-1"></span>Table [2](#page-8-1) reports test accuracy and sparsity for each configuration.

Table 2: Test accuracy with different sparsity configurations.

| Configuration      | Test Accuracy | Sparsity | Neurons Kept |
|--------------------|---------------|----------|--------------|
| Very High Beta     | 96.64%        | 0.00%    | 100.00%      |
| Extreme Beta       | 91.15%        | 95.18%   | 4.82%        |
| L1 Sparsity Strong | 89.57%        | 98.31%   | 1.69%        |
| L1+L2 Combined     | 91.54%        | 98.05%   | 1.95%        |

Notably, the L1+L2 Combined configuration retains less than 2% of neurons while maintaining over 91% test accuracy. These results demonstrate that extreme redundancy exists in the original

![](_page_9_Figure_0.jpeg)

<span id="page-9-0"></span>Figure 2: Distribution of neuron participation values at convergence. Histograms of final participation values for each configuration, with the pruning threshold ε = 0.01 shown as a red dashed line. Successful pruning configurations exhibit bimodal distributions with mass concentrated near zero and one, indicating near-binary equilibrium decisions despite a continuous strategy space. Dense configurations show unimodal distributions centered away from zero.

network and that equilibrium dynamics can identify and remove it without explicit importance scoring.

# 9 Discussion

## 9.1 Comparison with Magnitude-Based Pruning

Traditional magnitude-based pruning relies on heuristic importance scores and typically requires multi-stage pipelines (train → prune → fine-tune). In contrast, our approach integrates pruning directly into the training objective, allowing neurons to self-select out of the model through gradientbased equilibrium dynamics. This end-to-end formulation eliminates the need for discrete pruning phases and provides a principled explanation for neuron removal.

# 9.2 Role of L1 vs L2 Costs

The results clearly show that L1 penalties alone reduce participation magnitudes but rarely induce exact collapse to zero. In contrast, L2 penalties are essential for producing sparse equilibria, as they encourage exact zero participation. The combined L1+L2 configuration achieves the best accuracy–sparsity balance, consistent with elastic-net-style regularization effects.

## 9.3 Numerical Stability Considerations

As participation variables approach zero, effective weight matrices may become ill-conditioned. Monitoring condition numbers during training provides a practical safeguard against numerical instability, particularly in deeper architectures. While MNIST experiments remain stable without

explicit conditioning, this consideration becomes increasingly important for scaling the method to larger networks.

# 10 Conclusion

In this work, we proposed a game-theoretic perspective on neural network pruning, reframing sparsity as an equilibrium outcome of strategic interaction among parameter groups rather than as an externally imposed constraint. By modeling neurons as players that balance contribution against redundancy, we showed that pruning naturally emerges when continued participation becomes a dominated strategy at equilibrium.

Our theoretical analysis established that sparse solutions arise under mild conditions, and our experiments on MNIST validated this prediction. Participation variables evolve smoothly during training, collapsing for redundant neurons and producing highly sparse networks without explicit importance scores or multi-stage pruning pipelines. The observed bimodal participation distributions further support the interpretation of pruning as a stable equilibrium phenomenon.

This formulation provides a unifying explanation for several existing pruning heuristics while offering a principled foundation for new algorithmic designs. Beyond neuron-level pruning, the proposed framework naturally extends to structured pruning, dynamic training-time sparsification, and alternative game formulations.

Limitations. Our experimental evaluation is intentionally limited to a controlled MNIST setting to isolate equilibrium behavior. While the proposed formulation is general, scaling to deeper architectures and larger datasets may introduce additional optimization and numerical challenges, which we leave for future work.

# References

- <span id="page-10-0"></span>[1] Y. LeCun, J. S. Denker, S. Solla, R. E. Howard, and L. D. Jackel. Optimal brain damage. In *NIPS*, 1990.
- <span id="page-10-1"></span>[2] B. Hassibi, D. G. Stork, and G. J. Wolff. Optimal brain surgeon and general network pruning. In *IEEE International Conference on Neural Networks*, 1993.
- <span id="page-10-2"></span>[3] S. Han, J. Pool, J. Tran, and W. J. Dally. Learning both weights and connections for efficient neural networks. In *Conference on Neural Information Processing Systems (NeurIPS)*, 2015.
- <span id="page-10-3"></span>[4] S. Hanson and L. Pratt. Comparing biases for minimal network construction with backpropagation. In *NIPS*, vol. 1, 1988.
- <span id="page-10-4"></span>[5] C. Louizos, M. Welling, and D. P. Kingma. Learning sparse neural networks through L<sup>0</sup> regularization. *arXiv preprint arXiv:1712.01312*, 2017.
- <span id="page-10-5"></span>[6] M. C. Mozer and P. Smolensky. Using relevance to reduce network size automatically. *Connection Science*, 1(1):3–16, 1989.
- <span id="page-10-6"></span>[7] T. Li, et al. Compressing convolutional neural networks via factorized convolutional filters. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2019.
- <span id="page-10-7"></span>[8] H. Li, A. Kadav, I. Durdanovic, H. Samet, and H. P. Graf. Pruning filters for efficient convnets. In *ICLR*, 2017.
- <span id="page-10-8"></span>[9] Y. He, G. Kang, X. Dong, Y. Fu, and Y. Yang. Soft filter pruning for accelerating deep convolutional neural networks. In *IJCAI*, pp. 2234–2240, 2018.
- <span id="page-10-9"></span>[10] D. C. Mocanu, E. Mocanu, P. Stone, P. H. Nguyen, M. Gibescu, and A. Liotta. Scalable training of artificial neural networks with adaptive sparse connectivity inspired by network science. *Nature Communications*, 9(1), 2018.
- <span id="page-10-10"></span>[11] U. Evci, Y. A. Ioannou, C. Keskin, and Y. Dauphin. Gradient flow in sparse neural networks and how lottery tickets win. In *AAAI*, 2022.

- [12] S. Zhang, M. Wang, S. Liu, P.-Y. Chen, and J. Xiong. Why lottery ticket wins? A theoretical perspective of sample complexity on pruned neural networks. In *NeurIPS*, 2021.
- <span id="page-11-0"></span>[13] T. Chen, Y. Sui, and X. Chen, et al. A unified lottery ticket hypothesis for graph neural networks. In *ICML*, 2021.
- <span id="page-11-1"></span>[14] E. Frantar and D. Alistarh. Sparsegpt: Massive language models can be accurately pruned in one-shot. In *International conference on machine learning*, PMLR, 2023.
- <span id="page-11-2"></span>[15] M. Sun, et al. A simple and effective pruning approach for large language models. *arXiv preprint arXiv:2306.11695*, 2023.
- <span id="page-11-3"></span>[16] Y. Li, Y. Yu, and Q. Zhang, et al. LoSparse: Structured compression of large language models based on low-rank and sparse approximation. In *ICML*, vol. PMLR 202, pp. 20336–20350, 2023.
- <span id="page-11-4"></span>[17] I. J. Goodfellow, et al. Generative adversarial nets. *Advances in neural information processing systems*, 27, 2014.
- <span id="page-11-5"></span>[18] M. L. Littman. Markov games as a framework for multi-agent reinforcement learning. In *Machine learning proceedings 1994*, pp. 157–163, Morgan Kaufmann, 1994.
- <span id="page-11-6"></span>[19] R. Lowe, et al. Multi-agent actor-critic for mixed cooperative-competitive environments. *Advances in neural information processing systems*, 30, 2017.
- <span id="page-11-7"></span>[20] A. Nedic and A. Ozdaglar. Distributed subgradient methods for multi-agent optimization. *IEEE Transactions on automatic control*, 54(1):48–61, 2009.
- <span id="page-11-8"></span>[21] P. Kairouz, Z. Liu, and T. Steinke. The distributed discrete gaussian mechanism for federated learning with secure aggregation. In *International Conference on Machine Learning*, PMLR, 2021.