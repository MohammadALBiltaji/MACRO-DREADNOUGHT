# MACRO-DREADNOUGHT

**MACRO-DREADNOUGHT** is an experimental neural-network architecture exploring a simple question:

> **What happens if a neural network is given mechanisms to observe parts of its own internal behavior and react when those parts become stagnant, underused, overly dominant, or repeatedly associated with failure?**

The project combines a three-path expert architecture with entropy-aware routing, controlled exploration, failure memory, selective weight reinitialization, feature reclamation, and a custom entropy-responsive activation.

MACRO-DREADNOUGHT **does not replace backpropagation**.

Gradient-based optimization remains the foundation of training. The project instead experiments with additional mechanisms around the normal training loop that use information such as routing behavior, uncertainty, rejected features, and difficult samples.

The current implementation should be considered a **research prototype**, not a state-of-the-art image-classification system.

---

## Research Motivation

Standard neural-network training is extremely effective, but much of the architecture remains passive during optimization.

A layer receives information, transforms that information, backpropagation computes gradients, and an optimizer updates the parameters.

MACRO-DREADNOUGHT explores whether additional internal signals can also be used during that process.

In particular, the project investigates whether a network can make practical use of:

- Expert utilization
- Routing entropy
- Residual information
- Rejected intermediate features
- Difficult or misclassified samples
- Signs of expert stagnation or routing monopoly

The central idea is not to remove gradient descent.

It is to experiment with giving the training system **more information about what is happening inside the network** and additional mechanisms for responding to that information.

---

# Core Architecture

MACRO-DREADNOUGHT is built around repeated **Highway-style blocks** containing three deliberately asymmetric processing paths.

Rather than giving three identical experts the same input and relying entirely on training dynamics to produce specialization, each path is designed to receive or emphasize a different information domain.

The three paths are:

1. **Anchor**
2. **Sniper**
3. **Ghost**

Their outputs are combined through an entropy-aware routing system.

---

## 1. Anchor Path

The **Anchor** is the primary local-processing path.

It receives the normal incoming feature representation and learns the dominant local transformation of that representation.

Conceptually:

```text
Input
  │
  ▼
Anchor
  │
  ▼
Primary Local Representation
```

The Anchor acts as the main processing path against which the behavior of the other experts can be defined.

---

## 2. Sniper Path

The **Sniper** is designed as a residual-correction expert.

Instead of simply receiving the same information as the Anchor, it operates on information remaining after the Anchor produces its representation.

Conceptually:

```text
Residual = Input - Anchor(Input)
```

The Sniper therefore receives a different information domain from the Anchor.

Its purpose is to encourage specialization around information that the primary path did not strongly represent.

This does **not** guarantee that the Sniper learns a perfectly complementary representation.

It is an architectural bias intended to make expert specialization more likely by reducing symmetry between the processing paths.

---

## 3. Ghost Path

The **Ghost** path focuses on broader spatial context.

It uses dilated convolutional operations to obtain a larger effective receptive field than the more locally focused Anchor path.

Its processing role therefore differs from both:

- the normal local representation used by the Anchor,
- and the residual representation used by the Sniper.

Conceptually:

```text
Anchor → Local information
Sniper → Residual information
Ghost  → Wider-context information
```

The goal is to create three paths with meaningfully different processing roles rather than three interchangeable branches.

---

# SpLR_V2

MACRO-DREADNOUGHT also uses **SpLR_V2**, a custom activation derived from my SentinelPulse activation research.

Its general behavior is based around a localized nonlinear correction of the form:

\[
f(x)=a x e^{-k x^2}+cx
\]

The nonlinear pulse is concentrated within a localized region and decreases as the magnitude of the input increases.

Within MACRO-DREADNOUGHT, the activation is additionally connected to an entropy-derived signal from the network.

The intention is to allow aspects of the nonlinear response to adapt according to the network's current uncertainty rather than remaining completely static throughout the forward pass.

SpLR_V2 is experimental.

The current MACRO-DREADNOUGHT experiment demonstrates that it can operate as part of the complete architecture, but it does **not yet isolate its independent contribution** relative to conventional activation functions.

A controlled activation ablation is part of the planned future work.

---

# Entropy-Aware Routing

The outputs of the three processing paths are combined through a learned routing mechanism.

The router observes the current representation and determines how strongly each expert should contribute.

Routing uncertainty is measured using entropy.

This provides the architecture with an internal signal describing how concentrated or uncertain the current routing decision is.

Entropy is used as both a diagnostic signal and an input to other experimental mechanisms within the architecture.

---

# 70/30 Exploitation-Exploration Router

Mixture-of-Experts systems can develop uneven expert utilization.

If routing becomes highly concentrated, one expert may receive most of the useful learning signal while other experts receive relatively little.

MACRO-DREADNOUGHT addresses this experimentally with a bounded exploitation-exploration mixture:

\[
R = 0.7R_{dynamic} + 0.3R_{uniform}
\]

where:

- **70%** of the routing distribution is determined dynamically.
- **30%** is distributed uniformly across the available experts.

For three experts, the exploration component ensures that every expert continues to receive non-zero routing mass.

The mechanism is intended to reduce the possibility of complete expert starvation while still allowing the router to prefer experts that appear useful.

It does **not** guarantee perfectly balanced expert utilization.

One expert can still dominate the learned portion of the routing distribution.

The mechanism instead ensures that the dynamic router is not the sole source of expert traffic.

---

# Failure Buffer

MACRO-DREADNOUGHT maintains a small **failure buffer** during training.

Rather than allowing difficult or misclassified samples to exist only as temporary loss values, the architecture retains information derived from a limited set of hard failures.

Conceptually:

```text
Difficult Sample
      │
      ▼
Failure Detected
      │
      ▼
Representation Stored
      │
      ▼
Failure Buffer
```

The buffer provides information that can later be used by the targeted reinitialization system.

The objective is to make certain interventions dependent on actual examples associated with current model failure rather than relying entirely on arbitrary random resets.

The buffer is deliberately limited rather than acting as a large replay dataset.

---

# Targeted Weight Reinitialization

One of the more unusual components of MACRO-DREADNOUGHT is its ability to selectively intervene when internal conditions indicate that part of the expert system may have become stagnant or excessively dominant.

The architecture monitors routing behavior alongside model errors.

When configured trigger conditions are satisfied, a selected component can undergo **targeted weight reinitialization**.

The general process is:

```text
Monitor Network Behavior
          │
          ▼
Check Routing / Failure Conditions
          │
     ┌────┴────┐
     │         │
   Normal    Trigger
     │         │
     ▼         ▼
 Continue   Select Target
 Training      │
               ▼
          Reinitialize
               │
               ▼
      Use Failure Information
               │
               ▼
        Continue Training
```

The goal is to test whether a stagnant component can be pushed into a substantially different parameter region rather than waiting indefinitely for ordinary gradient updates to change its behavior.

The reinitialization mechanism can incorporate information from the failure buffer so that the intervention is connected to examples the current model is struggling with.

This mechanism should be interpreted as an **experimental training intervention**.

It is not presented as a replacement for gradient descent, nor has the current experiment established that targeted reinitialization always improves generalization.

---

# Temporal Gate

MACRO-DREADNOUGHT uses gating to control how much information from a representation should continue through the primary hidden state.

A sigmoid-based gate produces values conceptually similar to:

\[
z = \sigma(\cdot)
\]

The gate separates information into components that are strongly accepted and components that receive lower gate weight.

Instead of assuming that low-weight information is permanently useless, MACRO-DREADNOUGHT provides another mechanism for handling part of that rejected representation.

That mechanism is the **Forensic Reclamation Bus**.

---

# Forensic Reclamation Bus

Deep neural networks repeatedly transform their internal representations.

During these transformations, information considered unimportant at one depth may effectively disappear from later representations.

MACRO-DREADNOUGHT experiments with preserving part of this rejected information.

Features receiving low gate weight can be transferred onto the **Forensic Reclamation Bus**.

That information is propagated forward and later made available to another processing path, particularly the Ghost expert of a subsequent block.

Conceptually:

```text
Current Representation
         │
         ▼
     Temporal Gate
       /       \
      /         \
Accepted       Rejected
Features       Features
   │              │
   ▼              ▼
Hidden       Reclamation
State            Bus
                   │
                   ▼
              Later Layer
                   │
                   ▼
               Ghost Path
```

A simplified relationship can be written as:

\[
Ghost_{n+1}
=
GhostInput_{n+1}
+
Reclaimed_n
\]

The purpose is to test whether information considered relatively unimportant at one layer may still become useful when processed later or from a different contextual perspective.

The mechanism is therefore intended to **mitigate feature loss or feature washout**.

The current results do not establish that information loss is completely eliminated.

---

# Simplified Highway Block

A MACRO-DREADNOUGHT block can be visualized approximately as:

```text
                         INPUT
                           │
                           ▼
                ┌────────────────────┐
                │ Entropy / Routing  │
                │     Analysis       │
                └─────────┬──────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
      ┌────────┐      ┌────────┐      ┌────────┐
      │ ANCHOR │      │ SNIPER │      │ GHOST  │
      └────────┘      └────────┘      └────────┘
          │               │               │
       Local          Residual          Wider
       Path             Path           Context
          │               │               │
          └───────────────┼───────────────┘
                          │
                          ▼
                  ┌──────────────┐
                  │ 70/30 Router │
                  └──────┬───────┘
                         │
                         ▼
                  Combined Output
                         │
                         ▼
                   Temporal Gate
                     /       \
                    /         \
                   ▼           ▼
             Hidden State   Reclaimed
                              Features
                                 │
                                 ▼
                         Reclamation Bus
                                 │
                                 ▼
                        Later Ghost Path
```

Multiple Highway blocks are stacked behind the convolutional feature-extraction backbone.

---

# Training Loop

MACRO-DREADNOUGHT still uses the conventional deep-learning training pipeline at its foundation.

```text
Forward Pass
     │
     ▼
Compute Loss
     │
     ▼
Backpropagation
     │
     ▼
Optimizer Step
     │
     ▼
Update Parameters
```

The experimental mechanisms operate around this process:

```text
Forward Pass
     │
     ▼
Compute Loss
     │
     ▼
Backpropagation
     │
     ▼
Adam / Gradient Update
     │
     ▼
Collect Internal Diagnostics
     │
     ├── Routing Utilization
     ├── Routing Entropy
     ├── Difficult Samples
     └── Failure Information
     │
     ▼
Check Intervention Conditions
     │
 ┌───┴─────────────┐
 │                 │
No Trigger       Trigger
 │                 │
 ▼                 ▼
Continue       Selective
Training      Reinitialization
                   │
                   ▼
              Continue Training
```

This distinction is important:

> **MACRO-DREADNOUGHT augments ordinary backpropagation. It does not replace it.**

---

# Experimental Benchmark

The current main experiment was performed on **Tiny ImageNet**.

| Configuration | Value |
|---|---|
| Dataset | Tiny ImageNet |
| Number of classes | 200 |
| Image resolution | 64 × 64 |
| Highway depth | 10 layers |
| Width | 256 channels |
| Training duration | 50 epochs |
| Hardware | Single NVIDIA Tesla T4 |
| Peak evaluation accuracy | ~44.5% |

The experiment was designed primarily to determine whether the complete architecture could operate together during a substantial training run.

The run tests the interaction of:

- The three expert paths
- Entropy-aware routing
- The 70/30 routing mechanism
- SpLR_V2
- The failure buffer
- Targeted reinitialization
- Temporal gating
- The Forensic Reclamation Bus

The approximately **44.5% peak evaluation accuracy** should be interpreted as a result from the current experimental configuration.

It is **not presented as a state-of-the-art Tiny ImageNet result**.

The architecture has not undergone the extensive hyperparameter optimization, compute scaling, augmentation tuning, or architecture search normally associated with benchmark-focused systems.

The primary purpose of the experiment is architectural investigation.

---

# What the Current Experiment Demonstrates

The current experiment provides evidence that:

- The complete MACRO-DREADNOUGHT architecture can be trained end-to-end.
- The three-path expert system can operate within the same network.
- Entropy and routing statistics can be measured during training.
- Controlled exploration can be integrated into expert routing.
- Difficult samples can be retained in the failure buffer.
- Selective reinitialization can be executed during training.
- Reclaimed features can be propagated between layers.
- The complete system can remain trainable over a 50-epoch Tiny ImageNet run.

These are implementation-level and system-level observations.

They should **not** be interpreted as evidence that every individual mechanism independently improves accuracy or generalization.

---

# What Has Not Yet Been Established

MACRO-DREADNOUGHT currently combines several experimental mechanisms simultaneously.

Because of that, the existing full-system experiment cannot determine exactly how much each individual component contributes.

The current results do not yet establish that:

- SpLR_V2 outperforms conventional activations in this architecture.
- Targeted reinitialization consistently improves final accuracy.
- The failure buffer independently improves generalization.
- The Forensic Reclamation Bus independently improves performance.
- The 70/30 router is superior to other expert-balancing methods.
- Each expert develops a perfectly distinct specialization.
- The full architecture consistently outperforms simpler parameter-matched baselines.

Those questions require controlled ablation experiments.

---

# Planned Research

Future experiments are intended to isolate individual components of the architecture.

## Component Ablations

Planned comparisons include:

- Full model vs. no Failure Buffer
- Full model vs. no targeted reinitialization
- Full model vs. no Forensic Reclamation Bus
- Full model vs. no temporal reclamation
- Full model vs. conventional activation functions
- Full model vs. purely dynamic routing
- Full model vs. purely uniform routing

---

## Activation Comparisons

SpLR_V2 can be compared against commonly used alternatives such as:

- ReLU
- GELU
- Mish
- SiLU

under otherwise comparable training conditions.

---

## Routing Analysis

Future analysis can measure:

- Expert utilization
- Routing entropy
- Routing monopoly
- Expert starvation
- Specialization patterns
- Changes in routing behavior throughout training

---

## Reinitialization Analysis

Potential comparisons include:

- Targeted reinitialization
- Random reinitialization
- No reinitialization
- Alternative trigger conditions

This would help determine whether the intervention itself provides measurable value beyond simply perturbing the model.

---

## Reclamation Analysis

The Forensic Reclamation Bus can be tested independently to determine whether transferred features provide useful information to later layers.

This includes comparing equivalent architectures with and without reclaimed features.

---

## Multi-Seed Experiments

Major experiments should eventually be repeated across multiple random seeds to measure:

- Mean performance
- Variance
- Stability
- Sensitivity to initialization

---

## Baseline Comparisons

Future experiments can compare MACRO-DREADNOUGHT against simpler models with comparable:

- Parameter counts
- Depth
- Width
- Training schedules
- Optimization settings

This will help separate improvements caused by individual mechanisms from improvements caused simply by increased model complexity or parameter count.

---

## Additional Datasets

Future testing may include other vision datasets and potentially other problem domains where expert routing, failure memory, and information reclamation can be studied.

---

# Research Status

MACRO-DREADNOUGHT is currently an **experimental architecture under active development**.

The project is intended as an investigation into unconventional training and architectural mechanisms rather than as a finished production model.

The current work should therefore be interpreted as:

> **Architecture design + implementation + initial experimental validation**

rather than a definitive evaluation of the underlying hypotheses.

Several mechanisms remain to be isolated through controlled experiments.

---

# Repository Contents

```text
MACRO-DREADNOUGHT/
│
├── README.md
├── WHITEPAPER.md
├── LICENSE
│
├── docs/
│   ├── Part 1 breakdown.ipynb
│   ├── Part 2 breakdown.ipynb
│   ├── Part 3 breakdown.ipynb
│   └── Part 4 breakdown.ipynb
│
└── MACRO_DREADNOUGHT_TinyImageNet_Full_Run.ipynb
```

---

# Breakdown Notebooks

The `docs/` directory contains notebooks explaining different sections of the architecture.

The breakdown notebooks are intended to make the architecture easier to inspect without requiring the reader to work through the entire full-scale experiment.

They cover major components of MACRO-DREADNOUGHT, including:

- Activation behavior
- Expert routing
- Residual specialization
- Failure tracking
- Backbone design
- Entropy calculations
- Reclamation mechanisms
- Diagnostics
- Training-time intervention logic

---

# Full Tiny ImageNet Run

The repository also contains the executed full experiment:

```text
MACRO_DREADNOUGHT_TinyImageNet_Full_Run.ipynb
```

This notebook contains the complete Tiny ImageNet training experiment and its recorded outputs.

The full run was performed for **50 epochs on a single NVIDIA Tesla T4**.

Readers interested primarily in the architecture can begin with the breakdown notebooks.

Readers interested in the complete experimental run can inspect the full notebook directly.

---

# Design Philosophy

MACRO-DREADNOUGHT was not created around the question:

> **How can I obtain the highest possible benchmark score?**

The project instead began with questions such as:

> What happens when experts are deliberately given different information domains?

> What happens if routing uncertainty becomes an internal control signal?

> Can difficult failures be retained and later used to influence an intervention?

> Can information rejected at one depth still become useful later?

> Can a stagnant component be selectively pushed into a new parameter region without restarting the rest of the network?

These mechanisms may or may not ultimately prove useful under controlled evaluation.

Determining that experimentally is part of the purpose of the project.

---

# Summary

MACRO-DREADNOUGHT combines several experimental ideas into a single trainable architecture:

- **Three asymmetric expert paths**
  - Anchor
  - Sniper
  - Ghost
- **Entropy-aware routing**
- **70/30 exploitation-exploration routing**
- **SpLR_V2**
- **Failure memory**
- **Targeted expert reinitialization**
- **Temporal gating**
- **Forensic feature reclamation**
- **Training-time behavioral diagnostics**

At its foundation, the model still uses ordinary gradient-based learning.

The difference is that MACRO-DREADNOUGHT experiments with allowing information about the network's own internal behavior to influence what happens around that optimization process.

The broader research question is:

> **Can neural networks benefit from mechanisms that do more than update parameters — mechanisms that also observe, preserve, reroute, remember, and selectively intervene?**

That is what MACRO-DREADNOUGHT is built to explore.

---

# Author

**Mohammad AL-Biltaji**

AI & Data Science  
Independent AI Research

GitHub: [MohammadALBiltaji](https://github.com/MohammadALBiltaji)

---

# License

This project is released under the terms provided in the repository's `LICENSE` file.
