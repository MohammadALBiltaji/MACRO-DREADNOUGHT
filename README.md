# MACRO-DREADNOUGHT 

Traditional Deep Learning models are fundamentally passive. They pass data blindly forward through static layers and rely purely on the law of averages during backpropagation to fix their mistakes. They suffer from mode collapse, convolutional amnesia, and rigid geometric blind spots.

**MACRO-DREADNOUGHT** is a custom Mixture of Experts (MoE) architecture built from absolute zero to reject standard backpropagation. It is a dynamic, self-mutating routing matrix that calculates its own confusion in real-time, traps the exact tensors it fails to understand, and violently rewrites its own structural DNA during runtime to hunt its failures.

📖 **Read the Master Blueprint:** The complete theoretical physics, domain segregation logic, and architectural philosophy governing this engine are documented in the [WHITEPAPER.md](WHITEPAPER.md).
## ⚙️ The Core Engine

The architecture is divided into four primary physical systems:

### 1. SpLR_V2 (The Activation Physics)
A custom, dynamic activation function: `f(x) = a * x * e^(-k x^2) + c * x`. 
Unlike standard Activation Functions, SpLR_V2 calculates its own Shannon Entropy per forward pass. It actively widens or chokes the mathematical gradient of the layer based on the network's real time confidence, acting as a localized, non linear feature selector.

### 2. HighwayLayerV3 (The 3-Lane MoE Router)
Before processing a feature map, the network pools the spatial data, calculates normalized entropy, and actively routes the tensor across three specialized lanes:
* **Lane A (The Primary):** Extracts standard, high level features.
* **Lane B (The Shadow Mutator):** Processes pure mathematical error (`x - Path A`). It is mathematically forced to learn the microscopic details the Primary Lane failed to understand.
* **Lane C (The Ghost Highway):** Uses alternating dilated convolutions to process macro level shapes and wide angle context.

### 3. The Memory Spine (Temporal Gates & Forensic Bus)
MACRO-DREADNOUGHT cures Convolutional Amnesia. Every layer contains a dynamic Sigmoid Gate (`z`) that dictates whether features should overwrite long term memory (`hidden_state`), or if they are "garbage" that should be ejected onto the **Forensic Bus** to be recycled by the wide angle Ghost Highway of the next layer. 

### 4. The DNA Mutation Engine
The network does not just use Adam Optimizer. Every few epochs, the master training loop intercepts the learning process. It evaluates the router's psychology. If the network is arrogant (high monopoly) but failing (high entropy), the engine triggers a 3 factor mutation:
1. It scrubs the weights of Lane B, forcing it to be mathematically orthogonal to Lane A.
2. It extracts the raw geometry of the hardest failed images from the localized `failed_buffer`.
3. It converts those failures into targeted mutagen, violently rewriting the DNA of the layer to pre-align its weights against the images that defeated it.

## 📚 The Breakdown Notebooks
To understand the exact physical laws governing this engine, refer to the classified deep-dives in the `docs/` folder:
* `Part 1 breakdown.ipynb`: The SpLR_V2 Activation & Entropy Calculus.
* `part 2 breakdown.ipynb`: The MoE Router, Traffic Controller, and Hit-List Buffer.
* `Part 3 breakdown.ipynb`: The Vanguard Ingestion and Alternating Dilation Spine.
* `Part 4 breakdown.ipynb`: The Proving Grounds, Custom Loss Matrix, and Runtime Evolution.

## 🚀 Quick Start
```python
import torch
from core.dreadnought import MacroDreadnought

# Initialize the 10-Layer MoE Chassis
model = MacroDreadnought(num_classes=200, depth=10, channels=256)

# Generate a dummy tensor (Batch Size 16, 3 Channels, 64x64 Image)
x = torch.randn(16, 3, 64, 64)

# Execute the Forward Pass
# Returns the class logits and the real-time entropy/routing metrics per layer
outputs, metrics = model(x)
```
## 📊 Compute Constraints & Scaling
Current benchmarks (Tiny ImageNet) were executed under strict independent compute constraints (Single Tesla T4 GPU, 50 Epochs). The DNA Mutation Engine and 70/30 Elastic Router demonstrate aggressive early-stage convergence. The architecture is mathematically unbounded and open for high compute scaling.
