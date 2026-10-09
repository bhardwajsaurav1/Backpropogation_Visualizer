# 🧠 NeuroFlow Studio: Neural Network & Backpropagation Visualizer & Analyser

> A high-performance, interactive, educational, and analytical neural network visualization platform built from scratch with zero black-box autograd engines.

🚀 **Live Interactive Demo:** [https://backpropagation-visualizer-kohl.vercel.app](https://backpropagation-visualizer-kohl.vercel.app)

---

## 🌟 Why NeuroFlow Studio?

Most modern deep learning practitioners rely on `loss.backward()` and trust black-box automatic differentiation frameworks without fully understanding the underlying matrix calculus, gradient dynamics, and failure modes.

**NeuroFlow Studio** implements every forward tensor operation, analytical gradient, chain-rule propagation, and parameter update explicitly by hand in pure NumPy (and pure JavaScript for instant 60fps web browser interactivity).

---

## ✨ Features Breakdown

### 1. 🌐 Flexible & Arbitrary Multi-Layer Perceptron (MLP)
- Build networks of arbitrary depth (1 to 6+ layers) and widths ($1-64$ neurons per layer).
- Choose from 9 hand-derived activation functions:
  - **ReLU**, **LeakyReLU** ($\alpha=0.01$), **GELU** (Gaussian Error Linear Unit), **Swish/SiLU**, **ELU**, **Tanh**, **Sigmoid**, **Softmax**, **Linear**.
- Multiple weight initialization schemes:
  - **He / Kaiming Normal & Uniform** (optimal for ReLU/GELU)
  - **Xavier / Glorot Normal & Uniform** (optimal for Sigmoid/Tanh)
  - **Standard Normal**
  - **Zero Initialization** (demonstrating the symmetry breaking problem)
  - **Constant Initialization**

### 2. ⚡ First-Principles Optimizers & Regularization
- **Vanilla SGD**, **SGD + Polyak Momentum**, **Nesterov Accelerated Gradient (NAG)**, **RMSprop**, **Adam**, **AdamW** (Decoupled Weight Decay).
- Regularization: **L1 Penalty**, **L2 Weight Decay**, and **Inverted Dropout** with live mask visualization.

### 3. 📐 Interactive Computational Graph & Micro-Calculus Inspector
- Real-time animated signal particle waves flowing through synaptic weights.
- Hover or click on any neuron/edge to inspect:
  - Pre-activation $z = Wx + b$
  - Activation $a = f(z)$
  - Local derivative $f'(z)$
  - Error gradient $\delta = \frac{\partial \mathcal{L}}{\partial z}$
  - Weight gradient $\frac{\partial \mathcal{L}}{\partial W_{ij}} = a_i \cdot \delta_j$
- Step-by-Step Micro-Pass Walkthrough: Step through affine transformation, non-linear activation, loss computation, backpropagation chain rule, and parameter update.

### 4. 🗺️ 2D Real-Time Decision Boundary & Custom Data Painter
- 60fps canvas shader contour map.
- Interactive **Paint Custom Points** tool: Click or drag directly on the 2D plane to draw custom datasets and watch the network dynamically learn custom geometries!
- Built-in synthetic datasets: **Moons**, **Circles**, **Archimedean Spiral (2 & 3 Class)**, **XOR Quadrants**, **Gaussian Blobs**, and **Sine Wave Regression**.

### 5. 🩺 Deep Gradient Flow & Health Analyser Suite
- **Vanishing / Exploding Gradient Detector**: Monitors the ratio between early layer gradients and output layer gradients ($\frac{||\nabla W_1||_2}{||\nabla W_L||_2}$).
- **Dead Neuron Matrix**: Real-time monitor tracking dying ReLU neurons and activation saturation.
- **Saliency Map**: Computes $\frac{\partial \mathcal{L}}{\partial X}$ across the 2D coordinate plane to visualize the input sensitivity of the model.
- **Actionable Diagnosis**: Automatic recommendations (e.g. suggesting activation switch or learning rate adjustment).

### 6. 🏔️ 3D Loss Landscape & Trajectory Surface
- Real-time 3D projected wireframe & shaded loss surface $L(\alpha, \beta)$ around the network's weights.
- Interactive rotation, pitch, and zoom controls.
- Visualizes the full optimization path of the model descending through ravines, valleys, and saddle points.

### 7. 🧪 100% Mathematically Verified via Finite-Difference Gradient Checking
- Automated tests compare all analytical gradients $\nabla_{\theta} \mathcal{L}$ against numerical central difference approximations:
  $$\frac{\partial \mathcal{L}}{\partial \theta_i} \approx \frac{\mathcal{L}(\theta_i + \epsilon) - \mathcal{L}(\theta_i - \epsilon)}{2\epsilon}$$
- Guaranteed numerical exactness within $< 10^{-6}$ relative error.

---

## 🚀 Quickstart

### Running the Web Visualizer Locally

```bash
# Using Node.js
node app/server.js

# OR Using Python
python app/demo_server.py
```

Then open `http://localhost:8080` in your web browser!

---

### Running the Pure NumPy Engine in Python

```bash
# 1. Run the test suite (verifying math & convergence)
python -m pytest -v

# 2. Run the quickstart demo on the spiral dataset
python examples/quickstart.py

# 3. Run failure-case diagnostic comparisons (vanishing gradients vs healthy flow)
python examples/run_diagnostics.py
```

---

## 📁 Project Structure

```
Backpropogation visualizer/
├── README.md                           # Documentation & Calculus Guide
├── requirements.txt                    # Python dependencies
├── core/                               # Pure NumPy First-Principles Neural Engine
│   ├── __init__.py
│   ├── activations.py                  # 9 activations & exact analytical derivatives
│   ├── losses.py                       # BCE, CCE, MSE, Huber loss functions
│   ├── optimizers.py                   # SGD, Momentum, NAG, RMSProp, Adam, AdamW
│   ├── layer.py                        # DenseLayer & DropoutLayer with cache
│   ├── network.py                      # MultiLayerPerceptron orchestration & telemetry
│   ├── analyser.py                     # Gradient flow, dead neurons & saliency analyser
│   ├── landscape.py                    # Filter-normalized loss landscape sampler
│   └── datasets.py                     # Synthetic 2D dataset generators
├── tests/
│   ├── test_grad_check.py              # Finite-difference gradient checking tests
│   ├── test_network.py                 # Non-linear convergence tests (XOR, Moons, Spiral)
│   └── test_optimizers.py              # Convex quadratic optimizer tests
├── app/                                # Standalone 60fps Interactive Web App
│   ├── index.html                      # Single-page cyberpunk dashboard
│   ├── css/styles.css                  # Modern glassmorphism & glowing animations
│   ├── js/
│   │   ├── math_engine.js              # Pure JS matrix/tensor calculus engine
│   │   ├── computational_graph.js      # SVG/Canvas graph & particle signal renderer
│   │   ├── decision_boundary.js        # 60fps 2D shader & custom data painter
│   │   ├── loss_landscape.js           # Interactive 3D loss surface renderer
│   │   ├── analyser_dashboard.js       # Live health, gradient norms & dead ReLU matrix
│   │   ├── step_debugger.js            # Micro-step inspector with exact equations
│   │   └── main.js                     # State manager & training loop
│   ├── demo_server.py                  # Python static server
│   └── server.js                       # Node.js static server
└── examples/
    ├── quickstart.py                   # Complete Python usage demonstration
    └── run_diagnostics.py              # Vanishing gradient vs healthy flow comparison
```

---

## 📐 Mathematical Formulation

### Forward Propagation
For layer $l \in \{1, \dots, L\}$:
$$Z^{(l)} = A^{(l-1)} W^{(l)} + b^{(l)}$$
$$A^{(l)} = f^{(l)}(Z^{(l)})$$
where $A^{(0)} = X \in \mathbb{R}^{m \times d_{\text{in}}}$.

### Backward Propagation
Output layer error for Sigmoid + BCE or Softmax + CCE:
$$\delta^{(L)} = \frac{\partial \mathcal{L}}{\partial Z^{(L)}} = \frac{1}{m} (\hat{Y} - Y)$$

For hidden layers $l = L-1, \dots, 1$:
$$\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial Z^{(l)}} = (\delta^{(l+1)} (W^{(l+1)})^T) \odot (f^{(l)})'(Z^{(l)})$$

Weight and bias gradients:
$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = (A^{(l-1)})^T \delta^{(l)} + \lambda W^{(l)}$$
$$\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \sum_{i=1}^m \delta_i^{(l)}$$

---

## 📜 License
MIT License. Built for deep learning research, education, and intuitive mastery of backpropagation.
