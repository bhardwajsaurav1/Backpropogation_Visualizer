/**
 * Micro-Step Execution Controller & Calculus Formula Inspector.
 */

class StepDebugger {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.currentStep = 0;
    this.steps = [
      {
        title: "Step 1: Forward Affine Transformation",
        formula: "Z^[l] = A^[l-1] · W^[l] + b^[l]",
        desc: "Each neuron computes the linear weighted combination of the activations from the preceding layer plus its bias.",
        stage: "forward"
      },
      {
        title: "Step 2: Non-Linear Activation Function",
        formula: "A^[l] = σ(Z^[l]) / ReLU(Z^[l]) / Swish(Z^[l])",
        desc: "The linear combination Z is transformed through a non-linear activation function, enabling the network to learn non-linear decision boundaries.",
        stage: "forward"
      },
      {
        title: "Step 3: Loss Function & Output Error Calculation",
        formula: "L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]  =>  δ^[L] = ∂L/∂Z^[L] = ŷ - y",
        desc: "Calculates the discrepancy between predicted probability ŷ and true target y. For Sigmoid+BCE or Softmax+CCE, the pre-activation gradient simplifies beautifully to (ŷ - y).",
        stage: "loss"
      },
      {
        title: "Step 4: Backward Propagation & The Chain Rule",
        formula: "∂L/∂W^[l] = (A^[l-1])^T · δ^[l]   |   δ^[l-1] = (δ^[l] · (W^[l])^T) ⊙ f'(Z^[l-1])",
        desc: "The gradient of the loss flows backward from output to input. For every weight W_ij, the gradient is the upstream error δ_j multiplied by the incoming activation a_i.",
        stage: "backward"
      },
      {
        title: "Step 5: Optimizer Parameter Update",
        formula: "W^[l] ← W^[l] - η · m_t / (sqrt(v_t) + ε)",
        desc: "The weights and biases are updated using the optimizer's momentum and second moments to descend the loss surface.",
        stage: "update"
      }
    ];
  }

  setStep(stepIndex, sampleTrace = null) {
    this.currentStep = Math.max(0, Math.min(this.steps.length - 1, stepIndex));
    this.render(sampleTrace);
  }

  next(sampleTrace = null) {
    this.setStep((this.currentStep + 1) % this.steps.length, sampleTrace);
    return this.currentStep;
  }

  prev(sampleTrace = null) {
    this.setStep((this.currentStep - 1 + this.steps.length) % this.steps.length, sampleTrace);
    return this.currentStep;
  }

  render(sampleTrace = null) {
    const s = this.steps[this.currentStep];

    let liveValuesHtml = '';
    if (sampleTrace && sampleTrace.layers && sampleTrace.layers[0]) {
      const l0 = sampleTrace.layers[0];
      liveValuesHtml = `
        <div style="margin-top:10px; background:#070a10; border:1px solid #1e2638; border-radius:6px; padding:10px; font-size:11px;">
          <div style="color:#38bdf8; font-weight:600; margin-bottom:4px;">Live Single-Sample Numerical Probe:</div>
          <div style="color:#94a3b8;">Sample Loss: <span style="color:#f59e0b;">${sampleTrace.loss?.toFixed(5) || 0}</span></div>
          <div style="color:#94a3b8;">Output ŷ: <span style="color:#38bdf8;">${sampleTrace.y_pred?.[0]?.[0]?.toFixed(4) || 0}</span> | True y: <span style="color:#10b981;">${sampleTrace.y?.[0]?.[0] || 0}</span></div>
        </div>
      `;
    }

    this.container.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <span style="font-size:12px; font-weight:700; color:#38bdf8; text-transform:uppercase; letter-spacing:0.05em;">${s.title}</span>
        <span style="font-size:11px; font-family:var(--font-mono); color:#64748b;">Step ${this.currentStep + 1} of ${this.steps.length}</span>
      </div>
      <p style="font-size:12.5px; color:#cbd5e1; line-height:1.5; margin-bottom:10px;">${s.desc}</p>
      <div class="formula-box">
        <code style="color:#38bdf8;">${s.formula}</code>
      </div>
      ${liveValuesHtml}
    `;
  }
}
