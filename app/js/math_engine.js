/**
 * NeuroFlow Math & Neural Engine (Pure JS Implementation)
 * Zero external math dependencies. Mirroring all first-principles calculus from core/.
 */

const MathEngine = {
  // --- Matrix & Vector Ops ---
  zeros(rows, cols) {
    const m = [];
    for (let i = 0; i < rows; i++) {
      m.push(new Float64Array(cols));
    }
    return m;
  },

  randn(rows, cols, scale = 1.0) {
    const m = [];
    for (let i = 0; i < rows; i++) {
      const row = new Float64Array(cols);
      for (let j = 0; j < cols; j++) {
        // Box-Muller transform
        const u1 = Math.max(1e-15, Math.random());
        const u2 = Math.random();
        const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        row[j] = z0 * scale;
      }
      m.push(row);
    }
    return m;
  },

  matmul(A, B) {
    const rA = A.length;
    const cA = A[0].length;
    const rB = B.length;
    const cB = B[0].length;
    if (cA !== rB) throw new Error(`Shape mismatch: (${rA}x${cA}) vs (${rB}x${cB})`);

    const C = [];
    for (let i = 0; i < rA; i++) {
      const row = new Float64Array(cB);
      for (let k = 0; k < cA; k++) {
        const a_ik = A[i][k];
        const b_row = B[k];
        for (let j = 0; j < cB; j++) {
          row[j] += a_ik * b_row[j];
        }
      }
      C.push(row);
    }
    return C;
  },

  transpose(A) {
    const rows = A.length;
    const cols = A[0].length;
    const T = [];
    for (let j = 0; j < cols; j++) {
      const row = new Float64Array(rows);
      for (let i = 0; i < rows; i++) {
        row[i] = A[i][j];
      }
      T.push(row);
    }
    return T;
  },

  addBias(Z, b) {
    const rows = Z.length;
    const cols = Z[0].length;
    const out = [];
    for (let i = 0; i < rows; i++) {
      const row = new Float64Array(cols);
      for (let j = 0; j < cols; j++) {
        row[j] = Z[i][j] + b[0][j];
      }
      out.push(row);
    }
    return out;
  },

  // --- Activations & Analytical Derivatives ---
  sigmoid(z) {
    const clipped = Math.max(-500, Math.min(500, z));
    return 1.0 / (1.0 + Math.exp(-clipped));
  },
  sigmoidDeriv(z) {
    const a = MathEngine.sigmoid(z);
    return a * (1.0 - a);
  },

  tanh(z) {
    return Math.tanh(z);
  },
  tanhDeriv(z) {
    const a = Math.tanh(z);
    return 1.0 - a * a;
  },

  relu(z) {
    return Math.max(0.0, z);
  },
  reluDeriv(z) {
    return z > 0.0 ? 1.0 : 0.0;
  },

  leakyRelu(z, alpha = 0.01) {
    return z > 0.0 ? z : alpha * z;
  },
  leakyReluDeriv(z, alpha = 0.01) {
    return z > 0.0 ? 1.0 : alpha;
  },

  elu(z, alpha = 1.0) {
    return z > 0.0 ? z : alpha * (Math.exp(Math.max(-500, Math.min(500, z))) - 1.0);
  },
  eluDeriv(z, alpha = 1.0) {
    return z > 0.0 ? 1.0 : alpha * Math.exp(Math.max(-500, Math.min(500, z)));
  },

  gelu(z) {
    const k = Math.sqrt(2.0 / Math.PI);
    return 0.5 * z * (1.0 + Math.tanh(k * (z + 0.044715 * Math.pow(z, 3))));
  },
  geluDeriv(z) {
    const k = Math.sqrt(2.0 / Math.PI);
    const inner = k * (z + 0.044715 * Math.pow(z, 3));
    const t = Math.tanh(inner);
    const sech2 = 1.0 - t * t;
    const dInner = k * (1.0 + 3.0 * 0.044715 * z * z);
    return 0.5 * (1.0 + t) + 0.5 * z * sech2 * dInner;
  },

  swish(z) {
    return z * MathEngine.sigmoid(z);
  },
  swishDeriv(z) {
    const s = MathEngine.sigmoid(z);
    return s + z * s * (1.0 - s);
  },

  softmax(Z_row) {
    const maxVal = Math.max(...Z_row);
    const exps = Z_row.map(v => Math.exp(v - maxVal));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / (sum + 1e-12));
  },

  // Apply activation elementwise
  applyActivation(Z, actName) {
    const rows = Z.length;
    const cols = Z[0].length;
    const A = [];

    if (actName === 'softmax') {
      for (let i = 0; i < rows; i++) {
        A.push(Float64Array.from(MathEngine.softmax(Array.from(Z[i]))));
      }
      return A;
    }

    let fn = MathEngine.relu;
    if (actName === 'sigmoid') fn = MathEngine.sigmoid;
    else if (actName === 'tanh') fn = MathEngine.tanh;
    else if (actName === 'leaky_relu') fn = MathEngine.leakyRelu;
    else if (actName === 'elu') fn = MathEngine.elu;
    else if (actName === 'gelu') fn = MathEngine.gelu;
    else if (actName === 'swish') fn = MathEngine.swish;
    else if (actName === 'linear') fn = (z) => z;

    for (let i = 0; i < rows; i++) {
      const row = new Float64Array(cols);
      for (let j = 0; j < cols; j++) {
        row[j] = fn(Z[i][j]);
      }
      A.push(row);
    }
    return A;
  },

  applyActivationDeriv(Z, actName) {
    const rows = Z.length;
    const cols = Z[0].length;
    const dZ = [];

    let derivFn = MathEngine.reluDeriv;
    if (actName === 'sigmoid') derivFn = MathEngine.sigmoidDeriv;
    else if (actName === 'tanh') derivFn = MathEngine.tanhDeriv;
    else if (actName === 'leaky_relu') derivFn = MathEngine.leakyReluDeriv;
    else if (actName === 'elu') derivFn = MathEngine.eluDeriv;
    else if (actName === 'gelu') derivFn = MathEngine.geluDeriv;
    else if (actName === 'swish') derivFn = MathEngine.swishDeriv;
    else if (actName === 'linear') derivFn = () => 1.0;
    else if (actName === 'softmax') derivFn = (z) => 1.0;

    for (let i = 0; i < rows; i++) {
      const row = new Float64Array(cols);
      for (let j = 0; j < cols; j++) {
        row[j] = derivFn(Z[i][j]);
      }
      dZ.push(row);
    }
    return dZ;
  },

  norm(M) {
    let sum = 0;
    for (let i = 0; i < M.length; i++) {
      for (let j = 0; j < M[0].length; j++) {
        sum += M[i][j] * M[i][j];
      }
    }
    return Math.sqrt(sum);
  }
};

/**
 * JS Neural Network Model supporting dynamic architecture, full backpropagation,
 * telemetry recording, and 2D decision boundary sampling.
 */
class NeuralNetworkJS {
  constructor(config = {}) {
    this.layerSizes = config.layerSizes || [2, 16, 8, 1];
    this.activations = config.activations || ['relu', 'relu', 'sigmoid'];
    this.lossType = config.lossType || 'bce';
    this.optimizer = config.optimizer || 'adam';
    this.lr = config.lr !== undefined ? config.lr : 0.03;
    this.initMode = config.initMode || 'he_normal';
    this.l2 = config.l2 || 0.0;

    this.layers = [];
    this.history = {
      epochs: [],
      loss: [],
      metric: [],
      gradNorms: [],
      layerWeights: []
    };

    this.optState = {};
    this.stepCount = 0;
    this.initNetwork();
  }

  initNetwork() {
    this.layers = [];
    for (let i = 0; i < this.layerSizes.length - 1; i++) {
      const inDim = this.layerSizes[i];
      const outDim = this.layerSizes[i + 1];
      const act = this.activations[i];

      let scale = Math.sqrt(2.0 / inDim);
      if (this.initMode === 'xavier_normal' || this.initMode === 'xavier_uniform') {
        scale = Math.sqrt(2.0 / (inDim + outDim));
      } else if (this.initMode === 'normal') {
        scale = 0.1;
      } else if (this.initMode === 'zero') {
        scale = 0.0;
      } else if (this.initMode === 'constant') {
        scale = 0.0; // Handled below
      }

      const W = MathEngine.randn(inDim, outDim, scale);
      if (this.initMode === 'constant') {
        for (let r = 0; r < inDim; r++) {
          for (let c = 0; c < outDim; c++) W[r][c] = 0.5;
        }
      }
      const b = MathEngine.zeros(1, outDim);

      this.layers.push({
        inDim,
        outDim,
        activation: act,
        W,
        b,
        grad_W: MathEngine.zeros(inDim, outDim),
        grad_b: MathEngine.zeros(1, outDim),
        cache: {}
      });
    }
  }

  forward(X, isSingle = false) {
    let out = X;
    for (let i = 0; i < this.layers.length; i++) {
      const l = this.layers[i];
      const Z = MathEngine.addBias(MathEngine.matmul(out, l.W), l.b);
      const A = MathEngine.applyActivation(Z, l.activation);
      l.cache = { X: out, Z, A };
      out = A;
    }
    return out;
  }

  backward(yPred, yTrue) {
    const m = yTrue.length;
    const lastLayerIdx = this.layers.length - 1;
    let upstreamGrad = null;

    // Simplified output layer delta for standard pairs
    const lastLayer = this.layers[lastLayerIdx];
    let isDirectDZ = false;

    if (lastLayer.activation === 'sigmoid' && this.lossType === 'bce') {
      // dL/dZ = (yPred - yTrue) / m
      const dZ = MathEngine.zeros(m, 1);
      for (let i = 0; i < m; i++) {
        dZ[i][0] = (yPred[i][0] - yTrue[i][0]) / m;
      }
      upstreamGrad = dZ;
      isDirectDZ = true;
    } else if (lastLayer.activation === 'softmax' && this.lossType === 'cce') {
      const numClasses = lastLayer.outDim;
      const dZ = MathEngine.zeros(m, numClasses);
      for (let i = 0; i < m; i++) {
        const targetClass = typeof yTrue[i] === 'number' ? yTrue[i] : yTrue[i][0];
        for (let c = 0; c < numClasses; c++) {
          const y_ic = (c === targetClass) ? 1.0 : 0.0;
          dZ[i][c] = (yPred[i][c] - y_ic) / m;
        }
      }
      upstreamGrad = dZ;
      isDirectDZ = true;
    } else {
      // MSE / Linear
      const dZ = MathEngine.zeros(m, lastLayer.outDim);
      for (let i = 0; i < m; i++) {
        for (let c = 0; c < lastLayer.outDim; c++) {
          const target = Array.isArray(yTrue[i]) ? yTrue[i][c] : yTrue[i];
          dZ[i][c] = (yPred[i][c] - target) / m;
        }
      }
      upstreamGrad = dZ;
      isDirectDZ = true;
    }

    // Backward loop
    for (let lIdx = lastLayerIdx; lIdx >= 0; lIdx--) {
      const l = this.layers[lIdx];
      const { X, Z, A } = l.cache;

      let dZ;
      if (isDirectDZ) {
        dZ = upstreamGrad;
        isDirectDZ = false;
      } else {
        const dAct = MathEngine.applyActivationDeriv(Z, l.activation);
        dZ = MathEngine.zeros(m, l.outDim);
        for (let i = 0; i < m; i++) {
          for (let j = 0; j < l.outDim; j++) {
            dZ[i][j] = upstreamGrad[i][j] * dAct[i][j];
          }
        }
      }

      // dL/dW = X.T @ dZ + L2 * W
      const XT = MathEngine.transpose(X);
      const gradW = MathEngine.matmul(XT, dZ);
      if (this.l2 > 0) {
        for (let r = 0; r < l.inDim; r++) {
          for (let c = 0; c < l.outDim; c++) gradW[r][c] += this.l2 * l.W[r][c];
        }
      }
      l.grad_W = gradW;

      // dL/db = sum(dZ, axis=0)
      const grad_b = MathEngine.zeros(1, l.outDim);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < l.outDim; j++) {
          grad_b[0][j] += dZ[i][j];
        }
      }
      l.grad_b = grad_b;

      // Upstream grad for prior layer: dL/dX = dZ @ W.T
      const WT = MathEngine.transpose(l.W);
      upstreamGrad = MathEngine.matmul(dZ, WT);

      l.cache.dZ = dZ;
      l.cache.dX = upstreamGrad;
    }

    return upstreamGrad;
  }

  updateParameters() {
    this.stepCount++;
    const t = this.stepCount;
    const lr = this.lr;
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;

    for (let lIdx = 0; lIdx < this.layers.length; lIdx++) {
      const l = this.layers[lIdx];
      const wKey = `w_${lIdx}`;
      const bKey = `b_${lIdx}`;

      if (!this.optState[wKey]) {
        this.optState[wKey] = {
          mW: MathEngine.zeros(l.inDim, l.outDim),
          vW: MathEngine.zeros(l.inDim, l.outDim),
          mb: MathEngine.zeros(1, l.outDim),
          vb: MathEngine.zeros(1, l.outDim)
        };
      }
      const st = this.optState[wKey];

      if (this.optimizer === 'adam' || this.optimizer === 'adamw') {
        for (let r = 0; r < l.inDim; r++) {
          for (let c = 0; c < l.outDim; c++) {
            const g = l.grad_W[r][c];
            st.mW[r][c] = beta1 * st.mW[r][c] + (1 - beta1) * g;
            st.vW[r][c] = beta2 * st.vW[r][c] + (1 - beta2) * g * g;
            const mHat = st.mW[r][c] / (1 - Math.pow(beta1, t));
            const vHat = st.vW[r][c] / (1 - Math.pow(beta2, t));
            l.W[r][c] -= (lr / (Math.sqrt(vHat) + eps)) * mHat;
          }
        }
        for (let c = 0; c < l.outDim; c++) {
          const g = l.grad_b[0][c];
          st.mb[0][c] = beta1 * st.mb[0][c] + (1 - beta1) * g;
          st.vb[0][c] = beta2 * st.vb[0][c] + (1 - beta2) * g * g;
          const mHat = st.mb[0][c] / (1 - Math.pow(beta1, t));
          const vHat = st.vb[0][c] / (1 - Math.pow(beta2, t));
          l.b[0][c] -= (lr / (Math.sqrt(vHat) + eps)) * mHat;
        }
      } else {
        // Momentum / SGD
        for (let r = 0; r < l.inDim; r++) {
          for (let c = 0; c < l.outDim; c++) {
            const g = l.grad_W[r][c];
            st.mW[r][c] = beta1 * st.mW[r][c] + lr * g;
            l.W[r][c] -= st.mW[r][c];
          }
        }
        for (let c = 0; c < l.outDim; c++) {
          const g = l.grad_b[0][c];
          st.mb[0][c] = beta1 * st.mb[0][c] + lr * g;
          l.b[0][c] -= st.mb[0][c];
        }
      }
    }
  }

  computeLoss(yPred, yTrue) {
    const m = yTrue.length;
    let loss = 0;
    const eps = 1e-12;

    if (this.lossType === 'bce') {
      for (let i = 0; i < m; i++) {
        const p = Math.max(eps, Math.min(1.0 - eps, yPred[i][0]));
        const y = yTrue[i][0];
        loss -= (y * Math.log(p) + (1.0 - y) * Math.log(1.0 - p));
      }
      return loss / m;
    } else if (this.lossType === 'cce') {
      for (let i = 0; i < m; i++) {
        const targetClass = typeof yTrue[i] === 'number' ? yTrue[i] : yTrue[i][0];
        const p = Math.max(eps, Math.min(1.0 - eps, yPred[i][targetClass]));
        loss -= Math.log(p);
      }
      return loss / m;
    } else {
      // MSE
      for (let i = 0; i < m; i++) {
        const target = Array.isArray(yTrue[i]) ? yTrue[i][0] : yTrue[i];
        const diff = yPred[i][0] - target;
        loss += 0.5 * diff * diff;
      }
      return loss / m;
    }
  }

  computeMetric(yPred, yTrue) {
    const m = yTrue.length;
    let correct = 0;
    if (this.lossType === 'bce') {
      for (let i = 0; i < m; i++) {
        const predClass = yPred[i][0] >= 0.5 ? 1 : 0;
        const trueClass = yTrue[i][0] >= 0.5 ? 1 : 0;
        if (predClass === trueClass) correct++;
      }
      return correct / m;
    } else if (this.lossType === 'cce') {
      for (let i = 0; i < m; i++) {
        const predClass = yPred[i].indexOf(Math.max(...yPred[i]));
        const trueClass = typeof yTrue[i] === 'number' ? yTrue[i] : yTrue[i][0];
        if (predClass === trueClass) correct++;
      }
      return correct / m;
    } else {
      return 1.0 - Math.min(1.0, this.computeLoss(yPred, yTrue));
    }
  }

  trainStep(X, y) {
    const yPred = this.forward(X);
    const loss = this.computeLoss(yPred, y);
    const metric = this.computeMetric(yPred, y);
    this.backward(yPred, y);
    this.updateParameters();

    // Log telemetry
    const layerGradNorms = this.layers.map(l => MathEngine.norm(l.grad_W));
    const layerWeights = this.layers.map(l => l.W.map(row => Array.from(row)));

    this.history.epochs.push(this.history.epochs.length + 1);
    this.history.loss.push(loss);
    this.history.metric.push(metric);
    this.history.gradNorms.push(layerGradNorms);
    this.history.layerWeights.push(layerWeights);

    return { loss, metric, gradNorms: layerGradNorms };
  }
}
