/**
 * Main Application Controller & UI Orchestration for NeuroFlow Studio.
 */

document.addEventListener('DOMContentLoaded', () => {
  // --- App State ---
  const state = {
    datasetName: 'moons',
    datasetNoise: 0.1,
    layerSizes: [2, 16, 8, 1],
    hiddenAct: 'relu',
    outAct: 'sigmoid',
    optimizer: 'adam',
    lr: 0.03,
    initMode: 'he_normal',
    lossType: 'bce',
    epochsTotal: 200,
    isTraining: false,
    currentEpoch: 0,
    activeTab: 'graph',
    drawMode: false
  };

  let network = null;
  let dataset = null;
  let animId = null;

  // --- Initialise Renderers ---
  const graphRenderer = new ComputationalGraphRenderer('graphCanvasContainer', 'nodeTooltip');
  const boundaryRenderer = new DecisionBoundaryRenderer('boundaryCanvas');
  const landscapeRenderer = new LossLandscapeRenderer('landscapeCanvas');
  const analyserDashboard = new AnalyserDashboard('analyserContainer');
  const stepDebugger = new StepDebugger('stepDebuggerContainer');

  // --- Dataset Generator ---
  function generateDataset(name, noise = 0.1, nSamples = 200) {
    const X = [];
    const y = [];

    if (name === 'moons') {
      const nHalf = Math.floor(nSamples / 2);
      for (let i = 0; i < nHalf; i++) {
        const theta = (i / nHalf) * Math.PI;
        X.push([Math.cos(theta) + (Math.random() - 0.5) * noise, Math.sin(theta) + (Math.random() - 0.5) * noise]);
        y.push([0]);
      }
      for (let i = 0; i < nHalf; i++) {
        const theta = (i / nHalf) * Math.PI;
        X.push([1.0 - Math.cos(theta) + (Math.random() - 0.5) * noise, 1.0 - Math.sin(theta) - 0.5 + (Math.random() - 0.5) * noise]);
        y.push([1]);
      }
    } else if (name === 'circles') {
      const nHalf = Math.floor(nSamples / 2);
      for (let i = 0; i < nHalf; i++) {
        const theta = Math.random() * Math.PI * 2;
        const r = 1.0 + (Math.random() - 0.5) * noise;
        X.push([r * Math.cos(theta), r * Math.sin(theta)]);
        y.push([0]);
      }
      for (let i = 0; i < nHalf; i++) {
        const theta = Math.random() * Math.PI * 2;
        const r = 0.45 + (Math.random() - 0.5) * noise;
        X.push([r * Math.cos(theta), r * Math.sin(theta)]);
        y.push([1]);
      }
    } else if (name === 'spiral') {
      const nHalf = Math.floor(nSamples / 2);
      for (let c = 0; c < 2; c++) {
        for (let i = 0; i < nHalf; i++) {
          const r = (i / nHalf) * 1.5;
          const theta = (i / nHalf) * 4.0 + c * Math.PI + (Math.random() - 0.5) * noise;
          X.push([r * Math.sin(theta), r * Math.cos(theta)]);
          y.push([c]);
        }
      }
    } else if (name === 'xor') {
      for (let i = 0; i < nSamples; i++) {
        const x1 = (Math.random() - 0.5) * 2.0;
        const x2 = (Math.random() - 0.5) * 2.0;
        const label = (x1 > 0) !== (x2 > 0) ? 1 : 0;
        X.push([x1 + (Math.random() - 0.5) * noise, x2 + (Math.random() - 0.5) * noise]);
        y.push([label]);
      }
    } else if (name === 'blobs') {
      const nHalf = Math.floor(nSamples / 2);
      for (let i = 0; i < nHalf; i++) {
        X.push([-0.8 + (Math.random() - 0.5) * noise * 2, -0.8 + (Math.random() - 0.5) * noise * 2]);
        y.push([0]);
      }
      for (let i = 0; i < nHalf; i++) {
        X.push([0.8 + (Math.random() - 0.5) * noise * 2, 0.8 + (Math.random() - 0.5) * noise * 2]);
        y.push([1]);
      }
    }

    // Standardize data
    const meanX = [0, 0];
    for (let i = 0; i < X.length; i++) {
      meanX[0] += X[i][0];
      meanX[1] += X[i][1];
    }
    meanX[0] /= X.length;
    meanX[1] /= X.length;

    for (let i = 0; i < X.length; i++) {
      X[i][0] = (X[i][0] - meanX[0]) * 1.2;
      X[i][1] = (X[i][1] - meanX[1]) * 1.2;
    }

    return { X, y };
  }

  // --- Reset & Re-build Model ---
  function resetModel() {
    state.isTraining = false;
    document.getElementById('playPauseBtn').innerText = '▶ Train';

    dataset = generateDataset(state.datasetName, state.datasetNoise);

    const activations = [];
    for (let i = 0; i < state.layerSizes.length - 2; i++) {
      activations.push(state.hiddenAct);
    }
    activations.push(state.outAct);

    network = new NeuralNetworkJS({
      layerSizes: state.layerSizes,
      activations: activations,
      lossType: state.lossType,
      optimizer: state.optimizer,
      lr: state.lr,
      initMode: state.initMode
    });

    state.currentEpoch = 0;
    updateUI();
    renderCurrentState();
  }

  // --- UI Updates ---
  function updateUI() {
    // Render Layer Chips
    const layerListEl = document.getElementById('layerList');
    layerListEl.innerHTML = '';
    state.layerSizes.forEach((size, idx) => {
      const chip = document.createElement('div');
      chip.className = 'layer-chip';
      const isInput = idx === 0;
      const isOutput = idx === state.layerSizes.length - 1;
      const title = isInput ? `In: ${size}` : isOutput ? `Out: ${size}` : `H${idx}: ${size}`;
      chip.innerHTML = `<span>${title}</span>`;

      if (!isInput && !isOutput && state.layerSizes.length > 3) {
        const removeBtn = document.createElement('span');
        removeBtn.className = 'remove-btn';
        removeBtn.innerText = '×';
        removeBtn.onclick = (e) => {
          e.stopPropagation();
          state.layerSizes.splice(idx, 1);
          resetModel();
        };
        chip.appendChild(removeBtn);
      }
      layerListEl.appendChild(chip);
    });

    document.getElementById('lrVal').innerText = state.lr.toFixed(3);
    document.getElementById('noiseVal').innerText = state.datasetNoise.toFixed(2);
  }

  function renderCurrentState() {
    const history = network.history;
    const curLoss = history.loss.length > 0 ? history.loss[history.loss.length - 1] : 0;
    const curMetric = history.metric.length > 0 ? history.metric[history.metric.length - 1] : 0;

    document.getElementById('statLoss').innerText = curLoss.toFixed(4);
    document.getElementById('statMetric').innerText = `${(curMetric * 100).toFixed(1)}%`;
    document.getElementById('statEpoch').innerText = `${history.epochs.length} / ${state.epochsTotal}`;

    const sampleTrace = {
      x: [dataset.X[0]],
      y: [dataset.y[0]],
      y_pred: network.layers.length ? [[network.layers[network.layers.length - 1].cache.A?.[0]?.[0] || 0]] : [[0]],
      loss: curLoss,
      layers: network.layers.map(l => ({ A: l.cache.A, dZ: l.cache.dZ, W: l.W, grad_W: l.grad_W }))
    };

    graphRenderer.render(network, sampleTrace);
    boundaryRenderer.render(network, dataset);
    analyserDashboard.update(network, history);
    stepDebugger.render(sampleTrace);
  }

  // --- Animation Loop ---
  function trainingLoop() {
    if (state.isTraining && network.history.epochs.length < state.epochsTotal) {
      // Run 2 steps per frame for smooth convergence
      for (let s = 0; s < 2; s++) {
        if (network.history.epochs.length < state.epochsTotal) {
          network.trainStep(dataset.X, dataset.y);
          graphRenderer.spawnSignalParticles(network, s % 2 === 0 ? 'forward' : 'backward');
        }
      }
      renderCurrentState();

      if (network.history.epochs.length >= state.epochsTotal) {
        state.isTraining = false;
        document.getElementById('playPauseBtn').innerText = '▶ Train';
        landscapeRenderer.sampleLandscape(network, dataset);
      }
    }

    graphRenderer.render(network);
    animId = requestAnimationFrame(trainingLoop);
  }

  // --- Attach Controls Event Handlers ---
  document.getElementById('playPauseBtn').onclick = () => {
    state.isTraining = !state.isTraining;
    document.getElementById('playPauseBtn').innerText = state.isTraining ? '⏸ Pause' : '▶ Train';
    if (state.isTraining && network.history.epochs.length >= state.epochsTotal) {
      resetModel();
      state.isTraining = true;
      document.getElementById('playPauseBtn').innerText = '⏸ Pause';
    }
  };

  document.getElementById('stepBtn').onclick = () => {
    state.isTraining = false;
    document.getElementById('playPauseBtn').innerText = '▶ Train';
    network.trainStep(dataset.X, dataset.y);
    graphRenderer.spawnSignalParticles(network, 'forward');
    graphRenderer.spawnSignalParticles(network, 'backward');
    renderCurrentState();
  };

  document.getElementById('resetBtn').onclick = () => {
    resetModel();
  };

  document.getElementById('addLayerBtn').onclick = () => {
    if (state.layerSizes.length < 6) {
      state.layerSizes.splice(state.layerSizes.length - 1, 0, 8);
      resetModel();
    }
  };

  document.getElementById('datasetSelect').onchange = (e) => {
    state.datasetName = e.target.value;
    resetModel();
  };

  document.getElementById('activationSelect').onchange = (e) => {
    state.hiddenAct = e.target.value;
    resetModel();
  };

  document.getElementById('optimizerSelect').onchange = (e) => {
    state.optimizer = e.target.value;
    if (network) network.optimizer = state.optimizer;
  };

  document.getElementById('initSelect').onchange = (e) => {
    state.initMode = e.target.value;
    resetModel();
  };

  document.getElementById('lrSlider').oninput = (e) => {
    state.lr = parseFloat(e.target.value);
    document.getElementById('lrVal').innerText = state.lr.toFixed(3);
    if (network) network.lr = state.lr;
  };

  document.getElementById('noiseSlider').oninput = (e) => {
    state.datasetNoise = parseFloat(e.target.value);
    document.getElementById('noiseVal').innerText = state.datasetNoise.toFixed(2);
    resetModel();
  };

  // Tab navigation
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const targetPane = document.getElementById(btn.dataset.tab);
      if (targetPane) targetPane.classList.add('active');

      if (btn.dataset.tab === 'tabLandscape') {
        landscapeRenderer.sampleLandscape(network, dataset);
      }
    };
  });

  // Step Debugger Buttons
  document.getElementById('stepPrevBtn').onclick = () => stepDebugger.prev();
  document.getElementById('stepNextBtn').onclick = () => stepDebugger.next();

  // Custom Point Drawing Toggle
  document.getElementById('drawToggleBtn').onclick = () => {
    state.drawMode = !state.drawMode;
    boundaryRenderer.drawMode = state.drawMode;
    document.getElementById('drawToggleBtn').innerText = state.drawMode ? '🖌️ Paint Mode (Active)' : '🖌️ Paint Custom Points';
    document.getElementById('drawToggleBtn').className = state.drawMode ? 'btn btn-amber' : 'btn';
  };

  boundaryRenderer.onDataModified = (newDataset) => {
    dataset = newDataset;
    renderCurrentState();
  };

  // Export Code Modal
  document.getElementById('exportBtn').onclick = () => {
    const code = `
# Standalone Pure NumPy Model Generated by NeuroFlow Studio
import numpy as np

# Layer Architecture: ${state.layerSizes.join(' -> ')}
class TrainedNeuroFlowModel:
    def __init__(self):
        # Weights and biases export
        self.weights = ${JSON.stringify(network.layers.map(l => l.W))}
        self.biases = ${JSON.stringify(network.layers.map(l => l.b))}
    
    def forward(self, X):
        out = np.asarray(X)
        for W, b in zip(self.weights, self.biases):
            out = np.maximum(0, out @ np.array(W) + np.array(b)) # ReLU
        return 1.0 / (1.0 + np.exp(-out)) # Sigmoid

model = TrainedNeuroFlowModel()
print("Model loaded successfully!")
    `.trim();

    document.getElementById('exportCodeBlock').innerText = code;
    document.getElementById('exportModal').classList.add('open');
  };

  document.getElementById('closeModalBtn').onclick = () => {
    document.getElementById('exportModal').classList.remove('open');
  };

  // Initial load
  resetModel();
  trainingLoop();
});
