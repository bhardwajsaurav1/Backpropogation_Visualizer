/**
 * Computational Graph Renderer with Particle Signal Waves & Micro-Calculus Hover Inspector.
 */

class ComputationalGraphRenderer {
  constructor(containerId, tooltipId) {
    this.container = document.getElementById(containerId);
    this.tooltip = document.getElementById(tooltipId);
    this.particles = [];
    this.animationMode = 'forward'; // 'forward', 'backward', 'idle'
    this.selectedElement = null;
    this.initCanvas();
  }

  initCanvas() {
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.container.innerHTML = '';
    this.container.appendChild(this.canvas);
    this.resize();
    window.addEventListener('resize', () => this.resize());

    // Mouse interaction for tooltip
    this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
    this.canvas.addEventListener('mouseleave', () => this.hideTooltip());
  }

  resize() {
    const rect = this.container.getBoundingClientRect();
    this.width = rect.width || 600;
    this.height = rect.height || 420;
    this.canvas.width = this.width * window.devicePixelRatio;
    this.canvas.height = this.height * window.devicePixelRatio;
    this.canvas.style.width = `${this.width}px`;
    this.canvas.style.height = `${this.height}px`;
    this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  }

  computeNodePositions(layerSizes) {
    const positions = [];
    const numLayers = layerSizes.length;
    const paddingX = 70;
    const paddingY = 50;
    const layerSpacing = (this.width - paddingX * 2) / (numLayers - 1);

    for (let l = 0; l < numLayers; l++) {
      const numNodes = layerSizes[l];
      const layerNodes = [];
      const nodeSpacing = (this.height - paddingY * 2) / Math.max(1, numNodes - 1);

      for (let n = 0; n < numNodes; n++) {
        const x = paddingX + l * layerSpacing;
        const y = numNodes === 1 ? this.height / 2 : paddingY + n * nodeSpacing;
        layerNodes.push({ x, y, layer: l, index: n, radius: 14 });
      }
      positions.push(layerNodes);
    }
    this.nodePositions = positions;
    return positions;
  }

  spawnSignalParticles(network, direction = 'forward') {
    if (!this.nodePositions) return;
    const numLayers = this.nodePositions.length;

    for (let l = 0; l < numLayers - 1; l++) {
      const srcLayerIdx = direction === 'forward' ? l : l + 1;
      const dstLayerIdx = direction === 'forward' ? l + 1 : l;
      const srcNodes = this.nodePositions[srcLayerIdx];
      const dstNodes = this.nodePositions[dstLayerIdx];

      for (let i = 0; i < srcNodes.length; i++) {
        for (let j = 0; j < dstNodes.length; j++) {
          if (Math.random() < 0.25) {
            this.particles.push({
              src: srcNodes[i],
              dst: dstNodes[j],
              progress: 0,
              speed: 0.025 + Math.random() * 0.02,
              direction,
              color: direction === 'forward' ? '#38bdf8' : '#f59e0b'
            });
          }
        }
      }
    }
  }

  render(network, sampleTrace = null) {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    if (!network) return;
    const layerSizes = network.layerSizes;
    const nodePositions = this.computeNodePositions(layerSizes);

    // 1. Draw Synaptic Edges (Weights)
    for (let l = 0; l < network.layers.length; l++) {
      const layer = network.layers[l];
      const srcNodes = nodePositions[l];
      const dstNodes = nodePositions[l + 1];

      for (let i = 0; i < layer.inDim; i++) {
        for (let j = 0; j < layer.outDim; j++) {
          const w = layer.W[i][j];
          const src = srcNodes[i];
          const dst = dstNodes[j];

          ctx.beginPath();
          ctx.moveTo(src.x, src.y);
          ctx.lineTo(dst.x, dst.y);

          // Edge style based on weight sign & magnitude
          const alpha = Math.min(0.85, Math.max(0.12, Math.abs(w) * 0.7));
          const color = w >= 0 ? `rgba(56, 189, 248, ${alpha})` : `rgba(244, 63, 94, ${alpha})`;
          ctx.strokeStyle = color;
          ctx.lineWidth = Math.min(5, Math.max(1, Math.abs(w) * 2.5));
          ctx.stroke();
        }
      }
    }

    // 2. Draw Moving Particles
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.progress += p.speed;

      const curX = p.src.x + (p.dst.x - p.src.x) * p.progress;
      const curY = p.src.y + (p.dst.y - p.src.y) * p.progress;

      ctx.beginPath();
      ctx.arc(curX, curY, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.shadowColor = p.color;
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.shadowBlur = 0;

      if (p.progress >= 1.0) {
        this.particles.splice(i, 1);
      }
    }

    // 3. Draw Nodes (Neurons)
    for (let l = 0; l < nodePositions.length; l++) {
      const nodes = nodePositions[l];
      const isInput = l === 0;
      const isOutput = l === nodePositions.length - 1;

      for (let n = 0; n < nodes.length; n++) {
        const node = nodes[n];
        let valText = "";
        let fillStyle = '#111622';
        let strokeStyle = '#334155';

        if (sampleTrace && sampleTrace.layers) {
          if (isInput && sampleTrace.x && sampleTrace.x[0]) {
            valText = sampleTrace.x[0][n]?.toFixed(2) || "";
            fillStyle = '#0f172a';
            strokeStyle = '#38bdf8';
          } else if (!isInput) {
            const lIdx = l - 1;
            const actVal = sampleTrace.layers[lIdx]?.A?.[0]?.[n];
            if (actVal !== undefined) {
              valText = actVal.toFixed(2);
              const intensity = Math.min(1, Math.max(0, actVal));
              fillStyle = isOutput ? `rgba(139, 92, 246, ${0.3 + intensity * 0.7})` : `rgba(56, 189, 248, ${0.2 + intensity * 0.8})`;
              strokeStyle = isOutput ? '#a78bfa' : '#38bdf8';
            }
          }
        }

        // Draw Outer Glow & Circle
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fillStyle = fillStyle;
        ctx.strokeStyle = strokeStyle;
        ctx.lineWidth = 2.5;
        ctx.fill();
        ctx.stroke();

        // Node Label
        ctx.fillStyle = '#f8fafc';
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(valText || `a${l}_${n}`, node.x, node.y);

        // Layer Title below layer column
        if (n === nodes.length - 1) {
          ctx.fillStyle = '#64748b';
          ctx.font = '11px "Inter", sans-serif';
          let lTitle = isInput ? 'Input X' : isOutput ? 'Output ŷ' : `Hidden ${l}`;
          ctx.fillText(lTitle, node.x, this.height - 14);
        }
      }
    }
  }

  handleMouseMove(e) {
    if (!this.nodePositions) return;
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    let hoveredNode = null;
    for (let l = 0; l < this.nodePositions.length; l++) {
      for (let n = 0; n < this.nodePositions[l].length; n++) {
        const node = this.nodePositions[l][n];
        const dist = Math.hypot(node.x - mouseX, node.y - mouseY);
        if (dist <= node.radius + 4) {
          hoveredNode = node;
          break;
        }
      }
      if (hoveredNode) break;
    }

    if (hoveredNode) {
      this.showNodeTooltip(hoveredNode, e.clientX, e.clientY);
    } else {
      this.hideTooltip();
    }
  }

  showNodeTooltip(node, clientX, clientY) {
    const l = node.layer;
    const n = node.index;
    this.tooltip.style.display = 'block';
    this.tooltip.style.left = `${clientX + 14}px`;
    this.tooltip.style.top = `${clientY + 14}px`;

    let html = `<strong>Neuron [Layer ${l}, Index ${n}]</strong><br/>`;
    html += `<span style="color:#94a3b8">Role: ${l === 0 ? 'Input Feature' : 'Hidden/Output Unit'}</span><br/>`;
    html += `<div style="margin-top:4px;border-top:1px dashed #334155;padding-top:4px;">`;
    html += `Forward: <code>a = f(z) = f(W·x + b)</code><br/>`;
    html += `Gradient: <code>δ = ∂L/∂z = ∂L/∂a · f'(z)</code>`;
    html += `</div>`;
    this.tooltip.innerHTML = html;
  }

  hideTooltip() {
    this.tooltip.style.display = 'none';
  }
}
