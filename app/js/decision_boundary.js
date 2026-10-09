/**
 * 2D Decision Boundary Contour & Custom Interactive Data Painter.
 */

class DecisionBoundaryRenderer {
  constructor(canvasId, options = {}) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.gridResolution = options.gridResolution || 64;
    this.drawMode = false;
    this.activeDrawClass = 0;
    this.onDataModified = null;

    this.initEvents();
  }

  initEvents() {
    this.canvas.addEventListener('mousedown', (e) => this.handlePointer(e));
    this.canvas.addEventListener('mousemove', (e) => {
      if (e.buttons === 1) this.handlePointer(e);
    });
  }

  handlePointer(e) {
    if (!this.drawMode || !this.dataset) return;
    const rect = this.canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    // Convert pixel to data space [-2.5, 2.5]
    const x1 = ((px / rect.width) * 5.0) - 2.5;
    const x2 = -(((py / rect.height) * 5.0) - 2.5);

    this.dataset.X.push([x1, x2]);
    this.dataset.y.push(this.activeDrawClass === 0 ? [0] : [1]);

    if (this.onDataModified) {
      this.onDataModified(this.dataset);
    }
  }

  render(network, dataset) {
    this.dataset = dataset;
    const ctx = this.ctx;
    const width = this.canvas.width;
    const height = this.canvas.height;
    ctx.clearRect(0, 0, width, height);

    if (!network || !dataset) return;

    // 1. Render Decision Boundary Grid Heatmap
    const res = this.gridResolution;
    const imgData = ctx.createImageData(width, height);
    const data = imgData.data;

    // Sample grid points
    const span = 2.5;
    const gridInputs = [];
    for (let r = 0; r < res; r++) {
      const yVal = span - (r / (res - 1)) * (span * 2);
      for (let c = 0; c < res; c++) {
        const xVal = -span + (c / (res - 1)) * (span * 2);
        gridInputs.push([xVal, yVal]);
      }
    }

    const preds = network.forward(gridInputs);

    // Upscale grid predictions to image data
    for (let py = 0; py < height; py++) {
      const rGrid = Math.min(res - 1, Math.floor((py / height) * res));
      for (let px = 0; px < width; px++) {
        const cGrid = Math.min(res - 1, Math.floor((px / width) * res));
        const idx = rGrid * res + cGrid;
        const p = preds[idx];

        let r = 10, g = 13, b = 20;

        if (network.lossType === 'cce') {
          // Multi-class color blend
          const p0 = p[0] || 0;
          const p1 = p[1] || 0;
          const p2 = p[2] || 0;
          r = Math.floor(56 * p0 + 245 * p1 + 139 * p2);
          g = Math.floor(189 * p0 + 158 * p1 + 92 * p2);
          b = Math.floor(248 * p0 + 11 * p1 + 246 * p2);
        } else {
          // Binary classification (prob 0 = cyan, prob 1 = amber)
          const prob = p[0];
          // Cyan: [56, 189, 248] vs Amber: [245, 158, 11]
          r = Math.floor((1 - prob) * 14 + prob * 245 * 0.4);
          g = Math.floor((1 - prob) * 189 * 0.4 + prob * 158 * 0.3);
          b = Math.floor((1 - prob) * 248 * 0.4 + prob * 11 * 0.1);
        }

        const pixelIdx = (py * width + px) * 4;
        data[pixelIdx] = r;
        data[pixelIdx + 1] = g;
        data[pixelIdx + 2] = b;
        data[pixelIdx + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);

    // 2. Draw Subtle Axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(width / 2, height);
    ctx.stroke();

    // 3. Draw Dataset Points
    const X = dataset.X;
    const y = dataset.y;

    for (let i = 0; i < X.length; i++) {
      const ptX = ((X[i][0] + span) / (span * 2)) * width;
      const ptY = ((-X[i][1] + span) / (span * 2)) * height;
      const label = Array.isArray(y[i]) ? y[i][0] : y[i];

      ctx.beginPath();
      ctx.arc(ptX, ptY, 4.5, 0, Math.PI * 2);

      let color = '#38bdf8'; // Class 0
      if (label === 1 || label >= 0.5) color = '#f59e0b'; // Class 1
      if (label === 2) color = '#8b5cf6'; // Class 2

      ctx.fillStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 6;
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.2;
      ctx.stroke();
      ctx.shadowBlur = 0;
    }
  }
}
