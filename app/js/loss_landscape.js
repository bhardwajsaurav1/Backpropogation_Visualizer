/**
 * 3D Loss Landscape & Trajectory Surface Renderer on 2D Canvas.
 * Rotatable, zoomable 3D projection of loss contours and optimization trajectories.
 */

class LossLandscapeRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.rotX = 0.8;
    this.rotY = 0.6;
    this.zoom = 1.0;
    this.isDragging = false;
    this.lastMouse = { x: 0, y: 0 };
    this.surface = null;
    this.trajectory = [];

    this.initEvents();
  }

  initEvents() {
    this.canvas.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      this.lastMouse = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    this.canvas.addEventListener('mousemove', (e) => {
      if (!this.isDragging) return;
      const dx = e.clientX - this.lastMouse.x;
      const dy = e.clientY - this.lastMouse.y;
      this.rotY += dx * 0.01;
      this.rotX += dy * 0.01;
      this.lastMouse = { x: e.clientX, y: e.clientY };
      this.draw();
    });

    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.zoom = Math.max(0.4, Math.min(2.5, this.zoom - e.deltaY * 0.001));
      this.draw();
    });
  }

  sampleLandscape(network, dataset) {
    if (!network || !dataset) return;
    const res = 15;
    const span = 1.2;
    const grid = [];

    // Synthetic multimodal loss landscape equation reflecting network loss and non-linearities
    const baseLoss = network.history.loss[network.history.loss.length - 1] || 0.5;

    for (let i = 0; i < res; i++) {
      const row = [];
      const u = -span + (i / (res - 1)) * (span * 2);
      for (let j = 0; j < res; j++) {
        const v = -span + (j / (res - 1)) * (span * 2);
        // Multimodal function with ravines and saddle points
        const r2 = u * u + v * v;
        const lossVal = baseLoss + 0.3 * (u * u - 0.5 * v * v) * Math.sin(u * 2) + 0.4 * r2;
        row.push({ u, v, z: Math.min(3.0, Math.max(0.01, lossVal)) });
      }
      grid.push(row);
    }
    this.surface = grid;

    // Build trajectory from history
    this.trajectory = [];
    const losses = network.history.loss;
    const total = losses.length;
    for (let t = 0; t < total; t += Math.max(1, Math.floor(total / 30))) {
      const frac = t / total;
      const u = Math.cos(frac * Math.PI * 1.5) * (1.0 - frac * 0.85);
      const v = Math.sin(frac * Math.PI * 1.5) * (1.0 - frac * 0.85);
      this.trajectory.push({ u, v, z: losses[t] });
    }

    this.draw();
  }

  project(u, v, z) {
    const cx = this.canvas.width / 2;
    const cy = this.canvas.height / 2;
    const scale = 110 * this.zoom;

    // Rotate Y
    const cosY = Math.cos(this.rotY);
    const sinY = Math.sin(this.rotY);
    const x1 = u * cosY - v * sinY;
    const y1 = u * sinY + v * cosY;

    // Rotate X
    const cosX = Math.cos(this.rotX);
    const sinX = Math.sin(this.rotX);
    const y2 = y1 * cosX - (z * 0.4) * sinX;
    const z2 = y1 * sinX + (z * 0.4) * cosX;

    return {
      x: cx + x1 * scale,
      y: cy - y2 * scale,
      depth: z2
    };
  }

  draw() {
    const ctx = this.ctx;
    const width = this.canvas.width;
    const height = this.canvas.height;
    ctx.clearRect(0, 0, width, height);

    if (!this.surface) {
      ctx.fillStyle = '#64748b';
      ctx.font = '13px "Inter", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText("Click 'Train' to generate 3D Loss Landscape", width / 2, height / 2);
      return;
    }

    const res = this.surface.length;

    // 1. Draw 3D Wireframe / Shaded Polygons
    for (let i = 0; i < res - 1; i++) {
      for (let j = 0; j < res - 1; j++) {
        const p1 = this.surface[i][j];
        const p2 = this.surface[i + 1][j];
        const p3 = this.surface[i + 1][j + 1];
        const p4 = this.surface[i][j + 1];

        const s1 = this.project(p1.u, p1.v, p1.z);
        const s2 = this.project(p2.u, p2.v, p2.z);
        const s3 = this.project(p3.u, p3.v, p3.z);
        const s4 = this.project(p4.u, p4.v, p4.z);

        const avgZ = (p1.z + p2.z + p3.z + p4.z) / 4;
        const intensity = Math.min(1, Math.max(0.1, avgZ / 2.0));

        ctx.beginPath();
        ctx.moveTo(s1.x, s1.y);
        ctx.lineTo(s2.x, s2.y);
        ctx.lineTo(s3.x, s3.y);
        ctx.lineTo(s4.x, s4.y);
        ctx.closePath();

        ctx.fillStyle = `rgba(14, 165, 233, ${0.15 + (1 - intensity) * 0.35})`;
        ctx.fill();
        ctx.strokeStyle = `rgba(56, 189, 248, ${0.3 + (1 - intensity) * 0.4})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    // 2. Draw Optimization Trajectory
    if (this.trajectory.length > 1) {
      ctx.beginPath();
      for (let t = 0; t < this.trajectory.length; t++) {
        const pt = this.trajectory[t];
        const s = this.project(pt.u, pt.v, pt.z);
        if (t === 0) ctx.moveTo(s.x, s.y);
        else ctx.lineTo(s.x, s.y);
      }
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 3;
      ctx.stroke();

      // Draw point markers
      for (let t = 0; t < this.trajectory.length; t++) {
        const pt = this.trajectory[t];
        const s = this.project(pt.u, pt.v, pt.z);
        ctx.beginPath();
        ctx.arc(s.x, s.y, t === this.trajectory.length - 1 ? 5 : 2.5, 0, Math.PI * 2);
        ctx.fillStyle = t === this.trajectory.length - 1 ? '#ef4444' : '#f59e0b';
        ctx.fill();
      }
    }

    // Instructions
    ctx.fillStyle = '#64748b';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('Drag to rotate 3D • Scroll to zoom', 16, height - 16);
  }
}
