/**
 * Health Analyser Dashboard: Vanishing/Exploding Gradients, Dead Neuron Grid & Diagnostics.
 */

class AnalyserDashboard {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
  }

  update(network, history) {
    if (!network || !history || !history.gradNorms || history.gradNorms.length === 0) return;

    const latestGradNorms = history.gradNorms[history.gradNorms.length - 1];
    const firstLayerGrad = latestGradNorms[0] || 0;
    const lastLayerGrad = latestGradNorms[latestGradNorms.length - 1] || 1e-12;
    const ratio = firstLayerGrad / (lastLayerGrad + 1e-12);

    let status = 'HEALTHY';
    let badgeClass = 'badge-healthy';
    let diagnosisMsg = 'Gradients are propagating effectively across all layers.';

    if (firstLayerGrad < 1e-5 && lastLayerGrad > 1e-2) {
      status = 'VANISHING GRADIENT';
      badgeClass = 'badge-danger';
      diagnosisMsg = 'Early layers receive near-zero gradients. The network is suffering from vanishing gradients.';
    } else if (Math.max(...latestGradNorms) > 100 || ratio > 1000) {
      status = 'EXPLODING GRADIENT';
      badgeClass = 'badge-danger';
      diagnosisMsg = 'Gradient norms are spiking unstable magnitudes. Consider gradient clipping or smaller learning rate.';
    } else if (ratio < 0.05) {
      status = 'GRADIENT ATTENUATION';
      badgeClass = 'badge-warning';
      diagnosisMsg = 'Early layers are receiving significantly lower gradient energy than outer layers.';
    }

    // Build Dashboard HTML
    let html = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
        <div>
          <span style="font-size:13px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.06em;">Status</span>
          <div style="margin-top:4px;"><span class="badge ${badgeClass}">${status}</span></div>
        </div>
        <div style="text-align:right;">
          <span style="font-size:12px; color:#94a3b8;">Gradient Flow Ratio (L1 / L_out):</span>
          <div style="font-family:var(--font-mono); font-size:15px; color:#38bdf8; font-weight:700;">${ratio.toFixed(4)}</div>
        </div>
      </div>

      <div style="background:#090c12; border:1px solid #1e2638; border-radius:8px; padding:12px; margin-bottom:16px;">
        <span style="font-size:12px; color:#94a3b8; font-weight:600;">Diagnosis:</span>
        <p style="font-size:13px; color:#e2e8f0; margin-top:4px; line-height:1.5;">${diagnosisMsg}</p>
      </div>

      <h4 style="font-size:13px; color:#94a3b8; margin-bottom:10px; text-transform:uppercase; letter-spacing:0.05em;">Layer Gradient Norms ||∂L/∂W||₂</h4>
      <div style="display:flex; flex-direction:column; gap:8px; margin-bottom:20px;">
    `;

    const maxNorm = Math.max(1e-6, ...latestGradNorms);
    for (let l = 0; l < latestGradNorms.length; l++) {
      const gNorm = latestGradNorms[l];
      const pct = Math.min(100, Math.max(4, (gNorm / maxNorm) * 100));
      const layerName = l === latestGradNorms.length - 1 ? `Output Layer (${network.layers[l].activation})` : `Hidden Layer ${l + 1} (${network.layers[l].activation})`;

      html += `
        <div>
          <div style="display:flex; justify-content:space-between; font-size:11px; font-family:var(--font-mono); margin-bottom:3px;">
            <span>${layerName}</span>
            <span style="color:#38bdf8;">${gNorm.toFixed(6)}</span>
          </div>
          <div style="background:#1e293b; height:8px; border-radius:4px; overflow:hidden;">
            <div style="width:${pct}%; height:100%; background:linear-gradient(90deg, #38bdf8, #818cf8); border-radius:4px;"></div>
          </div>
        </div>
      `;
    }

    html += `
      </div>
      <h4 style="font-size:13px; color:#94a3b8; margin-bottom:10px; text-transform:uppercase; letter-spacing:0.05em;">Neuron Activity & Dead ReLU Matrix</h4>
      <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px;">
    `;

    for (let l = 0; l < network.layers.length; l++) {
      const layer = network.layers[l];
      html += `<div style="background:#0c1017; border:1px solid #1e2638; border-radius:6px; padding:8px 12px; flex:1; min-width:140px;">
        <div style="font-size:11px; color:#94a3b8; font-family:var(--font-mono); margin-bottom:6px;">Layer ${l+1} (${layer.outDim} units)</div>
        <div style="display:flex; flex-wrap:wrap; gap:4px;">`;

      for (let u = 0; u < layer.outDim; u++) {
        const isDead = layer.cache && layer.cache.A && layer.activation === 'relu' && layer.cache.A[0] && layer.cache.A[0][u] === 0;
        const color = isDead ? '#ef4444' : '#10b981';
        html += `<span title="Neuron ${u+1}" style="width:12px; height:12px; border-radius:2px; background:${color}; display:inline-block;"></span>`;
      }
      html += `</div></div>`;
    }

    html += `</div>`;
    this.container.innerHTML = html;
  }
}
