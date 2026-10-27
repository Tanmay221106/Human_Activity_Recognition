// =============================================================================
// HUMAN ACTIVITY RECOGNITION (HAR) — FRONTEND CONTROLLER
// Clean, intuitive workflow: Upload CSV -> Instant Activity Classification
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const uploadZone = document.getElementById('upload-zone');
  const fileInput = document.getElementById('file-input');
  const btnAnalyze = document.getElementById('btn-analyze');
  const activeFileLabel = document.getElementById('active-file-label');
  const sampleBtns = document.querySelectorAll('.sample-btn');

  // Hero Result Elements
  const activityIcon = document.getElementById('activity-icon');
  const primaryActivityName = document.getElementById('primary-activity-name');
  const confidenceVal = document.getElementById('confidence-val');
  const statDuration = document.getElementById('stat-duration');
  const statSamples = document.getElementById('stat-samples');
  const statWindows = document.getElementById('stat-windows');
  const statSecurity = document.getElementById('stat-security');
  const distributionBars = document.getElementById('distribution-bars');
  const secHash = document.getElementById('sec-hash');

  // Waveform Elements
  const canvas = document.getElementById('waveform-canvas');
  const ctx = canvas.getContext('2d');
  const tooltip = document.getElementById('waveform-tooltip');
  const toggleAcc = document.getElementById('toggle-acc');
  const toggleGyro = document.getElementById('toggle-gyro');
  const waveformLegend = document.getElementById('waveform-legend');

  // Modal Elements
  const btnOpenMetrics = document.getElementById('btn-open-metrics');
  const btnCloseMetrics = document.getElementById('btn-close-metrics');
  const metricsModal = document.getElementById('metrics-modal');
  const cmTable = document.getElementById('cm-table');
  const featuresList = document.getElementById('features-list');
  const metaAccuracy = document.getElementById('meta-accuracy');

  // Activity Icons Mapping
  const ACTIVITY_ICONS = {
    'walking': '🚶',
    'standing': '🧍',
    'sitting': '🪑',
    'sleeping': '🛌',
    'running': '🏃'
  };

  // State
  let currentFile = 'cleaned_sensor_data.csv';
  let uploadedBytes = null;
  let currentWaveform = null;
  let activeChannel = 'acc';

  // Initialize
  loadModelMetadata();
  analyzeFile(currentFile);

  // ─────────────────────────────────────────────────────────────────────────
  // User Actions
  // ─────────────────────────────────────────────────────────────────────────

  // Sample buttons
  sampleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      sampleBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentFile = btn.getAttribute('data-file');
      uploadedBytes = null;
      activeFileLabel.textContent = `Active file: ${currentFile}`;
      analyzeFile(currentFile);
    });
  });

  // Upload trigger
  uploadZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  });

  // Drag and drop
  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });

  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
  });

  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  // Analyze button
  btnAnalyze.addEventListener('click', () => {
    if (uploadedBytes) {
      analyzeUploadedBytes();
    } else {
      analyzeFile(currentFile);
    }
  });

  // Channel toggles
  toggleAcc.addEventListener('click', () => {
    activeChannel = 'acc';
    toggleAcc.classList.add('active');
    toggleGyro.classList.remove('active');
    drawWaveform();
  });

  toggleGyro.addEventListener('click', () => {
    activeChannel = 'gyro';
    toggleGyro.classList.add('active');
    toggleAcc.classList.remove('active');
    drawWaveform();
  });

  // Waveform canvas hover inspection
  canvas.addEventListener('mousemove', handleWaveformHover);
  canvas.addEventListener('mouseleave', () => tooltip.classList.add('hidden'));

  // Modal toggle
  btnOpenMetrics.addEventListener('click', () => metricsModal.classList.remove('hidden'));
  btnCloseMetrics.addEventListener('click', () => metricsModal.classList.add('hidden'));
  metricsModal.addEventListener('click', (e) => {
    if (e.target === metricsModal) metricsModal.classList.add('hidden');
  });

  // ─────────────────────────────────────────────────────────────────────────
  // File Upload & API Handling
  // ─────────────────────────────────────────────────────────────────────────

  function handleFileUpload(file) {
    sampleBtns.forEach(b => b.classList.remove('active'));
    activeFileLabel.textContent = `Uploaded custom file: ${file.name} (${Math.round(file.size / 1024)} KB)`;
    
    const reader = new FileReader();
    reader.onload = (evt) => {
      uploadedBytes = evt.target.result;
      analyzeUploadedBytes();
    };
    reader.readAsArrayBuffer(file);
  }

  async function analyzeFile(filename) {
    setLoading(true);
    try {
      const res = await fetch('/api/analyze-file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename })
      });
      const data = await res.json();
      setLoading(false);

      if (data.success) {
        renderResults(data);
      } else {
        alert('Could not analyze file: ' + (data.error || 'Unknown error'));
      }
    } catch (err) {
      setLoading(false);
      console.error(err);
      alert('Connection error with the local server.');
    }
  }

  async function analyzeUploadedBytes() {
    if (!uploadedBytes) return;
    setLoading(true);

    try {
      const res = await fetch('/api/analyze-upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: uploadedBytes
      });
      const data = await res.json();
      setLoading(false);

      if (data.success) {
        renderResults(data);
      } else {
        alert('Could not parse uploaded CSV: ' + (data.error || 'Please check file format'));
      }
    } catch (err) {
      setLoading(false);
      console.error(err);
      alert('Upload error. Ensure the server is running.');
    }
  }

  function setLoading(isLoading) {
    if (isLoading) {
      btnAnalyze.textContent = 'Classifying...';
      btnAnalyze.disabled = true;
    } else {
      btnAnalyze.textContent = 'Analyze Activity';
      btnAnalyze.disabled = false;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Render Results
  // ─────────────────────────────────────────────────────────────────────────

  function renderResults(data) {
    const act = (data.primary_activity || 'Unknown').toLowerCase();
    
    // 1. Hero Card
    activityIcon.textContent = ACTIVITY_ICONS[act] || '🏃';
    primaryActivityName.textContent = act;
    confidenceVal.textContent = `${data.overall_confidence}%`;

    statDuration.textContent = data.duration || '~1m';
    statSamples.textContent = `${data.total_samples.toLocaleString()} rows`;
    statWindows.textContent = data.total_windows;
    
    const isSafe = data.security && !data.security.spoofing_detected;
    statSecurity.textContent = isSafe ? 'Verified Safe' : 'Anomaly Alert';
    statSecurity.style.color = isSafe ? 'var(--text-main)' : '#d17a7a';

    // 2. Activity Breakdown Bars
    const dist = data.distribution || {};
    const activities = ['walking', 'standing', 'sitting', 'sleeping'];
    let distHtml = '';
    activities.forEach(a => {
      const pct = dist[a] || 0;
      const icon = ACTIVITY_ICONS[a] || '•';
      distHtml += `
        <div class="dist-row">
          <span class="dist-name">${icon} ${a}</span>
          <div class="dist-track">
            <div class="dist-fill" style="width: ${pct}%"></div>
          </div>
          <span class="dist-pct">${pct}%</span>
        </div>
      `;
    });
    distributionBars.innerHTML = distHtml;

    // 3. Security SHA
    secHash.textContent = (data.security && data.security.sha256) ? data.security.sha256 : '--';

    // 4. Waveform
    currentWaveform = data.waveform;
    drawWaveform();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Waveform Visualizer
  // ─────────────────────────────────────────────────────────────────────────

  function drawWaveform() {
    if (!currentWaveform || !currentWaveform.time) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padLeft = 40;
    const padRight = 16;
    const padTop = 16;
    const padBottom = 26;
    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    // Background horizontal grid lines
    ctx.strokeStyle = '#1e2634';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= 4; i++) {
      const y = padTop + (plotH / 4) * i;
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
    }
    ctx.stroke();

    // Trace definitions (Strictly restrained triple-tone)
    let traces = [];
    if (activeChannel === 'acc') {
      traces = [
        { key: 'ax', color: '#667d94', width: 1.2 },
        { key: 'ay', color: '#889eb3', width: 1.2 },
        { key: 'az', color: '#45596e', width: 1.2 },
        { key: 'acc_mag', color: '#3b82a6', width: 2.2 }
      ];
      waveformLegend.innerHTML = `
        <div class="legend-chip"><span class="legend-line" style="background:#667d94"></span> ax (g)</div>
        <div class="legend-chip"><span class="legend-line" style="background:#889eb3"></span> ay (g)</div>
        <div class="legend-chip"><span class="legend-line" style="background:#45596e"></span> az (g)</div>
        <div class="legend-chip"><span class="legend-line" style="background:#3b82a6; height: 4px;"></span> |a| Magnitude</div>
      `;
    } else {
      traces = [
        { key: 'gx', color: '#667d94', width: 1.2 },
        { key: 'gy', color: '#889eb3', width: 1.2 },
        { key: 'gz', color: '#45596e', width: 1.2 },
        { key: 'gyro_mag', color: '#3b82a6', width: 2.2 }
      ];
      waveformLegend.innerHTML = `
        <div class="legend-chip"><span class="legend-line" style="background:#667d94"></span> gx (°/s)</div>
        <div class="legend-chip"><span class="legend-line" style="background:#889eb3"></span> gy (°/s)</div>
        <div class="legend-chip"><span class="legend-line" style="background:#45596e"></span> gz (°/s)</div>
        <div class="legend-chip"><span class="legend-line" style="background:#3b82a6; height: 4px;"></span> |ω| Gyro Magnitude</div>
      `;
    }

    // Min / Max computation
    let minVal = Infinity;
    let maxVal = -Infinity;
    traces.forEach(t => {
      const arr = currentWaveform[t.key] || [];
      arr.forEach(v => {
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      });
    });

    if (minVal === Infinity) { minVal = -2; maxVal = 2; }
    if (Math.abs(maxVal - minVal) < 0.1) {
      minVal -= 1;
      maxVal += 1;
    }
    const valRange = maxVal - minVal;

    // Y-Axis labels
    ctx.fillStyle = '#576575';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const y = padTop + (plotH / 4) * i;
      const v = maxVal - (valRange / 4) * i;
      ctx.fillText(v.toFixed(1), padLeft - 6, y + 3);
    }

    // Draw traces
    const n = currentWaveform.time.length;
    traces.forEach(t => {
      const arr = currentWaveform[t.key];
      if (!arr || arr.length === 0) return;

      ctx.strokeStyle = t.color;
      ctx.lineWidth = t.width;
      ctx.beginPath();

      for (let i = 0; i < n; i++) {
        const x = padLeft + (plotW / (n - 1)) * i;
        const normY = (arr[i] - minVal) / valRange;
        const y = padTop + plotH - (normY * plotH);

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    });

    // Time labels
    ctx.textAlign = 'center';
    if (n > 0) {
      ctx.fillText(currentWaveform.time[0] || '0', padLeft, h - 8);
      if (n > 2) {
        ctx.fillText(currentWaveform.time[Math.floor(n / 2)] || '', padLeft + plotW / 2, h - 8);
      }
      ctx.fillText(currentWaveform.time[n - 1] || '', padLeft + plotW, h - 8);
    }
  }

  function handleWaveformHover(e) {
    if (!currentWaveform || !currentWaveform.time || currentWaveform.time.length === 0) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const padLeft = 40;
    const padRight = 16;
    const plotW = canvas.width - padLeft - padRight;
    const scaleRatio = canvas.width / rect.width;
    const scaledMouseX = mouseX * scaleRatio;

    if (scaledMouseX < padLeft || scaledMouseX > padLeft + plotW) {
      tooltip.classList.add('hidden');
      return;
    }

    const n = currentWaveform.time.length;
    const idx = Math.min(n - 1, Math.max(0, Math.round(((scaledMouseX - padLeft) / plotW) * (n - 1))));

    const t = currentWaveform.time[idx];
    if (activeChannel === 'acc') {
      tooltip.innerHTML = `<strong>${t}</strong><br>ax: ${currentWaveform.ax[idx]}g | ay: ${currentWaveform.ay[idx]}g | az: ${currentWaveform.az[idx]}g<br>|a|: ${currentWaveform.acc_mag[idx]}g`;
    } else {
      tooltip.innerHTML = `<strong>${t}</strong><br>gx: ${currentWaveform.gx[idx]} | gy: ${currentWaveform.gy[idx]} | gz: ${currentWaveform.gz[idx]}<br>|ω|: ${currentWaveform.gyro_mag[idx]}`;
    }
    tooltip.classList.remove('hidden');
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Modal Metadata Loading
  // ─────────────────────────────────────────────────────────────────────────

  async function loadModelMetadata() {
    try {
      const res = await fetch('/api/model-info');
      if (!res.ok) return;
      const meta = await res.json();
      
      if (meta.accuracy) {
        metaAccuracy.textContent = `${meta.accuracy}%`;
      }

      if (meta.confusion_matrix) {
        const labels = meta.confusion_matrix.labels;
        const matrix = meta.confusion_matrix.matrix;

        let cmHtml = '<thead><tr><th>Actual \\ Pred</th>';
        labels.forEach(l => { cmHtml += `<th>${l}</th>`; });
        cmHtml += '</tr></thead><tbody>';

        matrix.forEach((row, rIdx) => {
          cmHtml += `<tr><th>${labels[rIdx]}</th>`;
          row.forEach((val, cIdx) => {
            const isDiag = (rIdx === cIdx);
            cmHtml += `<td class="${isDiag ? 'diag-cell' : ''}">${val}</td>`;
          });
          cmHtml += `</tr>`;
        });
        cmHtml += '</tbody>';
        cmTable.innerHTML = cmHtml;
      }

      if (meta.top_features) {
        const maxImp = meta.top_features[0]?.importance || 0.1;
        let featHtml = '';
        meta.top_features.slice(0, 10).forEach(f => {
          const pct = (f.importance / maxImp) * 100;
          featHtml += `
            <div class="feat-row">
              <span class="feat-lbl" title="${f.feature}">${f.feature}</span>
              <div class="feat-track">
                <div class="feat-fill" style="width: ${pct}%"></div>
              </div>
              <span class="feat-num">${f.importance}</span>
            </div>
          `;
        });
        featuresList.innerHTML = featHtml;
      }
    } catch (e) {
      console.warn('Model metadata error:', e);
    }
  }
});
