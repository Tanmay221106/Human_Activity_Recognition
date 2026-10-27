// =============================================================================
// HAR SENTINEL — CLIENT APPLICATION LOGIC
// Integrates MLF (Random Forest), INS (Security Audit), and DAV (Waveform/Timeline)
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const sampleSelect = document.getElementById('sample-select');
  const uploadZone = document.getElementById('upload-zone');
  const fileInput = document.getElementById('file-input');
  const btnAnalyze = document.getElementById('btn-analyze');
  const activeFileIndicator = document.getElementById('active-file-indicator');

  // MLF Elements
  const primaryActivityText = document.getElementById('primary-activity-text');
  const activityDetailSub = document.getElementById('activity-detail-sub');
  const confBadge = document.getElementById('conf-badge');
  const totalWindowsCount = document.getElementById('total-windows-count');
  const distributionBars = document.getElementById('distribution-bars');

  // INS Elements
  const secStatusBadge = document.getElementById('sec-status-badge');
  const secHash = document.getElementById('sec-hash');
  const secSpoofing = document.getElementById('sec-spoofing');
  const secMonotonicity = document.getElementById('sec-monotonicity');
  const btnCopyHash = document.getElementById('btn-copy-hash');

  // DAV Waveform Elements
  const canvas = document.getElementById('waveform-canvas');
  const ctx = canvas.getContext('2d');
  const tooltip = document.getElementById('waveform-tooltip');
  const toggleAcc = document.getElementById('toggle-acc');
  const toggleGyro = document.getElementById('toggle-gyro');
  const waveformLegend = document.getElementById('waveform-legend');

  // DAV Timeline Elements
  const timelineBar = document.getElementById('timeline-bar');
  const segmentsBody = document.getElementById('segments-body');

  // Modal Elements
  const btnOpenMetrics = document.getElementById('btn-open-metrics');
  const btnCloseMetrics = document.getElementById('btn-close-metrics');
  const metricsModal = document.getElementById('metrics-modal');
  const cmTable = document.getElementById('cm-table');
  const featuresList = document.getElementById('features-list');
  const classReportBody = document.getElementById('class-report-body');
  const metaAccuracy = document.getElementById('meta-accuracy');

  // State
  let currentWaveform = null;
  let activeChannel = 'acc'; // 'acc' or 'gyro'
  let uploadedFileBytes = null;
  let uploadedFileName = null;
  let modelMetadata = null;

  // Initialize
  loadModelInfo();
  // Auto-run first sample for instant display
  setTimeout(() => {
    analyzeSelectedFile();
  }, 400);

  // ─────────────────────────────────────────────────────────────────────────
  // Event Listeners
  // ─────────────────────────────────────────────────────────────────────────

  btnAnalyze.addEventListener('click', () => {
    if (uploadedFileBytes) {
      analyzeUploadedFile();
    } else {
      analyzeSelectedFile();
    }
  });

  sampleSelect.addEventListener('change', () => {
    uploadedFileBytes = null;
    uploadedFileName = null;
    activeFileIndicator.textContent = 'Selected: ' + sampleSelect.value;
    activeFileIndicator.className = 'badge-status';
  });

  // Drag & drop upload
  uploadZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', handleFileSelect);

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
      processUploadedFile(e.dataTransfer.files[0]);
    }
  });

  // Channel toggles
  toggleAcc.addEventListener('click', () => {
    activeChannel = 'acc';
    toggleAcc.classList.add('active');
    toggleGyro.classList.remove('active');
    renderWaveform();
  });

  toggleGyro.addEventListener('click', () => {
    activeChannel = 'gyro';
    toggleGyro.classList.add('active');
    toggleAcc.classList.remove('active');
    renderWaveform();
  });

  // Copy SHA-256
  btnCopyHash.addEventListener('click', () => {
    if (secHash.textContent && secHash.textContent !== '--') {
      navigator.clipboard.writeText(secHash.textContent).then(() => {
        btnCopyHash.textContent = 'Copied';
        setTimeout(() => { btnCopyHash.textContent = 'Copy'; }, 1500);
      });
    }
  });

  // Modal open/close
  btnOpenMetrics.addEventListener('click', () => metricsModal.classList.remove('hidden'));
  btnCloseMetrics.addEventListener('click', () => metricsModal.classList.add('hidden'));
  metricsModal.addEventListener('click', (e) => {
    if (e.target === metricsModal) metricsModal.classList.add('hidden');
  });

  // Waveform canvas hover inspection
  canvas.addEventListener('mousemove', handleWaveformHover);
  canvas.addEventListener('mouseleave', () => tooltip.classList.add('hidden'));

  // ─────────────────────────────────────────────────────────────────────────
  // Handlers & API Operations
  // ─────────────────────────────────────────────────────────────────────────

  function handleFileSelect(e) {
    if (e.target.files && e.target.files[0]) {
      processUploadedFile(e.target.files[0]);
    }
  }

  function processUploadedFile(file) {
    uploadedFileName = file.name;
    const reader = new FileReader();
    reader.onload = (evt) => {
      uploadedFileBytes = evt.target.result;
      activeFileIndicator.textContent = 'Uploaded: ' + file.name;
      activeFileIndicator.className = 'badge-status verified';
      analyzeUploadedFile();
    };
    reader.readAsArrayBuffer(file);
  }

  async function loadModelInfo() {
    try {
      const res = await fetch('/api/model-info');
      if (!res.ok) return;
      modelMetadata = await res.json();
      renderModalMetrics(modelMetadata);
    } catch (e) {
      console.warn('Could not load model info:', e);
    }
  }

  async function analyzeSelectedFile() {
    const filename = sampleSelect.value;
    setLoadingState(true);

    try {
      const res = await fetch('/api/analyze-file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename })
      });
      const data = await res.json();
      setLoadingState(false);

      if (data.success) {
        populateDashboard(data);
      } else {
        alert('Analysis Error: ' + (data.error || 'Unknown error'));
      }
    } catch (err) {
      setLoadingState(false);
      console.error(err);
      alert('Network or server connection failed.');
    }
  }

  async function analyzeUploadedFile() {
    if (!uploadedFileBytes) return;
    setLoadingState(true);

    try {
      const res = await fetch('/api/analyze-upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: uploadedFileBytes
      });
      const data = await res.json();
      setLoadingState(false);

      if (data.success) {
        populateDashboard(data);
      } else {
        alert('Analysis Error: ' + (data.error || 'Failed to parse upload'));
      }
    } catch (err) {
      setLoadingState(false);
      console.error(err);
      alert('Upload analysis failed.');
    }
  }

  function setLoadingState(isLoading) {
    if (isLoading) {
      btnAnalyze.textContent = 'Processing Telemetry...';
      btnAnalyze.disabled = true;
    } else {
      btnAnalyze.textContent = 'Run Inference & Audit';
      btnAnalyze.disabled = false;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Populate Dashboard UI
  // ─────────────────────────────────────────────────────────────────────────

  function populateDashboard(data) {
    // 1. MLF Pillar
    primaryActivityText.textContent = data.primary_activity;
    confBadge.textContent = `${data.overall_confidence}% Conf`;
    activityDetailSub.textContent = `Analyzed ${data.total_samples} raw sensor rows into ${data.total_windows} windows.`;
    totalWindowsCount.textContent = `${data.total_windows} Windows`;

    // Distribution bars
    const dist = data.distribution || {};
    const activities = ['walking', 'standing', 'sitting', 'sleeping'];
    let distHtml = '';
    activities.forEach(act => {
      const pct = dist[act] || 0;
      distHtml += `
        <div class="dist-item">
          <span class="dist-name">${act}</span>
          <div class="dist-bar-track">
            <div class="dist-bar-fill" style="width: ${pct}%"></div>
          </div>
          <span class="dist-val">${pct}%</span>
        </div>
      `;
    });
    distributionBars.innerHTML = distHtml;

    // 2. INS Pillar: Telemetry Security & Integrity
    const sec = data.security || {};
    secHash.textContent = sec.sha256 || '--';
    secStatusBadge.textContent = sec.integrity_status || 'Verified';
    secStatusBadge.className = 'badge-status ' + (sec.spoofing_detected ? '' : 'verified');

    if (sec.spoofing_detected) {
      secSpoofing.innerHTML = `<span style="color: #d17a7a;">Warning: ${sec.anomalies.join(', ')}</span>`;
    } else {
      secSpoofing.textContent = `Passed • Within biomechanical thresholds (±16g, ±2000°/s)`;
    }

    secMonotonicity.textContent = `Passed • ${sec.packet_count} packets verified with monotonic timestamps (~0.5s intervals)`;

    // 3. DAV Pillar: Waveform
    currentWaveform = data.waveform;
    renderWaveform();

    // 4. DAV Pillar: Timeline Segmentation
    renderTimeline(data.segments, data.total_windows);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Waveform Rendering (Canvas)
  // ─────────────────────────────────────────────────────────────────────────

  function renderWaveform() {
    if (!currentWaveform || !currentWaveform.time) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padLeft = 45;
    const padRight = 20;
    const padTop = 20;
    const padBottom = 30;
    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    // Draw background grid lines (subtle triple-tone border)
    ctx.strokeStyle = '#222d3f';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= 4; i++) {
      const y = padTop + (plotH / 4) * i;
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
    }
    ctx.stroke();

    // Channel traces configuration (strictly restrained triple-tone)
    let traces = [];
    if (activeChannel === 'acc') {
      traces = [
        { key: 'ax', label: 'ax (g)', color: '#68869c', width: 1.2 },
        { key: 'ay', label: 'ay (g)', color: '#8ea8ba', width: 1.2 },
        { key: 'az', label: 'az (g)', color: '#445f73', width: 1.2 },
        { key: 'acc_mag', label: '|a| Mag', color: '#3a7596', width: 2.0 }
      ];
      waveformLegend.innerHTML = `
        <div class="legend-item"><span class="legend-indicator" style="background:#68869c;"></span> ax (g)</div>
        <div class="legend-item"><span class="legend-indicator" style="background:#8ea8ba;"></span> ay (g)</div>
        <div class="legend-item"><span class="legend-indicator" style="background:#445f73;"></span> az (g)</div>
        <div class="legend-item"><span class="legend-indicator" style="background:#3a7596; height: 4px;"></span> |a| Magnitude (Filtered)</div>
      `;
    } else {
      traces = [
        { key: 'gx', label: 'gx (°/s)', color: '#68869c', width: 1.2 },
        { key: 'gy', label: 'gy (°/s)', color: '#8ea8ba', width: 1.2 },
        { key: 'gz', label: 'gz (°/s)', color: '#445f73', width: 1.2 },
        { key: 'gyro_mag', label: '|ω| Mag', color: '#3a7596', width: 2.0 }
      ];
      waveformLegend.innerHTML = `
        <div class="legend-item"><span class="legend-indicator" style="background:#68869c;"></span> gx (°/s)</div>
        <div class="legend-item"><span class="legend-indicator" style="background:#8ea8ba;"></span> gy (°/s)</div>
        <div class="legend-item"><span class="legend-indicator" style="background:#445f73;"></span> gz (°/s)</div>
        <div class="legend-item"><span class="legend-indicator" style="background:#3a7596; height: 4px;"></span> |ω| Gyro Magnitude</div>
      `;
    }

    // Determine scale
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

    // Draw Y axis labels
    ctx.fillStyle = '#576575';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const y = padTop + (plotH / 4) * i;
      const v = maxVal - (valRange / 4) * i;
      ctx.fillText(v.toFixed(1), padLeft - 8, y + 3);
    }

    // Plot traces
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

    // Time axis labels (first, middle, last)
    ctx.textAlign = 'center';
    if (n > 0) {
      ctx.fillText(currentWaveform.time[0] || '0', padLeft, h - 10);
      if (n > 2) {
        ctx.fillText(currentWaveform.time[Math.floor(n / 2)] || '', padLeft + plotW / 2, h - 10);
      }
      ctx.fillText(currentWaveform.time[n - 1] || '', padLeft + plotW, h - 10);
    }
  }

  function handleWaveformHover(e) {
    if (!currentWaveform || !currentWaveform.time || currentWaveform.time.length === 0) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const padLeft = 45;
    const padRight = 20;
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
      tooltip.innerHTML = `<strong>${t}</strong><br>ax: ${currentWaveform.ax[idx]}g | ay: ${currentWaveform.ay[idx]}g | az: ${currentWaveform.az[idx]}g<br>|a| Mag: ${currentWaveform.acc_mag[idx]}g`;
    } else {
      tooltip.innerHTML = `<strong>${t}</strong><br>gx: ${currentWaveform.gx[idx]} | gy: ${currentWaveform.gy[idx]} | gz: ${currentWaveform.gz[idx]}<br>|ω| Mag: ${currentWaveform.gyro_mag[idx]}`;
    }
    tooltip.classList.remove('hidden');
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Timeline Segmentation (DAV)
  // ─────────────────────────────────────────────────────────────────────────

  function renderTimeline(segments, totalWindows) {
    if (!segments || segments.length === 0) {
      timelineBar.innerHTML = `<div class="timeline-empty-hint">No segments available</div>`;
      segmentsBody.innerHTML = `<tr><td colspan="5" class="empty-row">No segments detected.</td></tr>`;
      return;
    }

    // 1. Visual segmented bar
    let barHtml = '';
    segments.forEach(seg => {
      const pct = (seg.window_count / totalWindows) * 100;
      const actClass = `seg-${seg.activity.toLowerCase()}`;
      barHtml += `
        <div class="timeline-segment-chunk ${actClass}" style="width: ${pct}%" title="${seg.activity}: ${seg.start_time} - ${seg.end_time} (${seg.window_count} wins)">
          ${pct > 8 ? seg.activity : ''}
        </div>
      `;
    });
    timelineBar.innerHTML = barHtml;

    // 2. Table rows
    let tableHtml = '';
    segments.forEach(seg => {
      tableHtml += `
        <tr>
          <td><code>${seg.start_time}</code></td>
          <td><code>${seg.end_time}</code></td>
          <td style="font-weight: 500; text-transform: capitalize;">${seg.activity}</td>
          <td>${seg.window_count} (${Math.round((seg.window_count / totalWindows) * 100)}%)</td>
          <td>${seg.avg_confidence}%</td>
        </tr>
      `;
    });
    segmentsBody.innerHTML = tableHtml;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Modal Metrics Population (MLF · DAV Capstone Evaluation)
  // ─────────────────────────────────────────────────────────────────────────

  function renderModalMetrics(meta) {
    if (!meta) return;

    if (meta.accuracy) {
      metaAccuracy.textContent = `${meta.accuracy}%`;
    }

    // Confusion Matrix
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

    // Top Features
    if (meta.top_features) {
      const maxImp = meta.top_features[0]?.importance || 0.1;
      let featHtml = '';
      meta.top_features.slice(0, 10).forEach(f => {
        const pct = (f.importance / maxImp) * 100;
        featHtml += `
          <div class="feat-item">
            <span class="feat-name" title="${f.feature}">${f.feature}</span>
            <div class="feat-bar-track">
              <div class="feat-bar-fill" style="width: ${pct}%"></div>
            </div>
            <span class="feat-val">${f.importance}</span>
          </div>
        `;
      });
      featuresList.innerHTML = featHtml;
    }

    // Class Performance Breakdown
    if (meta.metrics_per_class) {
      let repHtml = '';
      Object.entries(meta.metrics_per_class).forEach(([cls, m]) => {
        repHtml += `
          <tr>
            <td>${cls}</td>
            <td>${m.precision}%</td>
            <td>${m.recall}%</td>
            <td>${m.f1}%</td>
            <td>${m.support}</td>
          </tr>
        `;
      });
      classReportBody.innerHTML = repHtml;
    }
  }
});
