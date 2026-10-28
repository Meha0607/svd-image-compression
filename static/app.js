// ============================================================
// SVD Image Lab - frontend logic
// Set USE_DUMMY = true to test the UI without the Flask backend.
// Set it to false when the real API is ready.
// ============================================================
const USE_DUMMY = false;

// ---------- Shared state ----------
const state = {
  imageId: null, file: null, origUrl: null,
  shape: null, sv: [], maxK: 100,
  k: 20, chart: null,
  compMode: 'manual',
  compLoaded: false, matrixLoaded: false,
  noisyB64: null, suggestedK: null, busy: false,
};

const $ = id => document.getElementById(id);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const png = b64 => 'data:image/png;base64,' + b64;   // API sends base64 PNG
const fmt = n => (typeof n === 'number' ? Math.round(n * 100) / 100 : n);
const bytes = n => n < 1024 ? `${n} B` : `${(n / 1024).toFixed(1)} KB`;

function setStatus(message, type = '') {
  const status = $('status');
  status.textContent = message;
  status.className = 'status ' + type;
}

function showSavings(id, sizes) {
  const change = sizes.bytes_saved >= 0
    ? `${bytes(sizes.bytes_saved)} saved (${sizes.percent_saved}%)`
    : `${bytes(Math.abs(sizes.bytes_saved))} larger (${Math.abs(sizes.percent_saved)}%)`;
  $(id).textContent = `Actual PNG file size: original ${bytes(sizes.original_bytes)}, processed ${bytes(sizes.processed_bytes)} — ${change}. Theoretical matrix ratio is shown separately.`;
}

// ---------- Small helpers: toast, spinner, debounce ----------
let toastTimer;
function showError(msg) {
  const t = $('toast');
  t.textContent = '⚠️ ' + msg;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 5000);
}

// Show a spinner on a panel while an async job runs; show friendly errors
async function run(panelId, job) {
  if (state.busy) return;
  state.busy = true;
  const panel = $(panelId);
  panel.classList.add('loading');
  try { await job(); }
  catch (e) { setStatus(e.message || 'Something went wrong.', 'error'); showError(e.message || 'Something went wrong. Please try again.'); }
  finally { panel.classList.remove('loading'); state.busy = false; }
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// ---------- API caller (real or dummy) ----------
async function api(path, body, method = 'POST') {
  if (USE_DUMMY) { await sleep(500); return dummyApi(path, body); }

  const opt = { method };
  let url = path;
  if (method === 'GET') url += '?' + new URLSearchParams(body);
  else if (body instanceof FormData) opt.body = body;
  else { opt.headers = { 'Content-Type': 'application/json' }; opt.body = JSON.stringify(body); }

  let res;
  try { res = await fetch(url, opt); }
  catch (e) { throw new Error("Can't reach the server. Check that Flask is running."); }

  let data = {};
  try { data = await res.json(); } catch (e) { /* no JSON body */ }
  if (!res.ok || data.error) throw new Error(data.error || 'Server error (' + res.status + ').');
  return data;
}

function uploadWithProgress(form) {
  if (USE_DUMMY) return api('/api/upload', form);
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload');
    xhr.upload.onprogress = event => {
      if (event.lengthComputable) {
        const percent = Math.round(event.loaded / event.total * 100);
        $('uploadProgress').value = percent;
        setStatus(`Uploading… ${percent}%`);
      }
    };
    xhr.onerror = () => reject(new Error("Can't reach the server. Check that Flask is running."));
    xhr.onload = () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch (_) { /* handled below */ }
      if (xhr.status < 200 || xhr.status >= 300 || data.error) reject(new Error(data.error || `Server error (${xhr.status}).`));
      else resolve(data);
    };
    xhr.send(form);
  });
}

// ============================================================
// Tabs
// ============================================================
function showTab(name) {
  document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.id === 'tab-' + name));
  if (!state.imageId) return;
  if (name === 'matrix' && !state.matrixLoaded) loadMatrix();
  if (name === 'values') drawChart();
  if (name === 'compress' && !state.compLoaded) doCompress();
}
document.querySelectorAll('.tab').forEach(b => b.addEventListener('click', () => showTab(b.dataset.tab)));

// ============================================================
// 1. Upload
// ============================================================
const drop = $('drop'), fileInput = $('fileInput');
fileInput.addEventListener('change', () => fileInput.files[0] && handleFile(fileInput.files[0]));
['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => e.dataTransfer.files[0] && handleFile(e.dataTransfer.files[0]));

function handleFile(file) {
  if (!file.type.startsWith('image/')) return showError('Please choose an image file (PNG, JPG or BMP).');
  if (file.size > 10 * 1024 * 1024) return showError('That image is over 10 MB. Please pick a smaller one.');

  run('tab-upload', async () => {
    const form = new FormData();
    form.append('file', file);
    state.file = file;                       // dummy mode reads this
    $('uploadProgress').classList.remove('hidden');
    $('uploadProgress').value = 0;
    setStatus('Uploading image…');
    const data = await uploadWithProgress(form);

    // Save results
    state.imageId = data.image_id;
    state.shape = data.shape;
    state.sv = data.singular_values;
    state.maxK = state.sv.length;
    state.k = Math.min(20, state.maxK);
    state.origUrl = URL.createObjectURL(file);
    state.compLoaded = state.matrixLoaded = false;
    state.noisyB64 = null;
    if (state.chart) { state.chart.destroy(); state.chart = null; }

    // Update the page
    $('previewImg').src = state.origUrl;
    $('preview').classList.remove('hidden');
    $('previewInfo').textContent = file.name + ' · ' + data.shape.join(' × ') + ' · ' + bytes(data.original_bytes || file.size);
    document.body.classList.add('has-image');
    setSliders();
    resetResults();
    $('uploadProgress').classList.add('hidden');
    setStatus('Upload complete. Choose a tab to analyse or process the image.', 'success');
  });
}

// Set slider ranges and clear old results after a new upload
function setSliders() {
  $('kSlider').max = state.maxK; $('kSlider').value = state.k;
  $('kLabel').textContent = state.k;
  $('cmpOrig').src = state.origUrl;
  $('nOrig').src = state.origUrl;
}
function resetResults() {
  $('cmpComp').removeAttribute('src');
  ['mK', 'mRatio', 'mEnergy', 'mMse', 'mPsnr', 'mSsim', 'dK', 'nMse', 'nPsnr', 'nSsim', 'dMse', 'dPsnr', 'dSsim'].forEach(id => $(id).textContent = '–');
  $('compReason').textContent = $('denReason').textContent = '';
  $('nProcessed').removeAttribute('src');
  $('nProcessedCaption').textContent = 'Processed';
  $('nInputCaption').textContent = 'Uploaded image';
  $('compFileStats').textContent = $('denFileStats').textContent = 'Actual PNG file savings: –';
  ['downloadCompressed', 'downloadDenoised', 'applySuggestedK', 'toggleNoiseCompare'].forEach(id => $(id).classList.add('hidden'));
  $('noiseCompare').classList.add('hidden');
}

// ============================================================
// 2. Matrix view
// ============================================================
// Read the top-left 6x6 pixels (as gray values) straight from the image
async function pixelBlock() {
  const img = await loadImg(state.origUrl);
  const c = document.createElement('canvas');
  c.width = img.naturalWidth; c.height = img.naturalHeight;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const n = Math.min(6, c.width, c.height);
  const d = ctx.getImageData(0, 0, n, n).data;
  const rows = [];
  for (let y = 0; y < n; y++) {
    const row = [];
    for (let x = 0; x < n; x++) {
      const i = (y * n + x) * 4;
      row.push(Math.round(0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]));
    }
    rows.push(row);
  }
  return rows;
}

function makeTable(rows) {
  const t = document.createElement('table');
  rows.forEach(r => {
    const tr = t.insertRow();
    r.forEach(v => { tr.insertCell().textContent = fmt(v); });
  });
  return t;
}

// Show a matrix, a list of vectors, or plain text inside an element
function renderValue(el, v) {
  el.innerHTML = '';
  if (Array.isArray(v) && v.length && Array.isArray(v[0])) el.appendChild(makeTable(v));
  else if (Array.isArray(v) && v.length) el.appendChild(makeTable([v]));
  else if (v && typeof v === 'object' && !Array.isArray(v)) {
    const p = document.createElement('pre'); p.textContent = JSON.stringify(v, null, 2); el.appendChild(p);
  } else el.textContent = (v === undefined || v === null || (Array.isArray(v) && !v.length)) ? 'None' : String(v);
}

function loadMatrix() {
  run('tab-matrix', async () => {
    $('pixTable').replaceWith(Object.assign(makeTable(await pixelBlock()), { id: 'pixTable' }));
    const d = await api('/api/matrix-basics', { image_id: state.imageId }, 'GET');
    $('rankVal').textContent = d.rank;
    $('nullVal').textContent = d.nullity;
    renderValue($('rrefOut'), d.rref_block);
    renderValue($('basisOut'), d.basis);
    renderValue($('gsOut'), d.gram_schmidt);
    state.matrixLoaded = true;
  });
}

// ============================================================
// 3. Singular values chart (with a vertical marker at k)
// ============================================================
const markerPlugin = {
  id: 'kMarker',
  afterDraw(chart) {
    const x = chart.scales.x.getPixelForValue(state.k - 1);
    const { top, bottom } = chart.chartArea;
    const ctx = chart.ctx;
    ctx.save();
    ctx.strokeStyle = '#7ee0c3'; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle = '#7ee0c3'; ctx.font = '14px sans-serif';
    ctx.fillText('k = ' + state.k, x + 6, top + 14);
    ctx.restore();
  },
};

function drawChart() {
  $('kMarkerText').textContent = 'k = ' + state.k;
  if (state.chart) { state.chart.update('none'); return; }
  state.chart = new Chart($('svChart'), {
    type: 'line',
    data: {
      labels: state.sv.map((_, i) => i + 1),
      datasets: [{ label: 'Singular value', data: state.sv, borderColor: '#78b4ff', backgroundColor: '#78b4ff', pointRadius: 0, borderWidth: 2 }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#e8ebf3' } } },
      scales: {
        x: { title: { display: true, text: 'Index', color: '#a7b1c7' }, ticks: { color: '#a7b1c7', maxTicksLimit: 10 }, grid: { color: '#36405a' } },
        y: { title: { display: true, text: 'Value', color: '#a7b1c7' }, ticks: { color: '#a7b1c7' }, grid: { color: '#36405a' } },
      },
    },
    plugins: [markerPlugin],
  });
}

// Called whenever a new k is used anywhere
function setK(k) {
  state.k = k;
  $('kLabel').textContent = k;
  $('kSlider').value = k;
  $('kMarkerText').textContent = 'k = ' + k;
  if (state.chart) state.chart.update('none');
}

// ============================================================
// 4. Compression
// ============================================================
function setupToggle(id, onChange) {
  $(id).querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    $(id).querySelectorAll('button').forEach(x => x.classList.toggle('active', x === b));
    onChange(b.dataset.mode);
  }));
}

// Manual mode shows the slider, Auto mode shows the "Find k" controls
setupToggle('compToggle', mode => {
  state.compMode = mode;
  $('compManual').classList.toggle('hidden', mode !== 'manual');
  $('compAuto').classList.toggle('hidden', mode !== 'auto');
  if (mode === 'manual') doCompress();
});

// Default value for the Auto input depends on the method
$('methodSel').addEventListener('change', () => {
  const m = $('methodSel').value, inp = $('paramInput');
  inp.disabled = (m === 'elbow');
  inp.value = m === 'energy' ? 95 : m === 'ratio' ? 5 : '';
});

// Slider: update the label right away, call the API 300 ms after the user stops
$('kSlider').addEventListener('input', () => { $('kLabel').textContent = $('kSlider').value; });
$('kSlider').addEventListener('input', debounce(() => { setK(+$('kSlider').value); doCompress(); }, 300));
$('findK').addEventListener('click', previewRank);
$('applySuggestedK').addEventListener('click', () => {
  if (!state.suggestedK) return;
  state.compMode = 'manual';
  $('compToggle').querySelectorAll('button').forEach(b => b.classList.toggle('active', b.dataset.mode === 'manual'));
  $('compManual').classList.remove('hidden');
  $('compAuto').classList.add('hidden');
  setK(state.suggestedK);
  doCompress();
});

function autoRankBody() {
  const body = { image_id: state.imageId, method: $('methodSel').value };
  if (body.method !== 'elbow') {
    body.param = parseFloat($('paramInput').value);
    if (isNaN(body.param) || body.param <= 0) throw new Error('Please type a valid number for the value.');
  }
  return body;
}

function previewRank() {
  run('tab-compress', async () => {
    setStatus('Finding an automatic rank suggestion…');
    const d = await api('/api/rank-suggestion', autoRankBody());
    state.suggestedK = d.k_used;
    $('compReason').textContent = `${d.reason} Preview: ${d.energy_percent}% retained energy and ${d.compression_ratio}:1 theoretical compression. Confirm with “Apply suggested k”.`;
    $('applySuggestedK').classList.remove('hidden');
    setStatus(`Suggested rank k = ${d.k_used}.`, 'success');
  });
}

function doCompress() {
  const body = { image_id: state.imageId };
  if (state.compMode === 'manual') { body.mode = 'manual'; body.k = state.k; }
  else {
    body.mode = 'auto';
    Object.assign(body, autoRankBody());
  }
  run('tab-compress', async () => {
    setStatus('Compressing image with SVD…');
    const d = await api('/api/compress', body);
    setK(d.k_used);
    $('cmpComp').src = png(d.image);
    $('compReason').textContent = d.reason || '';
    $('mK').textContent = d.k_used;
    $('mRatio').textContent = fmt(d.compression_ratio) + ' : 1';
    $('mEnergy').textContent = fmt(d.energy_percent) + ' %';
    $('mMse').textContent = fmt(d.mse);
    $('mPsnr').textContent = fmt(d.psnr) + ' dB';
    $('mSsim').textContent = (+d.ssim).toFixed(3);
    showSavings('compFileStats', d.file_sizes);
    $('downloadCompressed').href = `/api/download/${state.imageId}/compressed`;
    $('downloadCompressed').classList.remove('hidden');
    state.compLoaded = true;
    setStatus('Compression complete.', 'success');
  });
}

// ============================================================
// 5. Noise reduction
// ============================================================
$('levelSlider').addEventListener('input', () => { $('levelLabel').textContent = $('levelSlider').value; });
$('noiseKind').addEventListener('change', () => {
  const saltPepper = $('noiseKind').value === 'saltpepper';
  $('noiseLevelName').innerHTML = `${saltPepper ? 'Salt-and-pepper percentage' : 'Gaussian standard deviation'} = <b id="levelLabel">${$('levelSlider').value}</b>`;
});

$('addNoise').addEventListener('click', () => {
  run('tab-noise', async () => {
    setStatus('Adding noise…');
    const d = await api('/api/noise', {
      image_id: state.imageId, kind: $('noiseKind').value, level: +$('levelSlider').value,
    });
    state.noisyB64 = d.noisy_image;
    $('nProcessed').src = png(d.noisy_image);
    $('nProcessedCaption').textContent = 'Noisy';
    $('nCompareNoisy').src = png(d.noisy_image);
    $('nMse').textContent = fmt(d.metrics.mse);
    $('nPsnr').textContent = fmt(d.metrics.psnr) + ' dB';
    $('nSsim').textContent = (+d.metrics.ssim).toFixed(3);
    $('denReason').textContent = 'Noise added. Press Denoise current image to remove it.';
    setStatus('Noise added.', 'success');
  });
});

$('runDenoise').addEventListener('click', doDenoise);

function doDenoise() {
  const source = state.noisyB64 ? 'noisy' : 'original';
  const body = { image_id: state.imageId };
  run('tab-noise', async () => {
    setStatus('Denoising image with SVD…');
    const d = await api('/api/denoise', body);
    setK(d.k_used);
    $('nOrig').src = source === 'noisy' ? png(state.noisyB64) : state.origUrl;
    $('nInputCaption').textContent = source === 'noisy' ? 'Noisy input' : 'Uploaded image input';
    $('nProcessed').src = png(d.image);
    $('nProcessedCaption').textContent = 'Denoised';
    $('nCompareDenoised').src = png(d.image);
    $('denReason').textContent = d.reason || '';
    $('dK').textContent = d.k_used;
    $('nMse').textContent = fmt(d.input_metrics.mse);
    $('nPsnr').textContent = fmt(d.input_metrics.psnr) + ' dB';
    $('nSsim').textContent = (+d.input_metrics.ssim).toFixed(3);
    $('dMse').textContent = fmt(d.denoised_metrics.mse);
    $('dPsnr').textContent = fmt(d.denoised_metrics.psnr) + ' dB';
    $('dSsim').textContent = (+d.denoised_metrics.ssim).toFixed(3);
    showSavings('denFileStats', d.file_sizes);
    $('downloadDenoised').href = `/api/download/${state.imageId}/denoised`;
    $('downloadDenoised').classList.remove('hidden');
    $('toggleNoiseCompare').classList.remove('hidden');
    setStatus('Denoising complete.', 'success');
  });
}

$('toggleNoiseCompare').addEventListener('click', () => $('noiseCompare').classList.toggle('hidden'));

// ============================================================
// DUMMY DATA (only used when USE_DUMMY = true)
// ============================================================
function loadImg(src) {
  return new Promise((ok, fail) => {
    const i = new Image();
    i.onload = () => ok(i);
    i.onerror = () => fail(new Error('Could not read that image.'));
    i.src = src;
  });
}
async function toCanvas(src) {
  const img = await loadImg(src);
  const c = document.createElement('canvas');
  c.width = img.naturalWidth; c.height = img.naturalHeight;
  c.getContext('2d').drawImage(img, 0, 0);
  return c;
}
const toB64 = c => c.toDataURL('image/png').split(',')[1];

// Blur a canvas by shrinking it and stretching it back (fake "compression")
function blurFake(c, f) {
  const s = document.createElement('canvas');
  s.width = Math.max(2, Math.round(c.width * f));
  s.height = Math.max(2, Math.round(c.height * f));
  s.getContext('2d').drawImage(c, 0, 0, s.width, s.height);
  const o = document.createElement('canvas');
  o.width = c.width; o.height = c.height;
  o.getContext('2d').drawImage(s, 0, 0, o.width, o.height);
  return o;
}

async function dummyApi(path, body) {
  const [h, w] = state.shape || [100, 100];

  if (path === '/api/upload') {
    const c = await toCanvas(URL.createObjectURL(state.file));
    const n = Math.min(c.width, c.height, 200);
    const sv = Array.from({ length: n }, (_, i) => Math.round(6000 * Math.exp(-i / 10) + 40 * Math.exp(-i / 80) * 10));
    return { image_id: 'dummy-1', shape: [c.height, c.width], singular_values: sv };
  }

  if (path === '/api/compress') {
    let k = body.k, reason = 'Manual choice: you picked k = ' + k + '.';
    if (body.mode === 'auto') {
      if (body.method === 'energy') { k = Math.round(body.param / 100 * 25 * (body.param >= 99 ? 2 : 1)); reason = 'Smallest k that keeps ' + body.param + '% of the energy.'; }
      if (body.method === 'ratio') { k = Math.round(h * w / (body.param * (h + w + 1))); reason = 'Largest k that still gives a ' + body.param + ':1 compression ratio.'; }
      if (body.method === 'elbow') { k = 12; reason = 'k = 12 is where the singular value curve bends (the elbow).'; }
      k = Math.max(1, Math.min(state.maxK, k));
    }
    const c = blurFake(await toCanvas(state.origUrl), Math.min(1, 0.03 + k / (state.maxK * 0.6)));
    const total = state.sv.reduce((a, s) => a + s * s, 0);
    const kept = state.sv.slice(0, k).reduce((a, s) => a + s * s, 0);
    return {
      k_used: k, reason, image: toB64(c),
      compression_ratio: h * w / (k * (h + w + 1)),
      energy_percent: 100 * kept / total,
      psnr: 16 + 8 * Math.log10(k + 1),
      ssim: 1 - 0.6 * Math.exp(-k / 12),
    };
  }

  if (path === '/api/noise') {
    const c = await toCanvas(state.origUrl);
    const ctx = c.getContext('2d');
    const img = ctx.getImageData(0, 0, c.width, c.height), d = img.data;
    for (let i = 0; i < d.length; i += 4) {
      if (body.kind === 'gaussian') {
        const n = (Math.random() + Math.random() + Math.random() - 1.5) * body.level * 2.5;
        for (let j = 0; j < 3; j++) d[i + j] = Math.max(0, Math.min(255, d[i + j] + n));
      } else if (Math.random() < body.level / 200) {
        d[i] = d[i + 1] = d[i + 2] = Math.random() < 0.5 ? 0 : 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    return { noisy_image: toB64(c) };
  }

  if (path === '/api/denoise') {
    const k = body.mode === 'auto' ? Math.min(10, state.maxK) : body.k;
    const c = blurFake(await toCanvas(png(state.noisyB64)), Math.min(1, 0.05 + k / (state.maxK * 0.6)));
    return {
      k_used: k, image: toB64(c),
      reason: body.mode === 'auto' ? 'Automatic cutoff: singular values below the noise level were dropped.' : 'Manual choice: you picked k = ' + k + '.',
      psnr_before: 18.4, psnr_after: 24.7,
    };
  }

  if (path === '/api/matrix-basics') {
    return {
      rank: 5, nullity: 1,
      rref_block: [[1, 0, 0, 0, 0, 0.5], [0, 1, 0, 0, 0, -1.2], [0, 0, 1, 0, 0, 0.8], [0, 0, 0, 1, 0, 2], [0, 0, 0, 0, 1, -0.3], [0, 0, 0, 0, 0, 0]],
      basis: [[-0.5, 1.2, -0.8, -2, 0.3, 1]],
      gram_schmidt: [[0.6, 0.8, 0, 0, 0, 0], [-0.8, 0.6, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]],
    };
  }
  throw new Error('Unknown dummy endpoint: ' + path);
}
