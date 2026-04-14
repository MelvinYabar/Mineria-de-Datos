/* ══════════════════════════════════════════
   CONFIGURACIÓN
══════════════════════════════════════════ */
const API_BASE = `${window.location.origin}/api`;
const CHART_COLORS = {
  coseno:     '#6c8fff',
  euclidiana: '#4fd1a1',
  manhattan:  '#f5a742',
};

let csvActivo  = 'example.csv';
let charts     = {};
let serverOk   = false;

/* ══════════════════════════════════════════
   INIT
══════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initCsvPicker();
  initButtons();
  checkServer();
});

/* ══════════════════════════════════════════
   SERVER HEALTH CHECK
══════════════════════════════════════════ */
async function checkServer() {
  try {
    const res = await fetch(`${API_BASE}/estado?path=${encodeURIComponent(csvActivo)}`);
    const data = await res.json();
    serverOk = true;
    setStatus('ok', `Servidor activo · ${data.usuarios} usuarios`);

    await cargarCSV(csvActivo);
  } catch {
    serverOk = false;
    setStatus('err', 'Sin conexión al servidor');
  }
}

function setStatus(type, msg) {
  const dot  = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  dot.className  = `status-dot ${type}`;
  text.textContent = msg;
}

/* ══════════════════════════════════════════
   NAVIGATION
══════════════════════════════════════════ */
function initNavigation() {
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
      const page = item.dataset.page;
      document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
      document.getElementById(`page-${page}`).classList.add('active');
      item.classList.add('active');
    });
  });
}

/* ══════════════════════════════════════════
   CSV PICKER  →  llama a /api/cargar
══════════════════════════════════════════ */
function initCsvPicker() {
  document.querySelectorAll('.csv-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const csv = btn.dataset.csv;
      if (csv === csvActivo) return;
      csvActivo = csv;
      document.querySelectorAll('.csv-btn').forEach(b => b.classList.remove('selected'));
      btn.classList.add('selected');
      await cargarCSV(csv);
    });
  });
}

async function cargarCSV(path) {
  setStatus('', `Cargando ${path}…`);
  try {
    const data = await apiPost('/cargar', { path });
    csvActivo = data.path;
    setStatus('ok', `${data.path} · ${data.usuarios.toLocaleString()} usuarios`);
  } catch (e) {
    setStatus('err', e.message || 'Error al cargar CSV');
  }
}

/* ══════════════════════════════════════════
   BUTTON BINDINGS
══════════════════════════════════════════ */
function initButtons() {
  document.getElementById('btn-t1-run').addEventListener('click', runKNN);
  document.getElementById('btn-t2-run').addEventListener('click', runRecommend);
  document.getElementById('btn-t3-batch').addEventListener('click', crearBatch);
  document.getElementById('btn-t3-inf').addEventListener('click', crearInfluencer);
  document.getElementById('btn-t3-cmp').addEventListener('click', comparar);
  document.getElementById('btn-t4-run').addEventListener('click', runExperiment);
  document.getElementById('btn-analysis-run').addEventListener('click', analyzeDataset);
}

/* ══════════════════════════════════════════
   TAREA 1 — K-NN  →  /api/knn
   Usa: obtener_knn()
══════════════════════════════════════════ */
async function runKNN() {
  const uid    = parseInt(document.getElementById('t1-uid').value);
  const k      = parseInt(document.getElementById('t1-k').value);
  const metrica = document.getElementById('t1-metric').value;
  const isSim  = metrica === 'coseno';

  setLoading('btn-t1-run', true);

  try {
    const data = await apiPost('/knn', { path: csvActivo, usuario_id: uid, k, metrica });

    /* métricas */
    const maxScore = data.vecinos.length ? data.vecinos[0].puntaje : 0;
    showMetrics('t1-metrics', {
      't1-m-vecinos': data.vecinos.length,
      't1-m-maxsim':  maxScore.toFixed(4),
      't1-m-movs':    data.peliculas_usuario,
      't1-m-time':    `${data.tiempo_ms}ms`,
    });

    if (!data.vecinos.length) {
      showEmpty('t1-results', 'No se encontraron vecinos con similitud positiva');
      return;
    }

    /* tabla */
    const metricLabel = isSim ? 'Similitud' : 'Distancia';
    const scoreClass  = isSim ? 'cell-sim' : 'cell-score';

    const rows = data.vecinos.map((v, i) => `
      <tr>
        <td class="cell-rank">#${i + 1}</td>
        <td class="cell-uid">${v.usuario_id}</td>
        <td class="${scoreClass}">${v.puntaje.toFixed(6)}</td>
        <td>${v.peliculas}</td>
        <td>
          <div class="overlap-row">
            <div class="progress-wrap">
              <div class="progress-bar" style="width:${v.overlap_pct}%;background:var(--accent)"></div>
            </div>
            <span class="overlap-pct">${v.overlap_pct}%</span>
          </div>
        </td>
      </tr>`).join('');

    document.getElementById('t1-results').innerHTML = `
      <div class="table-wrap">
        <table>
          <thead><tr>
            <th>Rank</th><th>Usuario ID</th><th>${metricLabel}</th>
            <th>Películas</th><th>Overlap</th>
          </tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;

    /* gráfico */
    document.getElementById('t1-chart-card').classList.remove('hidden');
    renderBarChart('t1-chart', 't1',
      data.vecinos.map((_, i) => `#${i + 1}`),
      data.vecinos.map(v => parseFloat(v.puntaje.toFixed(4))),
      metricLabel, 'rgba(108,143,255,0.6)', '#6c8fff');

  } catch (e) {
    showError('t1-results', e.message);
  } finally {
    setLoading('btn-t1-run', false);
  }
}

/* ══════════════════════════════════════════
   TAREA 2 — RECOMENDACIONES  →  /api/recomendar
   Usa: recomendar()
══════════════════════════════════════════ */
async function runRecommend() {
  const uid    = parseInt(document.getElementById('t2-uid').value);
  const k      = parseInt(document.getElementById('t2-k').value);
  const umbral = parseFloat(document.getElementById('t2-thresh').value);
  const topN   = parseInt(document.getElementById('t2-topn').value);

  setLoading('btn-t2-run', true);

  try {
    const data = await apiPost('/recomendar', { path: csvActivo, usuario_id: uid, k, umbral, top_n: topN });
    const recs = data.recomendaciones;

    const maxPred = recs.length ? recs[0].prediccion : 0;
    const minPred = recs.length ? recs[recs.length - 1].prediccion : 0;

    showMetrics('t2-metrics', {
      't2-m-cands':   data.candidatas,
      't2-m-maxpred': maxPred.toFixed(4),
      't2-m-minpred': minPred.toFixed(4),
      't2-m-vecs':    data.vecinos_usados,
    });

    if (!recs.length) {
      showEmpty('t2-results', `No hay películas que superen el umbral ${umbral}`);
      return;
    }

    const rows = recs.map(r => {
      const conf = Math.round((r.prediccion / (maxPred || 1)) * 100);
      const pct  = Math.round((r.prediccion / 5) * 100);
      return `
        <tr>
          <td class="cell-rank">#${r.rank}</td>
          <td class="cell-uid">${r.movie_id}</td>
          <td class="cell-score">${r.prediccion.toFixed(4)}</td>
          <td>
            <div class="overlap-row">
              <div class="progress-wrap">
                <div class="progress-bar" style="width:${conf}%;background:var(--green)"></div>
              </div>
              <span class="overlap-pct">${pct}%</span>
            </div>
          </td>
        </tr>`;
    }).join('');

    document.getElementById('t2-results').innerHTML = `
      <div class="table-wrap">
        <table>
          <thead><tr>
            <th>Rank</th><th>Movie ID</th><th>Predicción</th><th>Confianza</th>
          </tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;

    /* gráfico de distribución */
    const buckets = [0, 0, 0, 0, 0];
    for (const r of data.recomendaciones) {
      const idx = Math.min(Math.max(Math.floor(r.prediccion) - 1, 0), 4);
      buckets[idx]++;
    }

    document.getElementById('t2-chart-card').classList.remove('hidden');
    const ctx2 = document.getElementById('t2-chart');
    if (charts['t2']) charts['t2'].destroy();
    charts['t2'] = new Chart(ctx2, {
      type: 'bar',
      data: {
        labels: ['1–2', '2–3', '3–4', '4–5', '5'],
        datasets: [{
          label: 'Películas',
          data: buckets,
          backgroundColor: ['rgba(240,100,112,0.6)','rgba(245,167,66,0.6)',
            'rgba(108,143,255,0.6)','rgba(79,209,161,0.6)','rgba(79,209,161,0.9)'],
          borderRadius: 4,
        }],
      },
      options: chartOptions(),
    });

  } catch (e) {
    showError('t2-results', e.message);
  } finally {
    setLoading('btn-t2-run', false);
  }
}

/* ══════════════════════════════════════════
   TAREA 3 — BATCH  →  /api/batch
   Usa: crear_usuarios_batch()
══════════════════════════════════════════ */
async function crearBatch() {
  const cantidad    = parseInt(document.getElementById('t3-cant').value);
  const num_ratings = parseInt(document.getElementById('t3-nrat').value);
  setLoading('btn-t3-batch', true);

  try {
    const data = await apiPost('/batch', { path: csvActivo, cantidad, num_ratings });
    const out  = document.getElementById('t3-batch-out');
    const badges = data.nuevos_ids.map(id => `<span class="badge badge-blue">${id}</span>`).join('');
    out.innerHTML = `
      <div class="batch-ids">
        <span class="badge badge-green">✔ ${cantidad} usuarios creados</span>
        ${badges}
      </div>
      <div class="log-box">IDs generados: [${data.nuevos_ids.join(', ')}]
Ratings por usuario: ${num_ratings}
Total usuarios en memoria: ${data.total_usuarios}</div>`;
  } catch (e) {
    document.getElementById('t3-batch-out').innerHTML = errorHtml(e.message);
  } finally {
    setLoading('btn-t3-batch', false);
  }
}

/* ══════════════════════════════════════════
   TAREA 3 — INFLUENCER  →  /api/influencer
   Usa: crear_influencer()
══════════════════════════════════════════ */
async function crearInfluencer() {
  const influencer_id = parseInt(document.getElementById('t3-iid').value);
  const top_n         = parseInt(document.getElementById('t3-itop').value);
  setLoading('btn-t3-inf', true);

  try {
    const data = await apiPost('/influencer', { path: csvActivo, influencer_id, top_n });
    document.getElementById('t3-inf-out').innerHTML = `
      <div class="influencer-pill">★ Influencer ID: ${data.influencer_id} creado</div>
      <div class="metrics" style="margin-top:0.75rem">
        <div class="metric">
          <div class="metric-label">películas en perfil</div>
          <div class="metric-val amber">${data.num_peliculas}</div>
        </div>
        <div class="metric">
          <div class="metric-label">rating promedio</div>
          <div class="metric-val green">${data.rating_promedio.toFixed(4)}</div>
        </div>
        <div class="metric">
          <div class="metric-label">total usuarios</div>
          <div class="metric-val blue">${data.total_usuarios}</div>
        </div>
      </div>`;
  } catch (e) {
    document.getElementById('t3-inf-out').innerHTML = errorHtml(e.message);
  } finally {
    setLoading('btn-t3-inf', false);
  }
}

/* ══════════════════════════════════════════
   TAREA 3 — COMPARATIVA (knn + recomendar post-influencer)
   Usa: obtener_knn() + recomendar()
══════════════════════════════════════════ */
async function comparar() {
  const uid = parseInt(document.getElementById('t3-uid').value);
  const k   = parseInt(document.getElementById('t3-k').value);
  const iid = parseInt(document.getElementById('t3-iid').value);
  setLoading('btn-t3-cmp', true);

  try {
    const [knnData, recData] = await Promise.all([
      apiPost('/knn',       { path: csvActivo, usuario_id: uid, k, metrica: 'coseno' }),
      apiPost('/recomendar', { path: csvActivo, usuario_id: uid, k, umbral: 3.0, top_n: 5 }),
    ]);

    const vecRows = knnData.vecinos.slice(0, 5).map((v, i) => {
      const isInf = v.usuario_id === iid;
      return `<tr>
        <td class="cell-rank">#${i + 1}</td>
        <td class="cell-uid ${isInf ? 'cell-inf' : ''}">${v.usuario_id}${isInf ? ' ★' : ''}</td>
        <td class="cell-sim">${v.puntaje.toFixed(4)}</td>
      </tr>`;
    }).join('');

    const recRows = recData.recomendaciones.map(r => `
      <tr>
        <td class="cell-rank">#${r.rank}</td>
        <td class="cell-uid">${r.movie_id}</td>
        <td class="cell-score">${r.prediccion.toFixed(4)}</td>
      </tr>`).join('');

    document.getElementById('t3-compare-out').innerHTML = `
      <div class="compare-grid">
        <div>
          <div class="compare-subtitle">Top vecinos</div>
          <table>
            <thead><tr><th>Rank</th><th>UID</th><th>Sim</th></tr></thead>
            <tbody>${vecRows || '<tr><td colspan="3" style="color:var(--muted)">Sin vecinos</td></tr>'}</tbody>
          </table>
        </div>
        <div>
          <div class="compare-subtitle">Top recomendaciones</div>
          <table>
            <thead><tr><th>Rank</th><th>Movie ID</th><th>Score</th></tr></thead>
            <tbody>${recRows || '<tr><td colspan="3" style="color:var(--muted)">Sin recomendaciones</td></tr>'}</tbody>
          </table>
        </div>
      </div>`;
  } catch (e) {
    showError('t3-compare-out', e.message);
  } finally {
    setLoading('btn-t3-cmp', false);
  }
}

/* ══════════════════════════════════════════
   TAREA 4 — EXPERIMENTO  →  /api/experimento
   Usa: ejecutar_experimento_completo()
══════════════════════════════════════════ */
async function runExperiment() {
  const uid      = parseInt(document.getElementById('t4-uid').value);
  const cantidades = document.getElementById('t4-cuts').value
    .split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));

  if (!cantidades.length) { alert('Ingresa al menos un corte'); return; }

  setLoading('btn-t4-run', true);
  document.getElementById('t4-table').innerHTML = `
    <div class="empty"><div class="spinner"></div><br>Ejecutando experimento…</div>`;

  try {
    const data = await apiPost('/experimento', {
      path:       csvActivo,
      usuario_id: uid,
      cantidades,
    });

    const rows = data.resultados.map(r => `
      <tr>
        <td>${r.registros.toLocaleString()}</td>
        <td class="dist-${r.distancia.toLowerCase()}">${r.distancia}</td>
        <td>${r.t_subida.toFixed(4)}s</td>
        <td>${r.t_distancia.toFixed(4)}s</td>
        <td>${r.t_reco.toFixed(4)}s</td>
        <td>${r.ram_mb.toFixed(2)} MB</td>
        <td style="color:var(--red)">${r.mae.toFixed(4)}</td>
      </tr>`).join('');

    document.getElementById('t4-table').innerHTML = `
      <div class="table-wrap">
        <table class="perf-table">
          <thead><tr>
            <th>Registros</th><th>Métrica</th>
            <th>T. Carga</th><th>T. Dist.</th><th>T. Rec.</th>
            <th>RAM</th><th>MAE</th>
          </tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;

    /* gráficos */
    buildExperimentCharts(data.resultados);

  } catch (e) {
    showError('t4-table', e.message);
  } finally {
    setLoading('btn-t4-run', false);
  }
}

function buildExperimentCharts(resultados) {
  const metrics = ['Coseno', 'Euclidiana', 'Manhattan'];
  const labels  = [...new Set(resultados.map(r => r.registros.toLocaleString()))];

  const timeDatasets = metrics.map(m => ({
    label: m,
    data:  resultados.filter(r => r.distancia === m).map(r => r.t_distancia),
    borderColor: CHART_COLORS[m.toLowerCase()],
    borderWidth: 2,
    pointRadius: 4,
    fill: false,
  }));

  const maeDatasets = metrics.map(m => ({
    label: m,
    data:  resultados.filter(r => r.distancia === m).map(r => r.mae),
    backgroundColor: CHART_COLORS[m.toLowerCase()].replace(')', ',0.6)').replace('rgb', 'rgba') + '',
    borderRadius: 3,
  }));

  // legend
  const legendHtml = metrics.map(m => `
    <span class="legend-item">
      <span class="legend-dot" style="background:${CHART_COLORS[m.toLowerCase()]}"></span>${m}
    </span>`).join('');
  document.getElementById('t4-legend-time').innerHTML = legendHtml;
  document.getElementById('t4-legend-mae').innerHTML  = legendHtml;

  document.getElementById('t4-charts-wrap').classList.remove('hidden');

  if (charts['t4-time']) charts['t4-time'].destroy();
  charts['t4-time'] = new Chart(document.getElementById('t4-time-chart'), {
    type: 'line',
    data: { labels, datasets: timeDatasets },
    options: chartOptions(),
  });

  if (charts['t4-mae']) charts['t4-mae'].destroy();
  charts['t4-mae'] = new Chart(document.getElementById('t4-mae-chart'), {
    type: 'bar',
    data: { labels, datasets: maeDatasets },
    options: chartOptions(),
  });
}

/* ══════════════════════════════════════════
   ANÁLISIS DATASET  →  /api/analisis
══════════════════════════════════════════ */
async function analyzeDataset() {
  document.getElementById('analysis-loading').classList.remove('hidden');
  document.getElementById('analysis-out').innerHTML = '';

  const prog = document.getElementById('analysis-prog');
  prog.style.width = '40%';

  try {
    const d = await apiGet(`/analisis?path=${encodeURIComponent(csvActivo)}`);
    prog.style.width = '100%';

    await sleep(200);
    document.getElementById('analysis-loading').classList.add('hidden');

    const topUsers = d.top_usuarios.map(u =>
      `<tr><td class="cell-uid">${u.usuario_id}</td><td class="cell-score">${u.ratings}</td></tr>`
    ).join('');

    document.getElementById('analysis-out').innerHTML = `
      <div class="metrics">
        <div class="metric"><div class="metric-label">usuarios</div>
          <div class="metric-val blue">${d.num_usuarios.toLocaleString()}</div></div>
        <div class="metric"><div class="metric-label">películas únicas</div>
          <div class="metric-val green">${d.num_peliculas.toLocaleString()}</div></div>
        <div class="metric"><div class="metric-label">ratings totales</div>
          <div class="metric-val amber">${d.num_ratings.toLocaleString()}</div></div>
        <div class="metric"><div class="metric-label">rating promedio</div>
          <div class="metric-val">${d.rating_promedio.toFixed(3)}</div></div>
        <div class="metric"><div class="metric-label">ratings/usuario promedio</div>
          <div class="metric-val blue">${d.avg_ratings_usuario.toFixed(1)}</div></div>
        <div class="metric"><div class="metric-label">sparsity</div>
          <div class="metric-val red">${d.sparsity_pct.toFixed(2)}%</div></div>
        <div class="metric"><div class="metric-label">rating mínimo</div>
          <div class="metric-val">${d.rating_min}</div></div>
        <div class="metric"><div class="metric-label">rating máximo</div>
          <div class="metric-val">${d.rating_max}</div></div>
      </div>

      <div class="card">
        <div class="card-title">Distribución de ratings</div>
        <div class="chart-wrap" style="height:200px">
          <canvas id="analysis-dist-chart" role="img"
            aria-label="Distribución de ratings del dataset">
            Ratings del ${d.rating_min} al ${d.rating_max}
          </canvas>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem">
        <div class="card">
          <div class="card-title">Usuarios más activos (top 10)</div>
          <table>
            <thead><tr><th>Usuario ID</th><th>Ratings</th></tr></thead>
            <tbody>${topUsers}</tbody>
          </table>
        </div>
        <div class="card">
          <div class="card-title">Resumen estadístico</div>
          <table>
            <tr><td style="color:var(--muted);font-size:12px">Densidad</td>
              <td class="cell-score">${(100 - d.sparsity_pct).toFixed(4)}%</td></tr>
            <tr><td style="color:var(--muted);font-size:12px">Desv. estándar rating</td>
              <td class="cell-score">${d.rating_std.toFixed(4)}</td></tr>
            <tr><td style="color:var(--muted);font-size:12px">CSV activo</td>
              <td><span class="badge badge-green">${d.csv_activo}</span></td></tr>
          </table>
        </div>
      </div>`;

    /* gráfico de distribución */
    await sleep(50);
    const labels = ['0.5','1.0','1.5','2.0','2.5','3.0','3.5','4.0','4.5','5.0'];
    const bgColors = [
      'rgba(240,100,112,0.7)', 'rgba(240,100,112,0.6)',
      'rgba(245,167,66,0.6)',  'rgba(245,167,66,0.6)',  'rgba(245,167,66,0.6)',
      'rgba(108,143,255,0.6)', 'rgba(108,143,255,0.7)',
      'rgba(79,209,161,0.6)',  'rgba(79,209,161,0.7)',  'rgba(79,209,161,0.9)',
    ];

    if (charts['analysis']) charts['analysis'].destroy();
    charts['analysis'] = new Chart(document.getElementById('analysis-dist-chart'), {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Ratings',
          data: d.distribucion_ratings,
          backgroundColor: bgColors,
          borderRadius: 4,
        }],
      },
      options: chartOptions(),
    });

  } catch (e) {
    document.getElementById('analysis-loading').classList.add('hidden');
    document.getElementById('analysis-out').innerHTML = `
      <div class="empty error">
        <div class="empty-icon">✗</div>${e.message}
      </div>`;
  }
}

/* ══════════════════════════════════════════
   API HELPERS
══════════════════════════════════════════ */
async function apiPost(endpoint, body) {
  let res;
  try {
    res = await fetch(`${API_BASE}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error('No se pudo conectar con la API. Verifica el despliegue de /api en Vercel.');
  }

  const text = await res.text();
  let json = {};
  try {
    json = text ? JSON.parse(text) : {};
  } catch {
    if (!res.ok) throw new Error(`Error ${res.status}`);
    throw new Error('La API devolvio una respuesta invalida.');
  }

  if (!res.ok) throw new Error(json.error || `Error ${res.status}`);
  return json;
}

async function apiGet(endpoint) {
  let res;
  try {
    res = await fetch(`${API_BASE}${endpoint}`);
  } catch {
    throw new Error('No se pudo conectar con la API. Verifica el despliegue de /api en Vercel.');
  }

  const text = await res.text();
  let json = {};
  try {
    json = text ? JSON.parse(text) : {};
  } catch {
    if (!res.ok) throw new Error(`Error ${res.status}`);
    throw new Error('La API devolvio una respuesta invalida.');
  }

  if (!res.ok) throw new Error(json.error || `Error ${res.status}`);
  return json;
}

/* ══════════════════════════════════════════
   UI HELPERS
══════════════════════════════════════════ */
function showMetrics(containerId, mapping) {
  document.getElementById(containerId).classList.remove('hidden');
  for (const [id, val] of Object.entries(mapping)) {
    document.getElementById(id).textContent = val;
  }
}

function showEmpty(id, msg) {
  document.getElementById(id).innerHTML = `
    <div class="empty"><div class="empty-icon">◎</div>${msg}</div>`;
}

function showError(id, msg) {
  document.getElementById(id).innerHTML = `
    <div class="empty error"><div class="empty-icon">✗</div>${msg}</div>`;
}

function errorHtml(msg) {
  return `<div class="empty error" style="padding:0.75rem 0"><div class="empty-icon">✗</div>${msg}</div>`;
}

function setLoading(btnId, loading) {
  const btn = document.getElementById(btnId);
  btn.disabled = loading;
  if (loading) {
    btn._origHTML = btn.innerHTML;
    btn.innerHTML = '<span class="spinner"></span> Procesando…';
  } else if (btn._origHTML) {
    btn.innerHTML = btn._origHTML;
  }
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

/* ══════════════════════════════════════════
   CHART HELPERS
══════════════════════════════════════════ */
function chartOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: '#7a8099', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
      y: { ticks: { color: '#7a8099', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
    },
  };
}

function renderBarChart(canvasId, key, labels, values, label, bgColor, borderColor) {
  if (charts[key]) charts[key].destroy();
  charts[key] = new Chart(document.getElementById(canvasId), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label,
        data: values,
        backgroundColor: bgColor,
        borderColor,
        borderWidth: 1,
        borderRadius: 3,
      }],
    },
    options: chartOptions(),
  });
}
