/* 主控制台：页签路由、任务运行与日志、统计仪表盘、数据集导出 */
import { get, post, imgUrl, fmt, toast, poller, el, escapeHtml, FLAG_META, COLOR_BGR } from '/static/api.js';
import { initReview, setReviewOut, refreshReviewBadge } from '/static/review.js';

const S = {
  state: null,
  out: 'AutoLabel',
  logOffset: 0,
  stopLog: null,
  stopJob: null,
  statsData: null,
  lastJobStatus: 'idle',
};

/* ------------------------------------------------------------------ 工具 */
function $(id) { return document.getElementById(id); }
function setTab(name) {
  document.querySelectorAll('.nav-item').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tab-panel').forEach((p) => p.classList.toggle('active', p.id === 'tab-' + name));
  if (name === 'review') initReview();
  if (name === 'dash') loadDash();
  if (name === 'export') syncExportFields();
}
document.querySelectorAll('.nav-item').forEach((btn) => btn.addEventListener('click', () => setTab(btn.dataset.tab)));

/* ------------------------------------------------------------------ 初始化 */
async function boot() {
  try {
    const st = await get('/api/state', { out: S.out });
    S.state = st;
    S.out = st.current_out || S.out;
    fillOutSelect(st.outs || []);
    fillImagesSelect(st.images_dirs || []);
    applyDefaults(st.defaults || {});
    renderTopbar(st);
    setReviewOut(S.out);
    refreshReviewBadge();
    syncExportFields();
    await refreshStatus();
    S.stopJob = poller(refreshStatus, 2000);
  } catch (e) {
    toast('初始化失败：' + e.message, 'error', 6000);
  }
}

function fillOutSelect(outs) {
  const sel = $('outSelect');
  sel.innerHTML = '';
  outs.forEach((o) => {
    const opt = el('option', null, `${o.path}（${o.n_meta} 张）`);
    opt.value = o.path;
    sel.appendChild(opt);
  });
  if (!outs.length) {
    const opt = el('option', null, S.out);
    opt.value = S.out;
    sel.appendChild(opt);
  }
  sel.value = S.out;
  sel.onchange = async () => {
    S.out = sel.value;
    setReviewOut(S.out);
    syncExportFields();
    await refreshStatus();
    refreshReviewBadge();
    if ($('tab-dash').classList.contains('active')) loadDash();
  };
}

function fillImagesSelect(dirs) {
  const sel = $('f_images');
  sel.innerHTML = '';
  dirs.forEach((d) => {
    const opt = el('option', null, d);
    opt.value = d;
    sel.appendChild(opt);
  });
}

function applyDefaults(d) {
  const map = {
    f_limit: d.limit, f_sample: d.sample, f_workers: d.workers, f_expand: d.expand,
    f_conf: d.conf, f_max_det: d.max_det, f_min_refine_plate_w: d.min_refine_plate_w,
    f_agree_iou: d.agree_iou, f_agree_kpt: (d.agree_kpt_pct * 100), f_consensus: d.consensus,
    f_class_mode: d.class_mode, f_refine_extra: d.refine_extra_thresholds, f_panels: d.panel_limit,
    f_sheet_cols: d.sheet_cols, f_out: d.out, e_dest: 'TrainSet/armor_pose_v1',
  };
  Object.entries(map).forEach(([id, v]) => { if ($(id) && v !== undefined && v !== null) $(id).value = v; });
  if ($('f_images') && d.images) $('f_images').value = d.images;
}

function renderTopbar(st) {
  $('sideTeachers').textContent = (st.defaults.teachers || []).map((t) => t.split('/').pop()).join(' + ');
  $('sideRoot').textContent = st.root;
  $('serverInfo').textContent = `:${st.port} v${st.version}`;
  $('outMetaBadge').textContent = st.n_meta !== undefined ? `${st.n_meta} 张已标注` : '无输出';
  const d = st.defaults;
  $('brandSub').textContent = `双教师伪标签 · 1.5× ROI 传统视觉精修 · kpt_shape=[4,2] · ${d.teachers.length} 教师`;
}

/* ------------------------------------------------------------------ 状态轮询 */
async function refreshStatus() {
  const st = await get('/api/state', { out: S.out });
  S.state = st;
  const job = st.job || {};
  const running = job.running;
  const badge = $('jobBadge');
  badge.textContent = running ? '运行中' : ({ idle: '空闲', done: '已完成', error: '失败', stopped: '已停止' }[job.status] || job.status);
  badge.dataset.state = running ? 'running' : (job.status || 'idle');
  $('btnStart').disabled = running;
  $('btnResume').disabled = running;
  $('btnStop').disabled = !running;
  $('btnStatsOnly').disabled = running;

  const p = job.progress || {};
  const bar = $('jobBar');
  bar.style.width = p.percent ? p.percent + '%' : (running ? '12%' : '0%');
  bar.dataset.idle = running ? '0' : '1';
  $('jobProgressText').textContent = running
    ? `已完成 ${p.done}${p.total ? ' / ' + p.total : ''}${p.percent ? '（' + p.percent + '%）' : ''}`
    : (job.status === 'done' ? '上次任务已完成' : job.status === 'error' ? '上次任务失败，详见日志' : '尚未运行任务');
  $('chipTotal').textContent = '总数 ' + (p.total ?? '—');
  $('chipMs').textContent = '单图 ' + fmt.ms(p.ms_per_image);
  $('chipElapsed').textContent = '耗时 ' + fmt.sec(p.elapsed_s);
  $('jobCmd').textContent = job.cmd ? job.cmd.split(' ').slice(-6).join(' ') : '';
  $('sidePreview').textContent = st.preview ? `${st.preview.rendered} 张 / ${Math.round(st.preview.ms)}ms` : '—';

  if (job.log_path) $('btnLogDownload').href = imgUrl('/api/asset', { path: job.log_path, download: 1 });
  if (running || S.logOffset === 0) await pullLog();
  if (running && !S.stopLog) S.stopLog = poller(pullLog, 1000);
  if (!running && S.stopLog) { S.stopLog(); S.stopLog = null; await pullLog(); }
  S.lastJobStatus = job.status;
}

async function pullLog() {
  try {
    const r = await get('/api/job/log', { since: S.logOffset, limit: 600 });
    if (r.lines && r.lines.length) {
      appendLog(r.lines);
      S.logOffset = r.next;
      if (r.dropped) appendLog([`[webui] 跳过 ${r.dropped} 行已被截断的历史日志`]);
    }
  } catch (e) { /* 轮询失败忽略 */ }
}

function logClass(line) {
  if (/error|Traceback|失败|Failed|\[E:/i.test(line)) return 'l-er';
  if (/^\s*===|完成|agree|已保存|ok\b/i.test(line)) return 'l-ok';
  if (/^\s*\[onnxruntime|libcublasLt/.test(line)) return 'l-dim';
  return '';
}

function appendLog(lines) {
  const view = $('logView');
  if (view.dataset.placeholder) { view.textContent = ''; delete view.dataset.placeholder; }
  const level = $('logLevel').value;
  lines.forEach((l) => {
    const cls = logClass(l);
    if (level === 'error' && cls !== 'l-er') return;
    if (level === 'info' && /^\s*(\[E:|libcublasLt|onnxruntime)/.test(l)) return;
    const span = el('span', cls, l + '\n');
    view.appendChild(span);
  });
  while (view.childNodes.length > 4000) view.removeChild(view.firstChild);
  if ($('logAuto').checked) view.scrollTop = view.scrollHeight;
}

$('logLevel').onchange = () => { $('logView').textContent = '（切换筛选后新日志将按级别显示）'; $('logView').dataset.placeholder = '1'; };
$('btnLogCopy').onclick = async () => {
  try { await navigator.clipboard.writeText($('logView').innerText); toast('日志已复制'); }
  catch (e) { toast('复制失败：' + e.message, 'error'); }
};
$('btnHelp').onclick = () => $('helpPop').classList.toggle('show');
$('btnRefreshAll').onclick = async () => { await refreshStatus(); refreshReviewBadge(); toast('已刷新'); };
$('modal').onclick = () => $('modal').classList.remove('show');

/* ------------------------------------------------------------------ 任务控制 */
function taskPayload(extra = {}) {
  return Object.assign({
    images: $('f_images').value,
    out: $('f_out').value || S.out,
    limit: Number($('f_limit').value || 0),
    sample: $('f_sample').value,
    seed: Number($('f_seed').value || 0),
    workers: Number($('f_workers').value || 1),
    no_val: $('f_no_val').checked,
    no_resume: $('f_no_resume').checked,
    expand: Number($('f_expand').value),
    conf: Number($('f_conf').value),
    max_det: Number($('f_max_det').value),
    min_refine_plate_w: Number($('f_min_refine_plate_w').value),
    agree_iou: Number($('f_agree_iou').value),
    agree_kpt_pct: Number($('f_agree_kpt').value) / 100,
    consensus: $('f_consensus').value,
    class_mode: $('f_class_mode').value,
    refine_extra_thresholds: $('f_refine_extra').value,
    panels: Number($('f_panels').value || 0),
    sheet_cols: Number($('f_sheet_cols').value || 5),
  }, extra);
}

async function startJob(extra, label) {
  try {
    const r = await post('/api/job/start', taskPayload(extra));
    toast(`${label}已启动（${r.job.id}）`);
    S.logOffset = 0;
    $('logView').textContent = '';
    if (r.job.out) S.out = r.job.out;
    await refreshStatus();
  } catch (e) {
    toast('启动失败：' + e.message, 'error', 6000);
  }
}
$('btnStart').onclick = () => startJob({}, '任务');
$('btnResume').onclick = () => startJob({ no_resume: false }, '续跑');
$('btnStatsOnly').onclick = async () => {
  try {
    const r = await post('/api/job/start', taskPayload({ stats_only: true, limit: 0 }));
    toast('统计重算已启动');
    S.logOffset = 0;
    await refreshStatus();
  } catch (e) { toast('启动失败：' + e.message, 'error'); }
};
$('btnStop').onclick = async () => {
  try { await post('/api/job/stop', {}); toast('已请求停止（SIGINT → 超时 SIGKILL）', 'warn'); await refreshStatus(); }
  catch (e) { toast('停止失败：' + e.message, 'error'); }
};

/* ------------------------------------------------------------------ 仪表盘 */
async function loadDash() {
  try {
    const r = await get('/api/stats', { out: S.out, force: 1 });
    S.statsData = r;
    renderMetrics(r.stats, r.index);
    drawChart('chartPlate', histItems(r.index.hist.plate_w, (e) => e >= 1e6 ? '≥1536' : e < 12 ? '<12' : `${e}~`), '#2563EB');
    drawChart('chartShift', histItems(r.index.hist.shift_pct, (e) => e >= 1e6 ? '≥20' : `${e}~`), '#7C3AED');
    drawChart('chartFlags', Object.entries(r.index.hist.flags).sort((a, b) => b[1] - a[1]).map(([k, v]) => ({
      label: (FLAG_META[k]?.label || k), value: v, color: tagColor(k),
    })), '#0EA5E9');
    drawChart('chartReview', Object.entries(r.index.hist.review_kind).map(([k, v]) => ({
      label: (FLAG_META[k]?.label || k), value: v, color: tagColor(k === 'refine_failed' ? 'refine_failed' : k),
    })), '#DC2626');
    renderDists(r.index.hist);
    renderTiming(r.stats);
    renderThumbs();
    $('dashStamp').textContent = '数据时间 ' + new Date().toLocaleTimeString('zh-CN', { hour12: false });
  } catch (e) {
    toast('统计加载失败：' + e.message, 'error');
  }
}

function histItems(hist, labeler) {
  if (!hist) return [];
  return hist.counts.map((c, i) => ({ label: labeler(hist.edges[i]), value: c, color: '#2563EB' }));
}

function tagColor(flag) {
  const meta = FLAG_META[flag];
  return { ok: '#16A34A', info: '#3B82F6', warn: '#F59E0B', danger: '#DC2626', muted: '#9AA3AF' }[meta?.cls] || '#64748B';
}

function renderMetrics(st, idx) {
  const im = st.images || {}, ob = st.objects || {}, cons = st.consistency || {};
  const cards = [
    ['图片', fmt.int(im.total), `无检出 ${fmt.int(im.no_detection)}`],
    ['目标', fmt.int(ob.total), `每图 ${fmt.num(ob.per_image)} 个`],
    ['待复核', fmt.int(ob.review), `${fmt.pct((ob.review_rate || 0) * 100)}`],
    ['精修接受率', fmt.pct((st.refine_accept_rate || 0) * 100), `refine=${st.source?.refine ?? 0} teacher=${st.source?.teacher ?? 0}`],
    ['双教师一致率', fmt.pct((cons.agree_rate_over_paired || 0) * 100), `agree=${cons.agree ?? 0} conflict=${cons.conflict ?? 0}`],
    ['单侧检出', `${cons.primary_only ?? 0}/${cons.secondary_only ?? 0}`, '仅主 / 仅副'],
    ['已废弃', fmt.int(idx.deprecated), '不参与训练导出'],
    ['已人工修改 / 已确认', `${fmt.int(idx.edited)} / ${fmt.int(idx.reviewed)}`, 'reviewed 标记为已确认'],
    ['板宽中位数', fmt.num(st.plate_w_px?.p50) + ' px', `p90 ${fmt.num(st.plate_w_px?.p90)}`],
    ['精修位移', fmt.num(st.refine_shift_pct_plate?.mean) + ' %', `p90 ${fmt.num(st.refine_shift_pct_plate?.p90)} %`],
  ];
  $('dashMetrics').innerHTML = cards.map(([l, v, s]) =>
    `<div class="metric"><div class="label">${l}</div><div class="value">${v}</div><div class="sub">${escapeHtml(String(s))}</div></div>`).join('');
}

function drawChart(id, items, accent) {
  const cv = $(id);
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth || 480, H = Number(cv.getAttribute('height')) || 180;
  cv.width = W * dpr; cv.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (!items.length) {
    ctx.fillStyle = '#9AA3AF'; ctx.font = '12px sans-serif'; ctx.fillText('暂无数据', 10, 24);
    return;
  }
  const padL = 34, padR = 10, padT = 12, padB = 34;
  const maxV = Math.max(...items.map((i) => i.value), 1);
  const bw = Math.max(6, (W - padL - padR) / items.length - 6);
  ctx.strokeStyle = '#EEF0F3'; ctx.lineWidth = 1; ctx.font = '10px sans-serif';
  for (let g = 0; g <= 4; g++) {
    const y = padT + (H - padT - padB) * g / 4;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillStyle = '#9AA3AF';
    ctx.fillText(String(Math.round(maxV * (1 - g / 4))), 4, y + 3);
  }
  const anim = 340, t0 = performance.now();
  const step = (t) => {
    const k = Math.min(1, (t - t0) / anim);
    const ease = 1 - Math.pow(1 - k, 3);
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = '#EEF0F3';
    for (let g = 0; g <= 4; g++) {
      const y = padT + (H - padT - padB) * g / 4;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
      ctx.fillStyle = '#9AA3AF'; ctx.font = '10px sans-serif';
      ctx.fillText(String(Math.round(maxV * (1 - g / 4))), 4, y + 3);
    }
    items.forEach((it, i) => {
      const x = padL + i * ((W - padL - padR) / items.length) + 3;
      const h = (H - padT - padB) * (it.value / maxV) * ease;
      const y = H - padB - h;
      const grad = ctx.createLinearGradient(0, y, 0, H - padB);
      grad.addColorStop(0, it.color || accent);
      grad.addColorStop(1, (it.color || accent) + '55');
      ctx.fillStyle = grad;
      ctx.beginPath();
      const r = Math.min(4, bw / 2);
      ctx.moveTo(x, H - padB); ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.lineTo(x + bw - r, y); ctx.quadraticCurveTo(x + bw, y, x + bw, y + r);
      ctx.lineTo(x + bw, H - padB); ctx.closePath(); ctx.fill();
      if (it.value) {
        ctx.fillStyle = '#4F5B67'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
        ctx.fillText(String(it.value), x + bw / 2, y - 3); ctx.textAlign = 'left';
      }
      ctx.save();
      ctx.translate(x + bw / 2, H - padB + 12); ctx.rotate(-Math.PI / 4.4);
      ctx.fillStyle = '#6B7280'; ctx.font = '10px sans-serif';
      ctx.fillText(it.label, 0, 0); ctx.restore();
    });
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function renderDists(hist) {
  const box = $('dashDist');
  const mk = (title, obj, color) => {
    const entries = Object.entries(obj || {}).sort((a, b) => b[1] - a[1]).slice(0, 10);
    const max = Math.max(...entries.map((e) => e[1]), 1);
    return `<div class="muted" style="margin:4px 0 8px">${title}</div>` + (entries.length ? entries.map(([k, v]) =>
      `<div class="dist-row"><span>${escapeHtml(k)}</span><span class="dist-bar"><i style="width:${(v / max) * 100}%;background:linear-gradient(90deg,${color},${color}88)"></i></span><b class="mono">${v}</b></div>`).join('')
      : '<p class="muted">暂无</p>');
  };
  box.innerHTML = mk('颜色', hist.colors, '#3B82F6') + mk('编号', hist.nums, '#7C3AED');
}

function renderTiming(st) {
  const t = st.timing || {};
  const rows = [
    ['处理记录', fmt.int(t.records)],
    ['单图均值', fmt.ms(t.ms_per_image?.mean)],
    ['单图 p90', fmt.ms(t.ms_per_image?.p90)],
    ['累计', (t.total_minutes ?? '—') + ' 分钟'],
    ['状态计数', escapeHtml(JSON.stringify(t.status || {}))],
    ['拒绝原因', escapeHtml(JSON.stringify(st.reasons || {}))],
    ['跳过精修', fmt.int((st.flags || {}).refine_skipped)],
  ];
  $('dashTiming').innerHTML = rows.map(([k, v]) =>
    `<div class="dist-row" style="grid-template-columns:96px 1fr"><span>${k}</span><b class="mono" style="color:var(--text-2)">${v}</b></div>`).join('');
}

function renderThumbs() {
  const box = $('dashThumbs');
  const items = [
    ['抽检面板', `${S.out}/figures/panels_check.png`],
    ['待复核面板', `${S.out}/figures/panels_review.png`],
    ['流程报告', `${S.out}/report.md`],
  ];
  box.innerHTML = '';
  items.forEach(([cap, p]) => {
    const fig = el('figure', 'thumb');
    const img = el('img');
    img.src = imgUrl('/api/asset', { path: p, t: Date.now() });
    img.onerror = () => { fig.remove(); };
    img.onclick = () => {
      if (p.endsWith('.md')) { window.open(imgUrl('/api/asset', { path: p }), '_blank'); return; }
      $('modalImg').src = img.src; $('modalCap').textContent = p; $('modal').classList.add('show');
    };
    const fc = el('figcaption', null, cap);
    fig.append(img, fc);
    box.appendChild(fig);
  });
}

$('btnDashReload').onclick = loadDash;
$('btnStatsRefresh').onclick = async () => {
  try {
    const r = await post('/api/stats/refresh', { out: S.out, panels: Number($('dashPanels').value || 0), cols: 6 });
    toast('已开始重算统计与面板');
    const stop = poller(async () => {
      const t = await get('/api/task', { id: r.task.id });
      if (t.task.status !== 'running') {
        stop();
        if (t.task.status === 'done') { toast('统计与面板已更新'); loadDash(); }
        else toast('重算失败：' + (t.task.error || ''), 'error', 6000);
      }
    }, 1200);
  } catch (e) { toast('启动失败：' + e.message, 'error'); }
};

/* ------------------------------------------------------------------ 导出 */
function syncExportFields() {
  $('e_out').value = S.out;
  $('e_mode').value = $('f_class_mode') ? $('f_class_mode').value : 'single';
}
function exportPayload() {
  return {
    out: $('e_out').value || S.out,
    dest: $('e_dest').value,
    filter: $('e_filter').value,
    object_filter: $('e_objfilter').value,
    class_mode: $('e_mode').value,
    order: $('e_order') ? $('e_order').value : 'balanced',
    image_mode: $('e_imgmode').value,
    val_ratio: Number($('e_val').value || 0),
    limit: Number($('e_limit').value || 0),
    seed: Number($('e_seed').value || 0),
    include_background: $('e_bg').checked,
    overwrite: $('e_overwrite').checked,
  };
}
function reportRows(o) {
  const orderLabel = { balanced: '交错均衡（R1→B1→R2…循环）', random: '随机打乱', key: '按文件名' }[o.order] || o.order || '—';
  const rows = [
    ['命中图片', fmt.int(o.n_images)],
    ['导出目标', fmt.int(o.n_objects)],
    ['排序方式', orderLabel + (o.n_classes_hit ? `（覆盖 ${o.n_classes_hit} 类）` : '')],
    ['交错周期', escapeHtml((o.cycle || []).join(' → ')) + (o.cycle ? ' …' : '')],
    ['已废弃被排除', fmt.int(o.n_deprecated_excluded) + (o.include_deprecated ? '（未排除）' : '')],
    ['剔除待复核目标', fmt.int(o.n_dropped_review_objects)],
    ['无可用目标图片', fmt.int(o.n_images_without_usable_obj)],
    ['train / val', `${fmt.int(o.n_train)} / ${fmt.int(o.n_val)}`],
    ['预计拷贝', (o.est_copy_mb ?? 0) + ' MB'],
    ['类别分布', escapeHtml(JSON.stringify(o.class_dist || {}))],
    ['前 12 张（按导出顺序）', escapeHtml(((o.order_preview || o.sample_keys) || []).join(', '))],
  ];
  return `<div class="dist-row" style="grid-template-columns:120px 1fr"><b>项目</b><b>值</b></div>` +
    rows.map(([k, v]) => `<div class="dist-row" style="grid-template-columns:120px 1fr"><span>${k}</span><b class="mono" style="color:var(--text-2)">${v}</b></div>`).join('');
}

$('btnPreflight').onclick = async () => {
  try {
    const r = await post('/api/export/preflight', exportPayload());
    $('exportReport').innerHTML = '<div class="card-head compact" style="border:0;padding:0 0 8px"><h3>预检结果</h3></div>' + reportRows(r);
    $('exportDestLabel').textContent = r.dest || '';
    toast('预检完成：' + r.n_images + ' 张 / ' + r.n_objects + ' 目标');
  } catch (e) { toast('预检失败：' + e.message, 'error', 6000); }
};

$('btnExport').onclick = async () => {
  try {
    const r = await post('/api/export/run', exportPayload());
    $('exportState').textContent = '导出中…';
    toast('导出已启动：' + r.dest);
    const stop = poller(async () => {
      const t = await get('/api/export/status', { id: r.task.id });
      const p = t.task.progress || 0;
      $('exportBar').style.width = Math.round(p * 100) + '%';
      $('exportState').textContent = `${Math.round(p * 100)}% ${t.task.message || ''}`;
      if (t.task.status !== 'running') {
        stop();
        if (t.task.status === 'done') {
          const s = t.task.result;
          $('exportReport').innerHTML = '<div class="card-head compact" style="border:0;padding:0 0 8px"><h3>导出结果</h3></div>' +
            reportRows({ n_images: s.counts.train.images + s.counts.val.images, n_objects: s.counts.train.objects + s.counts.val.objects, n_train: s.counts.train.images, n_val: s.counts.val.images, est_copy_mb: '—' }) +
            `<p class="muted">跳过 ${s.n_skipped} · 失败 ${s.n_errors} · data.yaml: <code>${escapeHtml(s.data_yaml)}</code></p>`;
          toast('导出完成');
          showYaml(s.data_yaml);
        } else {
          $('exportState').textContent = '失败';
          toast('导出失败：' + (t.task.error || ''), 'error', 6000);
        }
      }
    }, 1000);
  } catch (e) { toast('导出失败：' + e.message, 'error', 6000); }
};

async function showYaml(path) {
  try {
    const txt = await fetch(imgUrl('/api/asset', { path, t: Date.now() })).then((r) => r.text());
    $('exportYaml').textContent = txt;
  } catch (e) { /* 忽略 */ }
}

/* ------------------------------------------------------------------ 启动 */
boot();
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') { $('helpPop').classList.remove('show'); $('modal').classList.remove('show'); }
});
