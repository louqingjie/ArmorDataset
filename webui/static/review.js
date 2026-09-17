/* 复核工作台：列表 / Canvas 预览与拖拽编辑 / 对象面板 / 快捷键与批量操作 */
import { get, post, imgUrl, toast, fmt, FLAG_META, COLOR_BGR, COLOR_NAME, NUM_NAME } from '/static/api.js';

const R = {
  out: 'AutoLabel',
  filter: 'review', q: '', sort: 'priority',
  offset: 0, limit: 200, total: 0, items: [],
  key: null, meta: null, objs: [], reviewer: 'webui',
  sel: -1, selPt: -1, undo: [], dirty: false,
  autosave: true, saveTimer: null, saving: false, savePromise: null,
  saveQueued: false, lastSave: null, saveFail: 0,
  img: null, W: 0, H: 0, previewMax: 1600, imgLoading: false,
  showCoarse: true, showIdx: true,
  view: { scale: 1, ox: 0, oy: 0 },
  drag: null, inited: false,
};

const $ = (id) => document.getElementById(id);
const canvas = () => $('rvCanvas');

/* ------------------------------------------------------------------ 列表 */
export function setReviewOut(out) {
  R.out = out;
  R.offset = 0; R.items = []; R.key = null; R.meta = null; R.objs = [];
  if (R.inited) { loadList(true); }
}

export async function refreshReviewBadge() {
  try {
    const r = await get('/api/images', { out: R.out, filter: 'review', limit: 1 });
    $('navReviewBadge').textContent = r.facets.review ?? 0;
  } catch (e) { /* 忽略 */ }
}

export function initReview() {
  if (R.inited) return;
  R.inited = true;
  bindUi();
  loadList(true);
}

function bindUi() {
  $('rvSearch').addEventListener('input', debounce(() => { R.q = $('rvSearch').value.trim(); loadList(true); }, 300));
  $('rvFilter').onchange = () => { R.filter = $('rvFilter').value; loadList(true); };
  $('rvSort').onchange = () => { R.sort = $('rvSort').value; loadList(true); };
  $('rvMore').onclick = () => loadList(false);
  $('rvList').addEventListener('scroll', () => {
    const el = $('rvList');
    if (el.scrollTop + el.clientHeight > el.scrollHeight - 60 && R.items.length < R.total) loadList(false);
  });
  $('rvShowCoarse').onchange = () => { R.showCoarse = $('rvShowCoarse').checked; draw(); };
  $('rvShowIdx').onchange = () => { R.showIdx = $('rvShowIdx').checked; draw(); };
  $('rvFit').onclick = () => { fit(); draw(); };
  $('rvOneOne').onclick = async () => {
    R.previewMax = R.previewMax === 0 ? 1600 : 0;
    $('rvOneOne').textContent = R.previewMax === 0 ? '高清中(1:1)' : '1:1 高清';
    if (R.key) { await loadImage(); R.view.scale = 1; fit(); draw(); }
  };
  $('rvUndo').onclick = undo;
  $('rvSave').onclick = save;
  R.autosave = localStorage.getItem('rvAutosave') !== '0';
  $('rvAutoSave').checked = R.autosave;
  $('rvAutoSave').onchange = onAutosaveToggle;
  updateAutoStatus();
  window.addEventListener('beforeunload', onBeforeUnload);
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') flushSave();      // 切标签页也落盘
  });
  $('rvAcceptAll').onclick = () => acceptAll();
  if ($('rvBatch')) $('rvBatch').onclick = () => batchAccept(20);
  $('rvNext').onclick = () => step(1);
  $('rvBg').onclick = markBackground;
  $('rvDep').onclick = toggleDeprecate;
  $('rvBatchDep').onclick = () => batchDeprecate(20);
  $('rvRestore').onclick = restore;
  $('rvRestore').title = '恢复该图最近一次备份（自动保存的快照也在其中）';

  const c = canvas();
  c.addEventListener('mousedown', onDown);
  c.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup', onUp);
  c.addEventListener('wheel', onWheel, { passive: false });
  c.addEventListener('dblclick', () => { fit(); draw(); });
  window.addEventListener('resize', () => { syncCanvasSize(); draw(); });
  document.addEventListener('keydown', onKey);
}

function debounce(fn, ms) {
  let t = null;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

async function loadList(reset) {
  try {
    if (reset) { R.offset = 0; R.items = []; $('rvList').innerHTML = ''; }
    const r = await get('/api/images', {
      out: R.out, filter: R.filter, q: R.q, sort: R.sort,
      order: (R.sort === 'plate_w' || R.sort === 'score') ? 'desc' : 'asc',
      offset: R.offset, limit: R.limit,
    });
    R.total = r.total;
    R.items = R.items.concat(r.items);
    R.offset = R.items.length;
    $('rvCount').textContent = `${R.items.length}/${R.total}`;
    renderList(r.items);
    if (reset && !R.key && R.items.length) selectKey(R.items[0].key);
    if (r.index && r.index.building) setTimeout(() => loadList(true), 2500);
  } catch (e) {
    toast('列表加载失败：' + e.message, 'error');
  }
}

function renderList(items) {
  const box = $('rvList');
  items.forEach((it) => {
    const row = document.createElement('div');
    row.className = `list-row p${it.priority <= 3 ? it.priority : 9}`;
    row.dataset.key = it.key;
    const badge = it.n_review ? `<span class="tag danger">${it.n_review}</span>`
      : (it.reviewed ? '<span class="tag ok">已确认</span>' : (it.n_obj ? '<span class="tag info">OK</span>' : '<span class="tag muted">背景</span>'));
    row.className = `list-row p${it.priority <= 3 ? it.priority : 9}` + (it.deprecated ? ' deprecated' : '');
    row.innerHTML = `<span class="bar"></span>
      <span><div class="key">${escapeHtml(it.key)}</div>
      <div class="sub">${it.n_obj} 目标 · 板宽 ${fmt.num(it.plate_w)}px · ${it.sources.join('/')}</div></span>
      <span>${it.deprecated ? '<span class="tag muted">废弃</span>' : ''}${it.dep_objs ? `<span class="tag muted" title="含 ${it.dep_objs} 个已废弃目标（不参与训练）">弃${it.dep_objs}</span>` : ''}${badge}</span>`;
    row.onclick = () => selectKey(it.key);
    box.appendChild(row);
  });
}

function markActive() {
  document.querySelectorAll('.list-row').forEach((r) => r.classList.toggle('active', r.dataset.key === R.key));
}

/* ------------------------------------------------------------------ 选中与加载 */
async function selectKey(key) {
  if (key === R.key && R.meta) return;
  if (R.dirty) {                       // 切换前先把改动落盘（自动保存的兜底）
    const ok = await flushSave();
    if (!ok && !confirm('自动保存未成功，切换将丢失这些改动，继续？')) return;
  }
  R.key = key;
  markActive();
  try {
    const m = await get('/api/meta', { out: R.out, key });
    R.meta = m;
    R.W = m.size[1]; R.H = m.size[0];
    R.objs = (m.objects || []).map((o) => ({ ...o }));
    R.deprecated = !!m.deprecated;
    R.sel = R.objs.length ? 0 : -1; R.selPt = -1;
    R.undo = []; R.dirty = false;
    $('rvTitle').textContent = key + (R.deprecated ? '（已废弃）' : '');
    $('rvIdx').textContent = `${R.objs.length} 目标`;
    updateDepButton();
    await loadImage();
    fit(); draw(); renderObjects(); renderMetaInfo();
    updateDirty();
  } catch (e) {
    toast('加载失败：' + e.message, 'error');
  }
}

function loadImage() {
  return new Promise((resolve) => {
    R.imgLoading = true;
    const img = new Image();
    img.onload = () => { R.img = img; R.imgLoading = false; resolve(); };
    img.onerror = () => { R.img = null; R.imgLoading = false; toast('原图加载失败', 'error'); resolve(); };
    img.src = imgUrl('/api/image', { out: R.out, key: R.key, max: R.previewMax, t: Date.now() });
  });
}

/* ------------------------------------------------------------------ 画布 */
function syncCanvasSize() {
  const c = canvas(), wrap = $('rvCanvasWrap');
  const dpr = window.devicePixelRatio || 1;
  c.width = wrap.clientWidth * dpr;
  c.height = wrap.clientHeight * dpr;
  c.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w: wrap.clientWidth, h: wrap.clientHeight };
}

function fit() {
  const { w, h } = syncCanvasSize();
  if (!R.W || !R.H) return;
  const s = Math.min(w / R.W, h / R.H) * 0.94;
  R.view.scale = clamp(s, 0.02, 40);
  R.view.ox = (w - R.W * R.view.scale) / 2;
  R.view.oy = (h - R.H * R.view.scale) / 2;
}

function clamp(v, a, b) { return Math.min(b, Math.max(a, v)); }
function toScreen(p) { return [R.view.ox + p[0] * R.view.scale, R.view.oy + p[1] * R.view.scale]; }
function toImage(x, y) { return [(x - R.view.ox) / R.view.scale, (y - R.view.oy) / R.view.scale]; }

function draw() {
  const c = canvas(), ctx = c.getContext('2d');
  const wrap = $('rvCanvasWrap');
  ctx.clearRect(0, 0, wrap.clientWidth, wrap.clientHeight);
  if (!R.img) {
    ctx.fillStyle = '#9AA3AF'; ctx.font = '13px sans-serif';
    ctx.fillText(R.key ? (R.imgLoading ? '图像加载中…' : '无预览图') : '请选择左侧列表中的图片', 16, 28);
    return;
  }
  const { scale, ox, oy } = R.view;
  ctx.imageSmoothingEnabled = scale < 1.6;
  ctx.drawImage(R.img, ox, oy, R.W * scale, R.H * scale);
  ctx.lineJoin = 'round';

  R.objs.forEach((o, i) => {
    if (o.__deleted) return;
    drawQuad(ctx, o.quad_coarse_px, {
      stroke: 'rgba(120,130,145,.85)', dash: [5, 4], width: i === R.sel ? 1.6 : 1, show: R.showCoarse,
    });
    const color = o.source === 'refine' ? (COLOR_BGR[o.color] || '#16A34A') : '#F59E0B';
    drawQuad(ctx, o.quad_final_px, {
      stroke: color, width: i === R.sel ? 3 : 2,
      glow: i === R.sel ? 'rgba(37,99,235,.55)' : null,
    });
    R.objs[i]._screen = (o.quad_final_px || []).map(toScreen);
  });

  R.objs.forEach((o, i) => {
    if (o.__deleted || !o._screen) return;
    o._screen.forEach((p, k) => {
      const isSel = i === R.sel && k === R.selPt;
      ctx.beginPath();
      ctx.arc(p[0], p[1], isSel ? 8 : 5.5, 0, Math.PI * 2);
      ctx.fillStyle = isSel ? '#2563EB' : '#EF4444';
      ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = '#fff'; ctx.stroke();
      if (R.showIdx) {
        ctx.font = 'bold 11px ui-monospace, monospace';
        ctx.fillStyle = 'rgba(15,23,42,.85)';
        ctx.fillText(String(k), p[0] + 8, p[1] - 7);
      }
    });
    const p0 = o._screen[0];
    if (p0) {
      const txt = `${o.color_name || '?'}${o.num_name || '?'} ${fmt.num(o.score_primary ?? o.score_secondary)} ${o.source || ''}`;
      ctx.font = '12px sans-serif';
      const tw = ctx.measureText(txt).width + 12;
      ctx.fillStyle = 'rgba(15,23,42,.78)';
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(p0[0] - 2, p0[1] - 30, tw, 20, 5) : ctx.rect(p0[0] - 2, p0[1] - 30, tw, 20);
      ctx.fill();
      ctx.fillStyle = '#F8FAFC';
      ctx.fillText(txt, p0[0] + 4, p0[1] - 16);
    }
  });
}

function drawQuad(ctx, quad, opt) {
  if (!quad || quad.length !== 4 || opt.show === false) return;
  const pts = quad.map(toScreen);
  ctx.save();
  if (opt.glow) { ctx.shadowColor = opt.glow; ctx.shadowBlur = 12; }
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < 4; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath();
  ctx.lineWidth = opt.width || 2;
  ctx.strokeStyle = opt.stroke;
  ctx.setLineDash(opt.dash || []);
  ctx.stroke();
  ctx.restore();
}

/* ------------------------------------------------------------------ 交互 */
function hitTest(sx, sy) {
  for (let i = R.objs.length - 1; i >= 0; i--) {
    const o = R.objs[i];
    if (o.__deleted || !o._screen) continue;
    for (let k = 0; k < o._screen.length; k++) {
      const [px, py] = o._screen[k];
      if ((px - sx) ** 2 + (py - sy) ** 2 <= 100) return { obj: i, pt: k };
    }
    if (pointInPoly([sx, sy], o._screen)) return { obj: i, pt: -1 };
  }
  return null;
}
function pointInPoly(p, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if (((yi > p[1]) !== (yj > p[1])) && (p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi)) inside = !inside;
  }
  return inside;
}

function pushUndo() {
  R.undo.push(JSON.stringify(R.objs.map((o) => ({ quad: o.quad_final_px, color: o.color, num: o.num, del: !!o.__deleted, cr: !!o.__clearReview }))));
  if (R.undo.length > 60) R.undo.shift();
}

function onDown(e) {
  if (!R.key) return;
  const rect = canvas().getBoundingClientRect();
  const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
  const hit = hitTest(sx, sy);
  if (hit) {
    if (R.sel !== hit.obj) { R.sel = hit.obj; R.selPt = -1; renderObjects(); }
    pushUndo();
    R.drag = hit.pt >= 0
      ? { mode: 'pt', obj: hit.obj, pt: hit.pt }
      : { mode: 'body', obj: hit.obj, start: toImage(sx, sy), quad0: R.objs[hit.obj].quad_final_px.map((p) => [...p]) };
    R.selPt = hit.pt;
    draw();
  } else {
    R.drag = { mode: 'pan', sx, sy, ox: R.view.ox, oy: R.view.oy };
  }
}

function onMove(e) {
  if (!R.drag) return;
  const rect = canvas().getBoundingClientRect();
  const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
  const d = R.drag;
  if (d.mode === 'pan') {
    R.view.ox = d.ox + (sx - d.sx);
    R.view.oy = d.oy + (sy - d.sy);
    draw();
    return;
  }
  const [ix, iy] = toImage(sx, sy);
  const o = R.objs[d.obj];
  if (d.mode === 'pt') {
    const q = o.quad_final_px.map((p) => [...p]);
    q[d.pt] = [clamp(round2(ix), 0, R.W), clamp(round2(iy), 0, R.H)];
    o.quad_final_px = q;
    o.__quadChanged = true;
  } else {
    const dx = ix - d.start[0], dy = iy - d.start[1];
    o.quad_final_px = d.quad0.map((p) => [clamp(round2(p[0] + dx), 0, R.W), clamp(round2(p[1] + dy), 0, R.H)]);
    o.__quadChanged = true;
  }
  R.dirty = true;
  updateDirty();
  draw();
  renderObjects(true);
}
const round2 = (v) => Math.round(v * 100) / 100;

function onUp() {
  if (R.drag && R.drag.mode !== 'pan') { R.dirty = true; updateDirty(); }
  R.drag = null;
}

function onWheel(e) {
  if (!R.key) return;
  e.preventDefault();
  const rect = canvas().getBoundingClientRect();
  const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
  const [ix, iy] = toImage(sx, sy);
  const k = e.deltaY < 0 ? 1.12 : 1 / 1.12;
  R.view.scale = clamp(R.view.scale * k, 0.02, 40);
  R.view.ox = sx - ix * R.view.scale;
  R.view.oy = sy - iy * R.view.scale;
  draw();
}

/* ------------------------------------------------------------------ 对象面板 */
function renderObjects(light) {
  const box = $('rvObjects');
  if (!R.objs.length) {
    box.innerHTML = '<p class="muted">该图无目标（背景）。可点“标记背景”确认，或从列表选择其他图片。</p>';
    return;
  }
  if (light && box.children.length === R.objs.length) {
    R.objs.forEach((o, i) => {
      const card = box.children[i];
      if (!card) return;
      card.classList.toggle('active', i === R.sel);
      const w = card.querySelector('[data-role=plate]');
      if (w && o.quad_final_px) w.textContent = quadWidth(o.quad_final_px).toFixed(1) + ' px';
    });
    return;
  }
  box.innerHTML = '';
  R.objs.forEach((o, i) => {
    if (o.__deleted) return;
    const card = document.createElement('div');
    card.className = 'obj-card' + (i === R.sel ? ' active' : '') + (o.deprecated ? ' deprecated' : '');
    const tags = (o.deprecated ? '<span class="tag muted" title="已废弃：标注保留但不参与训练导出">废弃·不参与训练</span>' : '')
      + (o.flags || []).map((f) => {
        const m = FLAG_META[f] || { label: f, cls: 'muted' };
        return `<span class="tag ${m.cls}">${escapeHtml(m.label)}</span>`;
      }).join('') + (o.__clearReview ? '<span class="tag ok">待确认</span>' : '');
    card.innerHTML = `
      <div class="title"><span class="dot" style="background:${COLOR_BGR[o.color] || '#999'}"></span>
        <span>#${i} ${escapeHtml(o.color_name || '?')}${escapeHtml(o.num_name || '?')}</span>
        <span class="muted" style="margin-left:auto">${escapeHtml(o.source || '')}</span></div>
      <div class="tags">${tags}</div>
      <div class="obj-meta">
        <span>板宽</span><b data-role="plate">${o.quad_final_px ? quadWidth(o.quad_final_px).toFixed(1) : '—'} px</b>
        <span>教师分数</span><b>${fmt.num(o.score_primary)} / ${fmt.num(o.score_secondary)}</b>
        <span>两教师 IoU</span><b>${fmt.num(o.iou_two_teachers)}</b>
        <span>精修</span><b>${o.source === 'refine' ? '通过' : (o.reason || '—')}</b>
      </div>
      <div class="row gap-sm mt-sm">
        <select class="input input-sm" data-role="color" style="width:88px">
          ${Object.entries(COLOR_NAME).map(([k, v]) => `<option value="${k}" ${Number(k) === o.color ? 'selected' : ''}>${v}</option>`).join('')}
        </select>
        <select class="input input-sm" data-role="num" style="width:120px">
          ${Object.entries(NUM_NAME).map(([k, v]) => `<option value="${k}" ${Number(k) === o.num ? 'selected' : ''}>${v}</option>`).join('')}
        </select>
        <button class="btn btn-soft btn-sm" data-role="accept">确认</button>
        <button class="btn btn-danger-soft btn-sm" data-role="del">删除</button>
      </div>`;
    card.onclick = (ev) => {
      if (ev.target.closest('select,button')) return;
      R.sel = i; R.selPt = -1; renderObjects(); draw();
    };
    card.querySelector('[data-role=color]').onchange = (ev) => {
      pushUndo(); R.objs[i].color = Number(ev.target.value);
      R.objs[i].color_name = (COLOR_NAME[ev.target.value] || '?').split(' ')[0];
      R.objs[i].__colorChanged = true; R.dirty = true; updateDirty(); draw();
    };
    card.querySelector('[data-role=num]').onchange = (ev) => {
      pushUndo(); R.objs[i].num = Number(ev.target.value);
      R.objs[i].num_name = (NUM_NAME[ev.target.value] || '?').split('(')[0];
      R.objs[i].__numChanged = true; R.dirty = true; updateDirty(); draw();
    };
    card.querySelector('[data-role=accept]').onclick = () => { clearReview(i); renderObjects(); updateDirty(); };
    card.querySelector('[data-role=del]').onclick = () => { deleteObj(i); };
    box.appendChild(card);
  });
}

function quadWidth(q) {
  const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  return (d(q[3], q[0]) + d(q[2], q[1])) / 2;
}

function renderMetaInfo() {
  const m = R.meta || {};
  const o = R.objs[R.sel] || {};
  const info = {
    key: m.key, image: m.image, size: m.size, teachers: m.teachers,
    image_path: m._image_path, class_mode: m.class_mode,
    deprecated: !!m.deprecated,
    config: m.config, n_obj: m.n_obj, n_review: m.n_review,
    current: {
      source: o.source, flags: o.flags, reason: o.reason, plate_w: o.plate_w,
      scores: [o.score_primary, o.score_secondary], pcolor: o.pcolor, pnum: o.pnum,
      refine: o.refine, review_edit: o.review_edit,
    },
  };
  $('rvMeta').textContent = JSON.stringify(info, null, 1);
}

function clearReview(i) {
  const o = R.objs[i];
  if (!o) return;
  pushUndo();
  o.flags = (o.flags || []).filter((f) => !['conflict', 'no_match', 'refine_failed', 'tiny'].some((p) => f.startsWith(p)));
  if (!o.flags.includes('reviewed')) o.flags.push('reviewed');
  o.__clearReview = true;
  o.reason = null;
  R.dirty = true;
  updateDirty();
  draw();
}

function deleteObj(i) {
  if (!R.objs[i]) return;
  pushUndo();
  R.objs[i].__deleted = true;
  R.sel = -1; R.selPt = -1;
  R.dirty = true;
  renderObjects(); draw(); updateDirty();
  toast('已标记删除 #' + i + (R.autosave ? '（自动保存中…）' : '（Ctrl+S 保存后生效）'), 'warn');
}

function markBackground() {
  if (!R.key) return;
  if (!confirm('确认将该图标记为背景（清空全部目标）？保存后 labels 文件会被删除。')) return;
  pushUndo();
  R.objs.forEach((o) => { o.__deleted = true; });
  R.__background = true; R.dirty = true;
  renderObjects(); draw(); updateDirty();
  toast(R.autosave ? '已标记背景，自动保存中…' : '已标记背景，记得保存', 'warn');
}

function acceptAll() {
  if (!R.objs.length) { toast('当前图没有目标', 'warn'); return; }
  R.objs.forEach((o, i) => clearReview(i));
  toast('已接受全部目标，正在保存…');
  saveNow(true);
}

function undo() {
  const snap = R.undo.pop();
  if (!snap) { toast('没有可撤销的操作', 'warn'); return; }
  const arr = JSON.parse(snap);
  arr.forEach((s, i) => {
    if (!R.objs[i]) return;
    R.objs[i].quad_final_px = s.quad;
    R.objs[i].color = s.color;
    R.objs[i].num = s.num;
    R.objs[i].__deleted = s.del;
    R.objs[i].__clearReview = s.cr;
    R.objs[i].__quadChanged = true;
  });
  R.dirty = true;
  renderObjects(); draw(); updateDirty();
}

function updateDirty() {
  const n = R.objs.reduce((acc, o) => acc + ((o.__quadChanged || o.__colorChanged || o.__numChanged || o.__clearReview || o.__deleted) ? 1 : 0), 0);
  R.dirty = n > 0 || !!R.__background;
  $('rvDirty').textContent = R.dirty ? `未保存 ${n} 项` : '无改动';
  $('rvDirty').style.background = R.dirty ? '#FEF3C7' : '';
  $('rvDirty').style.color = R.dirty ? '#B45309' : '';
  if (R.dirty) scheduleAutoSave();
}

/* ------------------------------------------------------------------ 自动保存 */
/* 编辑后 debounce 落盘；切图/关页/切标签页时强制落盘（sendBeacon 兜底）。
   后端增量更新复核清单，单次保存约 3ms，所以可以放心自动保存。        */
const AUTOSAVE_MS = 1500;

function clockStr() { return new Date().toLocaleTimeString('zh-CN', { hour12: false }); }

function updateAutoStatus(state) {
  const el = $('rvAutoStatus');
  if (!el) return;
  let text = '自动保存：开', bg = '', fg = '';
  if (!R.autosave) { text = '自动保存：关'; bg = '#F3F4F6'; fg = '#6B7280'; }
  else if (state === 'saving' || R.saving) { text = '保存中…'; bg = '#FEF3C7'; fg = '#B45309'; }
  else if (R.saveTimer) { text = '待自动保存…'; bg = '#FEF3C7'; fg = '#B45309'; }
  else if (R.saveFail) { text = `保存失败 ×${R.saveFail}（Ctrl+S 重试）`; bg = '#FEE2E2'; fg = '#B91C1C'; }
  else if (R.lastSave) { text = `已自动保存 ${R.lastSave}`; bg = '#DCFCE7'; fg = '#15803D'; }
  el.textContent = text; el.style.background = bg; el.style.color = fg;
}

function scheduleAutoSave(delay) {
  if (!R.autosave || !R.dirty || R.saveFail >= 5) return;   // 连续失败则停手，等人干预
  const ms = (delay || AUTOSAVE_MS) * Math.min(2 ** R.saveFail, 8);
  clearTimeout(R.saveTimer);
  R.saveTimer = setTimeout(() => { R.saveTimer = null; saveNow(true); }, ms);
  updateAutoStatus();
}

function onAutosaveToggle() {
  R.autosave = $('rvAutoSave').checked;
  localStorage.setItem('rvAutosave', R.autosave ? '1' : '0');
  if (R.autosave) { R.saveFail = 0; scheduleAutoSave(600); }
  else { clearTimeout(R.saveTimer); R.saveTimer = null; }
  updateAutoStatus();
  toast(R.autosave ? '已开启自动保存（编辑后约 1.5s 落盘）' : '已关闭自动保存（Ctrl+S 手动保存）', 'warn');
}

async function flushSave() {
  clearTimeout(R.saveTimer); R.saveTimer = null;
  if (R.saving && R.savePromise) { try { await R.savePromise; } catch (e) { /* 内部已提示 */ } }
  if (!R.dirty) return true;
  return await saveNow(true);
}

function onBeforeUnload(e) {
  if (!R.dirty) return;
  const body = buildSaveBody(true);
  if (!body) return;
  if (navigator.sendBeacon
      && navigator.sendBeacon('/api/save', new Blob([JSON.stringify(body)], { type: 'application/json' }))) {
    R.dirty = false;                                    // 已尽力送达，无需拦截
    return;
  }
  e.preventDefault(); e.returnValue = '';
}

/* ------------------------------------------------------------------ 保存 / 恢复 */
async function save() { return saveNow(false); }

function buildSaveBody(auto) {
  const objects = [];
  R.objs.forEach((o, i) => {
    if (o.__deleted) { objects.push({ index: i, action: 'delete' }); return; }
    const a = { index: i, action: 'update' };
    if (o.__quadChanged) a.quad_final_px = o.quad_final_px;
    if (o.__colorChanged) a.color = o.color;
    if (o.__numChanged) a.num = o.num;
    if (o.__clearReview) a.clear_review = true;
    if (Object.keys(a).length > 2) objects.push(a);
  });
  if (!objects.length && !R.__background) return null;
  return { out: R.out, key: R.key, reviewer: R.reviewer, auto: !!auto,
           objects, background: !!R.__background };
}

function saveNow(auto) {
  if (!R.key) return Promise.resolve(false);
  if (R.saving) { R.saveQueued = true; return R.savePromise || Promise.resolve(false); }
  if (auto && R.drag) { R.saveQueued = true; return Promise.resolve(false); }   // 拖动中不打断
  const body = buildSaveBody(auto);
  if (!body) {
    if (!auto) toast('没有需要保存的改动', 'warn');
    return Promise.resolve(true);
  }
  R.saving = true;
  updateAutoStatus('saving');
  R.savePromise = (async () => {
    try {
      const r = await post('/api/save', body);
      R.meta = r.meta;
      R.__background = false;
      R.lastSave = clockStr(); R.saveFail = 0;
      if (R.drag) {
        R.saveQueued = true;                 // 拖动中：等本轮结束再刷新视图
      } else {
        R.objs = (r.meta.objects || []).map((o) => ({ ...o }));
        R.dirty = false;
        if (!auto) R.undo = [];              // 自动保存保留撤销栈，仍可 Ctrl+Z 回退
      }
      if (!auto) toast('已保存并备份 → ' + (r.backup?.dir || ''), 'ok', 3200);
      if (!R.drag) { renderObjects(); draw(); updateDirty(); renderMetaInfo(); }
      refreshReviewBadge();
      const row = document.querySelector(`.list-row[data-key="${cssEscape(R.key)}"]`);
      if (row) {
        const n = r.meta.n_review;
        row.querySelector('span:last-child').innerHTML = n ? `<span class="tag danger">${n}</span>` : '<span class="tag ok">已确认</span>';
      }
      return true;
    } catch (e) {
      R.saveFail += 1;
      toast('保存失败：' + e.message + (auto ? '（改动已保留，可 Ctrl+S 重试）' : ''), 'error', 6000);
      return false;
    } finally {
      R.saving = false; R.savePromise = null;
      updateDirty(); updateAutoStatus();
      if (R.saveQueued) { R.saveQueued = false; scheduleAutoSave(400); }
    }
  })();
  return R.savePromise;
}

function cssEscape(s) { return String(s).replace(/["\\]/g, '\\$&'); }

async function restore() {
  if (!R.key) return;
  if (!confirm('恢复该图最近一次备份（当前内容会先备份）？')) return;
  await flushSave();                     // 未保存的改动先落盘，避免被恢复覆盖
  try {
    const r = await post('/api/restore', { out: R.out, key: R.key });
    R.meta = r.meta;
    R.objs = (r.meta.objects || []).map((o) => ({ ...o }));
    R.undo = []; R.dirty = false;
    renderObjects(); draw(); updateDirty(); renderMetaInfo();
    toast('已恢复到备份 ' + r.restored_from, 'ok');
  } catch (e) {
    toast('恢复失败：' + e.message, 'error', 5000);
  }
}

/* ------------------------------------------------------------------ 导航与快捷键 */
function step(d) {
  if (!R.items.length) return;
  let i = R.items.findIndex((it) => it.key === R.key);
  if (i < 0) i = 0;
  const next = R.items[(i + d + R.items.length) % R.items.length];
  if (next) selectKey(next.key);
}

function onKey(e) {
  const tag = (e.target.tagName || '').toLowerCase();
  if (['input', 'select', 'textarea'].includes(tag)) return;
  if (!$('tab-review').classList.contains('active')) return;
  if (e.ctrlKey || e.metaKey) {
    if (e.key.toLowerCase() === 's') { e.preventDefault(); save(); }
    if (e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); }
    return;
  }
  switch (e.key) {
    case 'ArrowLeft': e.preventDefault(); step(-1); break;
    case 'ArrowRight': e.preventDefault(); step(1); break;
    case 'a': case 'A': acceptAll(); break;
    case 's': case 'S': step(1); break;
    case 'd': case 'D':                              // D=废弃/取消废弃本图；Shift+D=批量废弃前 20 张
      if (e.shiftKey) batchDeprecate(20);
      else toggleDeprecate();
      break;
    case 'Delete': case 'Backspace':
      if (R.sel >= 0) deleteObj(R.sel);
      break;
    case '1': case '2': case '3': case '4': {
      const k = Number(e.key) - 1;
      if (R.sel >= 0 && R.objs[R.sel]) { R.selPt = k; draw(); }
      break;
    }
    default: break;
  }
}

/* 批量接受：当前列表中仍待复核的前 N 张 */
export async function batchAccept(n = 20) {
  const targets = R.items.filter((it) => it.n_review > 0).slice(0, n);
  if (!targets.length) { toast('列表中已无待复核图片', 'warn'); return; }
  let done = 0, failed = 0;
  for (const it of targets) {
    try {
      const m = await get('/api/meta', { out: R.out, key: it.key });
      const objects = (m.objects || []).map((o, i) => ({
        index: i, action: 'update', clear_review: true, color: o.color, num: o.num,
      }));
      await post('/api/save', { out: R.out, key: it.key, reviewer: R.reviewer, objects, background: false });
      done += 1;
    } catch (err) { failed += 1; }
  }
  toast(`批量接受完成：成功 ${done} 张${failed ? '，失败 ' + failed : ''}`, failed ? 'warn' : 'ok', 3600);
  refreshReviewBadge();
  loadList(true);
}

function updateDepButton() {
  const b = $('rvDep');
  if (!b) return;
  b.textContent = R.deprecated ? '取消废弃' : '废弃本图';
  b.classList.toggle('btn-danger-soft', !R.deprecated);
  b.classList.toggle('btn-soft', !!R.deprecated);
}

function markRowDeprecated(key, dep) {
  const row = document.querySelector(`.list-row[data-key="${cssEscape(key)}"]`);
  if (!row) return;
  row.classList.toggle('deprecated', !!dep);
  const box = row.querySelector('span:last-child');
  if (!box) return;
  const old = box.querySelector('.tag.muted');
  if (dep) {
    if (!old) box.insertAdjacentHTML('afterbegin', '<span class="tag muted">废弃</span>');
  } else if (old) {
    old.remove();
  }
}

async function toggleDeprecate() {
  if (!R.key) { toast('请先选择图片', 'warn'); return; }
  const next = !R.deprecated;
  if (next && !confirm('标记为「废弃」？该图将不参与训练集导出（可随时取消）。')) return;
  try {
    await post('/api/deprecate', {
      out: R.out, key: R.key, deprecated: next, reviewer: R.reviewer, reason: '',
    });
    R.deprecated = next;
    updateDepButton();
    markRowDeprecated(R.key, next);
    $('rvTitle').textContent = R.key + (next ? '（已废弃）' : '');
    toast(next ? '已标记废弃：不参与训练导出' : '已取消废弃', next ? 'warn' : 'ok');
    refreshReviewBadge();
  } catch (e) {
    toast('操作失败：' + e.message, 'error', 5000);
  }
}

async function batchDeprecate(n = 20) {
  const targets = R.items.filter((it) => !it.deprecated).slice(0, n);
  if (!targets.length) { toast('列表中已没有可废弃的图片', 'warn'); return; }
  if (!confirm(`把当前列表前 ${targets.length} 张标记为废弃？\n（可随时在“已废弃”过滤中取消）`)) return;
  try {
    const r = await post('/api/deprecate', {
      out: R.out, keys: targets.map((t) => t.key), deprecated: true,
      reviewer: R.reviewer, bulk: true,
    });
    targets.forEach((t) => { t.deprecated = true; markRowDeprecated(t.key, true); });
    toast(`已废弃 ${r.n} 张（库内累计 ${r.total} 张）`, 'warn', 3400);
    refreshReviewBadge();
  } catch (e) {
    toast('批量废弃失败：' + e.message, 'error', 5000);
  }
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
