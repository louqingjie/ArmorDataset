/* 极简请求层与通用工具：无依赖、离线可用 */

export async function get(path, params) {
  const url = new URL(path, location.origin);
  Object.entries(params || {}).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, v);
  });
  const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
  const data = await res.json().catch(() => ({ ok: false, error: '响应不是 JSON' }));
  if (!res.ok || data.ok === false) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

export async function post(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json().catch(() => ({ ok: false, error: '响应不是 JSON' }));
  if (!res.ok || data.ok === false) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

export function imgUrl(path, params) {
  const url = new URL(path, location.origin);
  Object.entries(params || {}).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, v);
  });
  return url.toString();
}

/* ---------------- 格式化 ---------------- */
export const fmt = {
  pct: (v) => (v === null || v === undefined || isNaN(v)) ? '—' : (v * 1).toFixed(1) + '%',
  num: (v) => (v === null || v === undefined || isNaN(v)) ? '—' : Number(v).toFixed(2),
  int: (v) => (v === null || v === undefined || isNaN(v)) ? '—' : String(Math.round(v)),
  ms: (v) => (v === null || v === undefined) ? '—' : (v >= 1000 ? (v / 1000).toFixed(2) + ' s' : Math.round(v) + ' ms'),
  sec: (v) => {
    if (v === null || v === undefined) return '—';
    const s = Math.round(v);
    return s < 60 ? s + ' s' : `${Math.floor(s / 60)} m ${s % 60} s`;
  },
  time: (t) => t ? new Date(t * 1000).toLocaleTimeString('zh-CN', { hour12: false }) : '—',
  dt: (t) => t ? new Date(t * 1000).toLocaleString('zh-CN', { hour12: false }) : '—',
};

/* ---------------- toast ---------------- */
export function toast(msg, type = 'ok', ms = 2600) {
  const box = document.getElementById('toasts');
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  box.appendChild(el);
  requestAnimationFrame(() => el.classList.add('show'));
  setTimeout(() => {
    el.classList.remove('show');
    setTimeout(() => el.remove(), 260);
  }, ms);
}

/* ---------------- 简单轮询 ---------------- */
export function poller(fn, ms) {
  let timer = null, stop = false;
  const tick = async () => {
    if (stop) return;
    try { await fn(); } catch (e) { /* 轮询失败静默，避免刷屏 */ }
    if (!stop) timer = setTimeout(tick, ms);
  };
  tick();
  return () => { stop = true; clearTimeout(timer); };
}

export function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

export function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

/* 徽标配色：语义色与设计令牌一致 */
export const FLAG_META = {
  agree:        { label: '一致',       cls: 'ok' },
  conflict:     { label: '冲突',       cls: 'danger' },
  conflict_iou: { label: 'IoU不足',    cls: 'danger' },
  conflict_kpt: { label: '角点差',     cls: 'danger' },
  conflict_cls: { label: '类别冲突',   cls: 'danger' },
  no_match:     { label: '单侧检出',   cls: 'warn' },
  primary_only: { label: '仅主教师',   cls: 'warn' },
  secondary_only:{ label: '仅副教师',  cls: 'warn' },
  refine_ok:    { label: '精修完成',   cls: 'info' },
  refine_failed:{ label: '精修失败',   cls: 'danger' },
  refine_skipped:{ label: '跳过精修',  cls: 'muted' },
  plate_too_small:{ label: '板太小',   cls: 'muted' },
  roi_too_small:{ label: 'ROI太小',    cls: 'muted' },
  no_bar_pair:  { label: '无灯条对',   cls: 'muted' },
  iou_guard:    { label: '跳变拦截',   cls: 'danger' },
  size_guard:   { label: '尺度异常',   cls: 'danger' },
  tiny:         { label: '极小目标',   cls: 'warn' },
  reviewed:     { label: '已确认',     cls: 'ok' },
};

export const COLOR_BGR = { 0: '#3B82F6', 1: '#DC2626', 2: '#64748B', 3: '#A1A1AA' };
// 索引 2 实测是"灰白·未点亮"装甲板（灯条不亮/极暗），并非紫色：见 color_audit.py
export const COLOR_NAME = { 0: 'B 蓝', 1: 'R 红', 2: 'G 灰白·未点亮', 3: 'N 其他' };
export const NUM_NAME = { 0: '7(哨兵)', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: 'O(前哨)', 7: 'B(基地)', 8: 'LB(大基地)' };
