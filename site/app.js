// FewGaze website. Every number shown is read from data/results.json and data/extras.json, which are generated
// from the recorded experiments by scripts/generate_report_assets.py and scripts/export_site_data.py.
const THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, parent) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
};
const f3 = (x) => (x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(3));
const SERIES = { geometry: "--s-geometry", dinov2: "--s-dinov2", l2cs: "--s-l2cs", clip: "--s-clip" };
const colorOf = (m, fam) => css(SERIES[fam] || "--ink-2");

/* ---------------- theme + chrome ---------------- */
const root = document.documentElement;
try {
  const t = localStorage.getItem("fewgaze-theme");
  if (t) root.setAttribute("data-theme", t);
} catch (e) { /* storage unavailable */ }
const themeListeners = [];
document.getElementById("theme-btn").addEventListener("click", () => {
  const dark = root.getAttribute("data-theme") === "dark" || (!root.getAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
  root.setAttribute("data-theme", dark ? "light" : "dark");
  try { localStorage.setItem("fewgaze-theme", dark ? "light" : "dark"); } catch (e) { /* ignore */ }
  themeListeners.forEach((f) => f());
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => themeListeners.forEach((f) => f()));
const bar = document.getElementById("bar");
addEventListener("scroll", () => bar.classList.toggle("scrolled", scrollY > 20), { passive: true });

const cfg = window.FEWGAZE_CONFIG || {};
(async () => {
  const paper = document.getElementById("link-paper");
  if (cfg.paperUrl) {
    try {
      const ok = cfg.paperUrl.startsWith("http") || (await fetch(cfg.paperUrl, { method: "HEAD" })).ok;
      if (ok) { paper.href = cfg.paperUrl; paper.hidden = false; }
    } catch (e) { /* keep hidden */ }
  }
  if (cfg.repoUrl) { const c = document.getElementById("link-code"); c.href = cfg.repoUrl; c.hidden = false; }
})();

/* ---------------- reveal on scroll ---------------- */
const io = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
}, { threshold: 0.15 });
const watch = (n) => (reduced ? n.classList.add("in") : io.observe(n));
document.querySelectorAll(".reveal").forEach(watch);

/* ---------------- counters ---------------- */
document.querySelectorAll("[data-count]").forEach((b) => {
  const target = Number(b.dataset.count);
  if (reduced || target === 0) { b.textContent = target.toLocaleString("en-US"); return; }
  b.textContent = "0";
  const o = new IntersectionObserver(([e]) => {
    if (!e.isIntersecting) return;
    o.disconnect();
    const t0 = performance.now();
    const step = (t) => {
      const p = Math.min(1, (t - t0) / 1400);
      b.textContent = Math.round(target * (1 - (1 - p) ** 3)).toLocaleString("en-US");
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, { threshold: 0.6 });
  o.observe(b);
});

/* ---------------- hero: face mesh that looks at the pointer ---------------- */
const hud = { box: document.getElementById("hud"), h: document.getElementById("hud-h"), v: document.getElementById("hud-v"), label: document.getElementById("hud-label") };
const LOCK_H = 2.5, LOCK_V = 5; // Smith et al. (UIST 2013): estimated detector tolerance
function gridLayout(hero) {
  const w = hero.clientWidth, h = hero.clientHeight;
  const narrow = w < 760;
  const s = Math.min(narrow ? w / 8.5 : w / 16, 70); // px per 5 degrees horizontally; 10 degrees vertically = 2s
  // on phones the face sits in a fixed band under the top bar and the copy starts below it (see .hero in styles.css)
  const cx = narrow ? w * 0.5 : w * 0.66, cy = narrow ? 84 + 2.4 * s : h * 0.44;
  return { cx, cy, s };
}
function buildGrid(hero, gridEl) {
  gridEl.innerHTML = "";
  const { cx, cy, s } = gridLayout(hero);
  const dots = [];
  for (const v of [10, 0, -10]) for (const hh of [-15, -10, -5, 0, 5, 10, 15]) {
    const i = document.createElement("i");
    i.style.left = `${cx + (hh / 5) * s}px`;
    i.style.top = `${cy - (v / 10) * 2 * s}px`;
    if (v === 0 && hh === 0) i.className = "centre";
    gridEl.appendChild(i);
    dots.push({ i, v, h: hh });
  }
  return dots;
}

async function hero() {
  const heroEl = document.getElementById("top");
  const canvas = document.getElementById("hero-canvas");
  const gridEl = document.getElementById("hero-grid");
  let dots = buildGrid(heroEl, gridEl);
  let THREE;
  let mesh;
  try {
    [THREE, mesh] = await Promise.all([import(THREE_URL), fetch("assets/face_mesh.json").then((r) => r.json())]);
  } catch (e) {
    canvas.remove();
    return;
  }
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 200);
  camera.position.set(0, 0, 60);

  // centre the canonical mesh (units: cm-ish, y up, +z towards the viewer)
  const V = mesh.vertices;
  const c = V.reduce((a, p) => [a[0] + p[0], a[1] + p[1], a[2] + p[2]], [0, 0, 0]).map((x) => x / V.length);
  const pos = new Float32Array(V.length * 3);
  V.forEach((p, i) => { pos[3 * i] = p[0] - c[0]; pos[3 * i + 1] = p[1] - c[1]; pos[3 * i + 2] = p[2] - c[2]; });
  const faceWidth = Math.max(...V.map((p) => p[0])) - Math.min(...V.map((p) => p[0]));
  const head = new THREE.Group();
  scene.add(head);
  const pg = new THREE.BufferGeometry();
  pg.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const pmat = new THREE.PointsMaterial({ size: 0.16, sizeAttenuation: true, transparent: true, opacity: 0.9 });
  head.add(new THREE.Points(pg, pmat));
  const edges = new Set();
  const segs = [];
  for (const [a, b, d] of mesh.faces) for (const [u, v] of [[a, b], [b, d], [d, a]]) {
    const k = u < v ? `${u}_${v}` : `${v}_${u}`;
    if (!edges.has(k)) { edges.add(k); segs.push(u, v); }
  }
  const lg = new THREE.BufferGeometry();
  lg.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  lg.setIndex(segs);
  const lmat = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.13 });
  head.add(new THREE.LineSegments(lg, lmat));

  // eyes: iris discs inside each eye ring, plus gaze rays
  const P = (i) => new THREE.Vector3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
  const eyes = ["rightEye", "leftEye"].map((key) => {
    const ring = mesh[key].map(P);
    const centre = ring.reduce((a, p) => a.add(p), new THREE.Vector3()).divideScalar(ring.length);
    const width = Math.max(...ring.map((p) => p.x)) - Math.min(...ring.map((p) => p.x));
    const rg = new THREE.BufferGeometry().setFromPoints([...ring, ring[0]]);
    const rmat = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.55 });
    head.add(new THREE.Line(rg, rmat));
    const iris = new THREE.Mesh(new THREE.CircleGeometry(width * 0.2, 40), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.95 }));
    const pupil = new THREE.Mesh(new THREE.CircleGeometry(width * 0.08, 24), new THREE.MeshBasicMaterial());
    pupil.position.z = 0.02;
    iris.add(pupil);
    iris.position.copy(centre).add(new THREE.Vector3(0, 0, 0.35));
    head.add(iris);
    const ray = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), new THREE.LineBasicMaterial({ transparent: true, opacity: 0.7 }));
    scene.add(ray);
    return { centre, width, iris, pupil, rmat, ray };
  });

  function applyColors() {
    pmat.color.set(css("--ink"));
    lmat.color.set(css("--ink-3"));
    eyes.forEach((e) => { e.rmat.color.set(css("--ink-2")); e.pupil.material.color.set(css("--ink")); });
  }
  applyColors();
  themeListeners.push(applyColors);

  let target = { h: 0, v: 0 };
  const cur = { h: 6, v: 4 };
  let lastPointer = -1e9;
  let layout = gridLayout(heroEl);
  function resize() {
    const w = heroEl.clientWidth, h = heroEl.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    layout = gridLayout(heroEl);
    dots = buildGrid(heroEl, gridEl);
    // place the head so its centre projects onto the grid centre
    const ndc = new THREE.Vector3((layout.cx / w) * 2 - 1, -(layout.cy / h) * 2 + 1, 0.5).unproject(camera);
    const dir = ndc.sub(camera.position).normalize();
    head.position.copy(camera.position).add(dir.multiplyScalar(60));
    // face spans about the inner 3 x 3 targets, so the 7 x 3 grid stays visible around it
    const unitsPerPx = (2 * 60 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) / h;
    head.scale.setScalar((4.2 * layout.s * unitsPerPx) / faceWidth);
  }
  resize();
  // the hero's height changes when fonts and data arrive, not only on window resize
  new ResizeObserver(() => resize()).observe(heroEl);
  heroEl.addEventListener("pointermove", (ev) => {
    const r = heroEl.getBoundingClientRect();
    const dx = ev.clientX - r.left - layout.cx, dy = ev.clientY - r.top - layout.cy;
    target = { h: Math.max(-20, Math.min(20, (dx / layout.s) * 5)), v: Math.max(-14, Math.min(14, (-dy / (2 * layout.s)) * 10)) };
    lastPointer = performance.now();
  });

  const wander = [[0, 0], [10, 0], [15, 10], [5, -10], [-10, 0], [-15, 10], [0, 0], [-5, -10], [5, 0], [0, 0]];
  let wi = 0, wNext = 0;
  let visible = true;
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; }).observe(heroEl);
  let lastHud = "";
  function frame(t) {
    requestAnimationFrame(frame);
    if (!visible) return;
    if (t - lastPointer > 3500 && !reduced) {
      if (t > wNext) { const [h, v] = wander[wi++ % wander.length]; target = { h, v }; wNext = t + (h === 0 && v === 0 ? 2600 : 1500); }
    }
    const k = reduced ? 1 : 0.08;
    cur.h += (target.h - cur.h) * k;
    cur.v += (target.v - cur.v) * k;
    // the head carries part of the rotation and the eyes the rest, as in real gaze shifts
    head.rotation.y = THREE.MathUtils.degToRad(cur.h * 0.75);
    head.rotation.x = THREE.MathUtils.degToRad(-cur.v * 0.6);
    const lock = Math.abs(cur.h) <= LOCK_H && Math.abs(cur.v) <= LOCK_V;
    const sig = css("--signal"), cob = css("--cobalt");
    head.updateMatrixWorld();
    // gaze 0/0 means 'towards the viewer's camera', so aim along the face-to-camera line, then offset by the gaze angles
    const toCam = camera.position.clone().sub(head.position).normalize();
    const aim = head.position.clone().add(toCam.multiplyScalar(16)).add(new THREE.Vector3(Math.tan(THREE.MathUtils.degToRad(cur.h)) * 16, Math.tan(THREE.MathUtils.degToRad(cur.v)) * 16, 0));
    for (const e of eyes) {
      e.iris.position.set(e.centre.x + (cur.h / 15) * e.width * 0.18, e.centre.y + (cur.v / 10) * e.width * 0.1, e.centre.z + 0.35);
      e.iris.material.color.set(lock ? sig : cob);
      const start = e.iris.getWorldPosition(new THREE.Vector3());
      e.ray.geometry.setFromPoints([start, aim]);
      e.ray.material.color.set(lock ? sig : cob);
      e.ray.material.opacity = lock ? 0.95 : 0.45;
    }
    renderer.render(scene, camera);
    const hR = Math.round(cur.h / 5) * 5, vR = Math.round(cur.v / 10) * 10;
    const key = `${cur.h.toFixed(1)}|${cur.v.toFixed(1)}|${lock}`;
    if (key !== lastHud) {
      lastHud = key;
      hud.h.textContent = `${cur.h >= 0 ? "+" : ""}${cur.h.toFixed(1)}°`;
      hud.v.textContent = `${cur.v >= 0 ? "+" : ""}${cur.v.toFixed(1)}°`;
      hud.label.textContent = lock ? "positive" : "negative";
      hud.box.classList.toggle("lock", lock);
      for (const d of dots) {
        d.i.classList.toggle("near", d.h === hR && d.v === vR && !(d.h === 0 && d.v === 0));
        if (d.h === 0 && d.v === 0) d.i.classList.toggle("lock", lock);
      }
    }
  }
  requestAnimationFrame(frame);
}
hero();

/* ---------------- the 5 degree problem (real medians) ---------------- */
function eyeVisual(extras) {
  const svg = document.getElementById("eye-svg");
  const cells = extras.geometry_by_gaze.cells.filter((c) => c.gaze_v === 0);
  const byH = Object.fromEntries(cells.map((c) => [c.gaze_h, c]));
  const W = 520, cx = W / 2, cy = 112, ew = 400; // drawn eye width in px
  svg.innerHTML = "";
  const shape = `M ${cx - ew / 2} ${cy} Q ${cx} ${cy - 108} ${cx + ew / 2} ${cy} Q ${cx} ${cy + 96} ${cx - ew / 2} ${cy} Z`;
  el("path", { d: shape, fill: "var(--paper)", stroke: "var(--ink)", "stroke-width": 2.5 }, svg);
  const clip = el("clipPath", { id: "eyeclip" }, el("defs", {}, svg));
  el("path", { d: shape }, clip);
  const g = el("g", { "clip-path": "url(#eyeclip)" }, svg);
  const spread = el("rect", { y: cy - 92, height: 184, fill: "var(--cobalt-soft)" }, g);
  const iris = el("g", { style: "transition: transform .6s cubic-bezier(.2,.7,.1,1)" }, g);
  el("circle", { cx, cy, r: 62, fill: "var(--cobalt)" }, iris);
  el("circle", { cx, cy, r: 26, fill: "var(--ink)" }, iris);
  el("circle", { cx: cx + 16, cy: cy - 18, r: 7, fill: "#fff", opacity: 0.8 }, iris);
  el("line", { x1: cx, x2: cx, y1: 12, y2: 212, stroke: "var(--ink-3)", "stroke-dasharray": "3 4" }, svg);
  const lab = el("text", { x: cx, y: 232, "text-anchor": "middle", style: "font: 12px var(--mono); fill: var(--ink-3)" }, svg);
  const slider = document.getElementById("gaze-slider"), out = document.getElementById("gaze-out");
  const zero = byH[0].iris_x;
  function update() {
    const h = Number(slider.value);
    const cc = byH[h];
    const dx = (cc.iris_x - zero) * ew;
    iris.style.transform = `translateX(${dx}px)`;
    spread.setAttribute("x", cx + dx - (cc.iris_x_iqr * ew) / 2);
    spread.setAttribute("width", cc.iris_x_iqr * ew);
    out.textContent = `${h > 0 ? "+" : ""}${h}°`;
    lab.textContent = `iris shift ${cc.iris_x - zero >= 0 ? "+" : ""}${(cc.iris_x - zero).toFixed(3)} eye-widths · ${h === 0 ? "positive (gaze-locked)" : "negative"}`;
  }
  slider.addEventListener("input", update);
  update();
  document.getElementById("shift-5").textContent = (byH[5].iris_x - zero).toFixed(3);
  document.getElementById("spread-0").textContent = byH[0].iris_x_iqr.toFixed(3);
}

function heatmap(results) {
  const box = document.getElementById("heat");
  const rates = (results.error_rates || []).filter((r) => r.method === "geometry");
  if (!rates.length) { box.closest(".heat-wrap").hidden = true; return; }
  box.innerHTML = "";
  const Hs = [-15, -10, -5, 0, 5, 10, 15], Vs = [10, 0, -10];
  const lo = css("--heat-0"), hi = css("--heat-1");
  const mix = (t) => `color-mix(in srgb, ${hi} ${Math.round(t * 100)}%, ${lo})`;
  const label = (t) => Object.assign(document.createElement("div"), { className: "hl", textContent: t });
  box.appendChild(label(""));
  Hs.forEach((h) => box.appendChild(label(`H ${h > 0 ? "+" : ""}${h}°`)));
  Vs.forEach((v, vi) => {
    box.appendChild(label(`V ${v > 0 ? "+" : ""}${v}°`));
    Hs.forEach((h, j) => {
      const sel = rates.filter((r) => r.gaze_v === v && r.gaze_h === h);
      const n = sel.reduce((s, r) => s + r.n, 0);
      const rate = sel.reduce((s, r) => s + r.predicted_positive_rate * r.n, 0) / n;
      const d = document.createElement("div");
      d.className = "hc" + (v === 0 && h === 0 ? " pos" : "");
      d.textContent = `${Math.round(rate * 100)}%`;
      d.title = `V ${v}°, H ${h}°: ${(rate * 100).toFixed(1)}% predicted camera-directed (n = ${n})`;
      d.style.background = mix(rate);
      d.style.color = rate > 0.55 ? "#fff" : "var(--ink)";
      d.style.transitionDelay = `${(vi * 7 + j) * 35}ms`;
      box.appendChild(d);
    });
  });
  watch(box);
}

/* ---------------- charts ---------------- */
function tooltip(container) {
  let t = container.querySelector(".tooltip");
  if (!t) { t = document.createElement("div"); t.className = "tooltip"; container.appendChild(t); }
  return { show(html, x, y) { t.innerHTML = html; t.style.left = `${x}px`; t.style.top = `${y}px`; t.classList.add("on"); }, hide() { t.classList.remove("on"); } };
}
function drawPath(svg, d, attrs) {
  const p = el("path", { d, fill: "none", ...attrs, class: "draw" }, svg);
  const len = p.getTotalLength ? p.getTotalLength() : 1000;
  p.style.setProperty("--len", Math.ceil(len) + 1);
  return p;
}
const budgetLabel = (b) => (b === "all" ? "ALL" : b);

function learningCurve(container, data, metric) {
  container.innerHTML = "";
  container.classList.remove("in");
  const methods = Object.keys(data.methods).filter((m) => data.methods[m].primary && (metric === "auroc" || data.methods[m].kind === "trained" || m === "clip_zeroshot"));
  const curves = data.curves.filter((r) => r.metric === metric && r.condition === "clean");
  const budgets = data.protocol.budgets;
  const W = 900, H = 420, m = { l: 52, r: 24, t: 16, b: 46 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${metric} learning curves` }, container);
  const numeric = budgets.filter((b) => b !== "all").map(Number);
  const plotW = W - m.l - m.r, mainW = plotW * 0.84, xAll = m.l + plotW - 12;
  const x0 = Math.log(numeric[0]), x1 = Math.log(numeric[numeric.length - 1]);
  const X = (b) => (b === "all" ? xAll : m.l + ((Math.log(Number(b)) - x0) / (x1 - x0)) * mainW);
  const vals = curves.filter((r) => methods.includes(r.method)).flatMap((r) => [r.ci_low, r.ci_high]).filter((v) => v != null);
  const ymin = Math.floor(Math.min(0.48, ...vals) * 20) / 20, ymax = Math.min(1, Math.ceil(Math.max(...vals) * 20) / 20);
  const Y = (v) => m.t + (1 - (v - ymin) / (ymax - ymin)) * (H - m.t - m.b);
  const g = el("g", { class: "grid" }, svg);
  for (let v = ymin; v <= ymax + 1e-9; v += 0.05) { el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g); el("text", { x: m.l - 10, y: Y(v) + 4, "text-anchor": "end" }, svg).textContent = v.toFixed(2); }
  el("line", { class: "chance", x1: m.l, x2: W - m.r, y1: Y(0.5), y2: Y(0.5) }, svg);
  el("rect", { x: m.l + mainW + 14, y: m.t, width: xAll - m.l - mainW - 32, height: H - m.t - m.b, fill: "var(--rule-soft)", opacity: 0.6 }, svg);
  budgets.forEach((b) => { el("text", { x: X(b), y: H - m.b + 20, "text-anchor": "middle" }, svg).textContent = budgetLabel(b); });
  el("text", { x: m.l + plotW / 2, y: H - 6, "text-anchor": "middle" }, svg).textContent = "labelled training examples per class (log scale)";
  const tip = tooltip(container);
  const hits = [];
  methods.forEach((meth, mi) => {
    const info = data.methods[meth];
    const c = colorOf(meth, info.family);
    const rows = curves.filter((r) => r.method === meth);
    if (!rows.length) return;
    if (info.kind === "zeroshot") {
      const r = rows[0];
      el("rect", { x: m.l, width: plotW, y: Y(r.ci_high), height: Math.max(1, Y(r.ci_low) - Y(r.ci_high)), fill: c, "fill-opacity": 0.08, class: "fade" }, svg);
      el("line", { x1: m.l, x2: W - m.r, y1: Y(r.estimate), y2: Y(r.estimate), stroke: c, "stroke-width": 2.5, "stroke-dasharray": "7 5", class: "fade" }, svg);
      hits.push({ x: m.l + 40, y: Y(r.estimate), r, meth, zs: true });
      return;
    }
    const num = rows.filter((r) => r.budget !== "all").sort((a, b) => Number(a.budget) - Number(b.budget));
    const band = num.map((r) => `${X(r.budget)},${Y(r.ci_high)}`).concat(num.slice().reverse().map((r) => `${X(r.budget)},${Y(r.ci_low)}`));
    el("polygon", { points: band.join(" "), fill: c, "fill-opacity": 0.14, class: "fade" }, svg);
    const p = drawPath(svg, "M" + num.map((r) => `${X(r.budget)},${Y(r.estimate)}`).join(" L"), { stroke: c, "stroke-width": 3, "stroke-linejoin": "round" });
    p.style.transitionDelay = `${mi * 0.15}s`;
    rows.forEach((r) => {
      if (r.budget === "all") el("line", { x1: X("all"), x2: X("all"), y1: Y(r.ci_low), y2: Y(r.ci_high), stroke: c, "stroke-width": 2.5, class: "fade" }, svg);
      el("circle", { cx: X(r.budget), cy: Y(r.estimate), r: 5, fill: c, stroke: "var(--card)", "stroke-width": 2, class: "fade" }, svg);
      hits.push({ x: X(r.budget), y: Y(r.estimate), r, meth });
    });
  });
  const layer = el("rect", { x: m.l, y: m.t, width: plotW, height: H - m.t - m.b, fill: "transparent" }, svg);
  layer.addEventListener("pointermove", (ev) => {
    const pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());
    let best = null, bd = 1e9;
    hits.forEach((h) => { const d = h.zs ? Math.abs(h.y - p.y) * 2 : Math.hypot(h.x - p.x, h.y - p.y); if (d < bd) { bd = d; best = h; } });
    if (!best || bd > 44) return tip.hide();
    const s = svg.getBoundingClientRect().width / W, r = best.r;
    const what = best.zs ? "no training labels" : `k = ${budgetLabel(r.budget)} per class`;
    tip.show(`<b>${data.methods[best.meth].label}</b><br>${what}<br>${metric === "auroc" ? "AUROC" : "bal. acc."} ${f3(r.estimate)} [${f3(r.ci_low)}, ${f3(r.ci_high)}]`, (best.zs ? p.x : best.x) * s, best.y * s);
  });
  layer.addEventListener("pointerleave", () => tip.hide());
  const legend = document.getElementById("lc-legend");
  legend.innerHTML = "";
  methods.forEach((mm) => {
    const info = data.methods[mm];
    const s = document.createElement("span");
    s.innerHTML = `<i class="${info.kind === "zeroshot" ? "dash" : ""}" style="border-color:${colorOf(mm, info.family)}"></i>${info.label}`;
    legend.appendChild(s);
  });
  watch(container);
}

function dotChart(container, rows, groups, methods, data, groupLabel) {
  container.innerHTML = "";
  const W = 560, H = 300, m = { l: 44, r: 12, t: 14, b: 40 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "AUROC by head pose" }, container);
  const vals = rows.flatMap((r) => [r.lo, r.hi]);
  const ymin = Math.floor(Math.min(0.45, ...vals) * 10) / 10, ymax = Math.min(1, Math.ceil(Math.max(...vals) * 10) / 10);
  const Y = (v) => m.t + (1 - (v - ymin) / (ymax - ymin)) * (H - m.t - m.b);
  const band = (W - m.l - m.r) / groups.length;
  const g = el("g", { class: "grid" }, svg);
  for (let v = ymin; v <= ymax + 1e-9; v += 0.1) { el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g); el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, svg).textContent = v.toFixed(1); }
  el("line", { class: "chance", x1: m.l, x2: W - m.r, y1: Y(0.5), y2: Y(0.5) }, svg);
  const tip = tooltip(container);
  groups.forEach((gv, gi) => {
    const cx = m.l + band * (gi + 0.5);
    el("text", { x: cx, y: H - m.b + 20, "text-anchor": "middle" }, svg).textContent = groupLabel(gv);
    methods.forEach((meth, j) => {
      const r = rows.find((q) => q.g === gv && q.m === meth);
      if (!r) return;
      const x = cx + (j - (methods.length - 1) / 2) * Math.min(16, band / (methods.length + 1));
      const info = data.methods[meth], c = colorOf(meth, info.family);
      el("line", { x1: x, x2: x, y1: Y(r.lo), y2: Y(r.hi), stroke: c, "stroke-width": 2, class: "fade" }, svg);
      el("circle", { cx: x, cy: Y(r.v), r: 5, fill: info.kind === "zeroshot" ? "var(--card)" : c, stroke: c, "stroke-width": 2, class: "fade" }, svg);
      const hit = el("rect", { x: x - 8, y: Y(r.hi) - 4, width: 16, height: Y(r.lo) - Y(r.hi) + 8, fill: "transparent" }, svg);
      hit.addEventListener("pointerenter", () => { const s = svg.getBoundingClientRect().width / W; tip.show(`<b>${info.label}</b><br>${groupLabel(gv)} · AUROC ${f3(r.v)}<br>[${f3(r.lo)}, ${f3(r.hi)}]`, x * s, Y(r.v) * s); });
      hit.addEventListener("pointerleave", () => tip.hide());
    });
  });
  watch(container);
}

function lineChart(container, series, data, xs, xLabel) {
  container.innerHTML = "";
  const W = 560, H = 300, m = { l: 44, r: 12, t: 14, b: 40 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "AUROC under label noise" }, container);
  const vals = series.flatMap((s) => s.pts.flatMap((p) => [p.lo, p.hi]));
  const ymin = Math.floor(Math.min(0.45, ...vals) * 10) / 10, ymax = Math.min(1, Math.ceil(Math.max(...vals) * 10) / 10);
  const X = (v) => m.l + ((v - xs[0]) / (xs[xs.length - 1] - xs[0])) * (W - m.l - m.r);
  const Y = (v) => m.t + (1 - (v - ymin) / (ymax - ymin)) * (H - m.t - m.b);
  const g = el("g", { class: "grid" }, svg);
  for (let v = ymin; v <= ymax + 1e-9; v += 0.1) { el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g); el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, svg).textContent = v.toFixed(1); }
  xs.forEach((x) => { el("text", { x: X(x), y: H - m.b + 20, "text-anchor": "middle" }, svg).textContent = xLabel(x); });
  el("line", { class: "chance", x1: m.l, x2: W - m.r, y1: Y(0.5), y2: Y(0.5) }, svg);
  series.forEach((s, i) => {
    const c = colorOf(s.m, data.methods[s.m].family);
    el("polygon", { points: s.pts.map((p) => `${X(p.x)},${Y(p.hi)}`).concat(s.pts.slice().reverse().map((p) => `${X(p.x)},${Y(p.lo)}`)).join(" "), fill: c, "fill-opacity": 0.13, class: "fade" }, svg);
    const p = drawPath(svg, "M" + s.pts.map((q) => `${X(q.x)},${Y(q.v)}`).join(" L"), { stroke: c, "stroke-width": 3 });
    p.style.transitionDelay = `${i * 0.15}s`;
    s.pts.forEach((q) => el("circle", { cx: X(q.x), cy: Y(q.v), r: 4.5, fill: c, stroke: "var(--card)", "stroke-width": 2, class: "fade" }, svg));
  });
  watch(container);
}

/* ---------------- study 2 ---------------- */
function study2(extras) {
  const s2 = extras.study2;
  if (!s2) return;
  document.getElementById("s2-wait").hidden = true;
  document.getElementById("s2").hidden = false;
  const grid = document.getElementById("s2grid");
  const models = Object.keys(s2.models);
  const geoRow = (t) => s2.decoding.find((r) => r.model === "geometry" && r.target === t);
  function render(target) {
    const lo = css("--heat-0"), hi = css("--heat-1");
    grid.innerHTML = "";
    const axis = document.createElement("div");
    axis.className = "s2row axis";
    axis.innerHTML = `<div class="lab">layer →</div>` + Array.from({ length: 13 }, (_, i) => `<div class="c">${i}</div>`).join("");
    grid.appendChild(axis);
    for (const mk of models) {
      for (const rep of ["global", "eyes"]) {
        const row = document.createElement("div");
        row.className = "s2row";
        const lab = document.createElement("div");
        lab.className = "lab";
        lab.innerHTML = rep === "global" ? `<b>${s2.models[mk].label}</b>summary token` : "eye patches";
        row.appendChild(lab);
        for (let L = 0; L < 13; L++) {
          const r = s2.decoding.find((q) => q.model === mk && q.rep === rep && q.layer === L && q.target === target);
          const v = r ? Math.max(0, r.r2_pooled) : 0;
          const c = document.createElement("div");
          c.className = "c";
          c.style.background = `color-mix(in srgb, ${hi} ${Math.round(v * 100)}%, ${lo})`;
          c.textContent = r ? r.r2_pooled.toFixed(2) : "";
          if (r) c.title = `${s2.models[mk].label}, ${rep === "global" ? "summary token" : "eye patches"}, layer ${L}: R² = ${r.r2_pooled.toFixed(3)} (fold SD ${r.r2_fold_sd.toFixed(3)})`;
          row.appendChild(c);
        }
        grid.appendChild(row);
      }
    }
    const g = geoRow(target);
    const key = document.createElement("div");
    key.className = "s2key";
    key.innerHTML = `<span>R² 0</span><span class="ramp"></span><span>1</span>${g ? `<span>· nine geometry measurements: R² ${g.r2_pooled.toFixed(2)}</span>` : ""}`;
    grid.appendChild(key);
  }
  document.querySelectorAll("[data-s2]").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("[data-s2]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    render(b.dataset.s2);
  }));
  render("gaze_h");
  themeListeners.push(() => render(document.querySelector("[data-s2][aria-pressed='true']").dataset.s2));
  const best = (rep) => s2.decoding.filter((r) => r.target === "gaze_h" && r.rep === rep).sort((a, b) => b.r2_pooled - a.r2_pooled)[0];
  const bg = best("global"), be = best("eyes");
  const rd = s2.readout.filter((r) => r.model !== "geometry" && r.readout === "gaze_readout").sort((a, b) => b.auroc_mean - a.auroc_mean)[0];
  const geoRd = s2.readout.find((r) => r.model === "geometry" && r.readout === "gaze_readout");
  const items = [
    bg && [bg.r2_pooled.toFixed(2), "best horizontal-gaze R² from a summary token, over every model and layer: no better than always guessing the average"],
    be && [be.r2_pooled.toFixed(2), `best horizontal-gaze R² from eye patches (${s2.models[be.model].label}, layer ${be.layer})`],
    geoRow("gaze_h") && [geoRow("gaze_h").r2_pooled.toFixed(2), "the same decoder on the nine geometry measurements"],
    rd && [rd.auroc_mean.toFixed(3), `best eye-contact AUROC read out from any layer (${s2.models[rd.model].label}, ${rd.rep === "eyes" ? "eye patches" : "summary token"}, layer ${rd.layer}), picked with hindsight${geoRd ? `; geometry: ${geoRd.auroc_mean.toFixed(3)}` : ""}`],
  ];
  const nested = (s2.nested || []).filter((r) => r.model !== "geometry").sort((a, b) => b.auroc_mean - a.auroc_mean)[0];
  const nGeo = (s2.nested || []).find((r) => r.model === "geometry");
  if (nested && nGeo) {
    const d = nested.delta_vs_geometry;
    items.push([nested.auroc_mean.toFixed(3), `the same readout when the layer is chosen on training people only (${s2.models[nested.model].label}); geometry ${nGeo.auroc_mean.toFixed(3)}, paired difference ${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(3)} ± ${nested.delta_sd.toFixed(3)} (fold SD)`]);
  }
  items.splice(0, items.length, ...items.filter(Boolean));
  document.getElementById("s2facts").innerHTML = items.map(([b, s]) => `<article class="fact"><b>${b}</b><span>${s}</span></article>`).join("");
}

function study3(s3) {
  const z = s3.zeroshot, sub = s3.subset;
  const items = [
    [z.vlm_zeroshot.auroc.toFixed(3), `AUROC asking the VLM, zero-shot: the best label-free method (CLIP prompts ${z.clip_zeroshot.auroc.toFixed(3)}, L2CS-Net ${z.l2cs_zeroshot.auroc.toFixed(3)})`],
    [`${Math.round(z.vlm_zeroshot.says_yes_rate * 100)}%`, "of all images got a \"Yes\", though only 4.8% are looking at the camera. Its answers follow head pose more than gaze"],
  ];
  if (sub) items.push([sub.vlm_fewshot_k2.auroc.toFixed(3), `with two labelled examples per class in the prompt, down from ${sub.vlm_zeroshot.auroc.toFixed(3)} on the same images: the examples hurt`]);
  document.getElementById("s3facts").innerHTML = items.map(([b, t]) => `<article class="fact"><b>${b}</b><span>${t}</span></article>`).join("");
  document.getElementById("s3").hidden = false;
}

/* ---------------- data ---------------- */
function fill(data) {
  const get = (m, b, metric = "auroc") => data.curves.find((r) => r.method === m && r.budget === b && r.metric === metric && r.condition === "clean");
  const hp = (m, b, p) => data.head_pose.find((r) => r.method === m && r.budget === b && r.head_pose === p);
  const front = hp("geometry", "all", 0), l30 = hp("geometry", "all", -30), r30 = hp("geometry", "all", 30);
  const vals = {
    geo_all: f3(get("geometry", "all")?.estimate),
    dino_all: f3(get("dinov2", "all")?.estimate),
    l2cs_zero: f3(get("l2cs_zeroshot", "0")?.estimate),
    pose_drop: front && l30 && r30 ? `${front.auroc.toFixed(2)} → ${Math.min(l30.auroc, r30.auroc).toFixed(2)}` : "–",
    run_id: data.run_id,
    git_commit: (data.git_commit || "").slice(0, 7),
  };
  document.querySelectorAll("[data-fill]").forEach((e) => { const v = vals[e.dataset.fill]; if (v != null) e.textContent = v; });
}

async function load() {
  let data = null, extras = null;
  try { data = await (await fetch("data/results.json", { cache: "no-cache" })).json(); } catch (e) { /* missing */ }
  try { extras = await (await fetch("data/extras.json", { cache: "no-cache" })).json(); } catch (e) { /* missing */ }
  document.querySelectorAll("[data-needs-results]").forEach((e) => (e.hidden = !data));
  document.querySelectorAll("[data-no-results]").forEach((e) => (e.hidden = !!data));
  if (extras) eyeVisual(extras);
  if (!data) return;
  fill(data);
  heatmap(data);
  let metric = "auroc";
  const lc = document.getElementById("lc");
  learningCurve(lc, data, metric);
  document.querySelectorAll("[data-metric]").forEach((b) => b.addEventListener("click", () => {
    metric = b.dataset.metric;
    document.querySelectorAll("[data-metric]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    learningCurve(lc, data, metric);
    requestAnimationFrame(() => lc.classList.add("in"));
  }));
  const primary = Object.keys(data.methods).filter((m) => data.methods[m].primary);
  const poses = [...new Set(data.head_pose.map((r) => r.head_pose))].sort((a, b) => a - b);
  const poseRows = data.head_pose.filter((r) => r.budget === "all").map((r) => ({ g: r.head_pose, m: r.method, v: r.auroc, lo: r.ci_low, hi: r.ci_high }));
  dotChart(document.getElementById("pose"), poseRows, poses, primary, data, (p) => `${p > 0 ? "+" : ""}${p}°`);
  const trained = primary.filter((m) => data.methods[m].kind === "trained");
  const noise = trained.map((m) => ({
    m,
    pts: [0, 0.1, 0.2, 0.3]
      .map((rate) => data.curves.find((r) => r.method === m && r.metric === "auroc" && r.budget === "100" && Math.abs(r.noise_rate - rate) < 1e-9 && r.condition === (rate === 0 ? "clean" : "noise")))
      .filter(Boolean)
      .map((r) => ({ x: r.noise_rate, v: r.estimate, lo: r.ci_low, hi: r.ci_high })),
  }));
  lineChart(document.getElementById("noise"), noise, data, [0, 0.1, 0.2, 0.3], (x) => `${Math.round(x * 100)}%`);
  if (extras) study2(extras);
  if (extras && extras.study3) study3(extras.study3);
  themeListeners.push(() => { learningCurve(lc, data, metric); requestAnimationFrame(() => lc.classList.add("in")); heatmap(data); });
  let to;
  addEventListener("resize", () => { clearTimeout(to); to = setTimeout(() => { learningCurve(lc, data, metric); lc.classList.add("in"); }, 200); });
}
load();
