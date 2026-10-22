// FewGaze live demo: runs entirely in the viewer's browser. Camera frames and photos are never uploaded.
// Pipeline = the study's geometry method: MediaPipe Face Landmarker (tasks-vision 0.10.21, same model file
// as the Python run) -> 9 eye/head features (fewgaze-core.js, parity-tested) -> exported logistic regression.
import { analyse, EYES, IRIS_CENTRES } from "./fewgaze-core.js";

const BASE = new URL(".", import.meta.url);
const VENDOR = new URL("vendor/", BASE);
const MAX_SIDE = 1600; // same longest side as the Python pipeline (photos)
const FEATURE_LABELS = [
  "Right iris, along eye", "Right iris, across eye", "Right eye opening",
  "Left iris, along eye", "Left iris, across eye", "Left eye opening",
  "Head pitch", "Head yaw", "Head roll",
];
const SPARK_SECONDS = 10;

const $ = (id) => document.getElementById(id);
const ui = {
  stage: $("stage"), video: $("cam"), canvas: $("stage-canvas"), empty: $("stage-empty"),
  camBtn: $("cam-btn"), snapBtn: $("snap-btn"), file: $("demo-file"), status: $("demo-status"),
  gauge: $("gauge"), fill: $("g-fill"), needle: $("g-needle"), prob: $("demo-prob"), verdict: $("demo-verdict"),
  spark: $("spark-line"), checks: $("demo-checks"), table: $("demo-features"), meta: $("demo-model-meta"),
};

let landmarkers = {}; // by running mode
let modelPromise = null;
let stream = null;
let live = false;
let frozen = false;
let smooth = null;
let history = [];
let lastVideoTime = -1;

function status(text, kind = "") {
  ui.status.textContent = text;
  ui.status.dataset.kind = kind;
}

function loadModel() {
  modelPromise ||= fetch(new URL("model.json", BASE)).then((r) => {
    if (!r.ok) throw new Error(`model.json: HTTP ${r.status}`);
    return r.json();
  });
  return modelPromise;
}

// face_landmarker.task, or the identical bytes as base64 text on hosts that cannot serve .task
async function modelBytes() {
  const r = await fetch(new URL("face_landmarker.task", VENDOR)).catch(() => null);
  if (r && r.ok) return new Uint8Array(await r.arrayBuffer());
  const t = await fetch(new URL("face_landmarker.task.b64.txt", VENDOR));
  if (!t.ok) throw new Error(`face model: HTTP ${t.status}`);
  const bin = atob((await t.text()).trim());
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

let shared = null;
async function landmarker(mode) {
  if (!landmarkers[mode]) {
    landmarkers[mode] = (async () => {
      shared ||= Promise.all([import(new URL("vision_bundle.mjs", VENDOR).href), modelBytes()]);
      const [{ FaceLandmarker }, bytes] = await shared;
      return FaceLandmarker.createFromOptions(
        { wasmLoaderPath: new URL("vision_wasm_internal.js", VENDOR).href, wasmBinaryPath: new URL("vision_wasm_internal.wasm", VENDOR).href },
        { baseOptions: { modelAssetBuffer: bytes, delegate: "CPU" }, runningMode: mode, numFaces: 2, outputFacialTransformationMatrixes: true },
      );
    })();
    landmarkers[mode].catch(() => { delete landmarkers[mode]; shared = null; });
  }
  return landmarkers[mode];
}

function showModelMeta(m) {
  const cv = m.cross_validated;
  ui.meta.textContent = `Model: trained on ${m.training.n_images.toLocaleString("en-US")} images of ${m.training.n_subjects} people. ` +
    `On people it had not seen it scored AUROC ${cv.auroc.toFixed(3)} (95% CI ${cv.auroc_ci[0].toFixed(3)}–${cv.auroc_ci[1].toFixed(3)}), ` +
    `catching ${(cv.sensitivity * 100).toFixed(0)}% of camera-directed images and rejecting ${(cv.specificity * 100).toFixed(0)}% of the others.`;
}

/* ---------------- camera ---------------- */
async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) return status("This browser has no camera access here. Use Upload photo instead.", "error");
  status("Asking for camera permission…");
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (err) {
    const msg = err && err.name === "NotAllowedError"
      ? "Camera permission was refused or is blocked on this page. Use Upload photo, or allow the camera in your browser settings."
      : "The camera could not start. Use Upload photo instead.";
    return status(msg, "error");
  }
  ui.video.srcObject = stream;
  await ui.video.play();
  status("Loading the face model (about 13 MB, first time only)…");
  try {
    await Promise.all([landmarker("VIDEO"), loadModel()]);
  } catch (err) {
    console.error(err);
    stopCamera();
    return status("The face model couldn't start in this browser (it needs WebAssembly).", "error");
  }
  live = true; frozen = false; smooth = null; history = [];
  ui.empty.hidden = true;
  ui.camBtn.textContent = "Stop camera";
  ui.snapBtn.disabled = false;
  ui.snapBtn.textContent = "Freeze frame";
  status("Live. Look straight into your camera lens, then a little to the side.", "ok");
  requestAnimationFrame(loop);
}

function stopCamera() {
  live = false;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  ui.camBtn.textContent = "Start camera";
  ui.snapBtn.disabled = true;
  ui.stage.classList.remove("lock");
}

async function loop() {
  if (!live) return;
  requestAnimationFrame(loop);
  if (frozen || ui.video.readyState < 2 || ui.video.currentTime === lastVideoTime) return;
  lastVideoTime = ui.video.currentTime;
  const [lm, model] = await Promise.all([landmarker("VIDEO"), loadModel()]);
  const w = ui.video.videoWidth, h = ui.video.videoHeight;
  const res = lm.detectForVideo(ui.video, performance.now());
  drawFrame(ui.video, w, h, true);
  if (!res.faceLandmarks.length) { noFace(); return; }
  const a = analyse(model, res.faceLandmarks[0], res.facialTransformationMatrixes[0], w, h, "col");
  smooth = smooth == null ? a.prob : smooth + 0.35 * (a.prob - smooth);
  history.push([performance.now(), smooth]);
  drawPoints(a.pts, w, h, true);
  render(model, a, smooth, res.faceLandmarks.length);
}

/* ---------------- photo ---------------- */
async function decode(file) {
  try {
    return await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    const url = URL.createObjectURL(file);
    try { const img = new Image(); img.src = url; await img.decode(); return img; } finally { URL.revokeObjectURL(url); }
  }
}

async function runPhoto(file) {
  if (!file || !file.type.startsWith("image/")) return status("That file isn't an image. Choose a JPEG, PNG or WebP photo.", "error");
  if (live) stopCamera();
  status("Loading the face model (about 13 MB, first time only)…");
  let lm, model, src;
  try {
    [lm, model, src] = await Promise.all([landmarker("IMAGE"), loadModel(), decode(file)]);
  } catch (err) {
    console.error(err);
    return status("The face model couldn't start in this browser (it needs WebAssembly).", "error");
  }
  const s = Math.min(1, MAX_SIDE / Math.max(src.width, src.height));
  const w = Math.round(src.width * s), h = Math.round(src.height * s);
  const work = document.createElement("canvas");
  work.width = w; work.height = h;
  work.getContext("2d").drawImage(src, 0, 0, w, h);
  status("Finding the face…");
  await new Promise((r) => setTimeout(r, 20));
  const res = lm.detect(work);
  ui.empty.hidden = true;
  drawFrame(work, w, h, false);
  if (!res.faceLandmarks.length) { noFace(); return status("No face found. Use a photo with one clear, roughly frontal face.", "error"); }
  const a = analyse(model, res.faceLandmarks[0], res.facialTransformationMatrixes[0], w, h, "col");
  history = [];
  ui.spark.setAttribute("points", "");
  drawPoints(a.pts, w, h, false);
  render(model, a, a.prob, res.faceLandmarks.length);
  status("Done. The photo stayed in your browser.", "ok");
}

/* ---------------- drawing ---------------- */
function drawFrame(src, w, h, mirror) {
  const cv = ui.canvas;
  const box = cv.getBoundingClientRect();
  const dpr = Math.min(devicePixelRatio || 1, 2);
  cv.width = Math.round(box.width * dpr); cv.height = Math.round(box.height * dpr);
  const g = cv.getContext("2d");
  const sc = Math.min(cv.width / w, cv.height / h);
  const dw = w * sc, dh = h * sc, ox = (cv.width - dw) / 2, oy = (cv.height - dh) / 2;
  g.save();
  if (mirror) { g.translate(cv.width, 0); g.scale(-1, 1); }
  g.drawImage(src, mirror ? cv.width - ox - dw : ox, oy, dw, dh);
  g.restore();
  ui.canvas._map = { sc, ox, oy, mirror, W: cv.width };
}

function drawPoints(pts, w, h, mirror) {
  const g = ui.canvas.getContext("2d");
  const { sc, ox, oy, W } = ui.canvas._map;
  const X = (p) => (mirror ? W - (ox + p[0] * sc) : ox + p[0] * sc), Y = (p) => oy + p[1] * sc;
  const accent = getComputedStyle(document.documentElement).getPropertyValue("--cobalt").trim() || "#2a44f5";
  g.lineWidth = 2;
  for (const eye of ["right", "left"]) {
    const [a, b] = EYES[eye].corners.map((i) => pts[i]);
    g.strokeStyle = "rgba(255,255,255,0.9)";
    g.beginPath(); g.moveTo(X(a), Y(a)); g.lineTo(X(b), Y(b)); g.stroke();
    for (const i of [...EYES[eye].corners, ...EYES[eye].lids]) { g.fillStyle = "#fff"; g.beginPath(); g.arc(X(pts[i]), Y(pts[i]), 3, 0, 7); g.fill(); }
  }
  for (const i of IRIS_CENTRES) {
    g.strokeStyle = accent; g.lineWidth = 2.5;
    g.beginPath(); g.arc(X(pts[i]), Y(pts[i]), 8, 0, 7); g.stroke();
  }
}

function noFace() {
  ui.prob.textContent = "–";
  ui.verdict.textContent = "no face found";
  ui.stage.classList.remove("lock");
  ui.gauge.classList.remove("lock");
  setGauge(0);
}

function setGauge(p) {
  ui.fill.style.strokeDasharray = `${(p * 100).toFixed(1)} 100`;
  ui.needle.style.transform = `rotate(${-90 + p * 180}deg)`;
}

function render(model, a, p, nFaces) {
  setGauge(p);
  ui.prob.textContent = p.toFixed(2);
  const yes = p >= model.threshold;
  ui.verdict.textContent = yes ? "looks at the camera" : "looks away";
  ui.stage.classList.toggle("lock", yes);
  ui.gauge.classList.toggle("lock", yes);
  // sparkline of the last SPARK_SECONDS
  const now = performance.now();
  history = history.filter(([t]) => now - t < SPARK_SECONDS * 1000);
  ui.spark.setAttribute("points", history.map(([t, v]) => `${(300 * (1 - (now - t) / (SPARK_SECONDS * 1000))).toFixed(1)},${(60 - v * 60).toFixed(1)}`).join(" "));

  const lo = model.training.feature_p01, hi = model.training.feature_p99;
  const outside = a.features.map((v, i) => v < lo[i] || v > hi[i]);
  const checks = [];
  if (nFaces > 1) checks.push(["warn", "More than one face; only the first one MediaPipe returned is analysed."]);
  checks.push(outside.slice(6).some(Boolean)
    ? ["warn", "Head angle is outside the training range (Columbia only turns heads ±30° sideways, with no tilt)."]
    : ["ok", "Head angle is inside the training range."]);
  checks.push(outside.slice(0, 6).some(Boolean)
    ? ["warn", "Some eye measurements are outside the training range (closed eyes, glare or a strong gaze angle)."]
    : ["ok", "Eye measurements are inside the training range."]);
  checks.push(Math.abs(p - model.threshold) < 0.1 ? ["warn", "Close to the 0.5 threshold, so the answer can flip."] : ["ok", `${Math.abs(p - model.threshold).toFixed(2)} away from the 0.5 threshold.`]);
  const html = checks.map(([k, t]) => `<li data-kind="${k}">${t}</li>`).join("");
  if (ui.checks.innerHTML !== html) ui.checks.innerHTML = html;

  const fmt = (v, i) => (i >= 6 ? `${v.toFixed(1)}°` : v.toFixed(3));
  ui.table.innerHTML = a.features.map((v, i) =>
    `<tr class="${outside[i] ? "out" : ""}"><td>${FEATURE_LABELS[i]}</td><td class="num">${fmt(v, i)}</td><td class="num">${fmt(lo[i], i)} to ${fmt(hi[i], i)}</td></tr>`).join("");
}

/* ---------------- wiring ---------------- */
function init() {
  loadModel().then(showModelMeta).catch(() => {});
  ui.camBtn.addEventListener("click", () => (live ? stopCamera() : startCamera()));
  ui.snapBtn.addEventListener("click", () => {
    frozen = !frozen;
    ui.snapBtn.textContent = frozen ? "Resume" : "Freeze frame";
    status(frozen ? "Frame frozen. Nothing was saved." : "Live.", "ok");
  });
  ui.file.addEventListener("change", () => ui.file.files[0] && runPhoto(ui.file.files[0]));
  ui.stage.addEventListener("dragover", (e) => e.preventDefault());
  ui.stage.addEventListener("drop", (e) => {
    e.preventDefault();
    const f = [...(e.dataTransfer?.files || [])].find((x) => x.type.startsWith("image/"));
    if (f) runPhoto(f); else status("That file isn't an image.", "error");
  });
  addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((x) => x.type.startsWith("image/"));
    if (item) runPhoto(item.getAsFile());
  });
  document.addEventListener("visibilitychange", () => { if (document.hidden && live) stopCamera(); });
}

if (ui.camBtn) init();
