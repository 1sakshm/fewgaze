// FewGaze live demo: runs entirely in the viewer's browser. The photo is never uploaded anywhere.
// Pipeline = the study's geometry method: MediaPipe Face Landmarker (tasks-vision 0.10.21, same model file
// as the Python run) -> 9 eye/head features (fewgaze-core.js, parity-tested) -> exported logistic regression.
import { analyse, EYES, IRIS_CENTRES } from "./fewgaze-core.js";

const BASE = new URL(".", import.meta.url); // demo/ folder
const VENDOR = new URL("vendor/", BASE);
const MAX_SIDE = 1600; // same longest side as the Python pipeline
const FEATURE_LABELS = [
  "Right iris, along eye", "Right iris, across eye", "Right eye opening",
  "Left iris, along eye", "Left iris, across eye", "Left eye opening",
  "Head pitch", "Head yaw", "Head roll",
];

const $ = (id) => document.getElementById(id);
const els = {
  input: $("demo-file"), drop: $("demo-drop"), status: $("demo-status"), result: $("demo-result"),
  canvas: $("demo-canvas"), prob: $("demo-prob"), bar: $("demo-bar"), verdict: $("demo-verdict"),
  checks: $("demo-checks"), table: $("demo-features"), meta: $("demo-model-meta"),
};


let landmarkerPromise = null;
let modelPromise = null;

function status(text, kind = "") {
  els.status.textContent = text;
  els.status.dataset.kind = kind;
}

function loadModel() {
  modelPromise ||= fetch(new URL("model.json", BASE)).then((r) => {
    if (!r.ok) throw new Error(`model.json: HTTP ${r.status}`);
    return r.json();
  });
  return modelPromise;
}

// The landmarker model is face_landmarker.task. Hosts that cannot serve that file type get the identical bytes
// as base64 text (face_landmarker.task.b64.txt); either way MediaPipe receives the same buffer.
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

function loadLandmarker() {
  landmarkerPromise ||= (async () => {
    const [{ FaceLandmarker }, bytes] = await Promise.all([import(new URL("vision_bundle.mjs", VENDOR).href), modelBytes()]);
    return FaceLandmarker.createFromOptions(
      { wasmLoaderPath: new URL("vision_wasm_internal.js", VENDOR).href, wasmBinaryPath: new URL("vision_wasm_internal.wasm", VENDOR).href },
      { baseOptions: { modelAssetBuffer: bytes, delegate: "CPU" }, runningMode: "IMAGE", numFaces: 2, outputFacialTransformationMatrixes: true },
    );
  })();
  landmarkerPromise.catch(() => { landmarkerPromise = null; });
  return landmarkerPromise;
}

function init() {
  loadModel().then(showModelMeta).catch(() => {});
  els.input.addEventListener("change", () => els.input.files[0] && run(els.input.files[0]));
  ["dragenter", "dragover"].forEach((t) => els.drop.addEventListener(t, (e) => { e.preventDefault(); els.drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => els.drop.addEventListener(t, () => els.drop.classList.remove("over")));
  els.drop.addEventListener("drop", (e) => {
    e.preventDefault();
    const f = [...(e.dataTransfer?.files || [])].find((x) => x.type.startsWith("image/"));
    if (f) run(f); else status("That file isn't an image. Drop a JPEG, PNG or WebP photo.", "error");
  });
  addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((x) => x.type.startsWith("image/"));
    if (item && document.getElementById("demo")?.getBoundingClientRect().top < innerHeight) run(item.getAsFile());
  });
}

function showModelMeta(m) {
  const cv = m.cross_validated;
  els.meta.textContent = `Trained on ${m.training.n_images.toLocaleString("en-US")} images of ${m.training.n_subjects} people. ` +
    `On people it had not seen, this pipeline scored AUROC ${cv.auroc.toFixed(3)} (95% CI ${cv.auroc_ci[0].toFixed(3)}–${cv.auroc_ci[1].toFixed(3)}), ` +
    `catching ${(cv.sensitivity * 100).toFixed(0)}% of camera-directed images and correctly rejecting ${(cv.specificity * 100).toFixed(0)}% of the others (run ${cv.run_id}).`;
}

async function decode(file) {
  try {
    return await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    const url = URL.createObjectURL(file);
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      return img;
    } finally {
      URL.revokeObjectURL(url);
    }
  }
}

async function run(file) {
  els.result.hidden = true;
  if (!file.type.startsWith("image/")) return status("That file isn't an image. Choose a JPEG, PNG or WebP photo.", "error");
  status("Loading the face model (about 13 MB, first time only)…", "busy");
  let model, lm, src;
  try {
    [model, lm, src] = await Promise.all([loadModel(), loadLandmarker(), decode(file)]);
  } catch (err) {
    console.error(err);
    return status("The face model couldn't start in this browser. It needs WebAssembly, which this viewer may block. Open the project page from its own website to try the demo.", "error");
  }
  const w0 = src.width, h0 = src.height;
  const s = Math.min(1, MAX_SIDE / Math.max(w0, h0));
  const w = Math.round(w0 * s), h = Math.round(h0 * s);
  const work = document.createElement("canvas");
  work.width = w; work.height = h;
  work.getContext("2d").drawImage(src, 0, 0, w, h);
  status("Finding the face…", "busy");
  await new Promise((r) => setTimeout(r, 30));
  let res;
  try {
    res = lm.detect(work);
  } catch (err) {
    console.error(err);
    return status("The face model failed on this image. Try another photo.", "error");
  }
  if (!res.faceLandmarks.length) return status("No face found. Use a photo where one face is clearly visible and roughly frontal.", "error");
  const a = analyse(model, res.faceLandmarks[0], res.facialTransformationMatrixes[0], w, h, "col");
  render(model, a, work, res.faceLandmarks.length);
  status("Done. The photo stayed in your browser.", "ok");
}

function render(model, a, work, nFaces) {
  drawOverlay(work, a.pts);
  const p = a.prob;
  els.prob.textContent = p.toFixed(2);
  els.bar.style.setProperty("--p", (p * 100).toFixed(1) + "%");
  const yes = p >= model.threshold;
  els.verdict.textContent = yes ? "Looks camera-directed" : "Looks away from the camera";
  els.verdict.dataset.kind = yes ? "yes" : "no";

  const lo = model.training.feature_p01, hi = model.training.feature_p99;
  const outside = a.features.map((v, i) => v < lo[i] || v > hi[i]);
  const checks = [];
  if (nFaces > 1) checks.push(["warn", "More than one face found; only the first face MediaPipe returned was analysed."]);
  const poseOut = outside.slice(6).some(Boolean);
  const eyeOut = outside.slice(0, 6).some(Boolean);
  checks.push(poseOut
    ? ["warn", "Head angle is outside what the model saw in training (Columbia only turns the camera ±30° sideways, with no tilt). Treat this result as unreliable."]
    : ["ok", "Head angle is within the training range."]);
  checks.push(eyeOut
    ? ["warn", "Some eye measurements are outside the training range (closed eyes, glare, unusual crop or a strong gaze angle)."]
    : ["ok", "Eye measurements are within the training range."]);
  checks.push(Math.abs(p - model.threshold) < 0.1
    ? ["warn", "The score is close to the decision threshold, so the answer could easily flip."]
    : ["ok", `Score is ${Math.abs(p - model.threshold).toFixed(2)} away from the 0.5 threshold.`]);
  els.checks.innerHTML = "";
  for (const [kind, text] of checks) {
    const li = document.createElement("li");
    li.dataset.kind = kind;
    li.textContent = text;
    els.checks.appendChild(li);
  }

  const fmt = (v, i) => (i >= 6 ? `${v.toFixed(1)}°` : v.toFixed(3));
  els.table.innerHTML = "";
  a.features.forEach((v, i) => {
    const tr = document.createElement("tr");
    if (outside[i]) tr.className = "out";
    tr.innerHTML = `<td></td><td class="num"></td><td class="num muted"></td>`;
    tr.children[0].textContent = FEATURE_LABELS[i];
    tr.children[1].textContent = fmt(v, i);
    tr.children[2].textContent = `${fmt(lo[i], i)} to ${fmt(hi[i], i)}`;
    els.table.appendChild(tr);
  });
  els.result.hidden = false;
}

function drawOverlay(work, pts) {
  // crop around the eyes so the measured points are visible
  const ids = [EYES.right.corners, EYES.left.corners, EYES.right.lids, EYES.left.lids].flat().concat(IRIS_CENTRES);
  const xs = ids.map((i) => pts[i][0]), ys = ids.map((i) => pts[i][1]);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const wBox = (Math.max(...xs) - Math.min(...xs)) * 1.5;
  const hBox = wBox * 0.42;
  const cv = els.canvas;
  const W = 720, H = Math.round(W * 0.42);
  cv.width = W; cv.height = H;
  const g = cv.getContext("2d");
  const sx = cx - wBox / 2, sy = cy - hBox / 2;
  g.drawImage(work, sx, sy, wBox, hBox, 0, 0, W, H);
  const k = W / wBox;
  const P = (i) => [(pts[i][0] - sx) * k, (pts[i][1] - sy) * k];
  const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#b4441b";
  g.lineWidth = 2;
  for (const eye of ["right", "left"]) {
    const [a, b] = EYES[eye].corners.map(P);
    g.strokeStyle = "rgba(255,255,255,0.85)";
    g.beginPath(); g.moveTo(...a); g.lineTo(...b); g.stroke();
    for (const q of [a, b, ...EYES[eye].lids.map(P)]) { g.fillStyle = "#fff"; g.beginPath(); g.arc(q[0], q[1], 3.5, 0, 7); g.fill(); }
  }
  for (const i of IRIS_CENTRES) {
    const [x, y] = P(i);
    g.strokeStyle = accent; g.lineWidth = 2.5;
    g.beginPath(); g.arc(x, y, 7, 0, 7); g.stroke();
    g.fillStyle = accent; g.beginPath(); g.arc(x, y, 2.5, 0, 7); g.fill();
  }
}

if (els.input) init();
