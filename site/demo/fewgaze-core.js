// FewGaze demo core: a line-by-line port of fewgaze.geometry (eye_features, head_rotation) and of the
// exported scikit-learn pipeline (scripts/export_demo_model.py: predict_exported). Parity with the Python
// implementation is checked by site/demo/parity.html (see site/demo/README.md).

// MediaPipe Face Mesh indices, identical to fewgaze/geometry.py
export const EYES = {
  right: { corners: [33, 133], lids: [159, 145] },
  left: { corners: [362, 263], lids: [386, 374] },
};
export const IRIS_CENTRES = [468, 473];

const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const norm = (a) => Math.hypot(a[0], a[1]);

/** Six per-eye features from (478, 2) pixel landmarks: [iris_x, iris_y, aperture] for right then left eye. */
export function eyeFeatures(pts) {
  const out = [];
  const irises = IRIS_CENTRES.map((i) => pts[i]);
  for (const eye of ["right", "left"]) {
    let a = pts[EYES[eye].corners[0]];
    let b = pts[EYES[eye].corners[1]];
    if (a[0] > b[0]) [a, b] = [b, a]; // orient the axis towards image-right for both eyes
    const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const width = norm(sub(b, a));
    const u = [(b[0] - a[0]) / width, (b[1] - a[1]) / width];
    const v = [-u[1], u[0]];
    const iris = irises.reduce((best, p) => (norm(sub(p, mid)) < norm(sub(best, mid)) ? p : best));
    const d = sub(iris, mid);
    const up = pts[EYES[eye].lids[0]];
    const lo = pts[EYES[eye].lids[1]];
    out.push(dot(d, u) / width, dot(d, v) / width, norm(sub(up, lo)) / width);
  }
  return out;
}

function det3(m) {
  return (
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
  );
}

function invT3(m) {
  // inverse-transpose of a 3x3 matrix (cofactor matrix / determinant)
  const d = det3(m);
  const c = (r0, c0, r1, c1) => m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
  return [
    [c(1, 1, 2, 2), -c(1, 0, 2, 2), c(1, 0, 2, 1)],
    [-c(0, 1, 2, 2), c(0, 0, 2, 2), -c(0, 0, 2, 1)],
    [c(0, 1, 1, 2), -c(0, 0, 1, 2), c(0, 0, 1, 1)],
  ].map((row) => row.map((x) => x / d));
}

/**
 * Head rotation (deg) from a 4x4 facial transformation matrix given as rows[r][c].
 * Python: normalise columns, take the nearest orthonormal matrix (SVD U V^T), scipy as_euler("xyz").
 * Here: the same normalisation, the polar factor by Newton iteration (equal to U V^T), and the closed-form
 * extrinsic x-y-z Euler angles (R = Rz(c) Ry(b) Rx(a)), checked against scipy in the parity test.
 */
export function headRotation(rows) {
  let r = [0, 1, 2].map((i) => [0, 1, 2].map((j) => rows[i][j]));
  for (let j = 0; j < 3; j++) {
    const n = Math.hypot(r[0][j], r[1][j], r[2][j]);
    for (let i = 0; i < 3; i++) r[i][j] /= n;
  }
  for (let it = 0; it < 50; it++) {
    const t = invT3(r);
    r = r.map((row, i) => row.map((x, j) => 0.5 * (x + t[i][j])));
  }
  const deg = 180 / Math.PI;
  const b = -Math.asin(Math.max(-1, Math.min(1, r[2][0])));
  const a = Math.atan2(r[2][1], r[2][2]);
  const c = Math.atan2(r[1][0], r[0][0]);
  return [a * deg, b * deg, c * deg];
}

/** Convert MediaPipe's Matrix {rows, columns, data} into rows[r][c]; `layout` is "row" or "col". */
export function matrixRows(m, layout) {
  const R = m.rows || 4;
  const C = m.columns || 4;
  return Array.from({ length: R }, (_, i) => Array.from({ length: C }, (_, j) => (layout === "row" ? m.data[i * C + j] : m.data[j * R + i])));
}

/** P(camera-directed) from the exported pipeline: scale -> polynomial -> scale -> logistic. */
export function predict(model, x) {
  const z = x.map((v, i) => (v - model.scaler1.mean[i]) / model.scaler1.scale[i]);
  let logit = model.intercept;
  model.poly_powers.forEach((pw, k) => {
    let t = 1;
    for (let i = 0; i < pw.length; i++) if (pw[i]) t *= z[i] ** pw[i];
    logit += model.coef[k] * ((t - model.scaler2.mean[k]) / model.scaler2.scale[k]);
  });
  return 1 / (1 + Math.exp(-logit));
}

/** Landmarks (normalised) -> features + probability. */
export function analyse(model, landmarks, matrix, width, height, layout) {
  const pts = landmarks.map((p) => [p.x * width, p.y * height]);
  const features = eyeFeatures(pts).concat(headRotation(matrixRows(matrix, layout)));
  return { pts, features, prob: predict(model, features) };
}
