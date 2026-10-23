# FewGaze website

A static, dependency-light site: `index.html`, `styles.css`, `app.js`, `config.js`, `assets/`, `data/`, `demo/`.

* **Hero:** MediaPipe's canonical face mesh (Apache-2.0, `assets/face_mesh.json`), rendered with three.js 0.160.0
  from jsDelivr, looks at the pointer across Columbia's 7 × 3 gaze-target grid. The HUD shows the Columbia label
  for the current direction and "locks" inside the detector tolerance Smith et al. (UIST 2013) estimated
  (±2.5° horizontally, ±5° vertically). Without a pointer, the gaze wanders the grid. With reduced motion it stays still.
* **Data-driven sections:** every number and chart is read from `data/results.json`
  (`scripts/generate_report_assets.py`) and `data/extras.json` (`scripts/export_site_data.py`). When the files are
  missing, the page says so instead of drawing charts.
* **Live demo:** camera (getUserMedia, video mode) or photo upload. MediaPipe tasks-vision 0.10.21 and the parity-tested
  geometry classifier run on the device, and nothing is uploaded. See `demo/README.md`.

## Preview locally

```bash
python -m http.server 8000 --directory site      # http://localhost:8000
```

The camera needs a secure context: `localhost` and any HTTPS host qualify.

## Deploy

* **GitHub Pages:** Settings → Pages → deploy from branch `main`, folder `/site` (or the repo root, then visit
  `/site/`). Pages serves `.wasm` and `.task` correctly. Set `repoUrl` and `paperUrl` in `config.js`. If you deploy
  only `/site`, also copy `report/main.pdf` into `site/` and point `paperUrl` at `main.pdf`.
* **claude.ai Artifact:** run `python scripts/build_artifact.py --out <dir>` and publish `<dir>/index.html` with the
  listed files. The artifact viewer has no camera access for this account, so visitors there use photo upload.

No analytics, cookies or backend. The theme choice is the only thing stored (`localStorage`).
**No dataset images appear anywhere on the site.**
