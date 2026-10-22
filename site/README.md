# FewGaze project page

A dependency-free static page: `index.html`, `styles.css`, `app.js`, `config.js` and `data/results.json`.
Every chart and number is rendered client-side from `data/results.json`. That file is written by
`scripts/generate_report_assets.py` from the recorded experiment outputs. If it is missing, the page
shows "Experiments not yet run" instead of charts.

## Preview locally

```bash
# from the repository root, so ../report/main.pdf resolves
python -m http.server 8000
# open http://localhost:8000/site/
```

Opening `index.html` directly from disk will not load the JSON, because browsers block `fetch` on `file://`.

## Deploy

* **GitHub Pages:** Settings → Pages → *Deploy from a branch* → `main` / `(root)`. The page is then
  served at `https://<user>.github.io/<repo>/site/`, and the paper link `../report/main.pdf` works
  because `report/main.pdf` is committed.
* **Cloudflare Pages / Netlify:** set the build command to *none* and the output directory to `/` (repo
  root), then visit `/site/`. Alternatively, copy `report/main.pdf` into `site/` and set `paperUrl`
  accordingly.
* Set `repoUrl` in `config.js` to show the "Explore the code" link.

The page loads Google Fonts (Source Serif 4, Inter, JetBrains Mono). It has no analytics, cookies or
backend. The theme toggle stores the light/dark choice in `localStorage`.

**No dataset images are used anywhere on the page** (see `data/README.md` for the licence).
