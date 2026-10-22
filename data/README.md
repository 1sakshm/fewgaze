# Data

FewGaze does **not** include or redistribute any dataset images. You must obtain the data yourself under its terms.

## Columbia Gaze Data Set

* Project page: <https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/>
* Paper (cite when using the data): B. A. Smith, Q. Yin, S. K. Feiner, S. K. Nayar,
  *Gaze Locking: Passive Eye Contact Detection for Human–Object Interaction*, ACM UIST 2013,
  pp. 271–280, doi:10.1145/2501988.2501994.
* Terms stated on the project page: **non-commercial use only**; cite the paper. No redistribution
  permission is stated, so do not commit or publish the images (the `.gitignore` excludes `data/raw/`).

### Download

1. Open the project page and click the download link. A form asks for your e-mail address; submitting it
   is how you accept the terms. The form then links `columbia_gaze_data_set.zip`, which is
   2,378,561,401 bytes. The file we used was downloaded on 2026-10-22.
2. Put the zip in `data/raw/` and unpack it there, skipping the macOS metadata folder:

   ```bash
   cd data/raw
   unzip columbia_gaze_data_set.zip -x "__MACOSX/*"
   ```

   This creates `data/raw/Columbia Gaze Data Set/0001 … 0056/`. Each subject folder holds 105 JPEGs, plus
   some `Thumbs.db`, `.DS_Store` and `.picasa.ini` files, which the manifest ignores.
3. If you unpack it somewhere else, set `dataset.root` in `configs/experiment.yaml`.

### Filename convention

`0001_2m_-15P_10V_-5H.jpg` = subject `0001`, capture distance 2 m, head pose −15°, vertical gaze +10°,
horizontal gaze −5°. The paper does not document this convention. FewGaze parses filenames with the
configurable `dataset.filename_regex` and cross-checks the subject ID against the folder name.

### Audit

```bash
python scripts/audit_dataset.py --config configs/experiment.yaml
```

This writes `results/audit/audit.json`, `audit_summary.md`, `manifest.csv` (original annotations),
`labels.csv` (derived labels) and `results/figures/fig2_dataset_composition.png`. The audit of our
copy found 5,880 readable 5184×3456 JPEGs from 56 subjects, a complete 5 × 21 grid per subject, and
**one pair of byte-identical files with different gaze annotations**
(`0007_2m_-15P_-10V_0H` and `0007_2m_-15P_-10V_5H`). Both files are excluded from the study; see
`docs/dataset_card.md`.

## Derived data

`cache/` holds face crops (derived images, so they are also never committed), feature matrices and
model weights. Everything in it can be regenerated with `scripts/extract_features.py`.
