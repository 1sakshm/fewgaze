# Commit hash map

The history was rewritten once before the first push, to edit commit messages only. Every commit's
**tree** (its exact file contents) is unchanged, so provenance records that cite a pre-rewrite hash can still be
verified. Find the old hash below, then check that `git rev-parse <new>^{tree}` prints the tree listed.

Provenance files written before the rewrite (`results/runs/*/provenance.json`, `results/posthoc/runs/*/provenance.json`,
`site/data/results.json`, `site/demo/model.json`) keep the hash recorded at run time. A `-dirty` suffix means the run
started with uncommitted changes, as `docs/methodology.md` explains.

| Pre-rewrite | Published | Tree |
|---|---|---|
| `0c4ad7a` | `3964422` | `6911cd0fe1e38961aaef5b813c8bbf275fb0c0ae` |
| `0fb6dcf` | `0abeba2` | `01c7ae79e1a0965f3074d1a6b76b527cc4958e7f` |
| `b7bc7d1` | `1c30743` | `593c5f8ead7196b846a84bbd6e0372b6134dad21` |
| `e120ebe` | `e06398d` | `0700c180f59ada96349a40b3a4bd73d165c60921` |
| `fb79073` | `d86bd5f` | `2dd6ae211332d4359a43d3edacc0968801ca9b6c` |
| `5182900` | `aaf9842` | `a5b8c1a9e427ed39db3fbcf8deed3d3f3a64e4a7` |
| `cba4e35` | `0a65ac4` | `97220ad35fda4ed8a6f0e1bc1d9c6d292e2fbc17` |
| `dd38cd5` | `88d1489` | `592f7bab840b2f5279d38b7987f4ff251866a746` |
| `3fbfc55` | `5f6e0ba` | `6582947fad06e83e696f7d0e5119bef5612f3eea` |
| `d26428d` | `841a70a` | `4fee8439fa60696d90ed83ebe016efe0743c790d` |
