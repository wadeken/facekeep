# Real-photo test corpus

A small set of **real** photographs used by `tests/test_corpus.py` to verify
behavior that synthetic, geometrically-drawn fixtures cannot: actual face
detection, real-world compression ratios, and fidelity on photographic content.
This is the Phase 2 "real-photo test corpus" item from [../../docs/ROADMAP.md](../../docs/ROADMAP.md).

## The images are not in the repo

To keep the repository light and avoid redistributing third-party image
binaries, the photos are **downloaded on demand**, not committed. Fetch them:

```bash
python tests/corpus/download.py          # download into the local cache
python tests/corpus/download.py --list   # show cache location + status
python tests/corpus/download.py --force  # re-download everything
```

They land in `~/.cache/facekeep/test-corpus/` (override with the
`FACEKEEP_CORPUS_DIR` environment variable). `tests/test_corpus.py` **skips**
when the cache is absent, so the suite stays green offline / in CI without
network access — the trade-off of not vendoring the files. Run `download.py`
once to enable those tests locally.

Each file's bytes are verified against a SHA256 recorded in `manifest.json`, so
an upstream file that silently changes fails loudly during download instead of
corrupting a test.

## Sources, licenses & attribution

All images are from [Wikimedia Commons](https://commons.wikimedia.org/) and are
license-clear. Public Domain files need no attribution; the CC-BY / CC-BY-SA
files are credited here as their licenses require.

| File | Faces | License | Credit |
|------|-------|---------|--------|
| `obama_portrait.jpg` | 1 | CC BY 3.0 | Pete Souza / The Obama-Biden Transition Project |
| `einstein_head.jpg` | 1 | Public domain | Orren Jack Turner (restored) |
| `beatles_group.jpg` | 5 | Public domain | EMI (publicity photo) |
| `snake_river.jpg` | 0 | Public domain | Adumbvoget |
| `hopetoun_falls.jpg` | 0 | CC BY-SA 3.0 | Diliff |
| `migrant_mother.jpg` | 2 | Public domain | Dorothea Lange / U.S. Library of Congress |

The first five are 800px renders for the *faithful* corpus tests.
`migrant_mother.jpg` is intentionally **large** (3840x4929, via `?width=2400`)
and has faces occupying a small fraction of the frame, so *aggressive* mode's
downsample genuinely shrinks the file (ratio > 1) — it drives
`test_corpus_aggressive_regression.py` (the aggressive ratio + restore-LPIPS
regression lock). On the small 800px renders aggressive mode sits below its
design point (faces fill the frame, `.fkeep` can be larger than the input), which
is why this dedicated larger image exists.

`manifest.json` holds the exact source URL, source page, license, attribution,
and SHA256 for each. The `faces` count is what Haar detected on the loaded image
at capture time — it informs the test assertions but is not treated as a strict
oracle (real detection varies; see `test_corpus.py` for the tolerances used).

## Replacing a file

If an upstream URL 404s or its SHA changes, pick a replacement on Commons (a
clearly-licensed photo with the same role — single face / group / faceless),
download it through `Special:FilePath/<File_name>?width=800`, and update that
entry's `url`, `sha256`, `license`, `attribution`, and `faces` in
`manifest.json`. Keep the corpus small.
