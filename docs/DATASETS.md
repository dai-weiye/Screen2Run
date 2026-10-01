# Dataset acquisition and exact cohort

The fixed cohort comprises 250 Pix2Code-Easy, 250 ReDraw-Real, and 100
MobileViews-X screenshots. The backbone subset contains 50, 50, and 20 screens,
respectively. `data/screen_lists/screens.json` records each file's SHA256,
dimensions, source archive member, dataset, and subset membership.

Obtain the original datasets from their providers:

- [pix2code](https://github.com/tonybeltramelli/pix2code): reconstruct and unpack
  its split dataset archive as instructed upstream. Use the Android `all_data`
  images, not the iOS or web split.
- [ReDraw dataset](https://zenodo.org/records/2530277): unpack
  `ReDraw-Final-Google-Play-Dataset.tar.gz`. The released manifest maps each
  `rr` identifier to its original app and screenshot path.
- [MobileViews](https://huggingface.co/datasets/mllmTeam/MobileViews): obtain the
  specified archives under `MobileViews_Apps_CompleteTraces/Zip_Files/`.
  Each selected state is identified by its exact JPEG archive member and hash.

Original screenshot archives are not copied into this package. Dataset licenses
and application content rights remain with their providers. No ground-truth GUI
hierarchies, source XML, or `.gui` targets are used by the generation entry point.

After extracting the archives, assemble the selected screenshots locally:

```sh
python scripts/prepare_screenshots.py \
  --pix2code-root /path/to/extracted/pix2code \
  --redraw-root /path/to/extracted/redraw \
  --mobileviews-root /path/to/mobileviews/shard-zips \
  --out data/screenshots
```

This offline command copies selected Easy/Real images and converts selected
MobileViews JPEGs to RGB PNGs with Pillow. It verifies source hashes before
accepting every file and refuses to overwrite a different destination. It does
not download data or contact an API. `--verify-only` checks an existing prepared
directory. Use the versions in the requirements files for the recorded PNG
serialization. Runtime manifest image paths resolve beneath
`SCREEN2RUN_SCREENSHOTS`. Source-directory and `--out` CLI paths are relative to
the current working directory unless absolute paths are supplied.

The bundled element measurements preserve the snapshot used by the experiment;
only their machine-specific image path has been replaced with the portable
manifest path. OCR strings and numeric bounds are unchanged. Recomputing the
detector on acquired inputs is optional and is documented in `RUNTIME.md`.
