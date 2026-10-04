# LocateAnything to GroundAnything migration

LocateAnything's game SFT annotations use ShareGPT rows:

```json
{"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"<ref>button</ref><box><10><20><300><400></box>"}],"image":"screens/a.png"}
```

GroundAnything's training loaders require the canonical row contract:

```json
{"id":"...","messages":[{"role":"user","content":"<image> ...","loss":null},{"role":"assistant","content":"<|object_ref_start|>button<|object_ref_end|><|box_start|><10><20><299><399><|box_end|>","loss":null}],"images":["screens/a.png"],"source":"..."}
```

Use the converter without modifying the LA source file:

```powershell
python tools/convert_locany_to_gam.py `
  --input D:\data\locany\train.jsonl `
  --output data\game\train.gam.jsonl `
  --metadata-output data\game\train.gam.meta.jsonl `
  --rejects-output data\game\train.gam.rejects.jsonl `
  --source locateanything-game
python tools/validate_gam_jsonl.py --input data\game\train.gam.jsonl
```

The converter maps `human/gpt` to `user/assistant`, normalizes `<image-1>` to
`<image>`, inserts a missing image placeholder when the LA row has one image,
and checks that the number of placeholders equals the image list. LocateAnything
coordinates are inclusive integers on `[0,1000]`; GroundAnything uses `[0,999]`,
so each coordinate is rounded with `round(value * 999 / 1000)` and clipped.

LA `<box>none</box>` is represented as a GAM entry with a phrase and the payload
`None`. The phrase is taken from the target portion of the user prompt. For
ambiguous prompts, pass `--default-phrase` and inspect the metadata sidecar. When
`--rejects-output` is supplied, malformed rows are kept in a JSONL quarantine
with the original line and error instead of stopping the rest of the conversion;
without it, the first malformed row fails the command.

The output is still annotation JSONL. Build the Arrow cache in the next step:

```powershell
python tools/build_gam_cache.py `
  --input data\game\train.gam.jsonl `
  --cache data\game\cache\train `
  --manifest data\game\cache\cache_manifest.json `
  --source-id game_train `
  --tokenizer weights\base_model
```

GroundAnything DLM training consumes that Arrow cache plus `cache_manifest.json`.
Do not compare LA's `[0,1000]` metrics with GAM's `[0,999]` metrics without
converting both predictions to pixel space first.
