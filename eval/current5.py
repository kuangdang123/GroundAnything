"""GA baseline on the *existing* LA Current5 sample files (CPU HTTP client).

No training, sampling, model loading or mutation of the LA result tree occurs.
Run as python -m eval.current5; see docs/CURRENT5_BASELINE.md.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import statistics
import sys
import time
from urllib.request import Request, urlopen

from eval.current5_metrics import FORMS, parse_la_answer, score_records, summary_cells
from grounding_anything.protocol import build_dense_bbox_prompt, parse_answer

DEFAULT_PROTOCOL_ROOT = "/workspace/nas-data/Feishu-Codex-Agent/workspaces/oc_9adceb0944844d02e4dbfcd75331573d/analysis/current5_eval_20260927"
GA_REF = re.compile(r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>", re.S)
GA_BOX = re.compile(r"<\|box_start\|>(.*?)<\|box_end\|>", re.S)
GA_TUPLE = re.compile(r"<(\d+)><(\d+)><(\d+)><(\d+)>")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    # Only replace this run's generated snapshots, never input data.
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.new')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_tasks(protocol_root):
    root = Path(protocol_root).resolve()
    protocol = json.loads((root / 'protocol.json').read_text(encoding='utf-8'))
    if protocol.get('protocol') != 'current5-v1' or set(protocol['forms']) != set(FORMS):
        raise ValueError('Expected an existing Current5-v1 five-form protocol')
    tasks = []
    for game, info in protocol['games'].items():
        for form in FORMS:
            sample = info['forms'].get(form)
            if not sample:
                continue
            sample_path = Path(sample['sample'])
            observed = digest(sample_path)
            if observed != sample['sample_sha256']:
                raise ValueError(f'Sample SHA256 changed: {sample_path}')
            rows = [json.loads(line) for line in sample_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            if len(rows) != int(sample['sampled_rows']):
                raise ValueError(f'Sample row count changed: {sample_path}')
            image_dir = Path(info['run_dir']) / 'val/images'
            for index, row in enumerate(rows, 1):
                conv = row['conversations']
                question = str(conv[0]['value'])
                body = question.split('：', 1)[-1] if '：' in question else question.split(':', 1)[-1]
                items = [x.strip() for x in body.rstrip('。').strip().split('</c>') if x.strip()]
                if not items:
                    raise ValueError(f'Empty query in {sample_path}:{index}')
                image = (image_dir / row['image']).resolve()
                if not image.is_relative_to(image_dir.resolve()) or not image.is_file():
                    raise ValueError(f'Invalid or missing image: {image}')
                sid = f'{game}/{form}/{index:04d}'
                tasks.append(dict(id=sid, sample_id=sid, game=game, form=form,
                                  row_index=index, image=str(image), image_relative=row['image'],
                                  query_items=items, question=question, gt_answer=str(conv[-1]['value']),
                                  source_sha256=observed))
    return protocol, tasks


def build_prompt(task, prompt_mode):
    if prompt_mode == 'la_zh':
        # Preserve the actual original query, not a translated approximation.
        return task.get('question', '定位所有符合以下描述的目标：' + '</c>'.join(task['query_items']) + '。')
    if prompt_mode == 'ga_native':
        # An explicitly separate prompt ablation; preserves all-instance intent.
        return build_dense_bbox_prompt(task['query_items'])
    raise ValueError(f'Unknown prompt mode: {prompt_mode}')


def clean_ga_answer(text):
    text = text.strip()
    if text.startswith('<think>') and '</think>' in text:
        text = text.split('</think>', 1)[1].strip()
    if text.endswith('<|im_end|>'):
        text = text[:-len('<|im_end|>')].rstrip()
    return text


def parse_ga_answer(text, width, height):
    """Recover complete GA boxes, retaining duplicates, like LA's recovery parser.

    Do not change predictions to improve scores. Canonical validity is separate.
    GA grid 0..999 maps to 0..W-1, unlike LA's 0..1000 -> W conversion.
    """
    groups = defaultdict(list)
    current = 'unknown'
    saw_none = False
    units = sorted([(m.start(), 'ref', m.group(1)) for m in GA_REF.finditer(text)] +
                   [(m.start(), 'box', m.group(1)) for m in GA_BOX.finditer(text)])
    for _, kind, payload in units:
        if kind == 'ref':
            current = ' '.join(payload.split()).strip().casefold() or 'unknown'
            continue
        saw_none |= payload.strip().casefold() == 'none'
        for match in GA_TUPLE.finditer(payload):
            coords = [int(v) for v in match.groups()]
            if any(v > 999 for v in coords):
                continue
            x1, y1, x2, y2 = coords
            if x2 <= x1 or y2 <= y1:
                continue
            groups[current].append([x1 / 999 * (width - 1), y1 / 999 * (height - 1),
                                    x2 / 999 * (width - 1), y2 / 999 * (height - 1)])
    groups = dict(groups)
    return groups, 'boxes' if groups else ('none' if saw_none else 'empty')


def get_json(url, timeout=30):
    with urlopen(url, timeout=timeout) as response:
        return json.load(response)


def check_server(base_url, decoder):
    from infer.decoding import validate_server_decoder
    root = base_url.rstrip('/')
    if root.endswith('/v1'):
        root = root[:-3]
    info = get_json(root + '/server_info')
    validate_server_decoder(decoder, info)
    if int(info.get('max_running_requests', 1)) != 1:
        raise ValueError('Current5 single-request protocol requires max_running_requests=1')
    return info


def request(task, args):
    from PIL import Image
    started = time.perf_counter()
    with Image.open(task['image']) as opened:
        image = opened.convert('RGB')
        width, height = image.size
        buffer = BytesIO()
        # Same decoded RGB pixels; PNG is lossless (no new JPEG artifacts).
        image.save(buffer, format='PNG')
    gt_groups, gt_status = parse_la_answer(task['gt_answer'], width, height)
    prompt = build_prompt(task, args.prompt_mode)
    body = dict(model=args.model, temperature=args.temperature, top_p=args.top_p,
                max_tokens=args.max_tokens, seed=args.seed, stream=False,
                skip_special_tokens=False, spaces_between_special_tokens=False,
                messages=[{'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')}},
                    {'type': 'text', 'text': prompt}]}])
    if args.decoder == 'denoise':
        # Generic Current5 is NOT a registered benchmark tier; do not enforce
        # single-target cardinality on all-instance or negative examples.
        body['custom_params'] = {'gam_dlm_decode': dict(temperature=args.temperature,
                                                       top_p=args.top_p, sampling_seed=args.seed,
                                                       expected_cardinality=None)}
    image_sha256 = digest(task['image'])
    preparation_seconds = time.perf_counter() - started
    record = dict(sample_id=task['sample_id'], game=task['game'], form=task['form'],
                  row_index=task['row_index'], image=task['image'], image_size=[width, height],
                  image_sha256=image_sha256, source_sha256=task['source_sha256'],
                  query_labels=task['query_items'], prompt=prompt, gt_groups=gt_groups,
                  gt_status=gt_status, pred_groups={}, pred_status='empty', valid=False,
                  raw_answer='', finish_reason=None, usage={}, error=None,
                  transport_ok=False)
    call_start = time.perf_counter()
    try:
        req = Request(args.base_url.rstrip('/') + '/chat/completions',
                      data=json.dumps(body, ensure_ascii=False).encode('utf-8'),
                      headers={'Content-Type': 'application/json'})
        with urlopen(req, timeout=args.timeout) as response:
            payload = json.load(response)
        choice = payload['choices'][0]
        record.update(raw_answer=choice['message']['content'] or '',
                      finish_reason=choice.get('finish_reason'), usage=payload.get('usage') or {},
                      transport_ok=True)
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    request_seconds = time.perf_counter() - call_start
    parse_start = time.perf_counter()
    cleaned = clean_ga_answer(record['raw_answer'])
    record['pred_groups'], record['pred_status'] = parse_ga_answer(cleaned, width, height)
    try:
        parse_answer(cleaned, 'bbox', require_canonical=True)
        record['valid'] = record['finish_reason'] == 'stop'
        if not record['valid']:
            record['parse_error'] = 'Generation did not finish naturally'
    except ValueError as exc:
        record['parse_error'] = str(exc)
    record.update(preparation_seconds=preparation_seconds,
                  request_seconds=request_seconds, parse_seconds=time.perf_counter() - parse_start,
                  latency_seconds=time.perf_counter() - started)
    record['prediction_sha256'] = fingerprint(record['pred_groups'])
    return record


def load_records(path):
    if not Path(path).exists():
        return []
    records = []
    for index, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f'Damaged output {path}:{index}; preserve it and select a new run') from exc
    return records


def summarize_run(output, protocol, tasks):
    output = Path(output)
    latest = {}
    for path in sorted(output.glob('predictions-shard*.jsonl')):
        for record in load_records(path):
            sid = record['sample_id']
            if sid in latest and latest[sid].get('shard') != record.get('shard'):
                raise ValueError(f'Duplicate sample across shards: {sid}')
            latest[sid] = record
    cells = defaultdict(dict)
    for game in protocol['games']:
        for form in FORMS:
            expected = [t for t in tasks if t['game'] == game and t['form'] == form]
            if not expected:
                continue
            records = [latest[t['sample_id']] for t in expected if t['sample_id'] in latest]
            # Transport errors are NOT successful no-object predictions. Preserve
            # error rows, but require successful completion for final coverage.
            good = [r for r in records if r.get('transport_ok')]
            if not good:
                continue
            metrics = score_records(good, form)
            metrics['transport_errors'] = sum(not r.get('transport_ok') for r in records)
            metrics['canonical_valid_rate'] = sum(r.get('valid', False) for r in good) / len(good)
            metrics['sample_sha256'] = expected[0]['source_sha256']
            cells[form][game] = metrics
    result = summary_cells(dict(cells), protocol)
    result['cells'] = dict(cells)
    result['attempted_requests'] = len(latest)
    result['transport_errors'] = sum(not r.get('transport_ok') for r in latest.values())
    write_json(output / 'summary.json', result)
    return result


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol-root', default=DEFAULT_PROTOCOL_ROOT)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:8111/v1')
    p.add_argument('--model', default='groundinganything')
    p.add_argument('--decoder', choices=('denoise', 'causal', 'speculative'), default='denoise')
    p.add_argument('--prompt-mode', choices=('la_zh', 'ga_native'), default='la_zh')
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--limit', type=int, help='Smoke only; incomplete output is never a full baseline')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--summarize', action='store_true')
    args = p.parse_args(argv)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        p.error('Invalid shard index/count')
    if args.max_tokens < 1 or args.timeout <= 0 or not 0 <= args.temperature <= 2 or not 0 < args.top_p <= 1:
        p.error('Invalid generation parameters')
    if args.decoder in ('causal', 'speculative') and (args.temperature != 0 or args.top_p != 1):
        p.error('Controlled causal/speculative routes require --temperature 0 --top-p 1')
    return args


def main(argv=None):
    args = arguments(argv)
    protocol, tasks = load_tasks(args.protocol_root)
    cell_keys = list(dict.fromkeys((t['game'], t['form']) for t in tasks))
    shard_keys = {k for i, k in enumerate(cell_keys) if i % args.num_shards == args.shard_index}
    selected = [t for t in tasks if (t['game'], t['form']) in shard_keys]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('limit must be positive')
        selected = selected[:args.limit]
    spec = dict(protocol_sha256=digest(Path(args.protocol_root) / 'protocol.json'),
                samples={k: v['source_sha256'] for k, v in ((t['sample_id'], t) for t in tasks)},
                decoder=args.decoder, prompt_mode=args.prompt_mode, temperature=args.temperature,
                top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed, model=args.model,
                num_shards=args.num_shards, limit=args.limit,
                implementation_sha256={name: digest(Path(__file__).parent / name) for name in ('current5.py', 'current5_metrics.py')},
                parser='ga-complete-box-recovery-v1', coordinate_mapping='GA: /999*(W-1); LA-GT: /1000*W')
    if args.dry_run:
        print(json.dumps(dict(spec=spec, tasks=len(tasks), shard_tasks=len(selected), cells=len(cell_keys)), ensure_ascii=False, indent=2))
        return 0
    output = Path(args.output_dir).resolve()
    if output == Path(args.protocol_root).resolve() or output.is_relative_to(Path(args.protocol_root).resolve()):
        raise ValueError('Output must not mutate LA protocol/result tree')
    output.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        stored = json.loads((output / 'config-shard0.json').read_text(encoding='utf-8'))
        if stored['protocol_sha256'] != spec['protocol_sha256']:
            raise ValueError('Protocol differs from recorded run')
        result = summarize_run(output, protocol, tasks)
        print(json.dumps({k: result.get(k) for k in ('coverage','current5_equal_view_macro','current5_training_weighted')}, indent=2))
        return 0 if result.get('complete') else 2
    config_path = output / f'config-shard{args.shard_index}.json'
    for existing in output.glob('config-shard*.json'):
        if json.loads(existing.read_text(encoding='utf-8')) != spec:
            raise ValueError('Resume/shard configuration differs; use a fresh output directory')
    write_json(config_path, spec)
    info = check_server(args.base_url, args.decoder)
    server_path = output / f'server-shard{args.shard_index}.json'
    if server_path.exists() and json.loads(server_path.read_text(encoding='utf-8')) != info:
        raise ValueError('Effective server settings changed; use a fresh output directory')
    write_json(server_path, info)
    write_json(output / f'command-shard{args.shard_index}.json', dict(argv=sys.argv, started_at=datetime.now(timezone.utc).isoformat()))
    write_json(output / f'protocol-shard{args.shard_index}.json', protocol)
    path = output / f'predictions-shard{args.shard_index}.jsonl'
    previous = {r['sample_id']: r for r in load_records(path)}
    print(f'[OP:开始] baseline shard={args.shard_index}/{args.num_shards} requests={len(selected)}', flush=True)
    with path.open('a', encoding='utf-8') as stream:
        for index, task in enumerate(selected, 1):
            old = previous.get(task['sample_id'])
            if old and old.get('transport_ok'):
                if old['image_sha256'] != digest(task['image']):
                    raise ValueError('Image changed during resume: ' + task['image'])
                continue
            record = request(task, args)
            record['shard'] = args.shard_index
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            print(f"[current5] {index}/{len(selected)} {task['sample_id']} elapsed={record['latency_seconds']:.3f}s transport={record['transport_ok']} valid={record['valid']}", flush=True)
    # No shared summary writer until all shards have exited.
    if args.num_shards == 1:
        summarize_run(output, protocol, tasks)
    print(f'[OP:成功] shard={args.shard_index} predictions={path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
