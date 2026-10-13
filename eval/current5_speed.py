"""Sequential Current5 speed measurement: CPU HTTP client, never starts a model.

Run ``python -m eval.current5_speed --help``. Reuse --manifest across decoder
modes, but use a separate --output-dir for each configuration. GPU measurements
are optional nvidia-smi samples, NOT an actual peak-memory measurement.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

from eval.current5 import (DEFAULT_PROTOCOL_ROOT, check_server, digest,
                           fingerprint, load_tasks, request, write_json)
from eval.current5_metrics import parse_la_answer

SCHEMA = 'current5-speed-v1'


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol-root', default=DEFAULT_PROTOCOL_ROOT)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:8111/v1')
    p.add_argument('--model', default='groundinganything')
    p.add_argument('--mode', choices=('denoise', 'causal', 'speculative'), default='denoise')
    p.add_argument('--prompt-mode', choices=('la_zh', 'ga_native'), default='la_zh')
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--per-cell', type=int, default=1)
    p.add_argument('--selection', choices=('first', 'hash'), default='first')
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--manifest', help='Shared sample plan JSON (independent of decoder/prompt mode)')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--gpu', help='Optional nvidia-smi GPU index or UUID')
    p.add_argument('--require-isolated', action='store_true')
    p.add_argument('--allowed-gpu-pids', default='', help='Comma-separated allowed compute PIDs')
    p.add_argument('--service-parent-pid', type=int, help='Allow this PID and current /proc descendants')
    args = p.parse_args(argv)
    if args.max_tokens < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        p.error('max-tokens and timeout must be positive and finite')
    if args.per_cell < 1 or args.repeats < 1 or args.warmup < 0:
        p.error('per-cell/repeats must be positive; warmup must be nonnegative')
    try:
        args.allowed_gpu_pids = sorted(set(int(x.strip()) for x in args.allowed_gpu_pids.split(',') if x.strip()))
    except ValueError:
        p.error('allowed-gpu-pids must be comma-separated integers')
    if any(pid <= 0 for pid in args.allowed_gpu_pids) or (args.service_parent_pid is not None and args.service_parent_pid <= 0):
        p.error('PIDs must be positive')
    args.decoder, args.temperature, args.top_p = args.mode, 0.0, 1.0
    return args


def select_tasks(tasks, per_cell, selection, seed):
    """Sort cell keys; first uses source row order, hash uses SHA256(seed, sid)."""
    cells = defaultdict(list)
    for task in tasks:
        cells[(task['game'], task['form'])].append(task)
    selected = []
    for key in sorted(cells):
        rows = sorted(cells[key], key=lambda t: (t['row_index'], t['sample_id']))
        if selection == 'hash':
            rows.sort(key=lambda t: (fingerprint([seed, t['sample_id']]), t['sample_id']))
        selected.extend(rows[:per_cell])
    if not selected:
        raise ValueError('No tasks available for speed measurement')
    ids = [t['sample_id'] for t in selected]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate sample IDs in task plan')
    return selected


def make_manifest(args, selected):
    return dict(schema=SCHEMA, protocol_sha256=digest(Path(args.protocol_root) / 'protocol.json'),
                selection=args.selection, per_cell=args.per_cell, seed=args.seed,
                samples=[dict(sample_id=t['sample_id'], game=t['game'], form=t['form'],
                              row_index=t['row_index'], source_sha256=t['source_sha256'],
                              task_sha256=fingerprint(t), image_sha256=digest(t['image']))
                         for t in selected])


def ensure_manifest(path, plan):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != plan:
            raise ValueError('Manifest differs from requested sample plan; refusing replacement')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(plan, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    return plan


def _duration(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def completion_tokens(record):
    value = record.get('usage', {}).get('completion_tokens')
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError('Invalid completion_tokens; expected nonnegative integer or missing/null')
    return value


def validate_record(record):
    if not isinstance(record, dict) or not isinstance(record.get('sample_id'), str):
        raise ValueError('Record must contain a sample_id')
    if not _duration(record.get('latency_seconds')):
        raise ValueError('Record must contain finite nonnegative E2E latency_seconds')
    for name in ('request_seconds', 'preparation_seconds', 'parse_seconds'):
        if record.get(name) is not None and not _duration(record[name]):
            raise ValueError(f'Invalid {name}')
    for name in ('transport_ok', 'valid'):
        if not isinstance(record.get(name), bool):
            raise ValueError(f'Missing/invalid {name}')
    for name in ('gt_groups', 'pred_groups'):
        groups = record.get(name)
        if not isinstance(groups, dict):
            raise ValueError(f'Missing/invalid {name}')
        for label, boxes in groups.items():
            if not isinstance(label, str) or not isinstance(boxes, list):
                raise ValueError(f'Invalid groups in {name}')
            for box in boxes:
                if not isinstance(box, list) or len(box) != 4 or not all(_duration(v) for v in box):
                    raise ValueError(f'Invalid xyxy in {name}')
    if not isinstance(record.get('usage'), dict):
        raise ValueError('Missing/invalid usage')
    completion_tokens(record)


def load_results(path, selected, repeats):
    """Never truncate damaged JSONL or retry a persisted failure."""
    path = Path(path)
    if not path.exists():
        return []
    expected = {t['sample_id']: t for t in selected}
    records, seen = [], set()
    with path.open(encoding='utf-8') as stream:
        for lineno, line in enumerate(stream, 1):
            try:
                if not line.endswith('\n'):
                    raise ValueError('Unterminated final JSONL line (possible interrupted write)')
                row = json.loads(line)
                validate_record(row)
                repeat = row.get('repeat')
                if not isinstance(repeat, int) or isinstance(repeat, bool) or not 0 <= repeat < repeats:
                    raise ValueError('Invalid repeat index')
                sid = row['sample_id']
                if sid not in expected or any(row.get(k) != expected[sid][k] for k in ('game', 'form')):
                    raise ValueError('Record outside sample plan')
                key = repeat, sid
                if key in seen:
                    raise ValueError('Duplicate repeat/sample_id')
                seen.add(key)
                records.append(row)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f'Damaged JSONL {path}:{lineno}: {exc}; preserve output, use a new directory') from exc
    return records


def nearest(values, p):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * p)] if ordered else None


def latency_stats(values):
    return dict(mean=statistics.mean(values) if values else None,
                p50=nearest(values, .50), p90=nearest(values, .90), p95=nearest(values, .95))


def speed_stats(records):
    elapsed = [r['latency_seconds'] for r in records]
    http = [r['request_seconds'] for r in records if r.get('request_seconds') is not None]
    tokens = [completion_tokens(r) for r in records]
    missing = sum(t is None for t in tokens)
    total, http_total = sum(elapsed), sum(http)
    known_tokens = sum(t for t in tokens if t is not None)
    return dict(samples=len(records), sumtime_seconds=total,
                samples_per_second=len(records) / total if total else None,
                e2e_seconds=latency_stats(elapsed),
                http_only=dict(measured_samples=len(http), missing_seconds=len(records) - len(http),
                               sumtime_seconds=http_total, seconds=latency_stats(http),
                               samples_per_second=len(http) / http_total if http_total else None),
                completion_tokens=known_tokens, missing_completion_tokens=missing,
                output_tokens_per_second=known_tokens / total if total and not missing else None,
                known_output_tokens_per_second=known_tokens / total if total else None)


def gt_bucket(record):
    count = sum(len(boxes) for boxes in record['gt_groups'].values())
    if count <= 1:
        return str(count)
    return '2-5' if count <= 5 else ('6-10' if count <= 10 else '11+')


def summarize(records, expected_requests):
    def group(rows):
        good = [r for r in rows if r['transport_ok'] and r['valid']]
        result = speed_stats(rows)
        result.update(successful_samples=len(good), failed_samples=len(rows) - len(good),
                      transport_errors=sum(not r['transport_ok'] for r in rows),
                      invalid_samples=sum(r['transport_ok'] and not r['valid'] for r in rows),
                      success_only=speed_stats(good))
        return result
    forms, boxes = defaultdict(list), defaultdict(list)
    for row in records:
        forms[row['form']].append(row)
        boxes[gt_bucket(row)].append(row)
    sampled = [s['memory_used_mib'] for r in records for s in (r.get('gpu_before'), r.get('gpu_after'))
               if s and s.get('memory_used_mib') is not None]
    states = [s.get('isolation', 'unknown') for r in records for s in (r.get('gpu_before'), r.get('gpu_after')) if s]
    isolation = 'contaminated' if 'contaminated' in states else ('isolated' if states and all(s == 'isolated' for s in states) else 'unknown')
    return dict(schema=SCHEMA, expected_requests=expected_requests,
                complete=len(records) == expected_requests,
                definitions=dict(percentile='nearest sorted value at round((n-1)*p), Python round',
                                 primary='all attempts / sum full E2E latency; failures included; warmup excluded',
                                 success='transport_ok AND canonical valid',
                                 output_tps='sum completion_tokens / sum E2E; null if any token count missing',
                                 known_output_tps='known token counts / all E2E seconds; lower bound when counts missing',
                                 http_only='measured HTTP requests / sum request_seconds, not E2E throughput'),
                overall=group(records), by_form={k: group(v) for k, v in sorted(forms.items())},
                by_gt_box_count={k: group(v) for k, v in sorted(boxes.items())},
                gpu=dict(isolation=isolation, peak_sampled_memory_mib=max(sampled) if sampled else None,
                         memory_measurement='per-request before/after samples; NOT actual peak',
                         benchmark_isolated=isolation == 'isolated'))


def descendants(parent, proc_root=Path('/proc')):
    proc_root = Path(proc_root)
    if not (proc_root / str(parent) / 'stat').is_file():
        raise ValueError('service-parent-pid is not observable through /proc')
    parents = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # comm may contain spaces or parentheses; fields after final ')' start at state.
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            parents[int(entry.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue  # Processes may exit during enumeration.
    allowed = {parent}
    while True:
        expanded = allowed | {pid for pid, ppid in parents.items() if ppid in allowed}
        if expanded == allowed:
            return allowed
        allowed = expanded


class GPUProbe:
    """No threads, no CUDA imports, no process termination."""
    def __init__(self, args):
        self.args = args
        self.uuid = None

    def _query(self, *flags):
        done = subprocess.run(['nvidia-smi', *flags], capture_output=True, text=True,
                              timeout=10, check=True)
        return list(csv.reader(done.stdout.splitlines(), skipinitialspace=True))

    def sample(self):
        if not self.args.gpu:
            return dict(isolation='unknown', reason='GPU monitoring disabled')
        try:
            gpu = self._query('-i', self.args.gpu, '--query-gpu=uuid,memory.used,memory.total', '--format=csv,noheader,nounits')
            if len(gpu) != 1 or len(gpu[0]) != 3:
                raise ValueError('--gpu must identify exactly one GPU')
            uuid, used, total = [v.strip() for v in gpu[0]]
            if self.uuid is not None and self.uuid != uuid:
                raise ValueError('Selected GPU UUID changed')
            self.uuid = uuid
            apps = self._query('--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits')
            pids, processes = set(), []
            for row in apps:
                if len(row) != 3:
                    raise ValueError('Unrecognized nvidia-smi compute process row')
                if row[0].strip() == uuid:
                    pid = int(row[1])
                    pids.add(pid)
                    processes.append(dict(pid=pid, memory_used_mib=row[2].strip()))
            allowed = set(self.args.allowed_gpu_pids)
            if self.args.service_parent_pid is not None:
                allowed |= descendants(self.args.service_parent_pid)
            foreign = sorted(pids - allowed)
            return dict(uuid=uuid, memory_used_mib=float(used), memory_total_mib=float(total),
                        compute_processes=processes, allowed_pids=sorted(allowed), foreign_pids=foreign,
                        isolation='contaminated' if foreign else 'isolated')
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return dict(uuid=self.uuid, isolation='unknown', reason=f'{type(exc).__name__}: {exc}')

    def require(self, sample):
        if self.args.require_isolated and sample['isolation'] != 'isolated':
            raise ValueError(f"GPU isolation required, got {sample}; no processes were killed")


def perform_request(task, args):
    """Catch failures outside current5.request's HTTP catch, preserving E2E cost."""
    start = time.perf_counter()
    try:
        row = request(task, args)
        validate_record(row)
        if row['sample_id'] != task['sample_id']:
            raise ValueError('API returned a different sample_id')
        return row
    except Exception as exc:
        groups, status = parse_la_answer(task['gt_answer'], 1000, 1000)
        return dict(sample_id=task['sample_id'], game=task['game'], form=task['form'],
                    transport_ok=False, valid=False, pred_groups={}, gt_groups=groups,
                    gt_status=status, error=f'{type(exc).__name__}: {exc}', usage={}, finish_reason=None,
                    latency_seconds=time.perf_counter() - start, request_seconds=None,
                    preparation_seconds=None, parse_seconds=None)


def _configuration(args, manifest):
    values = vars(args).copy()
    for key in ('protocol_root', 'output_dir'):
        values[key] = str(Path(values[key]).resolve())
    values['manifest'] = str(Path(args.manifest).resolve()) if args.manifest else str(Path(args.output_dir).resolve() / 'manifest.json')
    return dict(schema=SCHEMA, arguments=values, manifest_sha256=fingerprint(manifest))


def main(argv=None):
    args = arguments(argv)
    protocol, tasks = load_tasks(args.protocol_root)
    selected = select_tasks(tasks, args.per_cell, args.selection, args.seed)
    plan = make_manifest(args, selected)
    output = Path(args.output_dir).resolve()
    root = Path(args.protocol_root).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else output / 'manifest.json'
    if output == root or output.is_relative_to(root) or manifest_path == root or manifest_path.is_relative_to(root):
        raise ValueError('Output/manifest must not mutate the protocol/result tree')
    config = _configuration(args, plan)
    config_path, results_path = output / 'config.json', output / 'requests.jsonl'
    reserved = {'config.json', 'requests.jsonl', 'warmup.jsonl', 'summary.json',
                'server-info.json', 'protocol-snapshot.json', 'dry-run.json'}
    if manifest_path.parent == output and manifest_path.name in reserved:
        raise ValueError('Manifest path conflicts with a run artifact')
    stored = None
    if config_path.exists():
        stored = json.loads(config_path.read_text(encoding='utf-8'))
        if stored.get('configuration') != config:
            raise ValueError('Resume configuration differs; use a fresh output directory')
    elif results_path.exists():
        raise ValueError('Existing JSONL without configuration; refusing resume')
    previous = load_results(results_path, selected, args.repeats)
    if args.require_isolated:
        for row in previous:
            if any((row.get(key) or {}).get('isolation') != 'isolated' for key in ('gpu_before', 'gpu_after')):
                raise ValueError('Persisted request has failed/unknown GPU isolation; preserve it and use a fresh directory')
    ensure_manifest(manifest_path, plan)
    output.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        # No server check, request, or nvidia-smi even with --require-isolated.
        if stored is None:
            write_json(config_path, dict(configuration=config, server_info=None))
        write_json(output / 'dry-run.json', dict(plan=plan, measured_requests=len(selected) * args.repeats,
                                               warmup_requests=args.warmup, isolation='unknown'))
        print(json.dumps(dict(dry_run=True, selected=len(selected), requests=len(selected) * args.repeats,
                              manifest=str(manifest_path)), ensure_ascii=False))
        return 0
    # Validate all persisted input before contacting a service or probing GPU.
    info = check_server(args.base_url, args.decoder)
    if stored is not None and stored.get('server_info') != info:
        raise ValueError('Resume server_info differs; use a fresh output directory')
    probe = GPUProbe(args)
    probe.require(probe.sample())
    if stored is None:
        write_json(config_path, dict(configuration=config, server_info=info,
                                    started_at=datetime.now(timezone.utc).isoformat(),
                                    command=[sys.executable, '-m', 'eval.current5_speed', *(argv if argv is not None else sys.argv[1:])]))
    write_json(output / 'server-info.json', info)
    write_json(output / 'protocol-snapshot.json', protocol)
    expected = len(selected) * args.repeats
    seen = {(r['repeat'], r['sample_id']) for r in previous}
    records = list(previous)
    pending = expected - len(seen)
    print(f'[OP:开始] speed mode={args.mode} pending={pending}/{expected}', flush=True)
    # Warmups happen per invocation only when there is remaining measured work.
    if pending:
        with (output / 'warmup.jsonl').open('a', encoding='utf-8') as stream:
            for index in range(args.warmup):
                before = probe.sample()
                probe.require(before)
                row = perform_request(selected[index % len(selected)], args)
                after = probe.sample()
                row.update(warmup_index=index, gpu_before=before, gpu_after=after)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                probe.require(after)
    with results_path.open('a', encoding='utf-8') as stream:
        for repeat in range(args.repeats):
            for task in selected:
                if (repeat, task['sample_id']) in seen:
                    continue  # Includes failed attempts: NEVER silently retry.
                before = probe.sample()
                if args.require_isolated and before['isolation'] != 'isolated':
                    partial = summarize(records, expected)
                    partial['interruption'] = {'phase': 'before_request', 'sample_id': task['sample_id'], 'gpu': before}
                    partial['gpu']['isolation'] = before['isolation']
                    partial['gpu']['benchmark_isolated'] = False
                    write_json(output / 'summary.json', partial)
                probe.require(before)
                row = perform_request(task, args)
                after = probe.sample()
                row.update(repeat=repeat, gpu_before=before, gpu_after=after)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                records.append(row)
                # Persist the measured request even if post-request isolation fails.
                if args.require_isolated and after['isolation'] != 'isolated':
                    write_json(output / 'summary.json', summarize(records, expected))
                probe.require(after)
                print(f"[speed] repeat={repeat} {task['sample_id']} e2e={row['latency_seconds']:.3f}s transport={row['transport_ok']} valid={row['valid']}", flush=True)
    result = summarize(records, expected)
    write_json(output / 'summary.json', result)
    print(f"[OP:成功] attempts={len(records)} samples/sec={result['overall']['samples_per_second']} output={output}", flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f'[OP:失败] {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(2)
