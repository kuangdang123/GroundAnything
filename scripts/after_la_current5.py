"""Condition-triggered GA work after the existing LA validation queue.

Never changes LA files or terminates LA/other GPU jobs. Checks every --interval
seconds, records durable state, and starts each GA job once in a fresh run.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.current5 import DEFAULT_PROTOCOL_ROOT, write_json
from eval.current5_metrics import FORMS, PRIMARY_METRIC
from scripts.run_current5 import gpu_snapshot


def utc():
    return datetime.now(timezone.utc).isoformat()


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def proc_identity(pid, proc_root=Path('/proc')):
    p = proc_root / str(pid)
    try:
        stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        cmd = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
        return {'pid': pid, 'start_ticks': stat[19], 'command': cmd}
    except (OSError, IndexError):
        return None


def la_workers(gpus, proc_root=Path('/proc')):
    workers = []
    for p in proc_root.iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
            if 'eval_current5.py' not in cmd or cmd.startswith(('bash -c', 'python -')):
                continue
            env = (p / 'environ').read_bytes().split(b'\0')
            value = next((x.decode().split('=', 1)[1] for x in env if x.startswith(b'CUDA_VISIBLE_DEVICES=')), None)
            if value is None or set(value.split(',')) & set(map(str, gpus)):
                workers.append({'pid': int(p.name), 'gpu': value, 'command': cmd})
        except OSError:
            pass
    return workers


def result_complete(root, key, protocol):
    """Require valid full summaries, not mere metrics file existence."""
    p = root / 'results' / key / 'summary.json'
    try:
        v = json.loads(p.read_text(encoding='utf-8'))
        if not isinstance(v.get('current5_equal_view_macro'), (int, float)):
            return False
        for form in FORMS:
            expected = {g for g, info in protocol['games'].items()
                        if int(info['forms'].get(form, {}).get('sampled_rows', 0)) > 0}
            scores = v.get('forms', {}).get(form, {})
            if scores.get('metric') != PRIMARY_METRIC[form] or set(scores.get('per_game', {})) != expected:
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def snapshot_conditions(plan, proc_root=Path('/proc')):
    root = Path(plan['la_root'])
    protocol = json.loads((root / 'protocol.json').read_text(encoding='utf-8'))
    if source_hash(root / 'protocol.json') != plan['protocol_sha256']:
        raise ValueError('LA protocol changed after arming; review and create a new schedule')
    models = json.loads((root / 'models.json').read_text(encoding='utf-8'))
    # Changes to annotations in models.json are harmless; adding a checkpoint is not.
    if set(models) != set(plan['model_keys']):
        raise ValueError('LA model inventory changed after arming; review queue scope')
    queue_alive = []
    for identity in plan['queue_processes']:
        now = proc_identity(identity['pid'], proc_root)
        if now and now['start_ticks'] == identity['start_ticks']:
            queue_alive.append(now)
    workers = la_workers(plan['gpus'], proc_root)
    missing = [key for key in plan['model_keys'] if not result_complete(root, key, protocol)]
    gpu_free, gpu_reason = False, None
    if not queue_alive and not workers and not missing:
        try:
            gpu_snapshot(plan['gpus'])
            gpu_free = True
        except (ValueError, subprocess.SubprocessError) as exc:
            gpu_reason = str(exc)
    return {'checked_at': utc(), 'queue_alive': queue_alive, 'active_la_workers': workers,
            'complete_models': len(plan['model_keys']) - len(missing), 'total_models': len(plan['model_keys']),
            'missing_models': missing, 'gpus_free': gpu_free, 'gpu_wait_reason': gpu_reason,
            'ready': not queue_alive and not workers and not missing and gpu_free}


def launch_commands(plan):
    return {
        'baseline': [sys.executable, '-u', 'scripts/run_current5.py', 'baseline',
                     '--protocol-root', plan['la_root'], '--gpus', ','.join(map(str, plan['gpus'][:2])),
                     '--output-dir', plan['baseline_output']],
        'speculative': [sys.executable, '-u', 'scripts/run_current5.py', 'speed',
                        '--protocol-root', plan['la_root'], '--gpus', str(plan['gpus'][2]),
                        '--speed-modes', 'speculative', '--output-dir', plan['speed_output']]
    }


def run_schedule(plan, state_dir, interval, max_wait_hours):
    deadline = time.monotonic() + max_wait_hours * 3600
    checks = 0
    with (state_dir / 'checks.jsonl').open('a', encoding='utf-8') as stream:
        while time.monotonic() < deadline:
            condition = snapshot_conditions(plan)
            stream.write(json.dumps(condition, ensure_ascii=False) + '\n')
            stream.flush()
            write_json(state_dir / 'status.json', dict(phase='waiting', **condition))
            checks += 1
            print(f"[WAIT] {utc()} checks={checks} models={condition['complete_models']}/{condition['total_models']} queues={len(condition['queue_alive'])} workers={len(condition['active_la_workers'])} ready={condition['ready']}", flush=True)
            if condition['ready']:
                break
            time.sleep(interval)
        else:
            write_json(state_dir / 'status.json', {'phase': 'wait_expired', 'checked_at': utc(), 'max_wait_hours': max_wait_hours})
            return 3
    # Durable exclusive intent: NEVER launch twice on restart/crash.
    commands = launch_commands(plan)
    with (state_dir / 'launch-intent.json').open('x', encoding='utf-8') as stream:
        json.dump({'created_at': utc(), 'commands': commands}, stream, indent=2)
    processes, handles, jobs = {}, [], {}
    try:
        for name, command in commands.items():
            log_path = state_dir / (name + '.log')
            handle = log_path.open('x', encoding='utf-8')
            handles.append(handle)
            process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED='1'),
                                       stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            processes[name] = process
            jobs[name] = {'pid': process.pid, 'command': command, 'log': str(log_path), 'started_at': utc()}
            write_json(state_dir / 'jobs.json', jobs)
        print('[START] LA complete and GPU1/2/3 free; owned GA work started', flush=True)
        while True:
            exits = {name: process.poll() for name, process in processes.items()}
            write_json(state_dir / 'status.json', {'phase': 'running' if any(v is None for v in exits.values()) else 'finished',
                                                 'checked_at': utc(), 'jobs': jobs, 'exit_codes': exits})
            if all(v is not None for v in exits.values()):
                print('[FINISH] ' + json.dumps(exits), flush=True)
                return 0 if all(v == 0 for v in exits.values()) else 1
            time.sleep(interval)
    finally:
        # Services belong to run_current5 launcher and clean themselves up.
        # Never terminate other jobs here; durable jobs.json supports inspection.
        for handle in handles:
            handle.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--la-root', default=DEFAULT_PROTOCOL_ROOT)
    p.add_argument('--state-dir', required=True)
    p.add_argument('--baseline-output', required=True)
    p.add_argument('--speed-output', required=True)
    p.add_argument('--gpus', default='1,2,3')
    p.add_argument('--queue-pids', default='246946,246947,246948')
    p.add_argument('--interval', type=int, default=120)
    p.add_argument('--max-wait-hours', type=float, default=48)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    gpus = [int(x) for x in a.gpus.split(',')]
    if len(gpus) != 3 or len(set(gpus)) != 3 or a.interval < 30 or a.max_wait_hours <= 0:
        p.error('Require three distinct GPUs, interval>=30, max-wait-hours>0')
    state = Path(a.state_dir).resolve()
    outputs = [Path(a.baseline_output).resolve(), Path(a.speed_output).resolve()]
    for q in [state, *outputs]:
        if not q.is_relative_to(ROOT / 'outputs'):
            p.error('State and GA outputs must remain under project outputs/')
    if state in outputs or outputs[0] == outputs[1]:
        p.error('State/baseline/speed paths must be distinct')
    root = Path(a.la_root).resolve()
    models = json.loads((root / 'models.json').read_text(encoding='utf-8'))
    parents = [proc_identity(int(x)) for x in a.queue_pids.split(',')]
    identities = [v for v in parents if v]
    for v in identities:
        if 'sweepU.sh' not in v['command']:
            p.error('Specified queue PID is not the inspected LA sweepU worker')
    plan = {'la_root': str(root), 'protocol_sha256': source_hash(root / 'protocol.json'),
            'model_keys': sorted(models), 'queue_processes': identities, 'gpus': gpus,
            'baseline_output': str(outputs[0]), 'speed_output': str(outputs[1]),
            'interval': a.interval, 'max_wait_hours': a.max_wait_hours,
            'armed_at': utc(), 'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    if a.dry_run:
        print(json.dumps({'plan': plan, 'conditions': snapshot_conditions(plan)}, ensure_ascii=False, indent=2))
        return 0
    if sys.platform != 'linux':
        p.error('Actual scheduling requires Linux')
    if state.exists() or any(q.exists() for q in outputs):
        p.error('Use new versioned state and output directories; no implicit resume/relaunch')
    state.mkdir(parents=True)
    write_json(state / 'plan.json', plan)
    print('[ARM] ' + str(state), flush=True)
    try:
        return run_schedule(plan, state, a.interval, a.max_wait_hours)
    except Exception as exc:
        write_json(state / 'failure.json', {'error': f'{type(exc).__name__}: {exc}', 'at': utc()})
        write_json(state / 'status.json', {'phase': 'failed', 'error': str(exc), 'at': utc()})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
