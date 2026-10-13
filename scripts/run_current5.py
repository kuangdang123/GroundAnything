"""Explicit, single-GPU-per-service launch plans for Current5 (never auto-started).

Starts only its own process groups; on exit shuts down only those groups.
Use --dry-run first. Outputs and logs live below the requested versioned run.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.current5 import DEFAULT_PROTOCOL_ROOT, check_server, digest, write_json


def gpu_snapshot(gpus):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
    selected = {line.split(',')[1].strip() for line in result.stdout.splitlines() if int(line.split(',')[0]) in gpus}
    if len(selected) != len(gpus):
        raise ValueError('Selected GPUs do not exist')
    foreign = [line for line in apps.splitlines() if len(line.split(',')) >= 2 and line.split(',')[1].strip() in selected]
    if foreign:
        raise ValueError('Selected GPU already has compute processes; not starting:\n' + '\n'.join(foreign))
    return {'gpus': result.stdout, 'compute_apps': apps}


def service_command(gpu, port, mode, output, model_path, memory_fraction):
    command = [sys.executable, '-u', '-m', 'infer.serve_sglang', '--model', model_path,
               '--decoder', mode, '--port', str(port), '--output', str(output.relative_to(ROOT))]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), GAM_SGLANG_MEM_FRACTION=str(memory_fraction), PYTHONUNBUFFERED='1')
    return command, env


def client_command(args, output, port, mode='denoise', shard=0, shards=1, smoke=False):
    command = [sys.executable, '-u', '-m', 'eval.current5', '--protocol-root', args.protocol_root,
               '--output-dir', str(output), '--base-url', f'http://127.0.0.1:{port}/v1',
               '--decoder', mode, '--prompt-mode', args.prompt_mode, '--num-shards', str(shards),
               '--shard-index', str(shard), '--max-tokens', str(args.max_tokens),
               '--temperature', str(args.temperature), '--top-p', str(args.top_p)]
    if smoke:
        command += ['--limit', '10']
    return command


def start_service(args, gpu, port, mode, output, processes, streams):
    # Refuse to attach clients to an unrelated already-running server.
    import socket
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', port)) == 0:
            raise ValueError(f'Port {port} already in use')
    command, env = service_command(gpu, port, mode, output, args.model_path, args.memory_fraction)
    output.mkdir(parents=True, exist_ok=True)
    log = (output / 'console.log').open('a', encoding='utf-8')
    streams.append(log)
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    processes.append(process)
    deadline = time.monotonic() + args.startup_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Service exited code={process.returncode}; see {output / "console.log"}')
        try:
            check_server(f'http://127.0.0.1:{port}/v1', mode)
            return process
        except Exception:
            time.sleep(2)
    raise TimeoutError(f'Service startup timed out; see {output / "console.log"}')


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('baseline', 'speed'))
    p.add_argument('--protocol-root', default=DEFAULT_PROTOCOL_ROOT)
    p.add_argument('--output-dir', required=True, help='New versioned directory under this checkout')
    p.add_argument('--gpus', required=True, help='baseline: two physical indices; speed: one')
    p.add_argument('--ports', help='baseline default 8111,8112; speed default 8113')
    p.add_argument('--model-path', default='weights/dlm_bundle')
    p.add_argument('--prompt-mode', choices=('la_zh', 'ga_native'), default='la_zh')
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--memory-fraction', type=float, default=0.4)
    p.add_argument('--startup-timeout', type=float, default=900)
    p.add_argument('--speed-per-cell', type=int, default=1)
    p.add_argument('--speed-repeats', type=int, default=3)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    gpus = [int(x) for x in args.gpus.split(',')]
    ports = [int(x) for x in (args.ports or ('8111,8112' if args.action == 'baseline' else '8113')).split(',')]
    count = 2 if args.action == 'baseline' else 1
    if not 0 < args.memory_fraction <= 1 or args.startup_timeout <= 0 or args.max_tokens < 1:
        p.error('Invalid memory fraction, startup timeout or token budget')
    if len(gpus) != count or len(set(gpus)) != count or len(ports) != count or len(set(ports)) != count:
        p.error(f'{args.action} requires {count} distinct GPUs and ports')
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(ROOT / 'outputs'):
        p.error('Use a versioned output directory under this checkout outputs/')
    plan = dict(action=args.action, gpus=gpus, ports=ports, model_path=args.model_path,
                output=str(output), baseline_decoder='denoise', prompt_mode=args.prompt_mode,
                baseline_sampling={'temperature': args.temperature, 'top_p': args.top_p},
                speed_sampling={'temperature': 0, 'top_p': 1}, max_tokens=args.max_tokens,
                batch_size=1, request_concurrency=1, tensor_parallel=1,
                baseline_sharding='whole game/form cells round-robin',
                speed_modes=['denoise', 'causal', 'speculative'],
                speed_per_cell=args.speed_per_cell, speed_repeats=args.speed_repeats)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if sys.platform != 'linux':
        p.error('Actual launch requires Linux; --dry-run is available on Windows')
    if output.exists():
        p.error('Launcher requires a fresh versioned output dir; resume clients explicitly without restarting services')
    snapshot = gpu_snapshot(gpus)
    output.mkdir(parents=True)
    write_json(output / 'plan.json', plan)
    write_json(output / 'gpu-before.json', snapshot)
    versions = {name: importlib.metadata.version(name) for name in ('torch','transformers','sglang','sgl-kernel')}
    versions['python'] = sys.executable
    versions['git_head'] = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    # Small config hashes plus weight stat: NOT a full multi-GB weight checksum.
    model = ROOT / args.model_path
    versions['checkpoint_config_sha256'] = digest(model / 'config.json')
    versions['checkpoint_weight_bytes'] = (model / 'model.safetensors').stat().st_size
    write_json(output / 'environment.json', versions)
    processes, streams = [], []
    try:
        if args.action == 'baseline':
            for index, (gpu, port) in enumerate(zip(gpus, ports)):
                start_service(args, gpu, port, 'denoise', output / f'service_gpu{gpu}', processes, streams)
                smoke = client_command(args, output / 'smoke', port, shard=index, shards=2, smoke=True)
                subprocess.run(smoke, cwd=ROOT, check=True)
            # Require 10/10 successful transports on EACH replica before full run.
            from eval.current5 import load_records
            for index in range(2):
                records = load_records(output / 'smoke' / f'predictions-shard{index}.jsonl')
                if len(records) != 10 or any(not r.get('transport_ok') for r in records):
                    raise RuntimeError('Smoke transport failed; no full baseline launched')
                print(f'[SMOKE] shard={index} canonical_valid={sum(r.get("valid",False) for r in records)}/10', flush=True)
            clients = []
            for index, port in enumerate(ports):
                cmd = client_command(args, output / 'baseline', port, shard=index, shards=2)
                log = (output / f'client-shard{index}.log').open('a', encoding='utf-8')
                streams.append(log)
                client = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                clients.append(client)
                processes.append(client)
            codes = [process.wait() for process in clients]
            if any(codes):
                raise RuntimeError(f'Baseline clients failed: {codes}')
            subprocess.run(client_command(args, output / 'baseline', ports[0], shards=2) + ['--summarize'], cwd=ROOT, check=True)
        else:
            for mode in ('denoise','causal','speculative'):
                if mode != 'denoise':
                    gpu_snapshot(gpus)  # Previous owned service must have released GPU.
                process = start_service(args, gpus[0], ports[0], mode, output / f'service_{mode}_gpu{gpus[0]}', processes, streams)
                cmd = [sys.executable, '-u', '-m', 'eval.current5_speed', '--protocol-root', args.protocol_root,
                       '--output-dir', str(output / mode), '--base-url', f'http://127.0.0.1:{ports[0]}/v1',
                       '--mode', mode, '--prompt-mode', args.prompt_mode, '--max-tokens', str(args.max_tokens),
                       '--per-cell', str(args.speed_per_cell), '--repeats', str(args.speed_repeats),
                       '--manifest', str(output / 'sample_manifest.json'), '--gpu', str(gpus[0]),
                       '--require-isolated', '--service-parent-pid', str(process.pid)]
                subprocess.run(cmd, cwd=ROOT, check=True)
                stop(process)
        print(f'[OP:成功] {args.action} -> {output}', flush=True)
    except Exception as exc:
        write_json(output / 'failure.json', {'error': f'{type(exc).__name__}: {exc}', 'command': sys.argv})
        print(f'[OP:失败] {args.action}: {exc}', file=sys.stderr, flush=True)
        raise
    finally:
        for process in reversed(processes):
            stop(process)
        for stream in streams:
            stream.close()
        snapshot = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        write_json(output / 'gpu-after.json', {'stdout': snapshot.stdout, 'stderr': snapshot.stderr, 'exit': snapshot.returncode})


if __name__ == '__main__':
    raise SystemExit(main())
