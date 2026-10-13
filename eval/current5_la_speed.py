"""LA base-checkpoint fast/hybrid/slow speed benchmark, single GPU and batch=1.

Uses the installed LA worker via its actual predict API; never copies upstream
model implementation. No GPU/model import occurs in --dry-run or CPU helpers.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from eval.current5 import DEFAULT_PROTOCOL_ROOT, load_tasks, digest, fingerprint, write_json
from eval.current5_metrics import parse_la_answer
from eval.current5_speed import select_tasks, make_manifest, ensure_manifest, speed_stats, GPUProbe

MODES=('fast','hybrid','slow')
STAT_RE=re.compile(r'([a-z_]+)=([0-9]+(?:\.[0-9]+)?)')


def parse_native_stats(value):
    result={key:float(number) for key,number in STAT_RE.findall(str(value))}
    for key in ('num_tokens','forward_step','num_boxes','switch_to_ar'):
        if key in result:
            number=result[key]
            if number!=int(number):raise ValueError('Nonintegral native count: '+key)
            result[key]=int(number)
    if any(key not in result for key in ('num_tokens','forward_step','num_boxes')):
        raise ValueError('LA verbose output lacks native token/forward/box counts')
    return result


@contextmanager
def synchronized_generate_probe(worker,torch):
    """Wrap boundaries only; never alters tokens, logits or generation settings.

    Generation timer includes vision encoding/prefill/decode/detokenization.
    Decode window starts after first LM forward, ends at generate return. It
    excludes first-forward cost, NOT its token selection; separate candidate
    round/paper c_A instrumentation is not claimed.
    """
    model=worker.model;language=model.language_model
    original_generate=model.generate;original_forward=language.forward
    trace={}
    def forward(*args,**kwargs):
        out=original_forward(*args,**kwargs)
        trace['forward_calls']+=1
        if trace['forward_calls']==1:
            torch.cuda.synchronize();trace['first_forward_end']=time.perf_counter()
        return out
    def generate(*args,**kwargs):
        torch.cuda.synchronize();trace.clear();trace.update(forward_calls=0)
        start=time.perf_counter()
        result=original_generate(*args,**kwargs)
        torch.cuda.synchronize();end=time.perf_counter()
        trace['generate_seconds']=end-start
        trace['prefill_window_seconds']=trace.get('first_forward_end',end)-start
        trace['decode_window_seconds']=end-trace.get('first_forward_end',end)
        trace.pop('first_forward_end',None)
        return result
    language.forward=forward;model.generate=generate
    try:yield trace
    finally:language.forward=original_forward;model.generate=original_generate


def summarize_mode(rows):
    result=speed_stats(rows)
    tokens=sum(r['usage']['completion_tokens'] for r in rows)
    boxes=sum(r['native_stats']['num_boxes'] for r in rows)
    valid_boxes=sum(len(b) for r in rows for b in r['pred_groups'].values())
    gen=sum(r['generate_seconds'] for r in rows)
    decode=sum(r['decode_window_seconds'] for r in rows)
    forwards=sum(r['forward_calls'] for r in rows)
    result.update(native_output_tokens=tokens,native_box_markers=boxes,parsed_valid_boxes=valid_boxes,
                  generate_seconds=gen,decode_window_seconds=decode,
                  generate_tps=tokens/gen if gen else None,
                  decode_window_output_tps=tokens/decode if decode else None,
                  generate_bps=boxes/gen if gen else None,
                  valid_boxes_per_generate_second=valid_boxes/gen if gen else None,
                  e2e_bps=boxes/result['sumtime_seconds'] if result['sumtime_seconds'] else None,
                  language_forward_calls=forwards,tokens_per_language_forward=tokens/forwards if forwards else None,
                  successful_samples=sum(r['valid'] for r in rows),
                  definitions={'tps':'sum actual generated_ids counts (native verbose stats, includes EOS) / sum synchronized generate wall time',
                               'bps':'sum literal <box> markers (includes None/invalid wrappers) / sum generate time; parsed valid boxes/s is separate',
                               'decode_window_output_tps':'all output tokens / window after first LM forward through generate return; includes first-token selection, NOT strict post-prefill new-token rate',
                               'forward_scope':'language forward only; not visual encoder calls',
                               'native_verbose_overhead':'per-step tokenizer decoding/history bookkeeping enabled equally in all modes'})
    return result


def pin_gpu(args):
    probe=GPUProbe(args);sample=probe.sample()
    pids=[p['pid'] for p in sample.get('compute_processes',[])]
    if sample['isolation']=='unknown' or len(pids)!=1:
        raise ValueError('After loading on prechecked empty GPU expected one driver process: '+str(sample))
    args.allowed_gpu_pids=pids
    return probe,sample


def measure(task,args,worker,torch):
    from PIL import Image
    start=time.perf_counter()
    with Image.open(task['image']) as image: image=image.convert('RGB')
    image_sha=digest(task['image'])
    # Same PNG/base64 CPU preparation as GA for an application-level envelope;
    # no HTTP round-trip exists here. Native timing remains separately reported.
    from io import BytesIO
    import base64
    buffer=BytesIO();image.save(buffer,format='PNG');base64.b64encode(buffer.getvalue())
    groups,status=parse_la_answer(task['gt_answer'],image.width,image.height)
    prep=time.perf_counter()-start
    with synchronized_generate_probe(worker,torch) as trace:
        response=worker.predict(image,task['question'],generation_mode=args.mode,
                                max_new_tokens=args.max_tokens,temperature=0.0,top_p=1.0,
                                top_k=0,repetition_penalty=1.0,verbose=True)
    native=parse_native_stats(response.get('stats'))
    raw=str(response['answer']);parse_start=time.perf_counter()
    predictions,pred_status=parse_la_answer(raw,image.width,image.height)
    structural_complete=raw.rstrip().endswith('<|im_end|>')
    valid=structural_complete and pred_status in ('boxes','none')
    row={'sample_id':task['sample_id'],'game':task['game'],'form':task['form'],
         'image_sha256':image_sha,'source_sha256':task['source_sha256'],'prompt':task['question'],
         'raw_answer':raw,'gt_groups':groups,'gt_status':status,'pred_groups':predictions,
         'pred_status':pred_status,'transport_ok':True,'valid':valid,
         'finish_reason':'stop' if structural_complete else 'unknown_or_budget',
         'usage':{'completion_tokens':native['num_tokens']},'native_stats':native,
         'preparation_seconds':prep,'request_seconds':None,
         'parse_seconds':time.perf_counter()-parse_start,'latency_seconds':time.perf_counter()-start,
         'prediction_sha256':fingerprint(predictions),'error':None,**trace}
    return row


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol-root',default=DEFAULT_PROTOCOL_ROOT)
    p.add_argument('--la-root',default='/workspace/nas-data/LocateAnything')
    p.add_argument('--model-path',default='/workspace/nas-data/LocateAnything/nvidia/LocateAnything-3B')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--gpu',required=True)
    p.add_argument('--per-cell',type=int,default=1)
    p.add_argument('--selection',choices=('first','hash'),default='first')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--max-tokens',type=int,default=512)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    if a.per_cell<1 or a.repeats<1 or a.warmup<0 or a.max_tokens<1:p.error('Invalid sample/repeat/token counts')
    a.allowed_gpu_pids=[];a.service_parent_pid=None;a.require_isolated=True
    return a


def main(argv=None):
    a=arguments(argv);protocol,tasks=load_tasks(a.protocol_root)
    selected=select_tasks(tasks,a.per_cell,a.selection,a.seed)
    plan=make_manifest(a,selected)
    output=Path(a.output_dir).resolve();la=Path(a.la_root).resolve();model=Path(a.model_path).resolve()
    if output.is_relative_to(la) or output.is_relative_to(Path(a.protocol_root).resolve()):raise ValueError('Never write into LA code/validation tree')
    spec={'model_path':str(model),'model_config_sha256':digest(model/'config.json'),
          'la_code_head':subprocess.check_output(['git','-C',str(la),'rev-parse','HEAD'],text=True).strip(),
          'worker_sha256':digest(la/'locateanything_worker.py'),'generation_code_sha256':digest(model/'modeling_locateanything.py'),
          'ga_client_sha256':digest(Path(__file__)),'sample_manifest':plan,'modes':MODES,
          'batch_size':1,'concurrency':1,'gpu':a.gpu,'warmup':a.warmup,'repeats':a.repeats,
          'temperature':0,'top_p':1,'repetition_penalty':1,'max_tokens':a.max_tokens,'seed':a.seed}
    if a.dry_run:
        print(json.dumps({'plan':spec,'requests_per_mode':len(selected)*a.repeats},ensure_ascii=False,indent=2));return 0
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu):raise ValueError('CUDA_VISIBLE_DEVICES must select exactly the physical --gpu index')
    if output.exists():raise ValueError('Use fresh versioned LA speed output; no silent retries/resume')
    probe=GPUProbe(a);before=probe.sample()
    if before['isolation']=='unknown' or before.get('compute_processes'):raise ValueError('GPU not verified empty before LA load: '+str(before))
    output.mkdir(parents=True);write_json(output/'plan.json',spec);ensure_manifest(output/'sample_manifest.json',plan)
    sys.path.insert(0,str(la))
    # Dependencies stay in LA /opt/venv, not GA .venv-serve.
    import torch
    from game_finetune.infer.ground import ChineseGroundingWorker
    from game_finetune.core.runlog import OperationLog
    log=OperationLog(output/'worker-logs',project_root=la)
    torch.manual_seed(a.seed)
    load_start=time.perf_counter()
    wrapper=ChineseGroundingWorker(str(model),env_name='cloud-b',log=log)
    worker=wrapper._worker
    load_time=time.perf_counter()-load_start
    probe,identity=pin_gpu(a);probe.require(probe.sample())
    environment={'python':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,
                 'transformers':importlib.metadata.version('transformers'),'load_seconds':load_time,
                 'gpu_before':before,'gpu_after_load':identity,'allowed_driver_pids':a.allowed_gpu_pids,
                 'identity_boundary':'prechecked-empty-card then single driver PID; not proven host/container translation',
                 'attention_config':getattr(worker.model.config,'_attn_implementation',None)}
    write_json(output/'environment.json',environment)
    all_stats={}
    try:
        for mode in MODES:
            a.mode=mode;rows=[];mode_dir=output/mode;mode_dir.mkdir()
            with (mode_dir/'warmup.jsonl').open('x',encoding='utf-8') as f:
                for i in range(a.warmup):
                    probe.require(probe.sample());row=measure(selected[i%len(selected)],a,worker,torch);probe.require(probe.sample())
                    f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            with (mode_dir/'requests.jsonl').open('x',encoding='utf-8') as f:
                for repeat in range(a.repeats):
                    for task in selected:
                        before=probe.sample();probe.require(before)
                        row=measure(task,a,worker,torch);after=probe.sample()
                        row.update(repeat=repeat,gpu_before=before,gpu_after=after)
                        f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();rows.append(row)
                        if after['isolation']!='isolated':write_json(mode_dir/'partial-summary.json',summarize_mode(rows))
                        probe.require(after)
                        print(f"[LA-speed] mode={mode} {len(rows)}/{len(selected)*a.repeats} {task['sample_id']} e2e={row['latency_seconds']:.3f}s gen={row['generate_seconds']:.3f}s tokens={row['usage']['completion_tokens']}",flush=True)
            summary=summarize_mode(rows);summary.update(complete=True,mode=mode,expected_requests=len(selected)*a.repeats)
            write_json(mode_dir/'summary.json',summary);all_stats[mode]=summary
        write_json(output/'summary.json',{'complete':True,'modes':all_stats,'scope':'Single base LA checkpoint, native standard worker, batch1/concurrency1; mode-conditional output workloads'})
        return 0
    except Exception as exc:
        write_json(output/'failure.json',{'error':f'{type(exc).__name__}: {exc}','mode':a.mode,'at':datetime.now(timezone.utc).isoformat()});raise
    finally:
        wrapper.close()
        write_json(output/'gpu-before-process-exit.json',probe.sample())
        # Model memory is released when this dedicated process exits; the outer
        # scheduler verifies GPU release, not this pre-exit sample.


if __name__=='__main__':raise SystemExit(main())
