"""CPU-only behavioral differential check against a pinned LA evaluator source.

Extracts only named pure metric/parser functions with AST. Never imports the LA
module (which would load GPU worker code), writes LA files, or runs inference.
"""
import argparse
import ast
from collections import Counter, defaultdict
from io import StringIO
import json
from pathlib import Path
import random
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.current5 import digest, write_json
from eval import current5_metrics as new


def extract(path, names, namespace):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {f.name for f in functions} != set(names):
        raise ValueError('Expected reference functions missing from ' + str(path))
    for f in functions:
        # Avoid evaluating unknown annotations; no module top-level executed.
        f.returns = None
        for arg in f.args.args + f.args.kwonlyargs:
            arg.annotation = None
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--la-evaluator', required=True)
    p.add_argument('--la-common', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if digest(a.la_evaluator) != new.SOURCE_SHA256['eval_current5.py'] or digest(a.la_common) != new.SOURCE_SHA256['la-common.py']:
        raise ValueError('LA reference source changed; explicitly review and repin before comparison')
    common = {'re': __import__('re'), 'defaultdict': defaultdict, 'BOXES_STATUS':'boxes', 'NONE_STATUS':'none', 'EMPTY_STATUS':'empty'}
    tree = ast.parse(Path(a.la_common).read_text(encoding='utf-8'))
    regex_node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_OUTPUT_TOKEN_RE' for t in n.targets))
    exec(compile(ast.Module(body=[regex_node], type_ignores=[]), a.la_common, 'exec'), common)
    extract(a.la_common, ['_parse_output_units','parse_model_output_with_status','box_iou','maximum_matches'], common)
    env = dict(common, st=st, json=json, Counter=Counter, THRESHOLDS=new.THRESHOLDS, FORMS=new.FORMS,
               ChineseGroundingWorker=object, Path=Path)
    extract(a.la_evaluator, ['normalized','extract_query_items','extract_description','gt_groups','pred_groups',
                            'iou_metrics','add_counts','finalize_counts','_box_count_bucket','_empty_bucket_counts',
                            '_finalize_box_count_buckets','score_category_rows','score_ground_rows'], env)
    class Image:
        width, height = 1000, 1000
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def convert(self,mode): return self
        @staticmethod
        def open(path): return Image()
    env['Image'] = Image
    class Clock:
        def __init__(self): self.i=0
        def perf_counter(self):
            self.i += 1
            return self.i / 4
    env['time'] = Clock()
    rng = random.Random(a.seed)
    def answer():
        chunks=[]
        for label in ('enemy','weapon'):
            count=rng.randrange(4)
            chunks.append('<ref>'+label+'</ref>')
            if count == 0: chunks.append('<box>none</box>')
            for _ in range(count):
                x,y=rng.randrange(700),rng.randrange(700)
                chunks.append(f'<box><{x}><{y}><{x+150}><{y+150}></box>')
        return ''.join(chunks)
    for _ in range(1000):
        raw=answer()
        assert new.parse_la_answer(raw,1000,1000)==env['gt_groups'](raw,1000,1000)
    comparisons=0
    for form in new.FORMS:
        for _ in range(100):
            rows=[]; records=[]; predictions=[]
            for i in range(rng.randrange(1,7)):
                gt,pred=answer(),answer()
                gg,gs=env['gt_groups'](gt,1000,1000)
                boxes,ps=common['parse_model_output_with_status'](pred,1000,1000,None)
                pg,_=env['pred_groups'](boxes,ps)
                rows.append({'image':'mock','conversations':[{'value':'定位所有符合以下描述的目标：enemy</c>weapon。'},{'value':gt}]})
                records.append(dict(gt_groups=gg,pred_groups=pg,pred_status=ps,query_labels=['enemy','weapon'],latency_seconds=0.25))
                predictions.append((boxes,ps))
            class Worker:
                def __init__(self): self.i=0
                def detect(self,image,labels):
                    boxes,self.last_parse_status=predictions[self.i];self.i+=1
                    return boxes
            if form in ('category','multiclass'):
                expected=env['score_category_rows'](rows,Path('.'),Worker(),StringIO())
            else:
                expected=env['score_ground_rows'](rows,Path('.'),Worker(),StringIO(),negative=form=='negative')
            actual=new.score_records(records,form)
            for key,value in expected.items():
                if actual[key]!=value:
                    raise AssertionError(f'{form} metric mismatch {key}: {actual[key]!r} != {value!r}')
            comparisons+=1
    report=dict(status='passed', evidence='CPU behavioral comparison with original pure functions, not model inference',
                parser_cases=1000,cell_cases=comparisons,seed=a.seed,
                reference_sha256={'evaluator':digest(a.la_evaluator),'common':digest(a.la_common)},
                declared_source_sha256=new.SOURCE_SHA256)
    write_json(a.output,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
