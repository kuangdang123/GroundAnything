import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('after_la_current5',Path(__file__).resolve().parents[1]/'scripts/after_la_current5.py')
a=importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.la=self.root/'la';self.la.mkdir()
        self.proc=self.root/'proc';self.proc.mkdir()
        self.protocol={'games':{'g':{'forms':{f:{'sampled_rows':1} for f in a.FORMS}}}}
        (self.la/'protocol.json').write_text(json.dumps(self.protocol))
        (self.la/'models.json').write_text(json.dumps({'m':{}}))
        self.plan={'la_root':str(self.la),'protocol_sha256':a.source_hash(self.la/'protocol.json'),'model_keys':['m'],'queue_processes':[],'gpus':[1,2,3], 'baseline_output':'baseline','speed_output':'speed'}

    def complete(self):
        path=self.la/'results/m';path.mkdir(parents=True)
        forms={f:{'metric':a.PRIMARY_METRIC[f],'per_game':{'g':0.5}} for f in a.FORMS}
        (path/'summary.json').write_text(json.dumps({'current5_equal_view_macro':0.5,'forms':forms}))

    def test_incomplete_models_do_not_probe_gpu(self):
        with patch.object(a,'gpu_snapshot',side_effect=AssertionError('no GPU probe')):
            state=a.snapshot_conditions(self.plan,self.proc)
        self.assertFalse(state['ready']);self.assertEqual(state['missing_models'],['m'])

    def test_all_complete_empty_gpu_is_ready(self):
        self.complete()
        with patch.object(a,'gpu_snapshot',return_value={}):state=a.snapshot_conditions(self.plan,self.proc)
        self.assertTrue(state['ready'])

    def test_external_gpu_process_blocks_launch(self):
        self.complete()
        with patch.object(a,'gpu_snapshot',side_effect=ValueError('occupied')):state=a.snapshot_conditions(self.plan,self.proc)
        self.assertFalse(state['ready']);self.assertEqual(state['gpu_wait_reason'],'occupied')

    def test_active_queue_identity_blocks_gpu_probe(self):
        self.complete();self.plan['queue_processes']=[{'pid':10,'start_ticks':'123','command':'bash sweepU.sh'}]
        with patch.object(a,'proc_identity',return_value=self.plan['queue_processes'][0]),patch.object(a,'gpu_snapshot',side_effect=AssertionError('GPU')):
            state=a.snapshot_conditions(self.plan,self.proc)
        self.assertFalse(state['ready']);self.assertEqual(len(state['queue_alive']),1)

    def test_pid_reuse_does_not_count_as_original_queue(self):
        self.complete();self.plan['queue_processes']=[{'pid':10,'start_ticks':'123'}]
        with patch.object(a,'proc_identity',return_value={'pid':10,'start_ticks':'124'}),patch.object(a,'gpu_snapshot',return_value={}):state=a.snapshot_conditions(self.plan,self.proc)
        self.assertTrue(state['ready'])

    def test_inventory_change_refuses_automatic_launch(self):
        (self.la/'models.json').write_text(json.dumps({'m':{},'new':{}}))
        with self.assertRaisesRegex(ValueError,'inventory changed'):a.snapshot_conditions(self.plan,self.proc)

    def test_partial_summary_missing_game_is_incomplete(self):
        self.complete();path=self.la/'results/m/summary.json';v=json.loads(path.read_text());v['forms']['negative']['per_game']={};path.write_text(json.dumps(v))
        self.assertFalse(a.result_complete(self.la,'m',self.protocol))

    def test_launch_mapping_and_only_speculative_speed(self):
        commands=a.launch_commands(self.plan)
        self.assertIn('1,2',commands['baseline']);self.assertIn('3',commands['speculative']);self.assertIn('speculative',commands['speculative'])
        self.assertNotIn('--force',commands['baseline'])


if __name__=='__main__':unittest.main()
