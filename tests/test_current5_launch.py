import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('run_current5', Path(__file__).resolve().parents[1] / 'scripts/run_current5.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class LaunchTests(unittest.TestCase):
    def test_dry_run_never_probes_gpu_or_starts_process(self):
        with patch.object(launch, 'gpu_snapshot', side_effect=AssertionError('GPU')), patch.object(launch.subprocess, 'Popen', side_effect=AssertionError('process')):
            self.assertEqual(launch.main(['baseline','--gpus','1,2','--output-dir',str(launch.ROOT/'outputs/eval/current5_plan_v1'),'--dry-run']),0)
            self.assertEqual(launch.main(['speed','--gpus','3','--output-dir',str(launch.ROOT/'outputs/eval/current5_speed_plan_v1'),'--dry-run']),0)

    def test_single_mode_retry_plan(self):
        import io,json
        from contextlib import redirect_stdout
        text=io.StringIO()
        with redirect_stdout(text):
            self.assertEqual(launch.main(['speed','--gpus','0','--speed-modes','speculative','--output-dir',str(launch.ROOT/'outputs/eval/retry_plan_v1'),'--dry-run']),0)
        self.assertEqual(json.loads(text.getvalue())['speed_modes'],['speculative'])

    def test_gpu_count_and_distinctness(self):
        with self.assertRaises(SystemExit):
            launch.main(['baseline','--gpus','1,1','--output-dir','outputs/x','--dry-run'])
        with self.assertRaises(SystemExit):
            launch.main(['speed','--gpus','1,2','--output-dir','outputs/x','--dry-run'])

    def test_speed_identity_pins_single_unobservable_host_pid(self):
        from types import SimpleNamespace
        replies = [SimpleNamespace(stdout='GPU-test\n'), SimpleNamespace(stdout='74778, GPU-test, 14000\n')]
        with patch.object(launch.subprocess, 'run', side_effect=replies), patch('eval.current5_speed.descendants', return_value={280463,281288}), patch.object(launch.Path, 'exists', return_value=False):
            identity = launch.speed_gpu_identity(0,280463)
        self.assertEqual(identity['allowed_driver_pids'], [74778])
        self.assertIn('Not a proven', identity['limitation'])

    def test_speed_identity_rejects_multiple_unknown_pids(self):
        from types import SimpleNamespace
        replies = [SimpleNamespace(stdout='GPU-test\n'), SimpleNamespace(stdout='74778, GPU-test, 14000\n88888, GPU-test, 1000\n')]
        with patch.object(launch.subprocess, 'run', side_effect=replies), patch('eval.current5_speed.descendants', return_value={280463,281288}):
            with self.assertRaisesRegex(RuntimeError, 'Cannot safely pin'):
                launch.speed_gpu_identity(0,280463)

    def test_service_plan_one_visible_device_one_instance(self):
        command, env = launch.service_command(3,8113,'causal',launch.ROOT/'outputs/eval/current5_v1/service', 'weights/dlm_bundle',0.4)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'3')
        self.assertEqual(env['GAM_SGLANG_MEM_FRACTION'],'0.4')
        self.assertIn('causal',command)
        self.assertIn('8113',command)


if __name__ == '__main__':
    unittest.main()
