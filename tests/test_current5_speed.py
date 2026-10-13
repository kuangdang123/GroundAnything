"""CPU-only mocks: no HTTP service, CUDA/model loading or actual nvidia-smi."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from eval import current5_speed as speed
from eval.current5_metrics import FORMS

BOX = [0, 0, 10, 10]


def record(task=None, latency=2, http=1, tokens=10, ok=True, valid=True, boxes=1):
    task = task or dict(sample_id='alpha/category/0001', game='alpha', form='category')
    return dict(sample_id=task['sample_id'], game=task['game'], form=task['form'],
                transport_ok=ok, valid=valid, latency_seconds=latency,
                request_seconds=http, preparation_seconds=.3, parse_seconds=.1,
                usage={} if tokens is None else {'completion_tokens': tokens},
                finish_reason='stop' if ok else None, pred_groups={},
                gt_groups={'a': [list(BOX) for _ in range(boxes)]})


class StatisticsTests(unittest.TestCase):
    def test_nearest_round_not_interpolation(self):
        values = [9, 1, 5, 3]
        self.assertEqual(speed.nearest(values, .5), 5)
        self.assertEqual(speed.nearest([1, 3], .5), 1)  # Python round(.5)==0.
        self.assertEqual(speed.nearest(values, .9), 9)
        self.assertIsNone(speed.nearest([], .95))

    def test_failed_cost_in_primary_and_success_only_separate(self):
        rows = [record(latency=2, tokens=12),
                record(latency=8, http=7, tokens=0, ok=False, valid=False)]
        result = speed.summarize(rows, 2)
        stats = result['overall']
        self.assertTrue(result['complete'])
        self.assertEqual(stats['sumtime_seconds'], 10)
        self.assertEqual(stats['samples_per_second'], .2)
        self.assertEqual(stats['e2e_seconds'], dict(mean=5, p50=2, p90=8, p95=8))
        self.assertEqual(stats['output_tokens_per_second'], 1.2)
        self.assertEqual(stats['http_only']['samples_per_second'], .25)
        self.assertEqual(stats['success_only']['samples_per_second'], .5)
        self.assertEqual(stats['success_only']['output_tokens_per_second'], 6)
        self.assertEqual(stats['failed_samples'], 1)
        self.assertEqual(stats['transport_errors'], 1)
        self.assertEqual(result['gpu']['isolation'], 'unknown')

    def test_missing_tokens_explicit_including_failed_rows(self):
        stats = speed.speed_stats([record(tokens=10), record(tokens=None, ok=False, valid=False)])
        self.assertEqual(stats['missing_completion_tokens'], 1)
        self.assertEqual(stats['completion_tokens'], 10)
        self.assertIsNone(stats['output_tokens_per_second'])
        self.assertEqual(stats['known_output_tokens_per_second'], 2.5)

    def test_invalid_transport_success_not_success_only(self):
        stats = speed.summarize([record(valid=False)], 1)['overall']
        self.assertEqual(stats['successful_samples'], 0)
        self.assertEqual(stats['invalid_samples'], 1)

    def test_groups_count_dict_values_not_keys_and_sampled_peak(self):
        rows = [record(boxes=count) for count in (0, 1, 2, 5, 6, 10, 11)]
        rows[-1]['form'] = 'multi_instance'
        rows[0]['gpu_before'] = dict(isolation='isolated', memory_used_mib=10)
        rows[0]['gpu_after'] = dict(isolation='contaminated', memory_used_mib=25)
        result = speed.summarize(rows, 8)
        self.assertFalse(result['complete'])
        self.assertEqual(set(result['by_gt_box_count']), {'0', '1', '2-5', '6-10', '11+'})
        self.assertEqual(result['by_gt_box_count']['2-5']['samples'], 2)
        self.assertEqual(result['by_form']['category']['samples'], 6)
        self.assertEqual(result['gpu']['peak_sampled_memory_mib'], 25)
        self.assertEqual(result['gpu']['isolation'], 'contaminated')

    def test_zero_and_empty_no_division_error(self):
        self.assertIsNone(speed.speed_stats([])['samples_per_second'])
        self.assertIsNone(speed.speed_stats([record(latency=0, http=0)])['samples_per_second'])

    def test_invalid_metrics_rejected(self):
        for key, value in [('latency_seconds', float('nan')), ('latency_seconds', -1),
                           ('request_seconds', float('inf')), ('usage', {'completion_tokens': -1}),
                           ('usage', {'completion_tokens': 1.5}), ('gt_groups', []),
                           ('gt_groups', {'a': [[0, 1]]}), ('valid', 1)]:
            row = record()
            row[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                speed.validate_record(row)


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.protocol_root = self.root / 'protocol'
        self.protocol_root.mkdir()
        (self.protocol_root / 'protocol.json').write_text('{"protocol":"current5-v1"}', encoding='utf-8')
        image = self.root / 'image.bin'
        image.write_bytes(b'CPU fixture')
        self.tasks = [dict(sample_id=f'{game}/{form}/{index:04d}', game=game, form=form,
                           row_index=index, image=str(image), source_sha256='samplehash',
                           gt_answer='<ref>a</ref><box><0><0><10><10></box>', query_items=['a'])
                      for game in ('beta', 'alpha') for form in FORMS for index in (2, 1, 3)]
        self.output = self.root / 'speed-denoise'
        self.argv = ['--protocol-root', str(self.protocol_root), '--output-dir', str(self.output),
                     '--warmup', '0', '--repeats', '2']
        self.loader = patch.object(speed, 'load_tasks', return_value=({'fixture': True}, self.tasks))
        self.loader.start()
        self.server = patch.object(speed, 'check_server', return_value={'mode': 'test', 'max_running_requests': 1})
        self.check = self.server.start()
        self.prober = patch.object(speed.GPUProbe, '_query', side_effect=AssertionError('No GPU calls in CPU tests'))
        self.prober.start()

    def tearDown(self):
        self.prober.stop()
        self.server.stop()
        self.loader.stop()
        self.temp.cleanup()

    def read_json(self, name):
        return json.loads((self.output / name).read_text(encoding='utf-8'))

    def test_cli_defaults_decoder_and_deterministic_parameters(self):
        args = speed.arguments(['--output-dir', str(self.output), '--mode', 'causal',
                                '--allowed-gpu-pids', '4,3,4'])
        self.assertEqual((args.decoder, args.temperature, args.top_p), ('causal', 0, 1))
        self.assertEqual((args.max_tokens, args.timeout, args.seed, args.warmup, args.repeats), (512, 180, 42, 3, 3))
        self.assertEqual(args.allowed_gpu_pids, [3, 4])

    def test_cli_bad_values(self):
        for option, value in [('--timeout', 'nan'), ('--max-tokens', '0'), ('--per-cell', '0'),
                              ('--warmup', '-1'), ('--repeats', '0'), ('--allowed-gpu-pids', 'bad')]:
            with self.subTest(option=option), self.assertRaises(SystemExit):
                speed.arguments(['--output-dir', str(self.output), option, value])

    def test_selection_first_and_hash_stable_under_input_permutation(self):
        first = speed.select_tasks(self.tasks, 1, 'first', 42)
        self.assertEqual(len(first), 10)
        self.assertTrue(all(t['row_index'] == 1 for t in first))
        a = speed.select_tasks(self.tasks, 2, 'hash', 42)
        b = speed.select_tasks(list(reversed(self.tasks)), 2, 'hash', 42)
        self.assertEqual(a, b)
        self.assertNotEqual(a, speed.select_tasks(self.tasks, 2, 'hash', 43))

    def test_dryrun_no_service_requests_or_gpu_even_if_isolation_required(self):
        with patch.object(speed, 'request', side_effect=AssertionError('HTTP forbidden')):
            self.assertEqual(speed.main(self.argv + ['--dry-run', '--gpu', '0', '--require-isolated']), 0)
        self.check.assert_not_called()
        self.assertIsNone(self.read_json('config.json')['server_info'])
        self.assertEqual(self.read_json('dry-run.json')['measured_requests'], 20)
        self.assertFalse((self.output / 'requests.jsonl').exists())

    def test_manifest_reused_across_modes_but_plan_change_refused(self):
        manifest = self.root / 'shared-plan.json'
        for mode in ('denoise', 'causal', 'speculative'):
            argv = ['--protocol-root', str(self.protocol_root), '--output-dir', str(self.root / mode),
                    '--mode', mode, '--manifest', str(manifest), '--dry-run']
            self.assertEqual(speed.main(argv), 0)
        original = manifest.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Manifest differs'):
            speed.main(self.argv + ['--manifest', str(manifest), '--per-cell', '2', '--dry-run'])
        self.assertEqual(manifest.read_bytes(), original)

    def test_resume_never_retries_failed_or_successful_and_warmup_excluded(self):
        count = [0]
        def response(task, args):
            count[0] += 1
            self.assertEqual((args.decoder, args.temperature, args.top_p), ('denoise', 0, 1))
            return record(task, ok=count[0] % 2 == 0, valid=count[0] % 2 == 0)
        argv = self.argv + ['--warmup', '3']
        with patch.object(speed, 'request', side_effect=response) as req:
            speed.main(argv)
            self.assertEqual(req.call_count, 23)
            before = (self.output / 'requests.jsonl').read_bytes()
            speed.main(argv)
            self.assertEqual(req.call_count, 23)
        self.assertEqual((self.output / 'requests.jsonl').read_bytes(), before)
        summary = self.read_json('summary.json')
        self.assertEqual(summary['overall']['samples'], 20)
        self.assertEqual(summary['overall']['failed_samples'], 10)
        config = self.read_json('config.json')
        self.assertEqual(config['server_info']['mode'], 'test')
        self.assertEqual(config['configuration']['arguments']['warmup'], 3)

    def test_resume_partial_completed_failures_skipped(self):
        with patch.object(speed, 'request', side_effect=lambda t, a: record(t, ok=False, valid=False)):
            speed.main(self.argv)
        path = self.output / 'requests.jsonl'
        lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
        path.write_text(''.join(lines[:3]), encoding='utf-8')
        with patch.object(speed, 'request', side_effect=lambda t, a: record(t)) as req:
            speed.main(self.argv)
            self.assertEqual(req.call_count, 17)
        self.assertEqual(self.read_json('summary.json')['overall']['transport_errors'], 3)

    def test_request_exception_recorded_and_cost_counted(self):
        with patch.object(speed, 'request', side_effect=RuntimeError('preparation failed')):
            speed.main(self.argv)
        stats = self.read_json('summary.json')['overall']
        self.assertEqual(stats['transport_errors'], 20)
        self.assertEqual(stats['missing_completion_tokens'], 20)
        self.assertEqual(stats['http_only']['missing_seconds'], 20)
        self.assertGreater(stats['sumtime_seconds'], 0)
        with patch.object(speed, 'request', side_effect=AssertionError('Must not retry')):
            speed.main(self.argv)

    def test_resume_different_config_refused_before_service(self):
        with patch.object(speed, 'request', side_effect=lambda t, a: record(t)):
            speed.main(self.argv)
        self.check.reset_mock()
        for extra in (['--mode', 'causal'], ['--timeout', '12'], ['--prompt-mode', 'ga_native'],
                      ['--seed', '99'], ['--repeats', '4'], ['--gpu', '0']):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, 'configuration differs'):
                speed.main(self.argv + extra)
        self.check.assert_not_called()

    def test_resume_changed_server_rejected_without_request(self):
        with patch.object(speed, 'request', side_effect=lambda t, a: record(t)):
            speed.main(self.argv)
        self.check.return_value = {'mode': 'different', 'max_running_requests': 1}
        with patch.object(speed, 'request') as req:
            with self.assertRaisesRegex(ValueError, 'server_info differs'):
                speed.main(self.argv)
            req.assert_not_called()

    def test_changed_image_manifest_and_artifact_collision_rejected(self):
        speed.main(self.argv + ['--dry-run'])
        Path(self.tasks[0]['image']).write_bytes(b'changed pixels')
        self.check.reset_mock()
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            speed.main(self.argv + ['--dry-run'])
        self.check.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'conflicts with a run artifact'):
            speed.main(['--protocol-root', str(self.protocol_root), '--output-dir', str(self.root / 'new-run'),
                        '--manifest', str(self.root / 'new-run' / 'config.json'), '--dry-run'])

    def test_warmup_cost_never_in_measured_summary(self):
        counter = [0]
        def response(task, args):
            counter[0] += 1
            return record(task, latency=999 if counter[0] <= 3 else 2)
        with patch.object(speed, 'request', side_effect=response):
            speed.main(self.argv + ['--warmup', '3'])
        self.assertEqual(self.read_json('summary.json')['overall']['sumtime_seconds'], 40)
        self.assertEqual(len((self.output / 'warmup.jsonl').read_text(encoding='utf-8').splitlines()), 3)

    def test_corrupt_jsonl_duplicate_foreign_and_unterminated_refused(self):
        with patch.object(speed, 'request', side_effect=lambda t, a: record(t)):
            speed.main(self.argv)
        path = self.output / 'requests.jsonl'
        first = path.read_text(encoding='utf-8').splitlines()[0]
        row = json.loads(first)
        foreign = copy.deepcopy(row)
        foreign['sample_id'] = 'foreign'
        wrong_repeat = copy.deepcopy(row)
        wrong_repeat['repeat'] = 999
        for text in ('{broken\n', first + '\n' + first + '\n', json.dumps(foreign) + '\n',
                     json.dumps(wrong_repeat) + '\n', first, '\n'):
            path.write_text(text, encoding='utf-8')
            self.check.reset_mock()
            with self.subTest(text=text[:20]), self.assertRaisesRegex(ValueError, 'Damaged JSONL'):
                speed.main(self.argv)
            self.assertEqual(path.read_text(encoding='utf-8'), text)
            self.check.assert_not_called()

    def test_post_request_isolation_failure_persists_row_and_does_not_kill(self):
        args = self.argv + ['--gpu', 'GPU-fixture', '--require-isolated']
        isolated = dict(isolation='isolated', uuid='GPU-fixture', memory_used_mib=10)
        foreign = dict(isolation='contaminated', uuid='GPU-fixture', memory_used_mib=20, foreign_pids=[999])
        with patch.object(speed.GPUProbe, 'sample', side_effect=[isolated, isolated, foreign]), \
                patch.object(speed, 'request', side_effect=lambda t, a: record(t)) as req:
            with self.assertRaisesRegex(ValueError, 'no processes were killed'):
                speed.main(args)
            self.assertEqual(req.call_count, 1)
        self.assertEqual(len((self.output / 'requests.jsonl').read_text(encoding='utf-8').splitlines()), 1)
        self.assertEqual(self.read_json('summary.json')['gpu']['isolation'], 'contaminated')
        with patch.object(speed, 'request') as req:
            with self.assertRaisesRegex(ValueError, 'Persisted request has failed/unknown GPU isolation'):
                speed.main(args)
            req.assert_not_called()

    def test_no_gpu_require_isolated_fails_before_request(self):
        with patch.object(speed, 'request') as req:
            with self.assertRaisesRegex(ValueError, 'unknown'):
                speed.main(self.argv + ['--require-isolated'])
            req.assert_not_called()

    def test_output_cannot_mutate_protocol_tree(self):
        with self.assertRaisesRegex(ValueError, 'must not mutate'):
            speed.main(self.argv + ['--output-dir', str(self.protocol_root / 'results'), '--dry-run'])


class GPUProbeTests(unittest.TestCase):
    def probe(self, extra=None):
        return speed.GPUProbe(speed.arguments(['--output-dir', 'unused', '--gpu', '0', *(extra or [])]))

    def test_uuid_filter_and_comma_allowed_pids(self):
        probe = self.probe(['--allowed-gpu-pids', '123,456'])
        with patch.object(probe, '_query', side_effect=[
                [['GPU-one', '100', '1000']],
                [['GPU-one', '123', '10'], ['GPU-other', '999', '10']]]):
            sample = probe.sample()
        self.assertEqual(sample['uuid'], 'GPU-one')
        self.assertEqual(sample['isolation'], 'isolated')
        self.assertEqual(sample['allowed_pids'], [123, 456])

    def test_foreign_pids_and_unknown_query(self):
        probe = self.probe()
        with patch.object(probe, '_query', side_effect=[
                [['GPU-one', '100', '1000']], [['GPU-one', '123', '10']]]):
            sample = probe.sample()
        self.assertEqual(sample['isolation'], 'contaminated')
        self.assertEqual(sample['foreign_pids'], [123])
        with patch.object(probe, '_query', side_effect=FileNotFoundError('no nvidia-smi')):
            self.assertEqual(probe.sample()['isolation'], 'unknown')

    def test_proc_descendants_comm_spaces_and_transitive(self):
        with tempfile.TemporaryDirectory() as temp:
            proc = Path(temp)
            for pid, parent in [(1, 0), (2, 1), (3, 2), (9, 0)]:
                (proc / str(pid)).mkdir()
                (proc / str(pid) / 'stat').write_text(f'{pid} (worker (with spaces)) S {parent} 0 0', encoding='utf-8')
            self.assertEqual(speed.descendants(1, proc), {1, 2, 3})
            with self.assertRaises(ValueError):
                speed.descendants(99, proc)

    def test_service_descendants_allowed_and_fail_closed_if_unavailable(self):
        probe = self.probe(['--service-parent-pid', '1'])
        data = [[['GPU-one', '100', '1000']], [['GPU-one', '3', '10']]]
        with patch.object(probe, '_query', side_effect=data), patch.object(speed, 'descendants', return_value={1, 2, 3}):
            self.assertEqual(probe.sample()['isolation'], 'isolated')
        with patch.object(probe, '_query', side_effect=data), patch.object(speed, 'descendants', side_effect=ValueError('no /proc')):
            self.assertEqual(probe.sample()['isolation'], 'unknown')


if __name__ == '__main__':
    unittest.main()
