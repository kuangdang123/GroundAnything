"""CPU-only fake-service and fixed-manifest checks; no model is exercised."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest

from PIL import Image
from eval import current5 as c


class Current5ClientTests(unittest.TestCase):
    def test_ga_recovery_preserves_duplicates_and_grid(self):
        raw = '<|object_ref_start|>敌人<|object_ref_end|><|box_start|><0><0><999><999>,<0><0><999><999><|box_end|>'
        groups, status = c.parse_ga_answer(raw, 101, 51)
        self.assertEqual(status, 'boxes')
        self.assertEqual(groups['敌人'], [[0, 0, 100, 50]] * 2)
        self.assertEqual(c.parse_ga_answer('garbage', 101, 51), ({}, 'empty'))
        self.assertEqual(c.parse_ga_answer('<|box_start|>None<|box_end|>', 101, 51), ({}, 'none'))

    def test_prompt_modes_are_separate(self):
        task = {'query_items': ['敌人', '投掷物'], 'question': '定位所有符合以下描述的目标：敌人</c>投掷物。'}
        self.assertEqual(c.build_prompt(task, 'la_zh'), task['question'])
        self.assertIn('all the instances', c.build_prompt(task, 'ga_native'))
        self.assertIn('敌人</c>投掷物', c.build_prompt(task, 'ga_native'))

    def test_transport_and_raw_predictions(self):
        bodies = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                bodies.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                data = json.dumps({'choices': [{'message': {'content': '<|object_ref_start|>敌人<|object_ref_end|><|box_start|><0><0><999><999><|box_end|>'}, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 8}}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def log_message(self, *args):
                pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as root:
                image = Path(root) / 'i.png'
                Image.new('RGB', (101, 51)).save(image)
                task = dict(sample_id='g/category/0001', game='g', form='category', row_index=1,
                            image=str(image), source_sha256='sample', query_items=['敌人'],
                            question='定位所有符合以下描述的目标：敌人。',
                            gt_answer='<ref>敌人</ref><box><0><0><1000><1000></box>')
                args = argparse.Namespace(base_url=f'http://127.0.0.1:{server.server_port}/v1', model='groundinganything', decoder='denoise',
                                          prompt_mode='la_zh', max_tokens=512, temperature=0.1, top_p=0.9, seed=42, timeout=2)
                record = c.request(task, args)
                self.assertTrue(record['transport_ok'])
                self.assertTrue(record['valid'])
                self.assertEqual(record['gt_groups']['敌人'][0], [0, 0, 101, 51])
                self.assertEqual(record['pred_groups']['敌人'][0], [0, 0, 100, 50])
                self.assertGreater(record['latency_seconds'], 0)
                self.assertEqual(bodies[0]['max_tokens'], 512)
                self.assertFalse(bodies[0]['skip_special_tokens'])
                self.assertIsNone(bodies[0]['custom_params']['gam_dlm_decode']['expected_cardinality'])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_bad_output_tail_is_not_silently_repaired(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'rows.jsonl'
            path.write_text('{bad', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Damaged output'):
                c.load_records(path)
            self.assertEqual(path.read_text(), '{bad')

    def test_existing_samples_are_hash_checked_and_not_resampled(self):
        from eval.current5_metrics import FORMS
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            image_dir = root / 'game-run/val/images'
            image_dir.mkdir(parents=True)
            Image.new('RGB', (8, 8)).save(image_dir / 'frame.png')
            sample = root / 'sample.jsonl'
            row = {'image': 'frame.png', 'conversations': [{'value': '定位所有符合以下描述的目标：敌人。'}, {'value': '<box>none</box>'}]}
            sample.write_text(json.dumps(row) + '\n', encoding='utf-8')
            protocol = {'protocol': 'current5-v1', 'forms': list(FORMS), 'games': {'g': {'run_dir': str(root / 'game-run'), 'forms': {'category': {'sample': str(sample), 'sample_sha256': c.digest(sample), 'sampled_rows': 1}}}}}
            (root / 'protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
            _, tasks = c.load_tasks(root)
            self.assertEqual(tasks[0]['query_items'], ['敌人'])
            self.assertEqual(tasks[0]['sample_id'], 'g/category/0001')
            sample.write_text(json.dumps(row) + '\n\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'SHA256 changed'):
                c.load_tasks(root)

    def test_greedy_speculation_guard(self):
        with self.assertRaises(SystemExit):
            c.arguments(['--output-dir', 'unused', '--decoder', 'speculative'])
        a = c.arguments(['--output-dir', 'unused', '--decoder', 'speculative', '--temperature', '0', '--top-p', '1'])
        self.assertEqual(a.max_tokens, 512)


if __name__ == '__main__':
    unittest.main()
