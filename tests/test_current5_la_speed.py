import types
import unittest
from eval.current5_la_speed import parse_native_stats,summarize_mode,synchronized_generate_probe,MODES


class LASpeedTests(unittest.TestCase):
    def test_native_counts_not_retokenized_text(self):
        stats=parse_native_stats('Statistic Info, num_tokens=12; generate_time(s)=1.23; tps=9.75; forward_step=3; num_boxes=2; bps=1.62; prefill_time=0.1; switch_to_ar=1')
        self.assertEqual(stats['num_tokens'],12);self.assertEqual(stats['num_boxes'],2)
        self.assertEqual(MODES,('fast','hybrid','slow'))
        with self.assertRaises(ValueError):parse_native_stats('num_tokens=3')

    def test_ratio_of_totals_and_box_markers_separate_from_valid(self):
        rows=[]
        for n,t,b,valid in [(10,2,3,1),(20,4,2,2)]:
            rows.append({'latency_seconds':t+1,'request_seconds':None,'usage':{'completion_tokens':n},'native_stats':{'num_boxes':b},'pred_groups':{'a':[[0,0,1,1]]*valid},'generate_seconds':t,'decode_window_seconds':t-0.5,'forward_calls':2,'valid':True})
        result=summarize_mode(rows)
        self.assertEqual(result['generate_tps'],5)
        self.assertEqual(result['generate_bps'],5/6)
        self.assertEqual(result['valid_boxes_per_generate_second'],0.5)
        self.assertEqual(result['language_forward_calls'],4)

    def test_probe_restores_model_and_counts_real_forward_calls(self):
        language=types.SimpleNamespace(forward=lambda x:x+1)
        model=types.SimpleNamespace(language_model=language)
        def generate():
            language.forward(1);language.forward(2)
            return 'answer'
        model.generate=generate;worker=types.SimpleNamespace(model=model)
        original_forward=language.forward
        cuda=types.SimpleNamespace(synchronize=lambda:None)
        with synchronized_generate_probe(worker,types.SimpleNamespace(cuda=cuda)) as trace:
            self.assertEqual(model.generate(),'answer')
        self.assertEqual(trace['forward_calls'],2)
        self.assertGreaterEqual(trace['generate_seconds'],trace['decode_window_seconds'])
        self.assertIs(model.generate,generate);self.assertIs(language.forward,original_forward)


if __name__=='__main__':unittest.main()
