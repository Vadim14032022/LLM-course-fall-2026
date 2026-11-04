import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src import pipeline
from src.provenance import generation_protocol, normalizer_protocol, input_fingerprints, assert_reusable


def replacement_normalizer(*args, **kwargs):
    return {'new':'postprocessing'}


class ProvenanceTests(unittest.TestCase):
    def test_generation_change_invalidates_cache_but_normalizer_is_separate(self):
        cfg={'runtime':'somewhere/runtime.json','generation':{'mode':'rag','max_tokens':4096},'retrieval':{'method':'bm25'}}
        runtime={'sampling':{'temperature':1},'model':{'thinking':True}}
        original=generation_protocol(cfg,runtime)
        with patch('src.pipeline.SYSTEM',pipeline.SYSTEM+' Changed instruction.'):
            self.assertNotEqual(original['sha256'],generation_protocol(cfg,runtime)['sha256'])
        base_normalizer=normalizer_protocol()
        with patch('src.pipeline.parse_answer',replacement_normalizer):
            self.assertEqual(original,generation_protocol(cfg,runtime))
            self.assertNotEqual(base_normalizer,normalizer_protocol())
        c=copy.deepcopy(cfg);c['generation']['response_strategy']='draft_then_prefill'
        self.assertNotEqual(original,generation_protocol(c,runtime))
        moved=copy.deepcopy(cfg);moved['runtime']='another/path.json'
        self.assertEqual(original,generation_protocol(moved,runtime))

    def test_logical_index_identity_ignores_clock_and_requires_data_and_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);q=root/'questions.jsonl';q.write_text('{"id":"q"}\n')
            chunks=root/'chunks.jsonl';chunks.write_text('chunk content')
            manifest={'format_version':1,'corpus_sha256':'corpus','chunking':{'target_chars':1600},'embedding':None,'build_seconds':.2}
            m=root/'manifest.json';m.write_text(json.dumps(manifest))
            cfg={'generation':{'mode':'rag'},'retrieval':{'method':'bm25'}};runtime={}
            snapshot={'inputs':input_fingerprints(q,root),'generation_protocol':generation_protocol(cfg,runtime)}
            assert_reusable(snapshot,cfg,runtime,q,root)
            manifest['build_seconds']=100;manifest['peak_ram_kib']=123;m.write_text(json.dumps(manifest))
            assert_reusable(snapshot,cfg,runtime,q,root)
            with patch('src.pipeline.SYSTEM','A different prompt'):
                with self.assertRaisesRegex(ValueError,'protocol'):assert_reusable(snapshot,cfg,runtime,q,root)
            with self.assertRaisesRegex(ValueError,'protocol'):assert_reusable({'inputs':snapshot['inputs']},cfg,runtime,q,root)
            chunks.write_text('different chunk')
            with self.assertRaisesRegex(ValueError,'input'):assert_reusable(snapshot,cfg,runtime,q,root)
