import csv
import json
import tempfile
import unittest
from pathlib import Path
from scripts.collect_experiments import collect, build_manifest, write_manifest, FIELDS
from scripts.run_experiments import check_input_fingerprints
from src.corpus import sha256
from src.provenance import digest


def make_run(root,name,mode='rag'):
    directory=root/name;directory.mkdir()
    answer={'id':'q','answer_ids':['A'],'abstain':False,'citations':['s'],'rationale':'Proof'}
    (directory/'answers.jsonl').write_text(json.dumps(answer)+'\n')
    (directory/'trace.jsonl').write_text('{"id":"q"}\n')
    protocol={'version':1,'components':{'test':'fixture'}};protocol['sha256']=digest(protocol)
    snapshot={'config':{'generation':{'mode':mode}},'runtime':{},'generation_protocol':protocol,
              'inputs':{'questions_sha256':'questions','corpus_sha256':'corpus'}}
    (directory/'trace.config.json').write_text(json.dumps(snapshot))
    review_hash=None
    if mode!='no_context':
        review={'answers_sha256':sha256(directory/'answers.jsonl'),'review_status':'reviewed',
                'reviewed_question_ids':['q'],'questions':{}}
        review_path=directory/'content_review.json';review_path.write_text(json.dumps(review))
        review_hash=sha256(review_path)
    metrics={key:1 for key in FIELDS if key!='name'}
    metrics['details']=[{'id':'q'}]
    if mode=='no_context':metrics['Q']=None
    metrics.update({'usage_verified':True,'efficiency_eligible':True,
       'cohort':{'questions_sha256':'questions','gold_sha256':'gold','corpus_sha256':'corpus',
                 'runtime_sha256':digest({}),'normalizer_id':'protocol:normalizer','evaluation_protocol_sha256':'evaluation'},
       'provenance':{'answers_sha256':sha256(directory/'answers.jsonl'),'trace_sha256':sha256(directory/'trace.jsonl'),
                     'review_sha256':review_hash,'review_status':'reviewed' if review_hash else None,
                     'reviewed_question_ids_sha256':digest(['q']) if review_hash else None,
                     'generation_protocol_sha256':protocol['sha256']}})
    (directory/'metrics.json').write_text(json.dumps(metrics))
    return {'name':name,'directory':str(directory),'review_status':'not_applicable' if mode=='no_context' else 'reviewed'}


class CollectionTests(unittest.TestCase):
    def test_reviewed_status_requires_hash_bound_attestation_of_every_question(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);spec=make_run(root,'reviewed');run=root/'reviewed'
            review_path=run/'content_review.json';review=json.loads(review_path.read_text())
            review['reviewed_question_ids']=[];review_path.write_text(json.dumps(review))
            metrics_path=run/'metrics.json';metrics=json.loads(metrics_path.read_text())
            metrics['provenance']['review_sha256']=sha256(review_path)
            metrics_path.write_text(json.dumps(metrics))
            manifest_path=root/'cohort.json';output=root/'table.csv'
            write_manifest(manifest_path,build_manifest([spec],manifest_path))
            with self.assertRaisesRegex(ValueError,'attest every question'):
                collect(manifest_path,output)
            self.assertFalse(output.exists())

    def test_same_question_ids_do_not_allow_reuse_after_text_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);question=root/'questions.jsonl'
            question.write_text('{"id":"same","question":"old"}\n')
            snapshot={'inputs':{'questions_sha256':sha256(question)}}
            check_input_fingerprints(snapshot,question,root,no_context=True)
            question.write_text('{"id":"same","question":"new"}\n')
            with self.assertRaises(ValueError):check_input_fingerprints(snapshot,question,root,no_context=True)

    def test_declared_new_ablation_names_survive_repeated_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'cohort.json';output=root/'experiments.csv'
            specs=[make_run(root,name) for name in ['bm25','final_new','ablation_new']]
            write_manifest(path,build_manifest(specs,path))
            collect(path,output);collect(path,output)
            with output.open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual({r['name'] for r in rows},{'bm25','final_new','ablation_new'})
            self.assertEqual(len(rows),3);self.assertNotIn(b'\r',output.read_bytes())
            make_run(root,'unrelated_run')
            self.assertEqual(len(collect(path,output)),3)  # Never glob undeclared runs.

    def test_mixed_cohort_and_pending_review_cannot_overwrite_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'cohort.json';output=root/'experiments.csv';output.write_text('KEEP')
            specs=[make_run(root,'a'),make_run(root,'b')]
            metrics_path=root/'b/metrics.json';metrics=json.loads(metrics_path.read_text())
            metrics['cohort']['corpus_sha256']='different';metrics_path.write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError,'Mixed'):build_manifest(specs,path)
            self.assertEqual(output.read_text(),'KEEP')
            specs=specs[:1];specs[0]['review_status']='pending'
            write_manifest(path,build_manifest(specs,path))
            with self.assertRaisesRegex(ValueError,'review pending'):collect(path,output)
            self.assertEqual(output.read_text(),'KEEP')

    def test_stale_outputs_and_missing_generation_proof_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'cohort.json';output=root/'experiments.csv';output.write_text('KEEP')
            spec=make_run(root,'a');manifest=build_manifest([spec],path);write_manifest(path,manifest)
            (root/'a/answers.jsonl').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Stale'):collect(path,output)
            self.assertEqual(output.read_text(),'KEEP')
            spec=make_run(root,'b');snapshot_path=root/'b/trace.config.json'
            snapshot=json.loads(snapshot_path.read_text());snapshot.pop('generation_protocol');snapshot_path.write_text(json.dumps(snapshot))
            metrics_path=root/'b/metrics.json';metrics=json.loads(metrics_path.read_text());metrics['provenance']['generation_protocol_sha256']=None;metrics_path.write_text(json.dumps(metrics))
            write_manifest(path,build_manifest([spec],path))
            with self.assertRaisesRegex(ValueError,'generation protocol'):collect(path,output)
            write_manifest(path,build_manifest([spec],path,historical=True))
            self.assertEqual(collect(path,output),['b'])
