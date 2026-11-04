"""Publish only an explicitly declared, fingerprint-checked cohort."""
import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.corpus import sha256
from src.provenance import digest

FIELDS=['name','answer_accuracy','Q','mean_input_tokens','mean_output_tokens','mean_llm_calls','mean_cost','median_latency_ms']
COMMON=('questions_sha256','corpus_sha256','gold_sha256','runtime_sha256','normalizer_id','evaluation_protocol_sha256')
FILES=('answers.jsonl','trace.jsonl','trace.config.json','metrics.json')


def build_manifest(specs, manifest_path, historical=False):
    """Construct a pending manifest from completed metrics, never a glob."""
    parent=Path(manifest_path).resolve().parent
    rows=[];common=None;names=set()
    for spec in specs:
        name=spec['name'];directory=Path(spec['directory']).resolve()
        if name in names:raise ValueError('Duplicate cohort name: '+name)
        names.add(name)
        metrics=json.loads((directory/'metrics.json').read_text())
        observed={k:metrics.get('cohort',{}).get(k) for k in COMMON}
        if any(v is None for v in observed.values()):raise ValueError('Incomplete cohort fingerprints: '+name)
        if common is None:common=observed
        elif common!=observed:raise ValueError('Mixed corpus/questions/gold/runtime/normalizer/evaluation cohort: '+name)
        snapshot=json.loads((directory/'trace.config.json').read_text())
        mode=snapshot['config']['generation']['mode']
        entry={'name':name,'directory':os.path.relpath(directory,parent),'mode':mode,
               'review_status':spec.get('review_status','not_applicable' if mode=='no_context' else 'pending'),
               'hashes':{file:sha256(directory/file) for file in FILES}}
        review=directory/'content_review.json'
        entry['review_file_sha256']=sha256(review) if review.exists() else None
        rows.append(entry)
    if not rows:raise ValueError('Empty cohort')
    if not historical and not common['normalizer_id'].startswith('protocol:'):
        raise ValueError('Legacy normalizer needs an explicitly historical cohort')
    return {'version':1,'historical':historical,'cohort':common,'runs':rows}


def write_manifest(path, manifest):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    atomic_write(path,json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')


def atomic_write(path, text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    handle=tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',newline='',dir=path.parent,delete=False)
    try:
        with handle:handle.write(text)
        os.replace(handle.name,path)
    finally:
        if os.path.exists(handle.name):os.unlink(handle.name)


def collect(manifest_path, output):
    manifest_path=Path(manifest_path).resolve()
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('version')!=1 or not manifest.get('runs') or type(manifest.get('historical')) is not bool or set(manifest.get('cohort',{}))!=set(COMMON):raise ValueError('Invalid cohort manifest')
    rows=[];names=set()
    for entry in manifest['runs']:
        name=entry['name'];directory=(manifest_path.parent/entry['directory']).resolve()
        if name in names:raise ValueError('Duplicate cohort name')
        names.add(name)
        if set(entry.get('hashes',{}))!=set(FILES):raise ValueError('Incomplete file fingerprints')
        for file,expected in entry['hashes'].items():
            if sha256(directory/file)!=expected:raise ValueError('Stale cohort file: '+name+'/'+file)
        metrics=json.loads((directory/'metrics.json').read_text())
        observed={k:metrics.get('cohort',{}).get(k) for k in COMMON}
        if observed!=manifest['cohort'] or any(v is None for v in observed.values()):raise ValueError('Mixed or incomplete cohort: '+name)
        snapshot=json.loads((directory/'trace.config.json').read_text())
        if snapshot['config']['generation']['mode']!=entry['mode']:raise ValueError('Mode mismatch')
        if digest(snapshot['runtime'])!=observed['runtime_sha256']:raise ValueError('Runtime fingerprint mismatch')
        protocol=snapshot.get('generation_protocol',{})
        if metrics.get('provenance',{}).get('generation_protocol_sha256')!=protocol.get('sha256'):raise ValueError('Metrics generation protocol mismatch')
        if not manifest.get('historical'):
            if not protocol or protocol.get('sha256')!=digest({k:v for k,v in protocol.items() if k!='sha256'}):raise ValueError('Missing/invalid generation protocol')
            if snapshot.get('inputs',{}).get('questions_sha256')!=observed['questions_sha256']:raise ValueError('Generation questions mismatch')
            if entry['mode']!='no_context' and snapshot.get('inputs',{}).get('corpus_sha256')!=observed['corpus_sha256']:raise ValueError('Generation corpus mismatch')
        review=directory/'content_review.json'
        actual_review=sha256(review) if review.exists() else None
        if actual_review!=entry.get('review_file_sha256') or actual_review!=metrics.get('provenance',{}).get('review_sha256'):raise ValueError('Stale content review: '+name)
        review_data=json.loads(review.read_text()) if review.exists() else None
        if review_data and review_data.get('answers_sha256')!=sha256(directory/'answers.jsonl'):raise ValueError('Review answers mismatch')
        if entry.get('review_status')=='reviewed':
            expected_ids={row['id'] for row in metrics.get('details',[]) if isinstance(row,dict) and isinstance(row.get('id'),str)}
            reviewed=review_data.get('reviewed_question_ids') if review_data else None
            if (not expected_ids or review_data.get('review_status')!='reviewed'
                    or not isinstance(reviewed,list) or any(not isinstance(q,str) or not q for q in reviewed)
                    or len(reviewed)!=len(set(reviewed)) or set(reviewed)!=expected_ids):
                raise ValueError('Review does not attest every question: '+name)
            if metrics.get('provenance',{}).get('review_status')!='reviewed' or metrics.get('provenance',{}).get('reviewed_question_ids_sha256')!=digest(sorted(reviewed)):
                raise ValueError('Metrics review attestation mismatch: '+name)
        if metrics.get('provenance',{}).get('answers_sha256')!=sha256(directory/'answers.jsonl') or metrics.get('provenance',{}).get('trace_sha256')!=sha256(directory/'trace.jsonl'):raise ValueError('Metrics refer to different outputs')
        accepted_review = entry['review_status']=='reviewed' or (entry['mode']=='no_context' and entry['review_status']=='not_applicable')
        historical_unreviewed = manifest.get('historical') and entry['review_status']=='historical_unreviewed'
        if not accepted_review and not historical_unreviewed:raise ValueError('Content review pending: '+name)
        if metrics.get('usage_verified') is not True or metrics.get('efficiency_eligible') is not True:raise ValueError('Unverified expense: '+name)
        for key in FIELDS[1:]:
            value=metrics.get(key)
            if key=='Q' and value is None and entry['mode']=='no_context':continue
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('Invalid metric: '+name+'/'+key)
        if metrics['answer_accuracy']>1 or metrics['Q'] is not None and metrics['Q']>1:raise ValueError('Invalid quality range')
        rows.append({'name':name,**{key:metrics[key] for key in FIELDS[1:]}})
    import io
    buffer=io.StringIO(newline='')
    writer=csv.DictWriter(buffer,fieldnames=FIELDS,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    atomic_write(output,buffer.getvalue())
    return [row['name'] for row in rows]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(collect(args.manifest,args.output))


if __name__=='__main__':main()
