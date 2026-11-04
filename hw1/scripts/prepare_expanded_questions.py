"""Compile source-authored QA annotations; never used by inference."""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.corpus import read_jsonl, write_jsonl, sha256

CORPUS_SHA='909b7b32472dc641acbe69c59748815c8baa16c92df48bc3189cd3118230ffb8'
CATEGORIES={'условия','сопоставление','применение','недостаточность данных'}


def normalized(text):
    return re.sub(r'\s+',' ',text.casefold()).strip()


def compile_case(case, blocks):
    if case['category'] not in CATEGORIES:raise ValueError('Unknown category')
    opts=case['options'];correct=case['correct']
    if not 4<=len(opts)<=6 or len(set(map(normalized,opts)))!=len(opts):raise ValueError('Invalid options')
    if (not isinstance(correct,list) or len(set(correct))!=len(correct)
        or any(type(i) is not int or not 0<=i<len(opts) for i in correct)):raise ValueError('Invalid correct indices')
    abstain=case['category']=='недостаточность данных'
    if abstain != (not correct):raise ValueError('Abstention/category mismatch')
    if not case['rationale'].strip() or abstain and not case.get('missing_information','').strip():raise ValueError('Missing explanation')
    sets=case['citation_sets']
    if not sets or any(not isinstance(s,list) or len(s)>4 or len(s)!=len(set(s)) or not set(s)<=set(blocks) for s in sets):raise ValueError('Invalid source set')
    if any(not s for s in sets):raise ValueError('Source-authored evidence must be nonempty even for refusal')
    identity=json.dumps({'question':case['question'],'options':opts},ensure_ascii=False,sort_keys=True)
    code=hashlib.sha256(identity.encode()).hexdigest()
    qid='qa_'+code[:12]
    order=list(range(len(opts)));random.Random(int(code[:16],16)).shuffle(order)
    question={'id':qid,'question':case['question'],'options':[{'id':chr(65+i),'text':opts[j]} for i,j in enumerate(order)]}
    gold={'id':qid,'category':case['category'],'answer_ids':[chr(65+i) for i,j in enumerate(order) if j in correct],
          'abstain':abstain,'citation_sets':[[]] if abstain else sets,'rationale':case['rationale'],
          'missing_information':case.get('missing_information',''),'origin':'source_authored_expanded_v1'}
    meta={'id':qid,'topic':case['topic'],'family_id':case['family_id'],'evidence_sets':sets,
          'source_sha256':{bid:hashlib.sha256(blocks[bid]['text'].encode()).hexdigest() for s in sets for bid in s}}
    return question,gold,meta


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--corpus',type=Path,default=ROOT/'artifacts/review2_corpus_v5/corpus.jsonl')
    p.add_argument('--directory',type=Path,default=ROOT/'questions/expanded_v2')
    p.add_argument('--draft',action='store_true',help='Export partial development data, not a frozen dataset')
    args=p.parse_args();directory=args.directory
    prior=directory/'manifest.json'
    if args.draft and prior.exists() and json.loads(prior.read_text()).get('status')=='frozen':
        raise ValueError('Cannot overwrite a frozen split with draft output')
    if sha256(args.corpus)!=CORPUS_SHA:raise ValueError('Source corpus changed; review annotations before rebuilding')
    blocks={x['block_id']:x for x in read_jsonl(args.corpus)}
    splits={s:json.loads((directory/f'annotations_{s}.json').read_text()) for s in ['dev','holdout'] if (directory/f'annotations_{s}.json').exists()}
    if not args.draft and set(splits)!={'dev','holdout'}:raise ValueError('Both complete splits are required')
    families={s:{c['family_id'] for c in rows} for s,rows in splits.items()}
    if families.get('dev',set())&families.get('holdout',set()):raise ValueError('Family leakage between splits')
    historical={normalized(q['question']) for f in ['questions/dev_questions.jsonl','own_questions.jsonl'] for q in read_jsonl(ROOT/f)}
    allids=set();texts=set();out={};annotations=[]
    for split,rows in splits.items():
        counts=Counter(c['category'] for c in rows)
        if not args.draft and (len(rows)!=40 or set(counts)!=CATEGORIES or set(counts.values())!={10}):raise ValueError('Expected 10 questions per category per split')
        compiled=[]
        for case in rows:
            q,g,m=compile_case(case,blocks)
            key=normalized(q['question'])
            if q['id'] in allids or key in texts or key in historical:raise ValueError('Duplicate question: '+q['id'])
            allids.add(q['id']);texts.add(key);compiled.append((q,g,m))
        compiled.sort(key=lambda x:x[0]['id'])
        out[split]=compiled
        annotations.extend({'split':split,**m} for q,g,m in compiled)
    if not args.draft:
        review=json.loads((directory/'annotation_review.json').read_text())
        if review.get('status')!='reviewed' or set(review.get('reviewed_question_ids',[]))!=allids:raise ValueError('Content review for all 80 questions is required')
        if review.get('annotation_hashes')!={s:sha256(directory/f'annotations_{s}.json') for s in splits}:raise ValueError('Stale annotation review')
    files={}
    for split,compiled in out.items():
        for suffix,rows in [('questions',[q for q,g,m in compiled]),('gold',[g for q,g,m in compiled])]:
            target=directory/f'{split}_{suffix}.jsonl';write_jsonl(target,rows);files[target.name]=sha256(target)
    source_map={'corpus_sha256':CORPUS_SHA,'questions':annotations}
    (directory/'source_map.json').write_text(json.dumps(source_map,ensure_ascii=False,indent=2)+'\n')
    files['source_map.json']=sha256(directory/'source_map.json')
    manifest={'version':1,'status':'draft' if args.draft else 'frozen','corpus_sha256':CORPUS_SHA,
              'annotation_hashes':{s:sha256(directory/f'annotations_{s}.json') for s in splits},'files':files,
              'splits':{s:{'count':len(rows),'categories':dict(Counter(g['category'] for q,g,m in rows)),
                'topics':dict(Counter(m['topic'] for q,g,m in rows))} for s,rows in out.items()},
              'policy':'Disjoint source-authored task families; deterministic content IDs and option permutation. Draft data must not be used as a frozen holdout.'}
    if not args.draft:
        manifest['frozen_at']=review['reviewed_at']
        manifest['annotation_review_sha256']=sha256(directory/'annotation_review.json')
    (directory/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest['splits'],ensure_ascii=False));print('status:',manifest['status'])


if __name__=='__main__':main()
