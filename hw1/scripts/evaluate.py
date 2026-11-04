"""Evaluate canonical records strictly; never normalize a submitted answer."""
import argparse
import json
import statistics
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.corpus import read_jsonl, sha256
from src.validation import answer_errors, trace_errors, accounting_errors, final_context
from src.provenance import digest

CATEGORIES = {'условия', 'сопоставление', 'применение', 'недостаточность данных'}


def keyed(rows, label):
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id']:
            raise ValueError(label + ': invalid/missing question ID')
        if row['id'] in result:
            raise ValueError(label + ': duplicate question ID ' + row['id'])
        result[row['id']] = row
    return result


def f1(predicted, expected):
    a, b = set(predicted), set(expected)
    return 2*len(a & b)/(len(a)+len(b)) if a or b else 1.0


def evaluate(answers, gold, traces, no_context=False, reviews=None,
             questions=None, diagnostics=None, source_ids=None):
    answers, gold, traces = keyed(answers, 'answers'), keyed(gold, 'gold'), keyed(traces, 'trace')
    if not gold or {g.get('category') for g in gold.values()} != CATEGORIES:
        raise ValueError('Gold must contain all four categories')
    if (set(answers) | set(traces)) - set(gold):
        raise ValueError('Unknown answer/trace question IDs')
    qs = keyed(questions, 'questions') if questions is not None else None
    ds = keyed(diagnostics, 'diagnostics') if diagnostics is not None else None
    if qs is not None and set(qs) != set(gold):
        raise ValueError('Questions and gold IDs differ')
    if ds is not None and set(ds)-set(gold):
        raise ValueError('Unknown diagnostic question IDs')
    if qs is not None:
        for q in qs.values():
            options = q.get('options')
            if not isinstance(options, list) or not options or any(not isinstance(o, dict) or not isinstance(o.get('id'), str) or not o['id'] for o in options) or len({o['id'] for o in options}) != len(options):
                raise ValueError('Invalid question options')
    for qid, g in gold.items():
        sets = g.get('citation_sets')
        choices = g.get('answer_ids')
        if type(g.get('abstain')) is not bool or not isinstance(choices, list) or any(not isinstance(v, str) or not v for v in choices) or len(set(choices)) != len(choices):
            raise ValueError('Invalid gold choices/abstain')
        if g['abstain'] != (not choices):
            raise ValueError('Gold abstain/choices mismatch')
        if qs is not None and not set(choices) <= {o['id'] for o in qs[qid]['options']}:
            raise ValueError('Gold contains unknown option IDs')
        if not isinstance(sets, list) or not sets or any(not isinstance(s, list) or len(s)>4 or any(not isinstance(v,str) or not v for v in s) or len(set(s))!=len(s) for s in sets):
            raise ValueError('Invalid gold citation sets')
        if any(bool(s) == g['abstain'] for s in sets):
            raise ValueError('Gold citation sets/abstain mismatch')
    reviews = reviews or {}
    if not set(reviews) <= set(gold):
        raise ValueError('Content review contains unknown question IDs')
    for review in reviews.values():
        if review.get('zero_score') is not True or not isinstance(review.get('reason'), str) or not review['reason'].strip():
            raise ValueError('Each content override requires zero_score=true and a reason')
    groups, automatic_groups, details = {}, {}, []
    verified = True
    trace_valid = True
    all_contexts = ds is not None
    for qid, g in gold.items():
        a, t = answers.get(qid), traces.get(qid)
        d = ds.get(qid) if ds is not None else t
        allowed = set(source_ids) if source_ids is not None else None
        reference_errors = []
        has_call_context=isinstance(d,dict) and isinstance(d.get('calls'),list) and any(
            isinstance(call,dict) and 'context_block_ids' in call for call in d['calls'])
        if not no_context and d is not None and ('context_block_ids' in d or has_call_context):
            context,context_errors=final_context(d,allowed)
            reference_errors.extend(context_errors)
            allowed=set(context) if allowed is None else set(context)&allowed
        else:
            all_contexts = False
        errors = answer_errors(a, qs[qid] if qs is not None else None, allowed, no_context)
        terrors = trace_errors(t)
        errors += terrors + reference_errors
        cost_errors = accounting_errors(t, d)
        trace_valid = trace_valid and not terrors
        verified = verified and not cost_errors
        exact = not errors and not t.get('technical_error', False) and a['abstain'] == g['abstain'] and set(a['answer_ids']) == set(g['answer_ids'])
        sf1 = max(f1(a['citations'], s) for s in g['citation_sets']) if not errors else 0.0
        score = float(exact) if g['abstain'] else float(exact)*(0.8+0.2*sf1)
        automatic_score = score
        automatic_groups.setdefault(g['category'], []).append(automatic_score)
        if qid in reviews:
            score = 0.0
        groups.setdefault(g['category'], []).append(score)
        over = isinstance(t,dict) and any(type(t.get(k)) is int and t[k]>cap for k,cap in [('llm_calls',4),('input_tokens',16384),('output_tokens',4096)])
        details.append({'id':qid, 'exact':bool(exact), 'citation_f1':sf1, 'score':score,
                        'automatic_score':automatic_score, 'over_budget':over,
                        'validation_errors':errors, 'accounting_errors':cost_errors,
                        'content_review':reviews.get(qid)})
    ts = list(traces.values())
    # Missing/invalid trace records cannot disappear from a cost denominator.
    cost_known = verified and len(ts) == len(gold)
    elapsed_known = trace_valid and len(ts) == len(gold)
    return {'answer_accuracy':statistics.mean(float(d['exact']) for d in details),
            'Q':None if no_context else statistics.mean(statistics.mean(v) for v in groups.values()),
            'automatic_Q':None if no_context else statistics.mean(statistics.mean(v) for v in automatic_groups.values()),
            'content_overrides':len(reviews), 'category_scores':{k:statistics.mean(v) for k,v in groups.items()},
            'mean_input_tokens':statistics.mean(t['input_tokens'] for t in ts) if cost_known else None,
            'mean_output_tokens':statistics.mean(t['output_tokens'] for t in ts) if cost_known else None,
            'mean_llm_calls':statistics.mean(t['llm_calls'] for t in ts) if cost_known else None,
            'mean_cost':statistics.mean(t['input_tokens']+4*t['output_tokens']+256*t['llm_calls'] for t in ts) if cost_known else None,
            'median_latency_ms':statistics.median(t['elapsed_ms'] for t in ts) if elapsed_known else None,
            'usage_verified':bool(cost_known), 'efficiency_eligible':bool(cost_known),
            'validation_scope':'schema_only' if qs is None else 'schema_options_no_context' if no_context else 'schema_options_context' if all_contexts else 'schema_options_corpus' if source_ids is not None else 'schema_options',
            'details':details}


def main():
    p = argparse.ArgumentParser()
    for name in ['answers','gold','trace','output','questions']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--corpus',type=Path)
    p.add_argument('--diagnostics',type=Path)
    p.add_argument('--no-context',action='store_true')
    p.add_argument('--review',type=Path)
    args = p.parse_args()
    if not args.no_context and args.corpus is None:
        p.error('--corpus is required for RAG source validation')
    reviews = None
    if args.review:
        review = json.loads(args.review.read_text())
        if review.get('answers_sha256') != sha256(args.answers):
            raise ValueError('Content review answers hash mismatch: review this run again')
        reviews = review['questions']
        if review.get('review_status')=='reviewed':
            reviewed=review.get('reviewed_question_ids')
            expected={row['id'] for row in read_jsonl(args.gold)}
            if (not isinstance(reviewed,list) or any(not isinstance(q,str) or not q for q in reviewed)
                    or len(reviewed)!=len(set(reviewed)) or set(reviewed)!=expected):
                raise ValueError('A reviewed run must attest every gold question ID')
    evidence = args.diagnostics
    if evidence is None:
        evidence = next((x for x in [args.trace.with_suffix('.diagnostics.jsonl'),args.trace.with_suffix('.calls.jsonl')] if x.exists()),None)
    result = evaluate(read_jsonl(args.answers),read_jsonl(args.gold),read_jsonl(args.trace),args.no_context,reviews,
                      read_jsonl(args.questions),read_jsonl(evidence) if evidence is not None else None,
                      {b['block_id'] for b in read_jsonl(args.corpus)} if args.corpus else None)
    snapshot_path=args.trace.with_suffix('.config.json')
    snapshot=json.loads(snapshot_path.read_text()) if snapshot_path.exists() else {}
    post_path=next((x for x in [args.trace.with_suffix('.postprocessing.json'),args.trace.parent/'postprocessing.json'] if x.exists()),None)
    post=json.loads(post_path.read_text()) if post_path else {}
    if post.get('answers_sha256') and post['answers_sha256'] != sha256(args.answers):
        raise ValueError('Postprocessing metadata belongs to different answers')
    normalizer=post.get('normalizer_protocol_sha256') or snapshot.get('normalizer_protocol_sha256')
    normalizer_id='protocol:'+normalizer if normalizer else 'legacy_file:'+post['normalizer_sha256'] if post.get('normalizer_sha256') else None
    result['cohort']={'questions_sha256':sha256(args.questions),'gold_sha256':sha256(args.gold),
        'corpus_sha256':sha256(args.corpus) if args.corpus else None,
        'runtime_sha256':digest(snapshot['runtime']) if 'runtime' in snapshot else None,
        'normalizer_id':normalizer_id,
        'evaluation_protocol_sha256':digest({'evaluate':sha256(Path(__file__)), 'validation':sha256(ROOT/'src/validation.py')})}
    result['provenance']={'answers_sha256':sha256(args.answers),'trace_sha256':sha256(args.trace),
        'review_sha256':sha256(args.review) if args.review else None,
        'review_status':review.get('review_status') if args.review else None,
        'reviewed_question_ids_sha256':digest(sorted(review['reviewed_question_ids'])) if args.review and review.get('review_status')=='reviewed' else None,
        'generation_protocol_sha256':snapshot.get('generation_protocol',{}).get('sha256')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='details'},ensure_ascii=False))


if __name__ == '__main__':
    main()
