"""Validate an entire run against questions, corpus, budgets and API diagnostics."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.corpus import read_jsonl
from src.validation import answer_errors, accounting_errors, final_context


def main():
    p = argparse.ArgumentParser()
    for name in ('questions', 'corpus', 'answers', 'trace'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--no-context', action='store_true')
    args = p.parse_args()
    def keyed(path):
        rows = read_jsonl(path)
        d = {x['id']: x for x in rows}
        if len(rows) != len(d):
            raise ValueError(f'Duplicate IDs in {path}')
        return d
    questions, answers, traces = [keyed(x) for x in (args.questions, args.answers, args.trace)]
    diagnostics = keyed(args.trace.with_suffix('.diagnostics.jsonl'))
    expected = set(questions)
    if any(set(d) != expected for d in (answers, traces, diagnostics)):
        raise ValueError('Missing or extra IDs')
    sources = {b['block_id'] for b in read_jsonl(args.corpus)}
    technical = 0
    for qid, question in questions.items():
        answer, trace, diagnostic = answers[qid], traces[qid], diagnostics[qid]
        allowed,context_errors=final_context(diagnostic,sources)
        if context_errors:
            raise ValueError(qid+': '+'; '.join(context_errors))
        errors = answer_errors(answer, question, allowed, args.no_context) + accounting_errors(trace, diagnostic)
        if errors:
            raise ValueError(f'{qid}: ' + '; '.join(errors))
        technical += bool(trace.get('technical_error'))
    print(json.dumps({'records': len(questions), 'format_and_budget_valid': True, 'technical_errors': technical}))


if __name__ == '__main__':
    main()
