"""Run configurations sequentially and collect verified metrics, without tuning gold."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.corpus import sha256
from src.provenance import assert_reusable, input_fingerprints
from scripts.collect_experiments import build_manifest, write_manifest, collect


def check_input_fingerprints(snapshot, questions, index, no_context=False):
    expected = input_fingerprints(questions,None if no_context else index)
    if any(snapshot.get('inputs', {}).get(k) != v for k, v in expected.items()):
        raise ValueError('Existing run input fingerprints differ or are missing; preserve it as a historical run')


def run(*arguments):
    subprocess.run([sys.executable, *map(str, arguments)], check=True, cwd=ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--configs', nargs='+', type=Path, required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--questions', type=Path, default=ROOT/'questions/dev_questions.jsonl')
    p.add_argument('--gold', type=Path, default=ROOT/'questions/dev_gold.jsonl')
    p.add_argument('--corpus', type=Path, default=ROOT/'artifacts/corpus/corpus.jsonl')
    p.add_argument('--runs', type=Path, default=ROOT/'artifacts/runs')
    p.add_argument('--csv', type=Path, help='Explicit reviewed-cohort CSV destination')
    p.add_argument('--manifest',type=Path,help='Cohort manifest; defaults to RUNS/cohort.json')
    p.add_argument('--reuse', action='store_true')
    p.add_argument('--wait-for-pid', type=int)
    args = p.parse_args()
    if args.wait_for_pid:
        status = Path('/proc')/str(args.wait_for_pid)/'stat'
        if status.exists():
            start_id = status.read_text().split()[21]
            print(f'Waiting for existing process {args.wait_for_pid}', flush=True)
            while status.exists():
                try:
                    if status.read_text().split()[21] != start_id:
                        break
                except FileNotFoundError:
                    break
                time.sleep(1)
    completed_specs=[]
    manifest_path=args.manifest.resolve() if args.manifest else args.runs.resolve()/'cohort.json'
    for config_path in args.configs:
        config_path = config_path.resolve()
        config = json.loads(config_path.read_text())
        target = args.runs.resolve()/config_path.stem
        target.mkdir(parents=True, exist_ok=True)
        answers, trace = target/'answers.jsonl', target/'trace.jsonl'
        if answers.exists():
            if not args.reuse:
                raise ValueError(f'Existing run: {target}; preserve or explicitly reuse it')
            snapshot = json.loads(trace.with_suffix('.config.json').read_text())
            old, new = dict(snapshot['config']), dict(config)
            old.pop('runtime', None)
            runtime_file = config_path.parent/new.pop('runtime')
            if old != new or snapshot['runtime'] != json.loads(runtime_file.read_text()):
                raise ValueError('Existing run configuration differs')
            check_input_fingerprints(snapshot, args.questions.resolve(), args.index.resolve(),
                                     config['generation']['mode'] == 'no_context')
            assert_reusable(snapshot,config,json.loads(runtime_file.read_text()),args.questions.resolve(),
                            None if config['generation']['mode']=='no_context' else args.index.resolve())
        else:
            run(ROOT/'predict.py', '--index', args.index.resolve(), '--questions', args.questions.resolve(),
                '--output', answers, '--trace', trace, '--config', config_path)
        extra = ['--no-context'] if config['generation']['mode'] == 'no_context' else []
        run(ROOT/'scripts/validate_run.py', '--questions', args.questions.resolve(), '--corpus', args.corpus.resolve(),
            '--answers', answers, '--trace', trace, *extra)
        review = target/'content_review.json'
        run(ROOT/'scripts/evaluate.py', '--answers', answers, '--gold', args.gold.resolve(),
            '--questions', args.questions.resolve(), '--corpus', args.corpus.resolve(),
            '--trace', trace, '--output', target/'metrics.json', *extra,
            *(['--review', review] if review.exists() else []))
        review_data=json.loads(review.read_text()) if review.exists() else {}
        completed_specs.append({'name':config_path.stem,'directory':str(target),
            'review_status':'not_applicable' if config['generation']['mode']=='no_context' else review_data.get('review_status','pending')})
        write_manifest(manifest_path,build_manifest(completed_specs,manifest_path))
    print('Cohort manifest saved:',manifest_path,flush=True)
    if args.csv:
        print(collect(manifest_path,args.csv.resolve()),flush=True)


if __name__ == '__main__':
    main()
