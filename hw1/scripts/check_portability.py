"""CLI integration audit from a foreign CWD with an endpoint-only network guard.

The endpoint below returns a synthetic protocol fixture, not a model answer or
an evaluation metric. No gold is loaded by either CLI.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.corpus import write_jsonl, read_jsonl
from src.llm import count_chat_tokens
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/portability_check.json')
    args = parser.parse_args()
    runtime = json.loads((ROOT/'runtime.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(runtime['model']['id'],
        revision=runtime['model']['revision'], local_files_only=True)
    requests_seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert self.path == '/replacement/v1/chat/completions'
            assert self.headers['X-Question-ID'].startswith('unseen-protocol-question-')
            assert self.headers['X-Submission-ID'] == 'portability-audit'
            assert body['model'] == 'replacement-model-alias'
            content = json.dumps({'answer_ids': ['X'], 'abstain': False,
                'rationale': 'Протокольная проверка на синтетическом источнике.',
                'citations': ['external:source-id']}, ensure_ascii=False)
            prefill_trial = self.headers['X-Question-ID'].endswith('-prefill')
            continuation = body.get('continue_final_message', False)
            if prefill_trial:
                emitted = content[1:] if continuation else 'Unfinished protocol draft'
                message = {'content':None, 'reasoning':emitted}
            else:
                emitted, message = content, {'content':content}
            usage = {'prompt_tokens': count_chat_tokens(tokenizer, body['messages'], runtime['model']['thinking'], continuation),
                     'completion_tokens': len(tokenizer.encode(emitted, add_special_tokens=False))}
            result = {'id': 'protocol-fixture', 'usage': usage,
                'choices': [{'message': message, 'finish_reason': 'stop'}]}
            data = json.dumps(result).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            requests_seen.append({'path': self.path, 'model': body['model'], 'usage': usage})
        def log_message(self, *args):
            pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    audits = []
    try:
        with tempfile.TemporaryDirectory(prefix='tfkp-cli-audit-') as temporary:
            directory = Path(temporary)
            write_jsonl(directory/'corpus.jsonl', [{'block_id': 'external:source-id', 'page': 9,
                'section': 'Синтетическая проверка', 'text': 'Тестовый источник для проверки протокола API: x=1.'}])
            questions = [{'id': f'unseen-protocol-question-{j}', 'question': 'Чему равно x?',
                'options': [{'id': k, 'text': str(i)} for i, k in enumerate(['W', 'X', 'Y', 'Z'])]} for j in range(4)]
            write_jsonl(directory/'questions.jsonl', questions)
            guard = directory/'guard'
            guard.mkdir()
            (guard/'sitecustomize.py').write_text('''import socket
_connect = socket.socket.connect
def guarded(self, address):
    if not isinstance(address, tuple) or address[:2] != ("127.0.0.1", PORT):
        raise RuntimeError("Audit blocked non-endpoint network connection: " + repr(address))
    return _connect(self, address)
socket.socket.connect = guarded
'''.replace('PORT', str(server.server_port)))
            env = {**os.environ, 'PYTHONPATH': str(guard), 'LLM_BASE_URL': f'http://127.0.0.1:{server.server_port}/replacement/v1',
                'LLM_MODEL': 'replacement-model-alias', 'LLM_API_KEY': 'fixture-key', 'SUBMISSION_ID': 'portability-audit'}
            def run(*command):
                completed = subprocess.run([sys.executable, *map(str, command)], env=env, cwd=directory,
                    capture_output=True, text=True, timeout=180)
                if completed.returncode:
                    raise RuntimeError(completed.stdout + completed.stderr)
            for label, source in [('bm25', ROOT/'configs/submission/bm25.json'),
                                  ('prefill', ROOT/'configs/submission/without_concise.json')]:
                trial_questions = [{**q, 'id':q['id']+'-'+label} for q in questions]
                write_jsonl(directory/'questions.jsonl', trial_questions)
                config = json.loads(source.read_text())
                config['runtime'] = str(ROOT/'runtime.json')
                config_path = directory/(label+'.json')
                config_path.write_text(json.dumps(config))
                index = directory/(label+'-index')
                answers, trace = directory/(label+'-answers.jsonl'), directory/(label+'-trace.jsonl')
                run(ROOT/'build_index.py', '--corpus', directory/'corpus.jsonl', '--output', index, '--config', config_path)
                run(ROOT/'predict.py', '--index', index, '--questions', directory/'questions.jsonl',
                    '--output', answers, '--trace', trace, '--config', config_path)
                run(ROOT/'scripts/validate_run.py', '--corpus', directory/'corpus.jsonl', '--questions', directory/'questions.jsonl',
                    '--answers', answers, '--trace', trace)
                output, usages = read_jsonl(answers), read_jsonl(trace)
                assert [a['id'] for a in output] == [q['id'] for q in trial_questions]
                assert all(a['citations'] == ['external:source-id'] and a['answer_ids'] == ['X'] for a in output)
                assert all(t['technical_error'] is False and t['llm_calls'] == (2 if label=='prefill' else 1) for t in usages)
                audits.append({'method': label, 'foreign_cwd': True, 'endpoint_replaced': True,
                    'non_endpoint_network_blocked': True, 'external_ids_preserved': True,
                    'trace_matches_endpoint_usage': True, 'questions': len(output),
                    'workers': config.get('execution', {}).get('workers', 1),
                    'index_files': sorted(p.name for p in index.iterdir())})
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {'type': 'synthetic_protocol_audit_not_quality_evaluation', 'checks': audits, 'requests': requests_seen}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
