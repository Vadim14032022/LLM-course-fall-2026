import copy
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.llm import LLMClient


class HTTPClientTests(unittest.TestCase):
    def test_endpoint_override_headers_and_unknown_usage_reservation(self):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                received.append((self.path, dict(self.headers), body))
                if len(received) == 3:
                    self.send_response(302)
                    self.send_header('Location', 'http://127.0.0.1:1/forbidden')
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                response = {'id': 'fake-server-id', 'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}]}
                if len(received) in (1,4):
                    response['usage'] = {'prompt_tokens': 17, 'completion_tokens': 5}
                if len(received) == 4:
                    response['choices'][0]['message'] = {'content':None,'reasoning':'"ok":true}'}
                data = json.dumps(response).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def log_message(self, *args):
                pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        runtime = json.loads((ROOT/'runtime.json').read_text())
        tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: {'input_ids': [1]*17})
        env = {'LLM_BASE_URL': f'http://127.0.0.1:{server.server_port}/alternate/v1',
               'LLM_MODEL': 'grader-served-alias', 'LLM_API_KEY': 'test-key', 'SUBMISSION_ID': 'test-submission'}
        fresh = {'llm_calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'usage_verified': True}
        try:
            with patch.dict(os.environ, env), patch('src.llm.AutoTokenizer.from_pretrained', return_value=tokenizer):
                client = LLMClient(runtime)
                budget = copy.deepcopy(fresh)
                _, diagnostic = client.complete('question-external-id', [], 64, budget)
                self.assertEqual(budget, {**fresh, 'llm_calls': 1, 'input_tokens': 17, 'output_tokens': 5})
                path, headers, body = received[0]
                self.assertEqual(path, '/alternate/v1/chat/completions')
                self.assertEqual(body['model'], 'grader-served-alias')
                self.assertEqual(headers['X-Question-ID'], 'question-external-id')
                self.assertEqual(headers['X-Submission-ID'], 'test-submission')
                self.assertEqual(body['temperature'], runtime['sampling']['temperature'])
                self.assertNotIn('profile', body)
                self.assertEqual(diagnostic['request_id'], 'fake-server-id')
                uncertain = copy.deepcopy(fresh)
                with self.assertRaises(ValueError):
                    client.complete('unknown-usage', [], 64, uncertain)
                self.assertFalse(uncertain['usage_verified'])
                self.assertEqual(uncertain['input_tokens'], 49)
                self.assertEqual(uncertain['output_tokens'], 64)
                with self.assertRaisesRegex(ValueError, 'redirects'):
                    client.complete('redirect-not-followed', [], 64, copy.deepcopy(fresh))
                exhausted = {**fresh, 'llm_calls': 4}
                with self.assertRaises(ValueError):
                    client.complete('over-budget', [], 64, exhausted)
                self.assertEqual(len(received), 3)
                messages = [{'role':'user','content':'Question'},
                            {'role':'assistant','reasoning_content':'Draft','content':'{'}]
                text, meta = client.complete('prefilled',messages,64,copy.deepcopy(fresh),continue_final=True)
                self.assertEqual(json.loads(text), {'ok':True})
                self.assertEqual(meta['completion_prefix'], '{')
                body = received[3][2]
                self.assertIs(body['add_generation_prompt'],False)
                self.assertIs(body['continue_final_message'],True)
                self.assertEqual(body['chat_template_kwargs']['enable_thinking'],runtime['model']['thinking'])
                for key,value in runtime['sampling'].items():
                    if key != 'profile': self.assertEqual(body[key],value)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
