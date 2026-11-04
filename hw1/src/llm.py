"""Chat Completions client with local template counting and per-question budgets."""
import os
import threading
from contextlib import nullcontext
import requests
from transformers import AutoTokenizer


def count_chat_tokens(tokenizer, messages, thinking, continue_final=False):
    encoded = tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=not continue_final, enable_thinking=thinking,
        **({'continue_final_message': True} if continue_final else {}))
    ids = encoded['input_ids'] if hasattr(encoded, 'keys') else encoded
    return len(ids)


class LLMClient:
    def __init__(self, runtime):
        self.runtime = runtime
        self._tokenizer_lock = threading.Lock()
        self.url = os.environ['LLM_BASE_URL'].rstrip('/') + '/chat/completions'
        self.model = os.environ['LLM_MODEL']
        self.key = os.environ['LLM_API_KEY']
        self.submission = os.environ['SUBMISSION_ID']
        model = runtime['model']
        self.tokenizer = AutoTokenizer.from_pretrained(model['id'], revision=model['revision'], local_files_only=True)

    def count(self, messages, continue_final=False):
        # Transformers 5 returns BatchEncoding by default; len(encoded) counts
        # fields rather than tokens. Older versions returned a plain ID list.
        with getattr(self, '_tokenizer_lock', nullcontext()):
            return count_chat_tokens(self.tokenizer, messages, self.runtime['model']['thinking'], continue_final)

    def complete(self, qid, messages, max_tokens, budget, continue_final=False):
        limits = self.runtime['limits']
        prompt_tokens = self.count(messages, continue_final)
        if budget['llm_calls'] >= limits['max_llm_calls_per_question']:
            raise ValueError('LLM call budget exhausted')
        max_tokens = min(max_tokens, limits['max_completion_tokens_per_request'],
                         limits['max_output_tokens_per_question'] - budget['output_tokens'])
        if max_tokens <= 0 or prompt_tokens + max_tokens + 32 > limits['max_input_plus_output_tokens_per_request']:
            raise ValueError('Request token budget exceeded before submission')
        if prompt_tokens + 32 + budget['input_tokens'] > limits['max_input_tokens_per_question']:
            raise ValueError('Question input budget exhausted')
        body = {'model': self.model, 'messages': messages, 'max_tokens': max_tokens,
                **{k: v for k, v in self.runtime['sampling'].items() if k != 'profile'},
                'chat_template_kwargs': {'enable_thinking': self.runtime['model']['thinking']}}
        api = self.runtime['api']
        prefix = ''
        if continue_final:
            if not messages or messages[-1]['role'] != 'assistant' or messages[-1]['content'] != '{':
                raise ValueError('Final continuation requires the explicit JSON opening prefix')
            body.update({'add_generation_prompt': False, 'continue_final_message': True})
            prefix = '{'
        headers = {'Authorization': 'Bearer ' + self.key,
                   api['question_id_header']: qid, api['submission_id_header']: self.submission}
        budget['llm_calls'] += 1
        # Reservation is retained if the endpoint cannot report actual usage.
        budget['input_tokens'] += prompt_tokens + 32
        budget['output_tokens'] += max_tokens
        budget['usage_verified'] = False
        response = requests.post(self.url, json=body, headers=headers,
            timeout=(self.runtime['timeouts_seconds']['connect'], self.runtime['timeouts_seconds']['request']),
            allow_redirects=False)
        response.raise_for_status()
        if 300 <= response.status_code < 400:
            raise ValueError('Endpoint redirects are not allowed during inference')
        data = response.json()
        usage = data.get('usage')
        if not usage or any(type(usage.get(k)) is not int or usage[k] < 0 for k in ('prompt_tokens', 'completion_tokens')):
            raise ValueError('Endpoint returned no trustworthy token usage')
        budget['input_tokens'] += usage['prompt_tokens'] - prompt_tokens - 32
        budget['output_tokens'] += usage['completion_tokens'] - max_tokens
        budget['usage_verified'] = True
        if (usage['prompt_tokens'] + max_tokens > limits['max_input_plus_output_tokens_per_request']
                or budget['input_tokens'] > limits['max_input_tokens_per_question']
                or budget['output_tokens'] > limits['max_output_tokens_per_question']):
            raise ValueError('Endpoint usage exceeds reserved limits')
        choice = data['choices'][0]
        reasoning = choice['message'].get('reasoning') or choice['message'].get('reasoning_content') or ''
        content = choice['message'].get('content') or ''
        # Some reasoning parsers inspect only newly generated tokens and miss
        # the closing think tag already in the prefill prompt. Their reasoning
        # field then contains the JSON continuation, not a fresh reasoning turn.
        continuation_channel = 'reasoning' if continue_final and not content else 'content'
        if continue_final and not content:
            content = reasoning
        return prefix + content, {
            'request_id': data.get('id'), 'finish_reason': choice['finish_reason'],
            'completion_prefix': prefix,
            'continuation_channel': continuation_channel,
            'reasoning_text': reasoning,
            'usage': usage, 'counted_prompt_tokens': prompt_tokens, 'max_tokens': max_tokens}

    def tail_tokens(self, text, cap):
        with self._tokenizer_lock:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            return self.tokenizer.decode(ids[-cap:])
