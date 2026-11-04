"""Question-local draft/final protocol within the unchanged runtime limits."""
from .pipeline import parse_answer
from .validation import answer_errors
from .context import pack_context, block_ids, BasePromptTooLarge
import json


def canonical_gate(text, question, source_ids, no_context=False):
    """Fixed generation routing, independent of postprocessing repairs."""
    try:
        data = json.loads(text)
        if not isinstance(data,dict):
            return False
        data = {**data, 'id':question['id']}
        return not answer_errors(data,question,source_ids,no_context)
    except (ValueError,TypeError):
        return False


def generate_answer(client, question, messages, source_ids, config, runtime, budget,
                    diagnostic, no_context=False, context_chunks=None,
                    message_builder=None, context_cap=None):
    strategy = config.get('response_strategy', 'single')
    if strategy not in ('single', 'draft_then_finalize', 'draft_then_prefill'):
        raise ValueError('Unknown response strategy: ' + strategy)
    limit = min(config['max_tokens'], runtime['limits']['max_output_tokens_per_question'])
    initial = min(config.get('draft_tokens', 2048), limit) if strategy != 'single' else limit
    diagnostic['calls'] = []
    original_messages = messages
    current_source_ids = list(source_ids)
    current_chunks = context_chunks
    for attempt in range(2 if strategy != 'single' else 1):
        remaining = limit - budget['output_tokens']
        if attempt and context_chunks is not None and message_builder is not None:
            request_cap=runtime['limits']['max_input_plus_output_tokens_per_request']-remaining-32
            question_cap=runtime['limits']['max_input_tokens_per_question']-budget['input_tokens']-32
            cap=min(context_cap if context_cap is not None else request_cap,request_cap,question_cap)
            def with_continuation(q,chunks,no_ctx,concise):
                base=message_builder(q,chunks,no_ctx,concise)
                if strategy=='draft_then_prefill':
                    return base+[{'role':'assistant','reasoning_content':draft,'content':'{'}]
                return base+[{'role':'assistant','content':'Незавершённый черновик (может содержать ошибки):\n'+draft},
                    {'role':'user','content':'Заверши ответ одним JSON. Используй только исходные источники как доказательства; черновик не является источником. Не повторяй разбор: кратко проверь вывод и ID ссылок и выдай результат.'}]
            full_draft=draft
            count_prompt=lambda items:client.count(items,continue_final=strategy=='draft_then_prefill')
            try:
                packed,messages,skipped=pack_context(question,context_chunks,with_continuation,
                    count_prompt,cap,no_context,config.get('concise_analysis',False))
            except BasePromptTooLarge:
                # Preserve the continuation protocol while shrinking only its
                # generated working material, never the source or JSON prefix.
                fitted=False
                for draft_size in range(700,-1,-100):
                    draft=client.tail_tokens(full_draft,draft_size)
                    try:
                        packed,messages,skipped=pack_context(question,context_chunks,with_continuation,
                            count_prompt,cap,no_context,config.get('concise_analysis',False))
                        fitted=True
                        diagnostic['second_call_draft_tokens_requested']=draft_size
                        break
                    except BasePromptTooLarge:
                        continue
                if not fitted:
                    raise
            current_source_ids=block_ids(packed)
            current_chunks=packed
            diagnostic.setdefault('context_skipped_second_call',[]).extend(skipped)
        text, api = client.complete(question['id'], messages, initial if attempt == 0 else remaining, budget,
                                   **({'continue_final': True} if attempt and strategy == 'draft_then_prefill' else {}))
        diagnostic['calls'].append({**api, 'response_text': text,
                                    'context_block_ids':current_source_ids,
                                    'context_chunk_ids':[c.get('chunk_id') for c in current_chunks] if current_chunks is not None else []})
        diagnostic.update(api)
        diagnostic['response_text'] = text
        # Only strict canonical validity can short-circuit the second call.
        # Updating a repair rule changes postprocessing, not the request plan.
        if strategy == 'single' or attempt or canonical_gate(text,question,source_ids,no_context) or budget['output_tokens'] >= limit:
            return parse_answer(text, question, current_source_ids, no_context)
        # This is generated working material, never a new evidentiary source.
        draft = client.tail_tokens(api.get('reasoning_text') or text, 800)
        if strategy == 'draft_then_prefill':
            # The first call has already generated reasoning. Continue its
            # assistant turn at the JSON prefix instead of starting new thought.
            # This changes the prompt, not runtime thinking/sampling settings.
            messages = original_messages + [{'role': 'assistant', 'reasoning_content': draft, 'content': '{'}]
            continue
        messages = original_messages + [
            {'role': 'assistant', 'content': 'Незавершённый черновик (может содержать ошибки):\n' + draft},
            {'role': 'user', 'content': 'Заверши ответ одним JSON. Используй только исходные источники как доказательства; черновик не является источником. Не повторяй разбор: кратко проверь вывод и ID ссылок и выдай результат.'}]
