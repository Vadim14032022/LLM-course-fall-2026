"""Portable inference entry point. QA gold is never read here."""
import argparse
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from functools import partial
from src.corpus import read_jsonl, sha256
from src.llm import LLMClient
from src.pipeline import messages_for
from src.retrieval import Retriever
from src.generation import generate_answer
from src.provenance import generation_protocol, normalizer_protocol, input_fingerprints
from src.context import pack_context, block_ids


def main():
    parser = argparse.ArgumentParser()
    for arg in ('index', 'questions', 'output', 'trace', 'config'):
        parser.add_argument('--' + arg, type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    runtime_path = Path(config['runtime'])
    if not runtime_path.is_absolute():
        runtime_path = args.config.resolve().parent / runtime_path
    runtime = json.loads(runtime_path.read_text())
    client = LLMClient(runtime)
    questions = read_jsonl(args.questions)
    ids = [q['id'] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate input question IDs')
    no_context = config['generation']['mode'] == 'no_context'
    retriever = None if no_context else Retriever(args.index, config)
    message_builder = partial(messages_for, system_prompt=config['generation'].get('system_prompt'))
    for path in (args.output, args.trace):
        path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    code_hashes = {name: sha256(root/name) for name in ['predict.py', 'src/llm.py', 'src/pipeline.py', 'src/generation.py', 'src/retrieval.py', 'src/dense.py', 'src/context.py', 'src/math_search.py', 'src/validation.py']}
    inputs = input_fingerprints(args.questions,None if no_context else args.index)
    args.trace.with_suffix('.config.json').write_text(json.dumps({'config': config, 'runtime': runtime, 'code_sha256': code_hashes, 'inputs': inputs,
        'generation_protocol':generation_protocol(config,runtime),'normalizer_protocol_sha256':normalizer_protocol()},
        ensure_ascii=False, indent=2), encoding='utf-8')
    search_lock = threading.Lock()
    def answer_one(question):
        started = time.perf_counter()
        budget = {'llm_calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'usage_verified': True}
        with search_lock:
            chunks = [] if no_context else retriever.search(question, config['retrieval']['top_k'])
        requested_ids = list(dict.fromkeys(c['block_id'] for c in chunks))
        diagnostic = {'id': question['id'], 'retrieved_block_ids': requested_ids}
        try:
            max_tokens = min(config['generation']['max_tokens'], runtime['limits']['max_output_tokens_per_question'])
            concise = config['generation'].get('concise_analysis', False)
            messages = message_builder(question, chunks, no_context, concise)
            request_cap=runtime['limits']['max_input_plus_output_tokens_per_request']-max_tokens-32
            context_cap=config['generation'].get('context_input_tokens',8192)
            input_cap=min(request_cap,context_cap)
            chunks,messages,skipped=pack_context(question,chunks,message_builder,client.count,input_cap,no_context,concise)
            source_ids = block_ids(chunks)
            diagnostic['context_block_ids'] = source_ids
            diagnostic['context_skipped'] = skipped
            diagnostic['context'] = [{k: c[k] for k in ('chunk_id', 'block_id', 'page', 'text')} for c in chunks]
            answer = generate_answer(client, question, messages, source_ids, config['generation'],
                                     runtime, budget, diagnostic, no_context,
                                     context_chunks=chunks,message_builder=message_builder,context_cap=context_cap)
        except Exception as error:
            diagnostic['error'] = type(error).__name__ + ': ' + str(error)
            answer = {'id': question['id'], 'answer_ids': [], 'citations': [], 'abstain': True,
                      'rationale': 'Технический сбой обработки ответа; см. диагностический журнал.'}
        trace = {'id': question['id'], **budget, 'elapsed_ms': round((time.perf_counter()-started)*1000),
                 'retrieved_block_ids': requested_ids, 'technical_error': 'error' in diagnostic}
        print(question['id'], 'error' if 'error' in diagnostic else 'ok', trace['output_tokens'], flush=True)
        return answer, trace, diagnostic
    workers = config.get('execution', {}).get('workers', 1)
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError('execution.workers must be between 1 and 32')
    with args.output.open('w', encoding='utf-8') as answers, args.trace.open('w', encoding='utf-8') as traces, \
            args.trace.with_suffix('.diagnostics.jsonl').open('w', encoding='utf-8') as diagnostics, \
            ThreadPoolExecutor(max_workers=workers) as executor:
        for answer, trace, diagnostic in executor.map(answer_one, questions):
            for stream, row in ((answers, answer), (traces, trace), (diagnostics, diagnostic)):
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()


if __name__ == '__main__':
    main()
