"""Strict canonical validation. No normalization, gold or network access."""


def answer_errors(answer, question=None, source_ids=None, no_context=False):
    if not isinstance(answer, dict):
        return ['answer: expected object']
    errors = []
    if not isinstance(answer.get('id'), str) or not answer['id']:
        errors.append('id: expected nonempty string')
    if question is not None and answer.get('id') != question['id']:
        errors.append('id: question mismatch')
    if type(answer.get('abstain')) is not bool:
        errors.append('abstain: expected boolean')
    valid_lists = {}
    for field in ('answer_ids', 'citations'):
        values = answer.get(field)
        valid = isinstance(values, list) and all(isinstance(v, str) and v.strip() for v in values)
        valid_lists[field] = valid
        if not valid:
            errors.append(field + ': expected list of nonempty strings')
            continue
        if len(values) != len(set(values)):
            errors.append(field + ': duplicate ID')
        allowed = ({o['id'] for o in question['options']} if question is not None else None) if field == 'answer_ids' else (set(source_ids) if source_ids is not None else None)
        if allowed is not None and not set(values) <= allowed:
            errors.append(field + ': unknown ID')
    rationale = answer.get('rationale')
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 1200:
        errors.append('rationale: expected nonempty string of at most 1200 characters')
    elif any(ord(c) < 32 and c not in '\n\t' for c in rationale):
        errors.append('rationale: control character')
    if valid_lists['citations']:
        if len(answer['citations']) > 4:
            errors.append('citations: too many sources')
        if no_context and answer['citations']:
            errors.append('citations: no_context requires empty list')
    if all(valid_lists.values()) and type(answer.get('abstain')) is bool:
        if answer['abstain'] and (answer['answer_ids'] or answer['citations']):
            errors.append('abstain: both lists must be empty')
        elif not answer['abstain'] and (not answer['answer_ids'] or not no_context and not answer['citations']):
            errors.append('answer: missing choices or sources')
    return errors


def trace_errors(trace):
    if not isinstance(trace, dict):
        return ['trace: expected object']
    errors = []
    for field, maximum in [('llm_calls', 4), ('input_tokens', 16384), ('output_tokens', 4096), ('elapsed_ms', None)]:
        value = trace.get(field)
        if type(value) is not int or value < 0 or maximum is not None and value > maximum:
            errors.append(field + ': invalid counter or exceeded limit')
    if type(trace.get('usage_verified')) is not bool:
        errors.append('usage_verified: expected boolean')
    if 'technical_error' in trace and type(trace['technical_error']) is not bool:
        errors.append('technical_error: expected boolean')
    if trace.get('llm_calls') == 0 and (trace.get('input_tokens') != 0 or trace.get('output_tokens') != 0 or trace.get('technical_error') is not True):
        errors.append('zero calls: expected explicit preflight failure and zero usage')
    return errors


def accounting_errors(trace, diagnostic):
    errors = trace_errors(trace)
    if errors:
        return errors
    if not isinstance(diagnostic, dict):
        return ['accounting: missing call evidence']
    n = trace['llm_calls']
    calls = diagnostic.get('calls')
    if n == 0:
        return [] if calls in (None, []) and diagnostic.get('error') else ['zero calls: missing preflight failure evidence']
    if calls is None and n == 1 and 'usage' in diagnostic:
        calls = [diagnostic]  # Explicit legacy single-call format.
    if not isinstance(calls, list) or len(calls) != n:
        return ['calls: missing or incomplete evidence']
    totals = [0, 0]
    for index, call in enumerate(calls):
        if not isinstance(call, dict) or not isinstance(call.get('usage'), dict):
            errors.append(f'calls[{index}]: missing usage')
            continue
        p, c, reserved = call['usage'].get('prompt_tokens'), call['usage'].get('completion_tokens'), call.get('max_tokens')
        if any(type(x) is not int for x in (p, c, reserved)) or p < 0 or c < 0 or not 1 <= reserved <= 4096:
            errors.append(f'calls[{index}]: invalid usage/reservation')
            continue
        if c > reserved or p + reserved > 8192:
            errors.append(f'calls[{index}]: reservation exceeded')
        totals[0] += p
        totals[1] += c
    if totals != [trace['input_tokens'], trace['output_tokens']]:
        errors.append('calls: totals do not match trace')
    if trace['usage_verified'] is not True:
        errors.append('usage_verified: cost remains unverified')
    return errors


def final_context(diagnostic, corpus_ids=None):
    """Return sources visible to the final answer and any trace-integrity errors."""
    errors=[]
    diagnostic=diagnostic if isinstance(diagnostic,dict) else {}
    initial=diagnostic.get('context_block_ids',[])
    def checked(value,label):
        if not isinstance(value,list) or any(not isinstance(v,str) or not v for v in value):
            errors.append(label+': invalid source IDs');return []
        if len(value)!=len(set(value)):
            errors.append(label+': duplicate source IDs')
        if corpus_ids is not None and not set(value)<=set(corpus_ids):
            errors.append(label+': unknown corpus source')
        return value
    initial=checked(initial,'context')
    initial_chunks={}
    packed_chunks=diagnostic.get('context')
    if packed_chunks is not None:
        if not isinstance(packed_chunks,list):
            errors.append('context: invalid chunk records')
        else:
            for item in packed_chunks:
                if not isinstance(item,dict) or not isinstance(item.get('chunk_id'),str) or not isinstance(item.get('block_id'),str):
                    errors.append('context: invalid chunk record');continue
                initial_chunks[item['chunk_id']]=item['block_id']
    calls=diagnostic.get('calls')
    if not isinstance(calls,list) or not calls:
        return initial,errors
    final=initial
    for i,call in enumerate(calls):
        if not isinstance(call,dict):
            errors.append(f'call {i}: invalid metadata');continue
        if 'context_block_ids' in call:
            ids=checked(call['context_block_ids'],f'call {i} context')
            if not set(ids)<=set(initial):errors.append(f'call {i}: source not in packed context')
            final=ids
        if 'context_chunk_ids' in call:
            chunks=call['context_chunk_ids']
            if not isinstance(chunks,list) or any(not isinstance(v,str) or not v for v in chunks):
                errors.append(f'call {i}: invalid chunk IDs')
            elif len(chunks)!=len(set(chunks)):
                errors.append(f'call {i}: duplicate chunk IDs')
            elif initial_chunks:
                if not set(chunks)<=set(initial_chunks):errors.append(f'call {i}: chunk not in packed context')
                mapped=list(dict.fromkeys(initial_chunks[c] for c in chunks if c in initial_chunks))
                if 'context_block_ids' in call and set(mapped)!=set(call['context_block_ids']):
                    errors.append(f'call {i}: chunk/source ID mismatch')
    return final,errors
