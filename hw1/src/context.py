"""Shared greedy context packing for inference and retrieval evaluation."""


class BasePromptTooLarge(ValueError):
    pass


def pack_context(question, candidates, build_messages, count_tokens, input_limit,
                 no_context=False, concise_analysis=False):
    """Keep the highest-ranked candidates that individually fit the real prompt.

    Oversized candidates are skipped rather than causing all lower-ranked
    candidates to be discarded. The full chat template is counted at each step.
    """
    base_messages=build_messages(question,[],no_context,concise_analysis)
    base_count=count_tokens(base_messages)
    if base_count>input_limit:
        raise BasePromptTooLarge(f'Base prompt needs {base_count} tokens; limit is {input_limit}')
    kept=[]; skipped=[]
    for candidate in candidates:
        attempt=kept+[candidate]
        messages=build_messages(question,attempt,no_context,concise_analysis)
        count=count_tokens(messages)
        if count<=input_limit:
            kept=attempt
        else:
            skipped.append({'chunk_id':candidate.get('chunk_id'),
                            'block_id':candidate.get('block_id'),
                            'prompt_tokens_if_added':count,
                            'reason':'input_limit'})
    messages=build_messages(question,kept,no_context,concise_analysis)
    return kept,messages,skipped


def block_ids(chunks):
    return list(dict.fromkeys(c['block_id'] for c in chunks))
