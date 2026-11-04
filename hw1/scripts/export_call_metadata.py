"""Export verifiable call accounting without source text or model reasoning."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.corpus import read_jsonl, write_jsonl


def public_record(diagnostic):
    fields = ('request_id', 'finish_reason', 'usage', 'counted_prompt_tokens',
              'max_tokens', 'completion_prefix', 'continuation_channel',
              'context_block_ids', 'context_chunk_ids')
    calls = diagnostic.get('calls')
    if calls is None:
        calls = [diagnostic] if 'usage' in diagnostic else []
    record = {'id': diagnostic['id'], 'calls': [
        {k: call[k] for k in fields if k in call} for call in calls]}
    if 'context_block_ids' in diagnostic:
        record['context_block_ids'] = diagnostic['context_block_ids']
    if 'context' in diagnostic:
        record['context'] = [{k: c[k] for k in ('chunk_id', 'block_id')}
            for c in diagnostic['context'] if 'chunk_id' in c and 'block_id' in c]
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write_jsonl(args.output, [public_record(d) for d in read_jsonl(args.diagnostics)])


if __name__ == '__main__':
    main()
