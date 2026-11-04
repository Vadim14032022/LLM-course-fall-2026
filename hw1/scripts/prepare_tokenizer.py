"""Download the pinned tokenizer before prediction, without generation weights."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', type=Path, default=Path(__file__).resolve().parents[1]/'runtime.json')
    args = p.parse_args()
    model = json.loads(args.runtime.read_text())['model']
    tokenizer = AutoTokenizer.from_pretrained(model['id'], revision=model['revision'])
    print(f"Prepared tokenizer {model['id']} at {model['revision']}; vocabulary={len(tokenizer)}")


if __name__ == '__main__':
    main()
