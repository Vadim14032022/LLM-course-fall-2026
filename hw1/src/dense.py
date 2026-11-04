"""Pinned multilingual E5 ONNX encoder, CPU only, no model downloads here."""
from pathlib import Path
import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


class DenseEncoder:
    def __init__(self, config):
        self.config = config
        self.path = Path(snapshot_download(config['model'], revision=config['revision'], local_files_only=True,
            allow_patterns=[config['onnx_file'], 'tokenizer.json', 'tokenizer_config.json',
                            'special_tokens_map.json', 'config.json']))
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = config['threads']
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.path/config['onnx_file']),
            sess_options=options, providers=['CPUExecutionProvider'])
        self.input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, texts, kind):
        if kind not in ('query', 'passage'):
            raise ValueError('Expected query or passage prefix')
        vectors = []
        for start in range(0, len(texts), self.config['batch_size']):
            batch = [kind+': '+t for t in texts[start:start+self.config['batch_size']]]
            tokens = self.tokenizer(batch, padding=True, truncation=True,
                max_length=self.config['max_length'], return_tensors='np')
            inputs = {k: np.asarray(v, dtype=np.int64) for k, v in tokens.items() if k in self.input_names}
            if 'token_type_ids' in self.input_names and 'token_type_ids' not in inputs:
                inputs['token_type_ids'] = np.zeros_like(inputs['input_ids'])
            hidden = self.session.run(None, inputs)[0]
            if hidden.ndim != 3:
                raise ValueError('Expected token-level ONNX output')
            mask = tokens['attention_mask'][..., None].astype(np.float32)
            pooled = (hidden*mask).sum(axis=1)/mask.sum(axis=1).clip(min=1)
            pooled /= np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-12)
            vectors.append(pooled.astype(np.float32))
        return np.concatenate(vectors, axis=0)
