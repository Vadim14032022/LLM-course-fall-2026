"""Content fingerprints for request protocol, data and postprocessing."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from .corpus import sha256


def digest(value):
    data=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(data).hexdigest()


def generation_protocol(config, runtime):
    from . import pipeline, generation, retrieval, math_search, llm, validation
    normalized={k:v for k,v in config.items() if k!='runtime'}
    components={name:sha256(module.__file__) for name,module in
                [('generation',generation),('retrieval',retrieval),('math_search',math_search),('llm',llm)]}
    components['messages_for']=digest(inspect.getsource(pipeline.messages_for))
    components['canonical_answer_validation']=digest(inspect.getsource(validation.answer_errors))
    prompt=pipeline.effective_system_prompt(config['generation']['mode']=='no_context',config['generation'].get('system_prompt'))
    components['effective_system_prompt']=digest(inspect.getsource(pipeline.effective_system_prompt))
    components['system_prompt']=digest(prompt)
    if importlib.util.find_spec('src.context') is not None:
        from . import context
        components['context_packer']=sha256(context.__file__)
    payload={'version':2,'effective_system_prompt':prompt,'components':components,'config_sha256':digest(normalized),'runtime_sha256':digest(runtime)}
    return {**payload,'sha256':digest(payload)}


def normalizer_protocol():
    from . import pipeline, validation
    return digest({name:inspect.getsource(obj) for name,obj in [
        ('parse_answer',pipeline.parse_answer),('load_response_json',pipeline.load_response_json),
        ('escape_math_commands',pipeline.escape_math_commands),('answer_errors',validation.answer_errors)]})


def input_fingerprints(questions, index=None):
    result={'questions_sha256':sha256(questions)}
    if index is not None:
        index=Path(index)
        manifest=json.loads((index/'manifest.json').read_text())
        identity={key:manifest.get(key) for key in ('format_version','corpus_sha256','chunking','embedding')}
        result.update({'index_identity_sha256':digest(identity),'corpus_sha256':manifest['corpus_sha256'],
                       'chunks_sha256':sha256(index/'chunks.jsonl')})
        if (index/'embeddings.npy').exists():
            result['embeddings_sha256']=sha256(index/'embeddings.npy')
    return result


def assert_reusable(snapshot,config,runtime,questions,index=None):
    if snapshot.get('generation_protocol',{}).get('sha256') != generation_protocol(config,runtime)['sha256']:
        raise ValueError('Generation protocol differs or is missing; keep this run historical')
    if snapshot.get('inputs') != input_fingerprints(questions,index):
        raise ValueError('Existing run input fingerprints differ or are missing')
