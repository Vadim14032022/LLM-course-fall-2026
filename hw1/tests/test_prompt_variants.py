import copy
import json
import unittest
from functools import partial
from pathlib import Path
from src.pipeline import SYSTEM, SYSTEM_NO_CONTEXT, messages_for, effective_system_prompt
from src.provenance import generation_protocol, normalizer_protocol


class PromptVariantTests(unittest.TestCase):
    def test_default_prompt_is_preserved_and_custom_is_used_in_all_messages(self):
        question={'id':'unseen','question':'Вопрос','options':[{'id':'Z','text':'Вариант'}]}
        self.assertEqual(messages_for(question,[])[0]['content'],SYSTEM)
        self.assertEqual(messages_for(question,[],True)[0]['content'],SYSTEM_NO_CONTEXT)
        builder=partial(messages_for,system_prompt='Custom instruction')
        self.assertEqual(builder(question,[],False,False)[0]['content'],'Custom instruction')
        self.assertEqual(builder(question,[{'block_id':'source','page':9,'text':'Source'}],False,False)[0]['content'],'Custom instruction')
        for invalid in ['', '   ', 42, []]:
            with self.assertRaises(ValueError):effective_system_prompt(False,invalid)

    def test_actual_prompt_is_recorded_and_invalidates_protocol(self):
        cfg={'generation':{'mode':'rag','system_prompt':'P0'},'retrieval':{'method':'bm25'}}
        before=normalizer_protocol(); a=generation_protocol(cfg,{})
        cfg=copy.deepcopy(cfg);cfg['generation']['system_prompt']='P1'
        b=generation_protocol(cfg,{})
        self.assertEqual(a['effective_system_prompt'],'P0')
        self.assertEqual(b['effective_system_prompt'],'P1')
        self.assertNotEqual(a['sha256'],b['sha256'])
        self.assertEqual(before,normalizer_protocol())
