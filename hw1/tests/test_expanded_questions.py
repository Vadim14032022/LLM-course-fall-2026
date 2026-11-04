import copy
import unittest
from scripts.prepare_expanded_questions import compile_case


class ExpandedQuestionsTests(unittest.TestCase):
    def setUp(self):
        self.case={'topic':'topic','family_id':'family','category':'условия','question':'Выберите условия.',
            'options':['Первая верна','Вторая ложна','Третья верна','Четвёртая ложна'],
            'correct':[0,2],'citation_sets':[['block']], 'rationale':'Обоснование','missing_information':''}
        self.blocks={'block':{'text':'Содержание источника'}}

    def test_option_shuffle_preserves_correct_answers_and_keeps_metadata_out_of_input(self):
        q,g,m=compile_case(self.case,self.blocks)
        self.assertEqual((q,g,m),compile_case(self.case,self.blocks))
        selected={o['text'] for o in q['options'] if o['id'] in g['answer_ids']}
        self.assertEqual(selected,{'Первая верна','Третья верна'})
        self.assertEqual(set(q),{'id','question','options'})
        self.assertEqual(m['family_id'],'family')
        self.assertEqual(g['citation_sets'],[['block']])

    def test_unknown_evidence_duplicate_options_and_inconsistent_refusal_are_rejected(self):
        for update in [{'citation_sets':[['absent']]}, {'options':['a','a','b','c']},
                       {'category':'недостаточность данных'}, {'correct':[0,0]}]:
            bad=copy.deepcopy(self.case);bad.update(update)
            with self.assertRaises(ValueError):compile_case(bad,self.blocks)
