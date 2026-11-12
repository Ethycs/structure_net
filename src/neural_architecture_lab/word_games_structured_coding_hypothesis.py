"""Scoped record for bounded bridge verification and the synthetic coding pilot."""
from dataclasses import dataclass


@dataclass(frozen=True)
class StructuredCodingEvidence:
    proof_cases: int
    tasks: int
    native_correct: int
    translated_correct: int
    renamed_keys_recovered: int
    renamed_keys_total: int
    canaries_recovered: int

    def __post_init__(self):
        if self.proof_cases!=85176 or self.tasks!=8:
            raise ValueError('Complete declared verification and pilot required')
        for value in (self.native_correct,self.translated_correct,self.canaries_recovered):
            if type(value) is not int or not 0<=value<=self.tasks:
                raise ValueError('Invalid task/recovery count')
        if self.renamed_keys_total!=64 or not 0<=self.renamed_keys_recovered<=64:
            raise ValueError('Eight genuinely renamed keys in eight receiver packets required')

    def record(self):
        return {'id':'word-games-structured-proof-coding-recovery-v1',
            'name':'WORD_GAMES bounded commutation and synthetic coding recovery',
            'description':f'Actual Engine/Bridge commute over {self.proof_cases} declared state/context/transaction cases with replay; unrestricted JSON bijection has counterexamples. Native coding {self.native_correct}/8 versus translated {self.translated_correct}/8 on synthetic tasks, including actor resource failures. Schema-aware recovery {self.renamed_keys_recovered}/64 renamed-key guesses. Not a natural-language equivalence or secrecy proof.',
            'category':'information_flow','project':'WORD_GAMES','tested':True,'confirmed':False,
            'confirmation_status':'bounded_structure_passed_empirical_limits_measured',
            'tested_scope':'one frozen mapping; 21 states, 8 contexts, 22 events, atomic length <=2; eight related synthetic coding tasks; one Qwen model',
            'direct_experiment_count':8,'proof_cases':self.proof_cases,
            'native_correct':self.native_correct,'translated_correct':self.translated_correct,
            'renamed_keys_recovered':self.renamed_keys_recovered,'renamed_keys_total':self.renamed_keys_total,
            'canaries_recovered':self.canaries_recovered,'unrestricted_json_bijection':False,
            'general_secrecy_proved':False,'swe_bench_evaluated':False}
