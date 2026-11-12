"""Scoped evidence for coding with computationally equivalent labels."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RelabelCodingEvidence:
    native_tasks: int
    relabeled_tasks: int
    native_tests: int
    relabeled_tests: int
    paired_disagreements: int

    def __post_init__(self):
        for tasks,tests in ((self.native_tasks,self.native_tests),(self.relabeled_tasks,self.relabeled_tests)):
            if type(tasks) is not int or type(tests) is not int or not 0<=tasks<=8:
                raise ValueError('Invalid counts')
            if not 36*tasks<=tests<=36*tasks+35*(8-tasks):raise ValueError('Inconsistent task/test counts')
        if type(self.paired_disagreements) is not int or not 0<=self.paired_disagreements<=288:
            raise ValueError('Invalid paired disagreement count')

    def record(self):
        return {'id':'word-games-isomorphic-label-coding-v1',
            'name':'WORD_GAMES Qwen coding with typed equivalent labels',
            'description':f'Terra controller and Qwen translator/executor on eight known coding tasks with 36 cases each. Native {self.native_tasks}/8 tasks and {self.native_tests}/288 cases; relabeled {self.relabeled_tasks}/8 and {self.relabeled_tests}/288. Strict normal-stop completion required. Free text and identifier suffixes deliberately preserved. Paired case disagreements {self.paired_disagreements}. Not a general accuracy or privacy claim.',
            'category':'information_flow','project':'WORD_GAMES','tested':True,'confirmed':False,
            'confirmation_status':'paired_label_coding_evaluated',
            'tested_scope':'eight known synthetic task families, one label table, 288 cases; compact contracts, typed maps and disjoint protocol',
            'direct_experiment_count':8,'native_correct':self.native_tasks,'relabeled_correct':self.relabeled_tasks,
            'native_tests':self.native_tests,'relabeled_tests':self.relabeled_tests,
            'paired_case_disagreements':self.paired_disagreements,'source_target_reference_checks':288,
            'general_accuracy_established':False,'information_secrecy_tested':False,'production_deployed':False}
