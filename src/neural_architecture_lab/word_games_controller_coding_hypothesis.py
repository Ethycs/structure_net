"""Conservative registry record for the known coding regression set."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ControllerCodingEvidence:
    tasks: int
    native_tasks: int
    translated_tasks: int
    native_tests: int
    translated_tests: int
    translated_clarifications: int

    def __post_init__(self):
        if type(self.tasks) is not int or self.tasks != 8:
            raise ValueError('All eight task pairs required')
        for tasks, tests in ((self.native_tasks,self.native_tests),(self.translated_tasks,self.translated_tests)):
            if type(tasks) is not int or type(tests) is not int or not 0<=tasks<=8:
                raise ValueError('Invalid pass count')
            if not 4*tasks <= tests <= 4*tasks+3*(8-tasks):
                raise ValueError('Task and test counts disagree')
        if type(self.translated_clarifications) is not int or not 0<=self.translated_clarifications<=24:
            raise ValueError('Invalid clarification count')

    def record(self):
        return {'id':'word-games-strong-controller-coding-v1',
            'name':'WORD_GAMES Terra controller and Qwen coding regression',
            'description':f'Terra interprets eight known synthetic coding tasks; Qwen translates and executes with bounded clarification. Native {self.native_tasks}/8 ({self.native_tests}/32 tests); translated {self.translated_tasks}/8 ({self.translated_tests}/32). Translated clarification requests: {self.translated_clarifications}. Matched arms reuse the old data and opaque aliases. No general accuracy or isolated clarification-benefit claim.',
            'category':'information_flow','project':'WORD_GAMES','tested':True,'confirmed':False,
            'confirmation_status':'known_regression_comparison_complete',
            'tested_scope':'eight previously inspected synthetic tasks, four fixed tests each, one Terra route and one Qwen route',
            'direct_experiment_count':8,'native_correct':self.native_tasks,'translated_correct':self.translated_tasks,
            'native_tests':self.native_tests,'translated_tests':self.translated_tests,
            'translated_clarifications':self.translated_clarifications,
            'observed_task_count_parity':self.native_tasks==self.translated_tasks,
            'general_accuracy_established':False,'production_deployed':False}
