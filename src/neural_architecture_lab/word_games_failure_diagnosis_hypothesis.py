"""Known-failure diagnostic evidence, explicitly distinct from model accuracy."""
from dataclasses import dataclass


@dataclass(frozen=True)
class FailureDiagnosisEvidence:
    saved_repairs_passing: int
    original_roots: int
    corrected_roots: int
    original_formatting: int
    corrected_formatting: int

    def __post_init__(self):
        for value, maximum in ((self.saved_repairs_passing,5),(self.original_roots,2),
                               (self.corrected_roots,2),(self.original_formatting,2),(self.corrected_formatting,2)):
            if type(value) is not int or not 0 <= value <= maximum: raise ValueError('Invalid diagnostic count')

    def record(self):
        return {'id':'word-games-humaneval-failure-diagnosis-v1',
            'name':'WORD_GAMES selected HumanEval failure mechanisms',
            'description':f'Selected known failures on HumanEval 32, 113, 160. Researcher-authored local corrections pass official and targeted suites in {self.saved_repairs_passing}/5 saved programs. Fresh root actor replays: original {self.original_roots}/2, corrected interface {self.corrected_roots}/2. Formatting replays: original {self.original_formatting}/2, completed example {self.corrected_formatting}/2. No execution tool was available to actors. This diagnoses handoff and implementation defects, not general accuracy or autonomous repair.',
            'category':'information_flow','project':'WORD_GAMES','tested':True,'confirmed':False,
            'confirmation_status':'selected_failure_mechanisms_tested','selected_task_families':3,
            'saved_repairs_passing':self.saved_repairs_passing,
            'original_root_replays_passed':self.original_roots,'corrected_root_replays_passed':self.corrected_roots,
            'original_format_replays_passed':self.original_formatting,'corrected_format_replays_passed':self.corrected_formatting,
            'general_accuracy_established':False,'autonomous_repair_demonstrated':False,
            'primary_accuracy_unchanged':True,'production_deployed':False}
