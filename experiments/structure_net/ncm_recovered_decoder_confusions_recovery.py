"""Preserve the serial experiment while fixing its append-only progress writes."""
from pathlib import Path
from types import SimpleNamespace

from experiments.structure_net import ncm_recovered_decoder_confusions_serial as serial

data = serial.data
INCIDENT = serial.ROOT
ROOT = data.ROOT/'20261028_recovered_decoder_confusion_codes_v4'


def run():
    if ROOT.exists(): raise ValueError('fresh recovered encoding diagnostic required')
    receipt = data.read(INCIDENT/'actual_tool_receipt.json'); data.verify(receipt['artifacts'])
    if receipt['actual_tool_exit_code'] != 1: raise ValueError('preserved failed serial attempt required')
    counter = 0
    def write(path, value):
        nonlocal counter
        path = Path(path)
        if path == ROOT/'current_case.json':
            data.write(ROOT/f'case_{counter:03d}.json', value); counter += 1
            if counter == 128: data.write(path, value)
        elif path == ROOT/'cases.json':
            # The original producer attempts to rewrite its growing list.
            # Retain per-case diagnostics and commit the full list once.
            if len(value) == 128: data.write(path, value)
        else: data.write(path, value)
    original_freeze = serial.evaluation.freeze
    def freeze(root, stage, training, environment, paths):
        return original_freeze(root, stage, training, environment,
            [*paths, *[INCIDENT/n for n in ('producer.json', 'failure.json', 'actual_tool_receipt.json')]])
    serial.ROOT = ROOT
    serial.data = SimpleNamespace(**{**vars(data), 'write': write})
    serial.evaluation.freeze = freeze
    serial.run()


if __name__ == '__main__':
    existed = ROOT.exists()
    try: run()
    except Exception as error:
        if ROOT.exists() and not existed: data.write(ROOT/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
