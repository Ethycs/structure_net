"""Replay the cheaper greedy preview against the unchanged patch-1 control."""
import argparse
from pathlib import Path
from . import run_grammar_output as replay


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--build-manifest',type=Path,required=True)
    parser.add_argument('--native-tests',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--port',type=int,default=18088)
    replay.HYPOTHESIS='word-games-grammar-on-demand-v2'
    replay.EXTRA_PLANS=(Path('docs/07 - Status Reports/2026-10-31_word-games-grammar-reduction-amendment.md'),)
    replay.run(parser.parse_args())
