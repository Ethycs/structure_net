"""Plot measured NCM summary outcomes; never infer a score for an active run."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path('data/experiments/ncm_paper_regime')
DEST = ROOT / '20261024_accuracy_history_chart_v1'
SOURCE_SHA = '1395a35a259ec02073c41f77858a682596e44b66c13ccb505a8cd9da18b2130f'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    if DEST.exists():
        raise ValueError('fresh append-only chart snapshot required')
    used, rows = {}, []

    def load(path):
        path = Path(path)
        used[str(path)] = digest(path)
        return json.loads(path.read_text())

    def row(group, label, date, passed, cases, path, metric, **extra):
        if not 0 <= passed <= cases or cases == 0:
            raise ValueError('invalid outcome denominator')
        result = {'group': group, 'label': label, 'date': date, 'passed': passed, 'cases': cases,
            'accuracy_percent': 100 * passed / cases, 'campaign': str(path), 'metric_path': metric, **extra}
        rows.append(result)
        return result

    historical = []
    path = Path('data/experiments/ncm_quality/20261021_judge_v3/20261021_registered_v1/campaign_results.json')
    value = load(path)['primary']['ncm']
    historical.append(row('historical_changed_protocols', 'Initial NCM\nOct 21', '2026-10-21', value['passes'], value['n'], path, 'primary.ncm', judge='claude_code/claude-sonnet-4-6'))
    path = Path('data/experiments/ncm_memory/20261022_accuracy_v2/campaign_results.json')
    state = load(path)
    for name, label in [('original_ncm', 'Memory baseline\nOct 22'), ('trained_ncm', 'Memory fine-tuned\nOct 22')]:
        value = state['models'][name]
        if abs(value['task_pass_rate'] - value['passes'] / value['n']) > 1e-12:
            raise ValueError('historical reported rate differs')
        historical.append(row('historical_changed_protocols', label, '2026-10-22', value['passes'], value['n'], path, 'models.' + name, judge=state['primary_judge']))
    path = Path('data/experiments/ncm_recipe_update/20261022_test_recovery_v1/campaign_results.json')
    value = load(path)['models']['updated']
    if value['valid'] != value['requested']:
        raise ValueError('invalid historical judgments must not become zero accuracy')
    historical.append(row('historical_changed_protocols', 'Updated recipe\nOct 22', '2026-10-22', value['passes'], value['requested'], path, 'models.updated', judge='historical recovered candidate judgments'))

    current = []
    trials = [
        ('20261024_pinned_date_development_results_v1', 'Clean student\nfixed-date input', 'semantic_results'),
        ('20261024_lm1b_native_word_readout_results_v1', 'Released LM1B\nword readout', 'metrics'),
        ('20261024_lm1b_grounded_development_results_v1', 'Source-attention\nadaptation', 'metrics'),
        ('20261024_lm1b_word_gradient_development_results_v1', 'Word-gradient\nadaptation', 'metrics'),
        ('20261024_lm1b_native655_development_results_v1', 'Full native\n655-target training', 'metrics'),
    ]
    for directory, label, metric in trials:
        path = ROOT / directory / 'campaign_results.json'
        state = load(path)
        # Only aggregate hashes are examined; no selected final source is opened.
        matching = [h for p, h in state['artifacts'].items() if p.endswith('cases.json')
            and ('development_metadata' in p or 'pinned_date_development_sources' in p)]
        if SOURCE_SHA not in matching or state['final_test_opened']:
            raise ValueError('same128 source binding and closed final test required')
        cohorts = [state[metric][c]['secondary'] for c in ('short_development', 'long_diagnostic')]
        if any(c['cases'] != 64 or c['valid'] != 64 or c['judge'] != 'codex_sdk/gpt-5.6-terra' for c in cohorts):
            raise ValueError('complete comparable64+64 Terra judgments required')
        current.append(row('same128_source_disjoint_DEV', label, '2026-10-24', sum(c['passed'] for c in cohorts),
            128, path, metric + '.[short_development,long_diagnostic].secondary',
            judge='codex_sdk/gpt-5.6-terra', short_passed=cohorts[0]['passed'], short_cases=64,
            long_passed=cohorts[1]['passed'], long_cases=64, source_cases_sha256=SOURCE_SHA))

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titleweight': 'bold',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
        'axes.spines.bottom': False, 'xtick.major.size': 0, 'ytick.major.size': 0})
    fig, axes = plt.subplots(2, 1, figsize=(13.8, 8.4), gridspec_kw={'hspace': .62})
    fig.patch.set_facecolor('#f7f9fc')
    fig.suptitle('NCM complete-summary accuracy', x=.075, y=.97, ha='left', fontsize=23, weight='bold', color='#152638')
    fig.text(.075, .918, 'Measured held-out outcomes. The latest comparable development trials remain at 0%.', fontsize=12, color='#46596b')
    for ax, records, title, color in zip(axes, [historical, current],
        ['Earlier evaluations — different datasets, splits and grading protocols',
         'Latest development evaluations — same 128 sources and Terra judge'], ['#77879d', '#167d9a']):
        x = np.arange(1, len(records) + 1)
        y = [r['accuracy_percent'] for r in records]
        ax.set_facecolor('#ffffff')
        ax.set_ylim(-6, 106)
        ax.set_yticks([0, 25, 50, 75, 95], ['0%', '25%', '50%', '75%', '95%'])
        ax.grid(axis='y', color='#e5eaf0', linewidth=.8)
        ax.set_axisbelow(True)
        ax.axhline(95, color='#c47a13', linestyle=(0, (5, 4)), linewidth=1.5)
        ax.text(.99, .87, '95% goal', transform=ax.transAxes, ha='right', color='#a6660e', fontsize=10)
        if records is current:
            ax.plot(x, y, color=color, linewidth=2, zorder=3)
        ax.scatter(x, y, color=color, s=58, zorder=4)
        for xx, yy, record in zip(x, y, records):
            text = f"{record['passed']}/{record['cases']}"
            if yy:
                text += f"  ({yy:.2f}%)"
            ax.annotate(text, (xx, yy), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10, color='#24384b')
        labels = [r['label'] for r in records]
        if records is current:
            ax.axvspan(5.55, 6.45, color='#eef1f5')
            ax.text(6, 43, 'Training\nUnmeasured', ha='center', va='center', color='#687787', fontsize=11)
            x = np.arange(1, 7)
            labels.append('FP16 + level noise\ncurrent run')
        ax.set_xlim(.55, len(labels) + .45)
        ax.set_xticks(x, labels, fontsize=10)
        ax.set_title(title, loc='left', fontsize=13, pad=15, color='#24384b')
    fig.subplots_adjust(left=.075, right=.97, top=.835, bottom=.15)
    fig.text(.075, .057, 'Pass = a complete, faithful, useful and readable summary within its output budget. Historical protocols differ.', fontsize=10, color='#46596b')
    fig.text(.075, .031, 'The 128-case DEV cohort contains 64 short + 64 long sources. The sealed final test is still unopened. No running model is assigned a score.', fontsize=10, color='#46596b')
    DEST.mkdir()
    png, svg = DEST / 'accuracy_history.png', DEST / 'accuracy_history.svg'
    fig.savefig(png, dpi=170, facecolor=fig.get_facecolor())
    fig.savefig(svg, facecolor=fig.get_facecolor())
    plt.close(fig)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (DEST / 'accuracy_history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (DEST / 'data.json').write_text(json.dumps(rows, indent=2))
    (DEST / 'producer_source.py').write_bytes(Path(__file__).read_bytes())
    summary = {'scope': 'measured NCM student complete-summary outcomes, historical and same128 DEV separated',
        'latest_measured_accuracy': current[-1]['accuracy_percent'], 'latest_passed': current[-1]['passed'],
        'latest_cases': current[-1]['cases'], 'pending_FP16_noise_accuracy': None, 'final_test_accuracy': None,
        'current_primary_judge_admitted': False, 'historical_protocols_not_one_learning_curve': True,
        'teacher_label_scores_and_TRAIN_fit_excluded': True, 'sealed_final_source_files_read': False,
        'input_campaign_sha256': used, 'source_sha256': digest(__file__),
        'artifacts': {str(p): digest(p) for p in [png, svg, DEST / 'accuracy_history.csv', DEST / 'data.json']}}
    (DEST / 'result.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'PNG': str(png), 'SVG': str(svg), 'CSV': str(DEST / 'accuracy_history.csv'),
        'latest_complete_summary_accuracy': '0/128', 'running_model_accuracy': None}), flush=True)


if __name__ == '__main__':
    main()
