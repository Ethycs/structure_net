"""Plot audited NCM component diagnostics without equating them to task accuracy.

Run with the existing plotting environment: pixi run --as-is python -m
experiments.structure_net.plot_ncm_component_radar_complete. No model is loaded.
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


ROOT = Path('data/experiments/ncm_paper_regime')
DEST = ROOT / '20261025_component_radar_v3'
FILES = {
    'codec': ROOT / '20261025_sequential_decoder_train_audit_v1/campaign_results.json',
    'parent': ROOT / '20261025_sequential_full732_pixi_train_v1/result.json',
    'semantic': ROOT / '20261025_sequential_full732_pixi_results_v1/campaign_results.json',
    'new': ROOT / '20261025_generated_code_decoder_train_audit_v1/campaign_results.json',
    'new_semantic': ROOT / '20261025_generated_code_decoder_results_v1/campaign_results.json',
}


def main():
    if DEST.exists():
        raise ValueError('Use a fresh artifact root; preserve prior charts')
    data = {k: json.loads(p.read_text()) for k, p in FILES.items()}
    codec, parent, semantic, new = (data[k] for k in ('codec', 'parent', 'semantic', 'new'))
    c, p, n = codec['oracle'], parent['final_TRAIN'], new['final_TRAIN_diagnostic']
    latest = data['new_semantic']
    if latest['cases'] != 128 or latest['valid'] != 128 or latest['passed'] != 0:
        raise ValueError('Expected completed latest semantic evaluation')
    if (c['cases'] != 732 or semantic['valid'] != 128
            or semantic['checkpoint_sha256'] != parent['checkpoint_sha256']
            or not parent['strict_reload_passed'] or not new['frozen_tensors_identical']):
        raise ValueError('Expected completed, matching component evidence')
    rows = [
        ('Reference-code reconstruction', c['active_word_matches'], c['active_words'], 'TRAIN',
         'Correct target codes supplied; exact native word IDs; source attention available'),
        ('Code prediction with correct history', p['teacher_correct_codes'], p['code_positions'], 'TRAIN',
         'Correct earlier target codes supplied; exact code IDs'),
        ('Free-running code prediction', p['rollout_correct_codes'], p['code_positions'], 'TRAIN',
         'Earlier codes predicted by model; exact code IDs'),
        ('Words from predicted codes', p['active_word_matches'], p['active_words'], 'TRAIN',
         'Source-only generated codes; exact native word IDs'),
        ('Complete unseen summaries', semantic['passed'], semantic['cases'], 'DEV',
         'Whole-summary semantic judgment; source-disjoint cases; Terra'),
    ]
    values = np.array([100 * num / den for _, num, den, _, _ in rows])
    updated = 100 * n['active_word_matches'] / n['active_words']
    angles = np.linspace(0, 2*np.pi, len(rows), endpoint=False)
    blue, orange, red, ink = '#3276b6', '#c96a17', '#b33c45', '#21324a'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(16, 10.2), facecolor='#f7f9fc')
    fig.text(.055, .95, 'NCM components: roles and measured performance',
             fontsize=21, fontweight='bold', color=ink)
    fig.text(.055, .913, '25 October 2026  |  Prior checkpoint + completed decoder adaptation',
             fontsize=12, color='#526278')

    ax = fig.add_axes([.09, .245, .48, .57], projection='polar', facecolor='#ffffff')
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 108)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9, color='#66768a')
    ax.set_rlabel_position(18)
    ax.grid(color='#d6deea', linewidth=.8)
    ax.spines['polar'].set_color('#d6deea')
    labels = [
        '1  Codec / decoder\nCorrect reference codes\nTRAIN',
        '2  Code predictor\nCorrect earlier codes\nTRAIN',
        '3  Code predictor\nIts own earlier codes\nTRAIN',
        '4  Word decoder\nPredicted codes\nTRAIN',
        '5  Whole pipeline\nUnseen summary passes\nDEV',
    ]
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10.5, color=ink)
    for i, label in enumerate(ax.get_xticklabels()):
        if i in (1, 2):
            label.set_horizontalalignment('left')
        elif i in (3, 4):
            label.set_horizontalalignment('right')
    ax.tick_params(axis='x', pad=22)
    ax.get_xticklabels()[-1].set_color(red)
    closed_a = np.r_[angles, angles[0]]
    closed_v = np.r_[values, values[0]]
    ax.plot(closed_a, closed_v, color=blue, linewidth=2.6)
    ax.fill(closed_a, closed_v, color=blue, alpha=.13)
    ax.scatter(angles, values, s=48, color=blue, edgecolors='white', linewidth=1.2, zorder=5)
    for i, (theta, value) in enumerate(zip(angles, values)):
        if i == 4:
            ax.annotate('0 / 128 passed', xy=(theta, value), xytext=(-58, 14),
                        textcoords='offset points', fontsize=11, fontweight='bold', color=red)
        else:
            offset = (-12, 12) if i == 1 else (0, 12 if i == 0 else -21)
            ax.annotate(f'{value:.1f}%', xy=(theta, value), xytext=offset,
                        textcoords='offset points', ha='right' if i == 1 else 'center', fontsize=12,
                        fontweight='bold', color=blue)
    ax.scatter([angles[3]], [updated], s=100, marker='D', color=orange,
               edgecolors='white', linewidth=1.3, zorder=6)
    ax.scatter([angles[4]], [0], s=90, marker='D', color=orange,
               edgecolors='white', linewidth=1.3, zorder=6)
    ax.annotate(f'New: {updated:.1f}%', xy=(angles[3], updated), xytext=(-80, 16),
                textcoords='offset points', color=orange, fontsize=11, fontweight='bold',
                arrowprops={'arrowstyle': '-', 'color': orange, 'lw': 1})

    panel = fig.add_axes([.66, .235, .30, .60])
    panel.set_axis_off()
    panel.set_xlim(0, 1)
    panel.set_ylim(0, 1)
    panel.text(0, 1.025, 'WHAT EACH COMPONENT DOES', color='#526278', fontsize=10,
               fontweight='bold')
    boxes = [
        (.82, .15, 'Frozen Llama body', 'Turns the source into contextual features.\nSource attention supplies these below.', '#edf1f7'),
        (.59, .15, 'NCM code generator', 'Predicts the hierarchy of summary codes.\n2  Correct history 94.6%  |  3  Own 66.1%', '#e8f1fa'),
        (.34, .17, 'Codec text decoder', 'Reconstructs words from codes + source.\n1  Correct codes 100% (prior, TRAIN)\n4  Predicted codes 36.9% → 42.4%', '#e8f1fa'),
        (.10, .14, 'Complete summary', 'Judged for faithfulness, salience, readability.\n5  Both checkpoints: 0 / 128 DEV passes', '#faebed'),
    ]
    for y, h, title, desc, color in boxes:
        panel.add_patch(FancyBboxPatch((.04, y), .91, h,
            boxstyle='round,pad=0.018,rounding_size=0.018', facecolor=color, edgecolor='#d6deea'))
        panel.text(.08, y+h-.045, title, color=ink, fontweight='bold', fontsize=11)
        panel.text(.08, y+h-.085, desc, color='#526278', fontsize=9.5, va='top', linespacing=1.6)
    for top, bottom in ((.80, .76), (.57, .53), (.32, .26)):
        panel.annotate('', xy=(.49, bottom), xytext=(.49, top),
                       arrowprops={'arrowstyle': '-|>', 'color': '#8093a9', 'lw': 1.5})
    panel.plot([.965, 1.01, 1.01, .965], [.87, .87, .42, .42],
               color='#879bb2', lw=1.1, linestyle='--', clip_on=False)
    panel.text(1.025, .64, 'Source features also reach the decoder', rotation=90,
               fontsize=8.5, color='#526278', ha='center', va='center')

    fig.text(.065, .17, 'Blue: prior checkpoint', color=blue, fontweight='bold', fontsize=11)
    fig.text(.32, .17, 'Orange: adapted decoder (two measured endpoints)', color=orange, fontweight='bold', fontsize=11)
    fig.text(.065, .125,
             'Different tests and denominators share a percentage scale. Polygon area is not a model score.\n'
             'The 100% reconstruction control uses correct codes on TRAIN; it does not establish unseen summary quality.',
             color='#526278', fontsize=10.5, linespacing=1.7)
    fig.text(.065, .065,
             'Training path: reference summary → codec encoder + vector quantizer → target codes. Generator training includes corrupted histories.\n'
             'Latest decoder oracle reconstruction is not included here. No standalone score for source attention; no speed score.',
             color='#526278', fontsize=10, linespacing=1.7)
    DEST.mkdir()
    fig.savefig(DEST/'component_radar.png', dpi=160, facecolor=fig.get_facecolor())
    fig.savefig(DEST/'component_radar.svg', facecolor=fig.get_facecolor())
    plt.close(fig)
    with (DEST/'measurements.csv').open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['measurement', 'correct', 'total', 'split', 'condition', 'percent'])
        for row, value in zip(rows, values):
            writer.writerow([*row, value])
        writer.writerow(['New decoder words from predicted codes', n['active_word_matches'],
                         n['active_words'], 'TRAIN', 'New decoder; generated codes', updated])
        writer.writerow(['New decoder complete unseen summaries', latest['passed'], latest['cases'],
                         'DEV', 'Whole-summary semantic judgment; source-disjoint cases; Terra', 0.0])
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    record = {'status': 'component_diagnostic_visualization', 'source_hashes': {str(p): sha(p) for p in FILES.values()},
        'plotter_sha256': sha(Path(__file__)), 'measurements': [dict(zip(
            ['measurement', 'correct', 'total', 'split', 'condition'], row), percent=float(v)) for row, v in zip(rows, values)],
        'new_decoder_TRAIN_word_agreement_percent': updated, 'new_decoder_full_DEV_score_included': True,
        'new_decoder_DEV_passes': latest['passed'], 'new_decoder_DEV_cases': latest['cases'],
        'metrics_are_not_interchangeable': True, 'speed_measured': False,
        'figure_hashes': {p.name: sha(p) for p in DEST.iterdir()}}
    (DEST/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'png': str(DEST/'component_radar.png'), 'svg': str(DEST/'component_radar.svg'),
                      'percentages': values.tolist(), 'new_decoder_word_percent': updated}))


if __name__ == '__main__':
    main()
