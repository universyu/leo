"""
Count lines of code for all Python files in the repository.
- Grouped by module (top-level directories)
- Generates beautiful visualisation charts
- Results saved to result/code_line_count.json and result/code_lines_visual.png
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------------------
# Counting helpers
# ---------------------------------------------------------------------------

def count_lines(filepath):
    """Count total, code, blank and comment lines in a single Python file."""
    total = blank = comment = 0
    in_docstring = False
    docstring_char = None

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            total += 1
            stripped = line.strip()

            if in_docstring:
                comment += 1
                if docstring_char in stripped and stripped.endswith(docstring_char):
                    in_docstring = False
                    docstring_char = None
                continue

            if stripped == '':
                blank += 1
                continue

            if stripped.startswith('"""') or stripped.startswith("'''"):
                dc = stripped[:3]
                comment += 1
                if stripped.count(dc) == 1:
                    in_docstring = True
                    docstring_char = dc
                continue

            if stripped.startswith('#'):
                comment += 1
                continue

    code = total - blank - comment
    return {'total': total, 'code': code, 'blank': blank, 'comment': comment}


def collect_stats(root_dir):
    """Walk the repo and return per-file and per-module statistics."""
    file_stats = {}
    module_stats = {}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != 'output' and d != '__pycache__']
        for fname in sorted(filenames):
            if not fname.endswith('.py'):
                continue
            fpath = os.path.join(dirpath, fname)
            relpath = os.path.relpath(fpath, root_dir)
            counts = count_lines(fpath)
            file_stats[relpath] = counts

            # Determine module name
            parts = relpath.split(os.sep)
            if len(parts) == 1:
                module = '(root)'
            elif parts[0] == 'leo_network' and len(parts) > 1:
                module = f"leo_network/{parts[1]}"
            elif parts[0] == 'tests' and len(parts) > 1:
                module = f"tests/{parts[1]}"
            else:
                module = parts[0]

            if module not in module_stats:
                module_stats[module] = {'files': 0, 'total': 0, 'code': 0, 'blank': 0, 'comment': 0}
            m = module_stats[module]
            m['files'] += 1
            for k in ('total', 'code', 'blank', 'comment'):
                m[k] += counts[k]

    summary = {'total_files': 0, 'total_lines': 0, 'total_code': 0, 'total_blank': 0, 'total_comment': 0}
    for m in module_stats.values():
        summary['total_files'] += m['files']
        summary['total_lines'] += m['total']
        summary['total_code'] += m['code']
        summary['total_blank'] += m['blank']
        summary['total_comment'] += m['comment']

    return file_stats, module_stats, summary


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

# A refined colour palette
PALETTE = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974',
           '#64B5CD', '#E5AE38', '#6D904F', '#D65F5F', '#B47CC7']

DARK_BG = '#1E1E2E'
CARD_BG = '#2A2A3C'
TEXT_CLR = '#CDD6F4'
ACCENT = '#89B4FA'
GRID_CLR = '#45475A'

# Chinese labels for each module
MODULE_CN = {
    '(root)':              '(根目录)',
    'leo_network/core':    '卫星通信模块',
    'leo_network/attacks':  '攻击模块',
    'leo_network/defense':  '防御模块',
    'tests/unit':           '单元测试',
    'tests/integration':    '集成测试',
    'examples':             '示例脚本',
}


def _cn(name):
    """Return 'english_name (中文名)' if a Chinese label exists."""
    cn = MODULE_CN.get(name)
    return f"{name}（{cn}）" if cn else name


def _add_watermark(fig):
    fig.text(0.99, 0.005, 'LEO Network Codebase', fontsize=7, color='#585B70',
             ha='right', va='bottom', style='italic')


def draw_visualisation(module_stats, summary, output_path):
    """Create a premium-looking dashboard with table + pie + bar charts."""
    # Sort modules by total lines (descending)
    sorted_modules = sorted(module_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    mod_names = [_cn(m[0]) for m in sorted_modules]
    mod_data = [m[1] for m in sorted_modules]

    fig = plt.figure(figsize=(18, 11), facecolor=DARK_BG)
    fig.suptitle('Python Codebase — Line Count Dashboard  代码行数统计面板',
                 fontsize=22, fontweight='bold', color=TEXT_CLR, y=0.97)

    # ── Layout: 3 rows ──
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30,
                          left=0.06, right=0.96, top=0.91, bottom=0.06)

    # ---------------------------------------------------------------
    # 1) Summary cards (top-left)
    # ---------------------------------------------------------------
    ax_cards = fig.add_subplot(gs[0, 0])
    ax_cards.set_facecolor(DARK_BG)
    ax_cards.axis('off')

    cards = [
        ('Total Files\n文件总数', summary['total_files'], '#89B4FA'),
        ('Total Lines\n总行数', summary['total_lines'], '#A6E3A1'),
        ('Code Lines\n代码行', summary['total_code'], '#F9E2AF'),
        ('Comment Lines\n注释行', summary['total_comment'], '#CBA6F7'),
        ('Blank Lines\n空行', summary['total_blank'], '#F38BA8'),
    ]

    for i, (label, value, color) in enumerate(cards):
        cx = 0.1 + i * 0.18
        rect = plt.Rectangle((cx - 0.07, 0.15), 0.14, 0.7,
                              transform=ax_cards.transAxes, facecolor=CARD_BG,
                              edgecolor=color, linewidth=1.5, clip_on=False,
                              zorder=2, joinstyle='round')
        rect.set_path_effects([])
        ax_cards.add_patch(rect)
        ax_cards.text(cx, 0.62, f'{value:,}', transform=ax_cards.transAxes,
                      ha='center', va='center', fontsize=16, fontweight='bold', color=color)
        ax_cards.text(cx, 0.32, label, transform=ax_cards.transAxes,
                      ha='center', va='center', fontsize=9, color=TEXT_CLR)

    # ---------------------------------------------------------------
    # 2) Donut chart — Code / Comment / Blank ratio (top-right)
    # ---------------------------------------------------------------
    ax_pie = fig.add_subplot(gs[0, 1])
    ax_pie.set_facecolor(DARK_BG)
    sizes = [summary['total_code'], summary['total_comment'], summary['total_blank']]
    labels = ['Code', 'Comment', 'Blank']
    colors_pie = ['#A6E3A1', '#CBA6F7', '#F38BA8']
    explode = (0.03, 0.03, 0.03)

    wedges, texts, autotexts = ax_pie.pie(
        sizes, labels=labels, autopct='%1.1f%%', startangle=140,
        colors=colors_pie, explode=explode, pctdistance=0.78,
        wedgeprops=dict(width=0.45, edgecolor=DARK_BG, linewidth=2))
    for t in texts:
        t.set_color(TEXT_CLR)
        t.set_fontsize(10)
    for t in autotexts:
        t.set_color(DARK_BG)
        t.set_fontsize(9)
        t.set_fontweight('bold')
    ax_pie.set_title('Code / Comment / Blank Ratio（代码/注释/空行占比）', fontsize=13,
                     fontweight='bold', color=TEXT_CLR, pad=12)

    # ---------------------------------------------------------------
    # 3) Horizontal stacked bar — per module breakdown (middle row, full width)
    # ---------------------------------------------------------------
    ax_bar = fig.add_subplot(gs[1, :])
    ax_bar.set_facecolor(CARD_BG)

    y_pos = np.arange(len(mod_names))
    code_vals = [d['code'] for d in mod_data]
    comment_vals = [d['comment'] for d in mod_data]
    blank_vals = [d['blank'] for d in mod_data]

    bar_h = 0.55
    bars1 = ax_bar.barh(y_pos, code_vals, bar_h, label='Code', color='#A6E3A1', edgecolor='none')
    bars2 = ax_bar.barh(y_pos, comment_vals, bar_h, left=code_vals, label='Comment',
                        color='#CBA6F7', edgecolor='none')
    left2 = [c + cm for c, cm in zip(code_vals, comment_vals)]
    bars3 = ax_bar.barh(y_pos, blank_vals, bar_h, left=left2, label='Blank',
                        color='#F38BA8', edgecolor='none')

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(mod_names, fontsize=10, color=TEXT_CLR)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('Lines', fontsize=11, color=TEXT_CLR)
    ax_bar.set_title('Lines of Code by Module (Stacked)  各模块代码行数堆叠图', fontsize=13,
                     fontweight='bold', color=TEXT_CLR, pad=10)
    ax_bar.tick_params(axis='x', colors=TEXT_CLR)
    ax_bar.xaxis.grid(True, color=GRID_CLR, linestyle='--', linewidth=0.5)
    ax_bar.set_axisbelow(True)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['bottom'].set_color(GRID_CLR)
    ax_bar.spines['left'].set_color(GRID_CLR)

    # Annotate total on each bar
    for idx in range(len(mod_names)):
        total_val = mod_data[idx]['total']
        ax_bar.text(total_val + max(summary['total_lines'] * 0.008, 10), y_pos[idx],
                    f'{total_val:,}', va='center', fontsize=9, color=TEXT_CLR, fontweight='bold')

    legend = ax_bar.legend(loc='lower right', fontsize=9, framealpha=0.6,
                           facecolor=CARD_BG, edgecolor=GRID_CLR)
    for t in legend.get_texts():
        t.set_color(TEXT_CLR)

    # ---------------------------------------------------------------
    # 4) Detailed table (bottom row, full width)
    # ---------------------------------------------------------------
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.set_facecolor(DARK_BG)
    ax_table.axis('off')

    col_labels = ['Module 模块', 'Files', 'Total', 'Code', 'Comment', 'Blank', 'Code %']
    table_data = []
    for name, d in sorted_modules:
        pct = f"{d['code'] / d['total'] * 100:.1f}%" if d['total'] > 0 else '—'
        table_data.append([_cn(name), str(d['files']), f"{d['total']:,}",
                           f"{d['code']:,}", f"{d['comment']:,}", f"{d['blank']:,}", pct])
    # Summary row
    s = summary
    pct_all = f"{s['total_code'] / s['total_lines'] * 100:.1f}%" if s['total_lines'] > 0 else '—'
    table_data.append(['TOTAL 合计', str(s['total_files']), f"{s['total_lines']:,}",
                       f"{s['total_code']:,}", f"{s['total_comment']:,}", f"{s['total_blank']:,}", pct_all])

    table = ax_table.table(cellText=table_data, colLabels=col_labels,
                           loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.45)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#45475A')
        cell.set_text_props(color='#CDD6F4', fontweight='bold')
        cell.set_edgecolor(GRID_CLR)

    # Style body
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            if i == len(table_data):  # total row
                cell.set_facecolor('#313244')
                cell.set_text_props(color=ACCENT, fontweight='bold')
            else:
                cell.set_facecolor(CARD_BG if i % 2 == 0 else '#313244')
                cell.set_text_props(color=TEXT_CLR)
            cell.set_edgecolor(GRID_CLR)

    ax_table.set_title('Detailed Breakdown  详细统计表', fontsize=13,
                       fontweight='bold', color=TEXT_CLR, pad=14)

    _add_watermark(fig)
    fig.savefig(output_path, dpi=180, facecolor=DARK_BG, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅  Visualisation saved to: {os.path.relpath(output_path, os.path.dirname(output_path))}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(root_dir, 'result')
    os.makedirs(output_dir, exist_ok=True)

    file_stats, module_stats, summary = collect_stats(root_dir)

    # ── Console output ──
    print()
    print("=" * 70)
    print(f"{'Python Codebase — Line Count Summary':^70}")
    print("=" * 70)
    print(f"  Total .py files  : {summary['total_files']}")
    print(f"  Total lines      : {summary['total_lines']:,}")
    print(f"  Code lines       : {summary['total_code']:,}")
    print(f"  Comment lines    : {summary['total_comment']:,}")
    print(f"  Blank lines      : {summary['total_blank']:,}")
    print("=" * 70)

    sorted_modules = sorted(module_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    print(f"\n  {'Module':<35} {'Files':>5} {'Total':>7} {'Code':>7} {'Comment':>7} {'Blank':>7} {'Code%':>6}")
    print("  " + "-" * 75)
    for name, d in sorted_modules:
        pct = f"{d['code'] / d['total'] * 100:.1f}" if d['total'] > 0 else '—'
        display = _cn(name)
        print(f"  {display:<35} {d['files']:>5} {d['total']:>7,} {d['code']:>7,} {d['comment']:>7,} {d['blank']:>7,} {pct:>5}%")
    print("  " + "-" * 75)
    s = summary
    pct_all = f"{s['total_code'] / s['total_lines'] * 100:.1f}" if s['total_lines'] > 0 else '—'
    print(f"  {'TOTAL 合计':<35} {s['total_files']:>5} {s['total_lines']:>7,} {s['total_code']:>7,} {s['total_comment']:>7,} {s['total_blank']:>7,} {pct_all:>5}%")
    print()

    # ── Save JSON ──
    json_path = os.path.join(output_dir, 'code_line_count.json')
    payload = {
        'summary': summary,
        'modules': {k: v for k, v in sorted_modules},
        'files': file_stats
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  ✅  JSON saved to: result/code_line_count.json")

    # ── Generate visualisation ──
    png_path = os.path.join(output_dir, 'code_lines_visual.png')
    draw_visualisation(module_stats, summary, png_path)

    print()


if __name__ == '__main__':
    main()
