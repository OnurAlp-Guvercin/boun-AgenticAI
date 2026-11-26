"""Draw the training curves of all runs as SVG files (standard library only).

Usage:
    python report/make_figures.py
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WINDOW = 10

RUNS = [  # (label, folder, color, dash)
    ("Custom few-shot", "exp_custom", "#2a78d6", ""),
    ("Original few-shot", "exp_fewshot", "#eb6834", ""),
    ("Original few-shot in system", "exp_fewshot_system", "#1baf7a", "6 4"),
    ("Custom no few-shot", "exp_custom_no_fewshot", "#eda100", "2 3"),
    ("Original no few-shot", "exp_no_fewshot", "#e87ba4", "10 4 2 4"),
]

WIDTH, HEIGHT = 480, 300
LEGEND_WIDTH = 640
LEFT, RIGHT, TOP, BOTTOM = 64, 14, 14, 50
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


def rolling_mean(values, window=WINDOW):
    means = []
    for i in range(len(values)):
        part = values[max(0, i - window + 1):i + 1]
        means.append(sum(part) / len(part))
    return means


def read_column(folder, column):
    with open(ROOT / "outputs" / folder / "train_log.csv") as f:
        return [float(row[column]) for row in csv.DictReader(f)]


def line_chart(column, y_max, y_ticks, y_label, tick_format):
    plot_w = WIDTH - LEFT - RIGHT
    plot_h = HEIGHT - TOP - BOTTOM

    def x_pos(step):
        return LEFT + (step - 1) / 99 * plot_w

    def y_pos(value):
        return TOP + plot_h - value / y_max * plot_h

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
             f'font-family="Helvetica, Arial, sans-serif" font-size="16">',
             f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#fcfcfb"/>']

    for tick in y_ticks:
        y = y_pos(tick)
        parts.append(f'<line x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{LEFT - 6}" y="{y + 5:.1f}" text-anchor="end" fill="{MUTED}">{tick_format(tick)}</text>')
    parts.append(f'<line x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y_pos(0):.1f}" y2="{y_pos(0):.1f}" stroke="{AXIS}" stroke-width="1"/>')

    for step in [1, 20, 40, 60, 80, 100]:
        parts.append(f'<text x="{x_pos(step):.1f}" y="{HEIGHT - BOTTOM + 20}" text-anchor="middle" fill="{MUTED}">{step}</text>')
    parts.append(f'<text x="{LEFT + plot_w / 2:.1f}" y="{HEIGHT - 6}" text-anchor="middle" fill="{MUTED}">Optimizer update</text>')
    parts.append(f'<text transform="translate(16 {TOP + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" fill="{MUTED}">{y_label}</text>')

    for label, folder, color, dash in RUNS:
        values = rolling_mean(read_column(folder, column))
        points = " ".join(f"{x_pos(i + 1):.1f},{y_pos(v):.1f}" for i, v in enumerate(values))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" '
                     f'stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>')

    parts.append("</svg>")
    return "\n".join(parts)


def legend():
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {LEGEND_WIDTH} 44" '
             f'font-family="Helvetica, Arial, sans-serif" font-size="11">',
             f'<rect width="{LEGEND_WIDTH}" height="44" fill="#fcfcfb"/>']
    for i, (label, _, color, dash) in enumerate(RUNS):
        x = 8 + (i % 3) * 210
        y = 14 + (i // 3) * 20
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{x}" x2="{x + 28}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="2"{dash_attr}/>')
        parts.append(f'<text x="{x + 36}" y="{y + 4}" fill="{INK}">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    out_dir = ROOT / "report"
    (out_dir / "training_accuracy.svg").write_text(
        line_chart("accuracy", 1.0, [0, 0.25, 0.5, 0.75, 1.0], "Training accuracy", lambda t: f"{t:.0%}") + "\n")
    (out_dir / "training_tool_calls.svg").write_text(
        line_chart("avg_tool_calls", 4.0, [0, 1, 2, 3, 4], "Tool calls per question", lambda t: f"{t:g}") + "\n")
    (out_dir / "legend.svg").write_text(legend() + "\n")
    print(f"Saved figures to {out_dir}")
