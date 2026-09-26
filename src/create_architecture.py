import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RESULTS_DIR = "results"


def add_box(ax, x, y, text):
    box = FancyBboxPatch(
        (x, y),
        2.4,
        0.8,
        boxstyle="round,pad=0.08",
        linewidth=1.5,
        facecolor="white"
    )
    ax.add_patch(box)

    ax.text(
        x + 1.2,
        y + 0.4,
        text,
        ha="center",
        va="center",
        fontsize=10
    )


def add_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=15,
        linewidth=1.3
    )
    ax.add_patch(arrow)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 8))

    boxes = [
        (0.5, 5.5, "Network Traffic"),
        (3.5, 5.5, "Data Collection"),
        (6.5, 5.5, "Preprocessing"),
        (9.5, 5.5, "Feature Engineering"),
        (12.5, 5.5, "Random Forest\nML Model"),
        (12.5, 3.5, "Risk Score"),
        (9.5, 3.5, "Decision\nThreshold"),
        (6.5, 3.5, "Alert\nClassification"),
        (3.5, 3.5, "Security\nAnalyst"),
        (3.5, 1.5, "Feedback"),
        (6.5, 1.5, "Model\nRetraining")
    ]

    for x, y, text in boxes:
        add_box(ax, x, y, text)

    # Main pipeline
    add_arrow(ax, 2.9, 5.9, 3.5, 5.9)
    add_arrow(ax, 5.9, 5.9, 6.5, 5.9)
    add_arrow(ax, 8.9, 5.9, 9.5, 5.9)
    add_arrow(ax, 11.9, 5.9, 12.5, 5.9)

    # Model to risk score
    add_arrow(ax, 13.7, 5.5, 13.7, 4.3)

    # Risk score to threshold
    add_arrow(ax, 12.5, 3.9, 11.9, 3.9)

    # Threshold to alert classification
    add_arrow(ax, 9.5, 3.9, 8.9, 3.9)

    # Alert classification to analyst
    add_arrow(ax, 6.5, 3.9, 5.9, 3.9)

    # Analyst to feedback
    add_arrow(ax, 4.7, 3.5, 4.7, 2.3)

    # Feedback to retraining
    add_arrow(ax, 5.9, 1.9, 6.5, 1.9)

    # Retraining back toward ML model
    ax.annotate(
        "",
        xy=(13.7, 5.5),
        xytext=(7.7, 2.3),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.3,
            connectionstyle="arc3,rad=0.25"
        )
    )

    ax.text(
        10.3,
        2.0,
        "Updated training data",
        fontsize=9,
        ha="center"
    )

    ax.set_xlim(0, 16)
    ax.set_ylim(0.5, 7)
    ax.axis("off")

    plt.tight_layout()

    output = f"{RESULTS_DIR}/architecture.png"
    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()