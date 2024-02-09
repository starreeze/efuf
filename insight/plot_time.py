# -*- coding: utf-8 -*-
# @Date    : 2024-02-09 15:49:44
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
from matplotlib import pyplot as plt


def main():
    labels = ["RLHF", "DPO", "contrastive learning", "FUME"]
    values = [20, 12, 10, 3]

    # Specify colors for each bar. All but the last one are blue, the last one is orange.
    colors = ["tab:blue"] * (len(values) - 1) + ["tab:orange"]

    # Create the bar chart
    plt.figure(figsize=(16, 12))
    plt.bar(labels, values, color=colors, width=0.5)  # type: ignore

    # Adding the title and labels
    # plt.xlabel("Method", fontsize=24)
    plt.ylabel("A100 GPU hours", fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
