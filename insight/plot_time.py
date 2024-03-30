# -*- coding: utf-8 -*-
# @Date    : 2024-02-09 15:49:44
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
from matplotlib import pyplot as plt


def main():
    labels = ["RLHF", "DPO", "CL", "EFUF"]
    values = [20, 12, 10, 3]
    colors = ["#45B39D"] * (len(values) - 1) + ["#F5B041"]

    # Create the bar chart
    plt.figure(figsize=(12, 9))
    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.12)
    plt.bar(labels, values, color=colors, width=0.6)  # type: ignore

    plt.plot(labels, values, color="darkred", marker="o", linestyle="-", linewidth=3, markersize=12)
    for label, value in enumerate(values):
        plt.text(label, value + 0.7, str(value), ha="center", fontsize=26)

    # Adding the title and labels
    # plt.xlabel("Method", fontsize=24)
    plt.ylabel("A100 GPU hours", fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.ylim(top=22.5)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
