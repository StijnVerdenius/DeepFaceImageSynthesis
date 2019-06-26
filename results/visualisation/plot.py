import argparse
import os
import sys

import numpy as np
import torch

from models.general.statistic import Statistic
from utils.general_utils import ensure_current_directory, mean
from utils.constants import DATA_MANAGER, PROGRESS_DIR, PIC_DIR
import matplotlib.pyplot as plt


def convert_loss_dict(loss_dict, operation):
    translation_dict = {
        "NonSaturatingGLoss": "adv",
        "PixelLoss": "pix",
        "PerceptualLoss": "pp",
        "TripleConsistencyLoss": "triple",
        "IdLoss": "id",
        "ConsistencyLoss": "self"
    }
    output_dict = {}
    for key, value in loss_dict.items():
        output_dict[translation_dict[key]] = float(operation(value))#float(np.log10(value))
    return output_dict


def main(args):
    data = DATA_MANAGER.load_python_obj(f"{args.model_date_path}/{PROGRESS_DIR}/{args.model_name}")
    DATA_MANAGER.set_date_stamp()

    final_plot_dict = {}

    element: Statistic
    for element in data:
        plottable_data = convert_loss_dict(element.loss_gen_train_dict, np.log10 if args.bar else np.log10)
        for key, value in plottable_data.items():
            if (not key in final_plot_dict):
                final_plot_dict[key] = []
            final_plot_dict[key].append(value)

    if (not args.bar):

        # smoothing
        window = args.smoothing_window
        for key, values in final_plot_dict.items():
            temp = []
            for i, value in enumerate(values):
                if (i > window and i < (len(values) + window)):
                    value = mean(values[i - window: i + window])
                if (i % args.smoothing_removal_frequency == 0):
                    temp.append(value)

            final_plot_dict[key] = np.array(temp)

        if (args.summing):
            running_sum = {}
            key_total = "total loss"
            for key, values in final_plot_dict.items():
                if (not key_total in running_sum):
                    running_sum[key_total] = values
                else:
                    running_sum[key_total] += values

            final_plot_dict = running_sum

        # plotting
        for (key, values) in final_plot_dict.items():
            plt.plot(range(len(values)), values, label=key)
        plt.xticks([0, len(list(final_plot_dict.values())[0])/3, 2*(len(list(final_plot_dict.values())[0])/3), len(list(final_plot_dict.values())[0])],
        ["0", str((8*len(data))/3)[:2]+"k", str((16*len(data))/3)[:2]+"k", str(len(data)*8)[:2]+"k"])
        plt.xlabel("Samples evaluated")
        plt.ylabel("Loss")
        plt.xlim((-1, len(list(final_plot_dict.values())[0]) + 1))
        yticks = [1, 2, 3, 4, 5, 6] if args.summing else [-1, 0, 1, 2]
        plt.yticks(yticks, [str(10 ** x) for x in yticks])
    else:

        last_data_points = {key: mean(values[-args.smoothing_window:]) for key, values in final_plot_dict.items()}
        for key, value in last_data_points.items():
            plt.bar(key, value+1, label=key)
        plt.xticks([])
        yticks = [0, 1, 2, 3]
        plt.yticks(yticks, [str(10**int((x-1))) for x in yticks])


    plt.legend()
    plt.grid()
    plot_string = 'bar' if args.bar else 'loss'
    plt.savefig(f"{DATA_MANAGER.directory}{args.model_date_path}/{PIC_DIR}/{DATA_MANAGER.stamp}_{plot_string}plot.png")
    plt.show()


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-date-path', type=str, default='stijn'
                        )
    parser.add_argument('--model-name', type=str, default='progress_list')

    parser.add_argument('--smoothing-window', type=int, default=3)
    parser.add_argument('--smoothing-removal-frequency', type=int, default=30)
    parser.add_argument('--summing', type=bool, default=True)
    parser.add_argument('--bar', type=bool, default=False)

    return parser.parse_args()


if __name__ == '__main__':
    print(
        'cuda_version:',
        torch.version.cuda,
        'pytorch version:',
        torch.__version__,
        'python version:',
        sys.version,
    )
    print('Working directory: ', os.getcwd())
    args = parse()
    ensure_current_directory()
    main(args)
