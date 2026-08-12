import numpy as np
import argparse
from sctp.utils import plotting
from pathlib import Path
import matplotlib.pyplot as plt


def extract_costs(file_path):
    sctp_cost = []
    base_cost = []
    seed_costs = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(' | ')
            seed = int(parts[0].split(' : ')[1].strip())
            planner = parts[1].split(' : ')[1].strip()
            cost = float(parts[2].split(' : ')[1].strip())
            if seed not in seed_costs:
                seed_costs[seed] = {"sctp": None, "base": None}
            seed_costs[seed][planner] = cost

    for seed in sorted(seed_costs.keys()):
        sctp_cost.append(seed_costs[seed]["sctp"])
        base_cost.append(seed_costs[seed]["base"])

    return sctp_cost, base_cost, seed_costs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/data/sctp/sctp_eval')
    parser.add_argument('--num_drones', type=int, default=1)
    args = parser.parse_args()
    file_path = Path(args.save_dir) / f'log_{args.num_drones}.txt'

    sctp_cost, base_cost, seed_costs = extract_costs(file_path)
    assert len(sctp_cost) == len(base_cost)
    print(f"The number of data {len(sctp_cost)}")

    plotting.make_scatter_plot_with_box(base_cost, sctp_cost, xlabel='Baseline', ylabel='SCTP')
    image_name = Path(args.save_dir) / f'log_{args.num_drones}_scatter.png'
    plt.tight_layout()
    plt.savefig(image_name)
