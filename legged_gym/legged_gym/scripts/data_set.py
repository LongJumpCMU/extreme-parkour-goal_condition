import torch
import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import math

patch_size = (11, 12)  # (y, x)


class TestDataset(Dataset):
    def __init__(self, transform_file):
        """
        In this file, any reference to X and Y directions refers to the global X and Y
        in the isaac gym frame
        """
        self.data = pd.read_csv(transform_file, header=None, on_bad_lines='warn')
        self.NUM_ITERATIONS_PER_EXPT = 10

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        start and end: In meters. We don't care about their values, only their difference
        energy and time: Need to be dealt specially when success is False
        success: Whether the robot reached the goal during data collection
        """
        csv_row_idx = idx

        row = self.data.iloc[csv_row_idx]

        ############## Read data from row ##############

        # [1:-1] gets rid of inverted commas at the two ends

        # split(", ") converts it into a list of 2 string elements
        start = np.array(row[0][1:-1].split(", "), dtype=np.float32)
        end = np.array(row[1][1:-1].split(", "), dtype=np.float32)
        heading = float(row[2])

        # split(", ") converts it into a list of 10 string elements
        successes = np.array(
            [result == "True" for result in row[3][1:-1].split(", ")], dtype=np.bool8
        )
        num_successes = successes.sum()
        avg_success = 1.0 * num_successes / self.NUM_ITERATIONS_PER_EXPT
        energies = np.array(row[4][1:-1].split(", "), dtype=np.float32)
        avg_energy = 1.0 * energies[successes].sum() / max(num_successes, 1e-6)
        times = np.array(row[5][1:-1].split(", "), dtype=np.float32)
        avg_time = 1.0 * times[successes].sum() / max(num_successes, 1e-6)

        ########################################################

        # import ipdb
        # ipdb.set_trace()

        # import ipdb
        # ipdb.set_trace()
        transform = []
        transform.append(float(end[0]) - float(start[0]))  # Delta X
        transform.append(float(end[1]) - float(start[1]))  # Delta Y
        # if transform[0] == 0:
        #     psi = -1 * float(heading)
        # else:
        #     psi = np.arctan(transform[1] / transform[0]) - float(heading)
        # while psi > math.pi:
        #     psi -= 2 * math.pi
        # while psi < -math.pi:
        #     psi += 2 * math.pi
        # transform.append(psi)
        # transform.append(float(heading))

        psi_a = heading
        psi_b = np.arctan2(transform[1], transform[0])
        delta_psi = (psi_b - psi_a) % (2 * math.pi)
        if delta_psi > math.pi:
            delta_psi -= (2 * math.pi)
        
        transform.append(delta_psi)
        transform.append(float(heading))

        patch = row[6][1:-1].split()
        patch = np.array(patch, dtype=float).reshape((1, 17, 17))
        # patch = patch[:, 5:16, 3:13] # 5 to 15, 3 to 12, i.e., 11 x 10
        patch = patch[:, 5:, 3:-3]  # 5 to the end, 3 to -3, i.e., 12 x 11

        # # 1mx1.6m rectangle (without center line)
        # measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2] # 12 pts
        # measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]   # 11 pts

        # measured_points_x_dataset = [-1.2, -1.05, -0.9, -0.75, -0.6, -0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2]
        # measured_points_y_dataset = [-1.2, -1.05, -0.9, -0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75, 0.9,1.05, 1.2]

        # print(patch.shape)

        costs = (avg_energy, avg_time, avg_success)

        return (patch, torch.tensor(transform)), torch.tensor(costs)
