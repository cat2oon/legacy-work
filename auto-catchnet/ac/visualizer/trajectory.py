import pandas as pd
import matplotlib.pyplot as plt

from ds.device.nexus5x import get_true_cam_xy


class ScreenTrajectory:
    def __init__(self):
        self.trajectory = None
        self.ground_truth = None
        self.init_trajectory()

    def init_trajectory(self):
        tdf = pd.DataFrame(index=range(10), columns=['x', 'y'])
        gdf = pd.DataFrame(index=range(10), columns=['x', 'y'])

        for i in range(0, 11):
            x, y = get_true_cam_xy(i)
            gdf.at[i, 'x'] = x
            gdf.at[i, 'y'] = y

        self.trajectory = tdf
        self.ground_truth = gdf

    def mark(self, pos_idx, x, y):
        df = self.trajectory
        df.at[pos_idx, 'x'] = x
        df.at[pos_idx, 'y'] = y

    def plot_scatter(self):
        tdf = self.trajectory
        gdf = self.ground_truth

        fig, ax = plt.subplots()
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()

        ax.scatter(tdf.x, tdf.y)
        ax.scatter(gdf.x, gdf.y)

        for i in range(1, gdf.shape[0]):
            tx, ty = tdf.at[i, 'x'], tdf.at[i, 'y']
            gx, gy = gdf.at[i, 'x'], gdf.at[i, 'y']
            ax.text(tx, ty, "{}".format(i), size=13)
            ax.text(gx, gy, "{}".format(i), size=13)

