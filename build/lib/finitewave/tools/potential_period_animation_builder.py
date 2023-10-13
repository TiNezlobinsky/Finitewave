import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import time
import sys
import os


class PotentialPeriodAnimationBuilder:
    def __init__(self):
        self.file_name_pot = ""
        self.file_name_per = ""
        self.skip = 0
        self.vmin_pot = 0
        self.vmax_pot = 1
        self.vmin_per = 0
        self.vmax_per = 1

        self.colormap_pot = ""
        self.colormap_per = ""

        # take into account the user path:
        self._prefix = os.getcwd()

    def write_2d_mp4(self, file_name, title="", fps=5, dpi=100):
        metadata = dict(title=title, artist='finitewave')
        writer = FFMpegWriter(fps=fps, metadata=metadata)

        pot_frames_list = os.listdir(self.file_name_pot)
        pot_frames_list.sort(key=lambda x: int(x.split(".")[0]))

        per_frames_list = os.listdir(self.file_name_per)
        per_frames_list.sort(key=lambda x: int(x.split(".")[0]))

        if not (len(pot_frames_list) and len(per_frames_list)):
            return

        fig, ax = plt.subplots(1,2)

        fig.subplots_adjust(wspace=0.5)

        ax_pot = ax[0]
        ax_per = ax[1]

        if not self.colormap_pot:
            self.colormap_pot = "viridis"

        if not self.colormap_per:
            self.colormap_per = "viridis"

        pot_frame_data = np.load(os.path.join(self._prefix, self.file_name_pot, pot_frames_list[0]))
        per_frame_data = np.load(os.path.join(self._prefix, self.file_name_per, per_frames_list[0]))

        anim_pot  = ax_pot.imshow(pot_frame_data, vmax=self.vmax_pot, vmin=self.vmin_pot, animated=True, cmap=self.colormap_pot)
        anim_per  = ax_per.imshow(per_frame_data, vmax=self.vmax_per, vmin=self.vmin_per, animated=True, cmap=self.colormap_per)

        fig.colorbar(anim_pot, ax=ax_pot, fraction=0.046, pad=0.04)
        fig.colorbar(anim_per, ax=ax_per, fraction=0.046, pad=0.04)

        fig.set_figheight(10)
        fig.set_figwidth(10)
        plt.tight_layout()


        with writer.saving(fig, os.path.join(self._prefix, file_name), dpi):
            start_time = time.time()
            N = len(pot_frames_list)
            for i in range(1, N):
                pot_frame_data = np.load(os.path.join(self._prefix, self.file_name_pot, pot_frames_list[i]))
                per_frame_data = np.load(os.path.join(self._prefix, self.file_name_per, per_frames_list[i]))
                anim_pot.set_array(pot_frame_data)
                anim_per.set_array(per_frame_data)

                writer.grab_frame()
                sys.stdout.write("Writing frames: %d  of %d\r" % (i, N))
                sys.stdout.flush()
