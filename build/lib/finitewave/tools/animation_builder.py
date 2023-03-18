import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import time
import sys
import os


class AnimationBuilder:
    def __init__(self):
        self.dir_name = ""
        self.skip = 0
        self.vmin = 0
        self.vmax = 1

        # take into account the user path:
        self._prefix = os.getcwd()

    def write_2d_mp4(self, file_name, title="", fps=5, dpi=100):
        metadata = dict(title=title, artist='finitewave')
        writer = FFMpegWriter(fps=fps, metadata=metadata)

        frames_list = os.listdir(self.dir_name)
        frames_list.sort(key=lambda x: int(x.split(".")[0]))

        if not len(frames_list):
            return
        fig = plt.figure()
        frame_data = np.load(os.path.join(self._prefix, self.dir_name, frames_list[0]))
        anim = plt.imshow(frame_data, vmax=self.vmax, vmin=self.vmin, animated=True)
        plt.colorbar(anim)
        with writer.saving(fig, os.path.join(self._prefix, file_name), dpi):
            start_time = time.time()
            N = len(frames_list)
            for i in range(1, N):
                frame_data = np.load(os.path.join(self._prefix, self.dir_name, frames_list[i]))
                anim.set_array(frame_data)
                writer.grab_frame()
                sys.stdout.write("Writing frames: %d  of %d\r" % (i, N))
                sys.stdout.flush()
        plt.close()
