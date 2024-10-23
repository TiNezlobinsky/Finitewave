from pathlib import Path
import shutil
import cv2
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt


class Animation2DBuilder:
    def __init__(self):
        pass

    def write(self, path, animation_name='animation', mask=None, shape_scale=1,
              fps=12, clim=[0, 1], shape=(100, 100), codec='mp4v',
              cmap="coolwarm", clear=False):
        """
        No operation. Exists to fulfill the interface requirements.

        Parameters
        ----------
        path : str
            Path to the snapshot folder.
        animation_name : str, optional
            Name of the animation. The default is 'animation'.
        mask : ndarray, optional
            Mask where the data is not valid. The default is None.
        shape_scale : int, optional
            Scale factor. The default is 1.
        fps : int, optional
            Frames per second. The default is 60.
        clim : list, optional
            Color limits. The default is [0, 1].
        codec : str, optional
            Codec. The default is 'mp4v'.
        cmap : str, optional
            Color map. The default is 'coolwarm'.
        clear : bool, optional
            Clear the snapshot folder. The default is False.
        """
        path = Path(path)
        path_save = path.parent.joinpath(animation_name).with_suffix(".mp4")

        files = natsorted(path.glob("*.npy"))

        height, width = np.array(shape) * shape_scale
        cmap = plt.get_cmap(cmap)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*codec)

        with VideoWriter(path_save, fourcc, fps, (width, height)) as out:
            file = files[0]
            for file in files:
                frame = np.load(file.with_suffix(".npy"))
                # Normalize the frame data to the colormap
                mask_ = (frame < clim[0]) | (frame > clim[1])

                if mask is not None:
                    mask_ |= mask

                frame[mask] = np.nan
                frame = (frame - clim[0]) / (clim[1] - clim[0])

                # Upscale the frame if necessary
                if shape_scale > 1:
                    frame = np.repeat(np.repeat(frame, shape_scale, axis=0),
                                      shape_scale, axis=1)
                # Convert the frame to an 8-bit RGB image
                frame_rgb = (cmap(frame, bytes=True)[:, :, :3]
                             ).astype(np.uint8)
                out.write(frame_rgb)

        if clear:
            shutil.rmtree(path)


class VideoWriter:
    """
    Context manager for the cv2.VideoWriter class.
    """
    def __init__(self, filename, fourcc, fps, frame_size):
        self.filename = filename
        self.fourcc = fourcc
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None

    def __enter__(self):
        self.writer = cv2.VideoWriter(self.filename, self.fourcc, self.fps,
                                      self.frame_size)
        if not self.writer.isOpened():
            message = f"Could not open the video file: {self.filename}"
            raise ValueError(message)
        return self.writer

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Release the VideoWriter when exiting the context
        if self.writer is not None:
            self.writer.release()
        # Optionally, handle exceptions here
        if exc_type:
            # Remove the file if an exception occurred
            Path(self.filename).unlink()
        return False
