import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

from clam.utils.logger import log


def check_optical_flow():
    data_dir = "/scr/shared/clam/tensorflow_datasets/tdmpc2-buffer_imgs_emb-r3m_flow_debug/mw-door-open-expert"
    ds = tf.data.Dataset.load(str(data_dir))

    vis_dir = os.path.join(data_dir, "flow_vis")
    os.makedirs(vis_dir, exist_ok=True)

    log(f"Loaded {len(ds)} episodes", "green")
    for i, data in tqdm(enumerate(ds), desc="Processing episodes", total=len(ds)):
        images = data["images"]
        points = data["points"]
        points = points.numpy()

        # make video of frames and points and flow at each step
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plt.tight_layout()
        ax.axis("off")

        def animate(i):
            ax.clear()
            ax.axis("off")
            ax.imshow(images[i])
            # Plot points at current timestep
            ax.scatter(points[i, :, 0], points[i, :, 1], color="orange", s=30)
            # Plot flow vectors
            if i < len(images) - 1:
                flow = points[i + 1] - points[i]
                ax.quiver(
                    points[i, :, 0],
                    points[i, :, 1],
                    flow[:, 0],
                    flow[:, 1],
                    angles="xy",
                    scale_units="xy",
                    scale=0.15,
                    color="green",
                )
            ax.set_title(f"Frame {i}")

        anim = animation.FuncAnimation(
            fig, animate, frames=len(images), interval=100, blit=False
        )

        # Save animation
        writer = animation.FFMpegWriter(fps=10)
        anim.save(os.path.join(vis_dir, f"traj_flow_{i}.mp4"), writer=writer)
        plt.close()

        if i > 10:
            break


if __name__ == "__main__":
    # check_dataset()
    check_optical_flow()
