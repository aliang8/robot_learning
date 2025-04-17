import torch
from einops import repeat


def get_track_displacement(tracks):
    """
    track: (B, T, N, 2)
    return: (B, N)

    caluculate the displacement of each track by taking the magnitude of the
    difference between each timestep, then summing over the timesteps.
    """
    b, t, c, n = tracks.shape
    diff_tracks = torch.diff(tracks, dim=1)
    mag_tracks = torch.norm(diff_tracks, dim=-1)
    disp_tracks = torch.sum(mag_tracks, dim=1)
    return disp_tracks


def sample_tracks(
    tracks, num_samples=16, uniform_ratio=0.25, vis=None, motion=False, h=None
):
    """
    tracks: (T, N, 2)
    num_samples: int, number of samples to take
    uniform_ratio: float, ratio of samples to take uniformly vs. according to displacement
    return: (T, num_samples, 2)

    sample num_samples tracks from the tracks tensor, using both uniform sampling and sampling according
    to the track displacement.
    """

    t, n, c = tracks.shape

    if motion:
        mask = (tracks > 0) & (tracks < h)
        mask = mask.all(dim=-1)  # if any of u, v is out of bounds, then it's false
        mask = mask.all(
            dim=0
        )  # if any of the points in the track is out of bounds, then it's false

        mask = repeat(mask, "n -> t n", t=t)
        tracks = tracks[mask]
        tracks = tracks.reshape(t, -1, c)

        if vis is not None:
            t, n = vis.shape
            vis = vis[mask]
            vis = vis.reshape(t, -1)

    num_uniform = int(num_samples * uniform_ratio)
    num_disp = num_samples - num_uniform

    uniform_idx = torch.randint(0, n, (num_uniform,))

    if num_disp == 0:
        idx = uniform_idx
    else:
        disp = get_track_displacement(tracks[None])[0]
        threshold = disp.min() + (disp.max() - disp.min()) * 0.1
        disp[disp < threshold] = 0
        disp[disp >= threshold] = 1
        disp_idx = torch.multinomial(disp, num_disp, replacement=True)

        idx = torch.cat([uniform_idx, disp_idx], dim=-1)

    sampled_tracks = tracks[:, idx]
    if vis is not None:
        t, n = vis.shape
        sampled_vis = vis[:, idx]

        return sampled_tracks, sampled_vis

    return sampled_tracks
