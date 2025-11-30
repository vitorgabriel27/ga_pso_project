import os
import numpy as np
import imageio
import matplotlib.pyplot as plt


def _fig_to_rgb_array(fig):
    """Render a Matplotlib figure (Qt / Agg backend) to an RGB numpy array."""
    fig.canvas.draw()
    # Use buffer_rgba which is available across interactive backends
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    # Drop alpha channel
    return arr[:, :, :3].copy()


def _downsample_history(history, max_frames=100):
    if len(history) <= max_frames:
        return history
    idx = np.linspace(0, len(history) - 1, max_frames).astype(int)
    return [history[i] for i in idx]


def generate_gif_2d(points_history, fitness_func, lower, upper, out_path, figsize=(5,5)):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sampled = _downsample_history(points_history)
    x = np.linspace(lower, upper, 200)
    y = np.linspace(lower, upper, 200)
    X, Y = np.meshgrid(x, y)
    pos_grid = np.vstack([X.ravel(), Y.ravel()]).T
    Z = fitness_func(pos_grid).reshape(X.shape)

    total = len(sampled)
    frames = []
    for idx, pts in enumerate(sampled):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(Z, extent=[lower, upper, lower, upper], origin='lower', cmap='viridis', alpha=0.85)
        ax.set_title(f'Evolução 2D - Iteração {idx+1}/{total}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        pts_arr = np.asarray(pts)
        if pts_arr.size:
            ax.scatter(pts_arr[:, 0], pts_arr[:, 1], c='white', s=20, edgecolors='black', linewidths=0.3)
        frame = _fig_to_rgb_array(fig)
        frames.append(frame)
        plt.close(fig)
    if frames:
        imageio.mimsave(out_path, frames, duration=0.15)
    return out_path


def generate_gif_3d(points_history, fitness_func, lower, upper, out_path, figsize=(6,6)):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sampled = _downsample_history(points_history)
    x = np.linspace(lower, upper, 60)
    y = np.linspace(lower, upper, 60)
    X, Y = np.meshgrid(x, y)
    pos_grid = np.vstack([X.ravel(), Y.ravel()]).T
    Z = fitness_func(pos_grid).reshape(X.shape)

    total = len(sampled)
    frames = []
    for idx, pts in enumerate(sampled):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
        ax.set_title(f'Evolução 3D - Iteração {idx+1}/{total}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x,y)')
        pts_arr = np.asarray(pts)
        if pts_arr.size:
            z_pts = fitness_func(pts_arr)
            ax.scatter(pts_arr[:, 0], pts_arr[:, 1], z_pts, c='white', s=15, depthshade=True, edgecolors='black', linewidths=0.3)
        frame = _fig_to_rgb_array(fig)
        frames.append(frame)
        plt.close(fig)
    if frames:
        imageio.mimsave(out_path, frames, duration=0.15)
    return out_path
