import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from tqdm.auto import tqdm

from .integrator import rk4
from .graphics import updatecolor, update_opacity, sample_background

def initq(l, r0, inclination):
    x0 = r0 * np.cos(inclination)
    z0 = r0 * np.sin(inclination)
    y0 = 0  # camera at mid-plane

    # Setup a simple camera window
    depth = 0.5
    width = 0.5
    u = -depth * np.array([x0, 0, z0]) / norm([x0, 0, z0])
    v = np.array([0, width, 0])
    w = np.cross(u / norm(u), v)
    A = np.array([x0, y0, z0]) + u - w - v

    # Create a grid of initial positions in the camera plane.
    i = np.arange(l)
    position = A[:, None, None] + (v[:, None, None] * i[None, :, None] + w[:, None, None] * i[None, None, :]) / l * 2
    velocity = position - np.array([x0, y0, z0])[:, None, None]
    velocity /= norm(velocity, axis=0)
    
    q = np.zeros((6, l, l))
    # add a little randomness to initial positions for realism
    q[:3] = position + velocity * np.random.random(size=(l, l)) * 0.01
    q[3:] = velocity

    # Compute angular momentum magnitude for each ray; this will be used in the integration.
    h = np.linalg.norm(np.cross(position, velocity, axis=0), axis=0)
    return q, h

def ray_tracing(l, r0, inclination, render_disk=True, background_image_path=None, dt=0.075,progress_bar=None):
    q, h = initq(l, r0, inclination)
    
    # Load background image if provided.
    if background_image_path != None:
        bg = plt.imread(background_image_path)

        if bg.shape[2] == 4:
            bg = bg[:, :, :3]

        if bg.dtype == np.uint8:
            bg = bg.astype(float) / 255.        

    else:
        bg = None

    if render_disk:
        disk_color_acc = np.zeros((l, l))
    opacity = np.ones((l, l))
    
    tmax = r0 * 2  # integration time can be adjusted
    t = 0
    dt = dt
    pbar = tqdm(total=tmax)
    
    while t < tmax:
        pbar.update(dt)

        if progress_bar:
            progress_bar.progress(min(int((t / tmax) * 100), 100))  # Progress in percentage
        
        if render_disk:
            disk_color_acc, opacity = updatecolor(q, disk_color_acc, opacity)
        else:
            opacity = update_opacity(q, opacity)  # ensure rays near BH are absorbed
        q = rk4(q, dt, h)
        t += dt
        
    # If a background image is provided, sample it using the deflected ray directions.
    if bg is not None:
        M = sample_background(q, bg)
        M = M * opacity[..., None]

        if render_disk:
            disk_norm = disk_color_acc / np.max(disk_color_acc + 1e-6)
            cmap = plt.get_cmap('gnuplot2')
            disk_colored = cmap(disk_norm)[..., :3]
            alpha = np.clip(disk_norm[..., None], 0, 1)
            M = M * (1 - alpha) + disk_colored * alpha
    
    else:
        if render_disk:
            M = disk_color_acc
        else:
            M = np.zeros((l, l))
    if M.ndim == 2:
        M = np.flip(M.T)
    else:
        M = np.flip(M.transpose(1, 0, 2), axis=0)

    return M

