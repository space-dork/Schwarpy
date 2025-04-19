import numpy as np
from numpy.linalg import norm

def disc_color(x, y, z):
    r = np.hypot(x, y)
    color = np.where(r > 1e-6, 1 / r, 0)
    color[np.abs(z) > 0.05] = 0
    color[np.abs(r) < 1.01] = 0
    return color

# Update the color based on the disk intensity
def updatecolor(q, color, opacity):
    c = disc_color(*q[:3])
    color += c * opacity
    opacity *= np.exp(-c / 200)  # you can adjust the factor to control fading effects
    r = norm(q[:3], axis=0)
    opacity[r < 0.4] = 0  # if too close to the black hole center, block the ray
    return color, opacity

def update_opacity(q, opacity):
    r = norm(q[:3], axis=0)
    opacity[r < 0.4] = 0  # block the ray if too close to black hole
    return opacity

def sample_background(q, background):
    # d is the final direction of the ray (normalized)
    d = q[3:]
    #god I hate UV unwrapping
    theta = np.arctan2(d[1], -d[0])  # azimuth
    phi = np.arcsin(np.clip(d[2], -1, 1))  # elevation

    H, W, _ = background.shape

    # Map theta from [-pi, pi] to [0, W)
    u = ((theta + np.pi) / (2 * np.pi) * W).astype(int) % W
    # Map phi from [-pi/2, pi/2] to [0, H)
    v = ((np.pi/2 - phi) / np.pi * H).astype(int)
    v = np.clip(v, 0, H - 1)

    return background[v, u]
