import numpy as np

def frk4(q, h):
    r_norm = np.sqrt(np.sum(q[:3]**2, axis=0))
    r = -1.5 * h**2 * r_norm**4
    velocity = q[3:]
    acceleration = q[:3] / r
    return np.concatenate((velocity, acceleration), axis=0)

def rk4(q, dt, h):
    k1 = frk4(q, h)
    k2 = frk4(q + k1 * dt / 2, h)
    k3 = frk4(q + k2 * dt / 2, h)
    k4 = frk4(q + k3 * dt, h)
    return q + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
