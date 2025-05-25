import matplotlib.pyplot as plt
import numpy as np

def planar_3D_setup(ax):
    # Draw XY grid manually at z = 0
    grid_size = 10
    grid_range = np.linspace(-1, 1, grid_size+1)
    for x in grid_range:
        ax.plot([x, x], [grid_range[0], grid_range[-1]], [0, 0], color='k', lw=0.5, alpha=0.5)
    for y in grid_range:
        ax.plot([grid_range[0], grid_range[-1]], [y, y], [0, 0], color='k', lw=0.5, alpha=0.5)

    # Draw axis arrows
    arrow_len = 1.2  # length of axis arrows
    # Arrows for X, Y, Z axes
    ax.quiver(0, 0, 0, arrow_len, 0, 0, color='k', arrow_length_ratio=0.1/arrow_len)
    ax.quiver(0, 0, 0, 0, arrow_len, 0, color='k', arrow_length_ratio=0.1/arrow_len)
    ax.quiver(0, 0, 0, 0, 0, arrow_len, color='k', arrow_length_ratio=0.1/arrow_len)
    # Labels at the tips
    ax.text(arrow_len + 0.05, 0.05, 0, 'X', color='k', fontsize=12)
    ax.text(0.05, arrow_len, 0, 'Y', color='k', fontsize=12)
    ax.text(0, 0, arrow_len + 0.05, 'Z', color='k', fontsize=12)

    # Axis settings
    
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([-0.4, 1.2])
    ax.set_box_aspect([1,1,1])
    ax.set_axis_off()
    ax.view_init(elev=30, azim=120)

