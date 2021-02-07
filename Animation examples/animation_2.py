import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
fig.set_tight_layout(True)

print('fig size: {0} DPI, size in inches {1}'.format(
    fig.get_dpi(), fig.get_size_inches()))

x = np.arange(0, 20, 0.1)
ax.scatter(x, x + np.random.normal(0, 3.0, len(x)))
line, = ax.plot(x, x - 5, 'r-', linewidth=2)


def update(i):
    label = 'timestep {0}'.format(i + 1)
    line.set_ydata(x - 5 + i)
    ax.set_xlabel(label)
    return line, ax


anim = FuncAnimation(fig, func=update, repeat=True,
                     frames=np.arange(0, 10), interval=500)

plt.show()
