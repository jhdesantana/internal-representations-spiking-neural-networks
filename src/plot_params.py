import shutil

import matplotlib.pyplot as plt

# Check if LaTeX is installed.
if shutil.which("latex"):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

# Font.
plt.rcParams["font.family"] = "Serif"

# Figsize.
plt.rcParams["figure.constrained_layout.use"] = True

# Axes.
# lw_axes = 2.25
# plt.rcParams['axes.grid'] = True
# plt.rcParams['grid.linewidth'] = lw_axes
# plt.rcParams['axes.linewidth'] = lw_axes
# plt.rcParams['xtick.major.width'] = lw_axes
# plt.rcParams['ytick.major.width'] = lw_axes
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 1

# Save fig.
plt.rcParams["savefig.bbox"] = "tight"
