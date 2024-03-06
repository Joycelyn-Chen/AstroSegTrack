import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

import yt


fig = plt.figure()

# See http://matplotlib.org/mpl_toolkits/axes_grid/api/axes_grid_api.html
# These choices of keyword arguments produce a four panel plot with a single
# shared narrow colorbar on the right hand side of the multipanel plot. Axes
# labels are drawn for all plots since we're slicing along different directions
# for each plot.
grid = AxesGrid(
    fig,
    (0.075, 0.075, 0.85, 0.85),
    nrows_ncols=(3, 3),
    axes_pad=0.05,
    label_mode="L",
    share_all=True,
    cbar_location="right",
    cbar_mode="each",
    cbar_size="10%",
    cbar_pad="0%",
)

z_centers = [-250, 0, 250]

ds = yt.load("/Users/joycelynchen/Desktop/UBC/Research/Presentations/CVPRW/sn34_smd132_bx5_pe300_hdf5_plt_cnt_0209")  # load data

for i, z_center in enumerate(z_centers):
    # Load the data and create a single plot
    
    p = yt.SlicePlot(ds, "z", "dens", center = [0, 0, z_center] * yt.units.pc)
    p.set_unit(("flash", "dens"), "g/cm**3")

    # Ensure the colorbar limits match for all plots
    # p.set_zlim("dens", 1e-29, 1e-24)

    # This forces the ProjectionPlot to redraw itself on the AxesGrid axes.
    plot = p.plots["dens"]
    plot.figure = fig
    plot.axes = grid[i].axes
    plot.cax = grid.cbar_axes[i]
    if i < 2:
        plot.hide_colorbar()

    # Finally, this actually redraws the plot.
    p.render()

for i, z_center in enumerate(z_centers):
    # Load the data and create a single plot
    
    p = yt.SlicePlot(ds, "z", "temp", center = [0, 0, z_center] * yt.units.pc)

    p.set_unit(("flash", "temp"), "K")

    # Ensure the colorbar limits match for all plots
    # p.set_zlim("temp", 1e4, 1e7)

    # This forces the ProjectionPlot to redraw itself on the AxesGrid axes.
    plot = p.plots["temp"]
    plot.figure = fig
    plot.axes = grid[i + 3].axes
    plot.cax = grid.cbar_axes[i + 3]
    if i < 2:
        plot.hide_colorbar()

    # Finally, this actually redraws the plot.
    p.render()

for i, z_center in enumerate(z_centers):
    # Load the data and create a single plot
    
    p = yt.SlicePlot(ds, "z", "velx", center = [0, 0, z_center] * yt.units.pc)
    p.set_unit(("flash", "velx"), "km/s")

    # Ensure the colorbar limits match for all plots
    # p.set_zlim("velx", 1e-7, 1e-7)

    # This forces the ProjectionPlot to redraw itself on the AxesGrid axes.
    plot = p.plots["velx"]
    plot.figure = fig
    plot.axes = grid[i + 6].axes
    plot.cax = grid.cbar_axes[i+6]
    if i < 2:
        plot.hide_colorbar()


    # Finally, this actually redraws the plot.
    p.render()

plt.savefig("dataset_grid.png")