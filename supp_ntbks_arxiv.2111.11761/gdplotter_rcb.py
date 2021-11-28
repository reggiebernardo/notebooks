import getdist.plots as gdplt

def plot_triangle(gdxs, params_list, ccs = ['red'], clst = ['-'],
                  lbls = None, font_size = 30, fig_legend_frame = False,
                  thickness = 5, title_fs = 20, width_inch = 7,
		  parlims = {}, lgd_font_size = 30, save = False, fname = None, folder = None):
    '''returns triangle plot of parameters
    
    Input:
    gdxs = list of cobaya/getdist samples, e.g., [gd_samples_1, ...]
    params_list = list of parameters, e.g., ["H0", "eta", ...]
    
    if superposing plots:
    ccs = list contour colors, e.g., ['red', 'blue', ...]
    clst = list of line styles, e.g., ['-', '--', '-.', ...]
    lbls = list of labels
    '''

    gdplot = gdplt.get_subplot_plotter(width_inch = width_inch)
    gdplot.settings.figure_legend_frame = fig_legend_frame
    gdplot.settings.legend_fontsize = lgd_font_size
    gdplot.settings.axes_labelsize = font_size
    gdplot.settings.axes_fontsize = font_size
    gdplot.settings.linewidth = thickness
    gdplot.settings.linewidth_contour = thickness
    gdplot.settings.alpha_filled_add = 0.9
    gdplot.settings.title_limit_fontsize = title_fs
    
    if len(gdxs) == 1:
        gdplot.triangle_plot(gdxs,
                             params_list,
                             contour_colors = [((1,1,1,0),
                                                ccs[0])],
                             title_limit = 1,
                             filled = True, param_limits = parlims)
    else:
        contour_colors = []
        for each in ccs:
            contour_colors.append(((1,1,1,0), each))

        gdplot.triangle_plot(gdxs,
                             params_list,
                             contour_ls = clst,
                             contour_colors = contour_colors,
                             filled = True,
                             legend_loc = 'upper right',
                             legend_labels = lbls, param_limits = parlims)
    if save == True:
        gdplot.export(fname, folder)
    
def plot_1d(gdxs, params_list, clrs = ['red'], lsty = ['-'],
            lbls = None, font_size = 30, fig_legend_frame = False,
            figs_per_row = 2, thickness = 5, title_fs = 20,
            width_inch = 7, lgd_font_size = 30, save = False, fname = None, folder = None):
    '''returns 1d posterior of parameters
    
    Input:
    gdxs = list of cobaya/getdist samples, e.g., [gd_samples_1, ...]
    params_list = list of parameters, e.g., ["H0", "eta", ...]
    
    if superposing plots:
    clrs = list line colors, e.g., ['red', 'blue', ...]
    lsty = list of line styles, e.g., ['-', '--', '-.', ...]
    lbls = list of labels
    '''
    
    gdplot = gdplt.get_subplot_plotter(width_inch = width_inch)
    gdplot.settings.figure_legend_frame = fig_legend_frame
    gdplot.settings.legend_fontsize = lgd_font_size
    gdplot.settings.axes_labelsize = font_size
    gdplot.settings.axes_fontsize = font_size
    gdplot.settings.linewidth = thickness
    gdplot.settings.alpha_filled_add = 0.9
    gdplot.settings.title_limit_fontsize = title_fs
    
    if len(gdxs) == 1:
        gdplot.plots_1d(gdxs,
                        params_list,
                        nx = figs_per_row,
                        title_limit = 1,
                        colors = [clrs[0]],
                        ls = [lsty[0]])
        
    else:
        gdplot.plots_1d(gdxs,
                        params_list,
                        nx = figs_per_row,
                        colors = clrs,
                        ls = lsty,
                        share_y = True,
                        legend_labels = lbls)
#        gdplot.add_legend(lbls,
#                          legend_loc = 'upper right')

    if save == True:
        gdplot.export(fname, folder)