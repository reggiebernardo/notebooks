import numpy as np
from pulsar_2p_stats import *


class Pulsar2pStats:
    def __init__(self, name, npsrs, n_sims, gwb_binary, rn_binary, \
                 log10_ra2_gw_min=-16, log10_ra2_gw_max=-8):
        self.name = name
        self.npsrs = npsrs
        self.n_sims = n_sims
        self.gwb_binary = gwb_binary
        self.rn_binary = rn_binary
        self.log10_ra2_gw_min = log10_ra2_gw_min
        self.log10_ra2_gw_max = log10_ra2_gw_max

    def load_data(self, print_input=True):
        ppkstats=load_ppkstats(self.name, npsrs=self.npsrs, n_sims=self.n_sims, \
                               gwb_binary=self.gwb_binary, rn_binary=self.rn_binary, \
                               print_input=print_input); print()
        self.ppkstats=ppkstats
        return ppkstats

    def plot_rkrk(self, ax, k_index=0, res2_unit=1e-12, \
                  color='red', fmt='.', markersize=1, alpha=0.3, elinewidth=1, capsize=1):
        stats_keys=['E1', 'V2', 'S3', 'K4']
        ensemble_stats_keys=['akak_ensemble_stats', 'bkbk_ensemble_stats']
        ppkstats=self.ppkstats
        for row, stats_key in enumerate(stats_keys):
            for col, ensemble_stats_key in enumerate(ensemble_stats_keys):
                for i in np.arange(1, ppkstats['input']['na_bins']):
                    zeta_ensemble_i = ppkstats['angles_ensemble'][:, i]
                    stats_ensemble_i = ppkstats[ensemble_stats_key][stats_key][i, k_index, :]
                    ax[row, col].errorbar(np.mean(zeta_ensemble_i) * 180 / np.pi, \
                                          xerr=np.std(zeta_ensemble_i) * 180 / np.pi, \
                                          y=np.mean(stats_ensemble_i)/(res2_unit**(row+1)), \
                                          yerr=np.std(stats_ensemble_i)/(res2_unit**(row+1)), \
                                          color=color, fmt=fmt, markersize=markersize, alpha=alpha, \
                                          ecolor=color, elinewidth=elinewidth, capsize=capsize)

    def plot_rkrk_model(self, ax, ra2, Erarb=Erarb_gwb, Vrarb=Vrarb_gwb, \
                        S3rarb=S3rarb_gwb, K4rarb=K4rarb_gwb, \
                        res2_unit=1e-12, ls='-', lw=1, color='blue', alpha=0.5, \
                        plot_one_point=False, one_point_marker='o', label=None):
        
        zta_vals=np.logspace(-3, np.log10(np.pi), 1000)
        for i in range(2):
            ax[0,i].plot(zta_vals*180/np.pi, ra2*Erarb(zta_vals)/res2_unit, \
                         ls=ls, lw=lw, color=color, alpha=alpha, label=label)
            ax[1,i].plot(zta_vals*180/np.pi, (ra2**2)*Vrarb(zta_vals)/(res2_unit**2), \
                         ls=ls, lw=lw, color=color, alpha=alpha, label=label)
            ax[2,i].plot(zta_vals*180/np.pi, (ra2**3)*S3rarb(zta_vals)/(res2_unit**3), \
                         ls=ls, lw=lw, color=color, alpha=alpha, label=label)
            ax[3,i].plot(zta_vals*180/np.pi, (ra2**4)*K4rarb(zta_vals)/(res2_unit**4), \
                         ls=ls, lw=lw, color=color, alpha=alpha, label=label)

        if plot_one_point:
            ax[0].plot(0, ra2*Erarb(0)/res2_unit, color=color, marker=one_point_marker)
            ax[1].plot(0, (ra2**2)*Vrarb(0)/(res2_unit**2), color=color, marker=one_point_marker)
            ax[2].plot(0, (ra2**3)*S3rarb(0)/(res2_unit**3), color=color, marker=one_point_marker)
            ax[3].plot(0, (ra2**4)*K4rarb(0)/(res2_unit**4), color=color, marker=one_point_marker)

    def get_rkrk_stats(self, stats_key='E1', ensemble_stats_key='akak_ensemble_stats', k_index=0):
        '''
        stats_key='E1', 'V2', 'S3', 'K4'
        ensemble_stats_key='akak_ensemble_stats', 'bkbk_ensemble_stats'
        '''
        ppkstats=self.ppkstats
        zeta_stats=[] # angular bins
        rkrk_stats=[]
        for i in np.arange(1, ppkstats['input']['na_bins']):
            zeta_ensemble_i = ppkstats['angles_ensemble'][:, i]
            stats_ensemble_i = ppkstats[ensemble_stats_key][stats_key][i, k_index, :]
            zeta_stats.append(zeta_ensemble_i)
            rkrk_stats.append(stats_ensemble_i)
        return zeta_stats, rkrk_stats

    def plot_rkrk_density(self, ax, stats_key='E1', ensemble_stats_key='akak_ensemble_stats', \
                          k_index=0, res2_unit=1e-12, zinds=[0,2,12,13], \
                          bins=15, histtype='step', ls='-', density=True, alpha=0.8):
        zeta_stats, rkrk_stats=self.get_rkrk_stats(stats_key=stats_key, \
                                                   ensemble_stats_key=ensemble_stats_key, \
                                                   k_index=k_index)
        stats_keys=['E1', 'V2', 'S3', 'K4']
        stats_key_idx=stats_keys.index(stats_key)
        line_widths=np.arange(0.5, 0.5*(len(zinds)+1), 0.5)
        for idx, (i, lw) in enumerate(zip(zinds, line_widths)):
            rkrk_stat = rkrk_stats[i]
            ax.hist(rkrk_stat/(res2_unit**(stats_key_idx+1)), \
                    bins=bins, histtype=histtype, lw=lw, ls=ls, density=density, alpha=0.8, \
                    label=fr'$\zeta\sim${int(np.mean(zeta_stats[i])*180/np.pi)}$^\circ$')