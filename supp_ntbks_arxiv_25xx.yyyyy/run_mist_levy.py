import numpy as np
from mist_levy import *


if __name__ == "__main__":

    ##### 2D runs #####

    # use ncpus=4 for box~2000^2; ncpus=cpu_count() for box~500^2; for memory allocation
    alpha_0 = 3/2; n_steps_per_pixel=40
    pixel_size=0.5; smoothing_scale_per_pix=4
    box_width=2000; n_bins_density=100; num_sims = 50; ncpus=4

    # for l_0 in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]:
    for l_0 in [0.001]:
        all_pdf_rho, all_bin_centers_rho, all_xi_0, all_s3, all_pdf_rho_mf, all_pdf_rho_bmf = \
        run_mist_levys(mode='2D', l_0 = l_0, alpha_0 = alpha_0, n_steps_per_pixel = n_steps_per_pixel, \
                       box_width = box_width, pixel_size=pixel_size, smoothing_scale_per_pix=smoothing_scale_per_pix, \
                       n_bins_density=n_bins_density, num_sims = num_sims, bmf=3, ncpus=ncpus, \
                       save_data=True, fname=f'levy_2d_l0_{l_0}_lO4', lmd_max=1e4)
        print('local variance xi_0 = ', np.mean(all_xi_0), '+/-', np.std(all_xi_0))
        print('local skewness s3   = ', np.mean(all_s3), '+/-', np.std(all_s3))
        print()
