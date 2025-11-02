import numpy as np
from levy_1d import *


if __name__ == "__main__":

    # b&p parameters
    input_dict={'l_0': 0.0004, 'alpha_0': 1/2, 'n_steps_per_pixel': 3, \
                'smoothing_scale_per_pix': 4, \
                'n_bins_density': 300, 'num_sims': 500}
    box_widths=np.arange(50000, 1000000+1, 50000)
    save_pickle=True
    fname='s3_1d_box_systematics_bp24params'
    s3_box_systematics=compute_s3_systematics(box_widths=box_widths, \
                                              input_dict=input_dict, \
                                              save_pickle=True, fname=fname)
    
    # near gaussian
    input_dict={'l_0': 0.1, 'alpha_0': 1/2, 'n_steps_per_pixel': 40, \
                'smoothing_scale_per_pix': 4, \
                'n_bins_density': 300, 'num_sims': 500}
    box_widths=np.arange(50000, 1000000+1, 50000)
    save_pickle=True
    fname='s3_1d_box_systematics_neargaussian'
    s3_box_systematics=compute_s3_systematics(box_widths=box_widths, \
                                              input_dict=input_dict, \
                                              save_pickle=True, fname=fname)
    