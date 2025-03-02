import os, pickle
import numpy as np
from scipy.ndimage import gaussian_filter
import mistree as mist
from multiprocess import Pool, cpu_count
from tqdm import tqdm
from levy_beyond_mean_field import PDF_rho_mf, PDF_rho_InvLaplaceMP, cgf_bmf, s3_exact_gaussian_filter


def run_mist_levy(l_0, alpha_0, n_steps_per_pixel, box_width, mode='2D', pixel_size=1, \
                  smoothing_scale_per_pix=1, n_bins_density=100, return_rho_density=False, periodic=True):
    
    # generate Levy flights
    grid_size = int( box_width/pixel_size )
    if mode=='2D':
        n_steps = int(n_steps_per_pixel*(grid_size**2))
        x, y=mist.get_levy_flight(n_steps, t_0=l_0, alpha=alpha_0, \
                                  box_size=box_width, periodic=periodic, mode=mode)
        density, _, _ = np.histogram2d(x, y, bins=grid_size, density=True)
    if mode=='3D':
        n_steps = int(n_steps_per_pixel*(grid_size**3))
        x, y, z=mist.get_levy_flight(n_steps, t_0=l_0, alpha=alpha_0, \
                                     box_size=box_width, periodic=periodic, mode=mode)
        sample = np.vstack((x, y, z)).T
        density, _ = np.histogramdd(sample, bins=grid_size, density=True)

    # gaussian smoothing to the density map
    density_smoothed = gaussian_filter((density-np.mean(density))/np.mean(density), \
                                       sigma=smoothing_scale_per_pix*pixel_size)+1
    
    # calculate mean of the density field
    mean_density = np.mean(density_smoothed)
    rho_density = density_smoothed/mean_density
    xi_0 = np.var(rho_density) # local density variance, P&B2024
    s3 = np.mean((rho_density - np.mean(rho_density))**3)/(xi_0**2)

    # flatten density field to calculate PDF
    rho_density_flat = rho_density.flatten()
    pdf_rho_density, bins_rho_density = \
    np.histogram(rho_density_flat, bins=n_bins_density, density=True)
    if return_rho_density==False:
        return bins_rho_density, pdf_rho_density, xi_0, s3
    else:
        return bins_rho_density, pdf_rho_density, xi_0, s3, rho_density

def run_mist_levy_simulation(args):
    l_0, alpha_0, n_steps_per_pixel, box_width, mode, \
    pixel_size, smoothing_scale_per_pix, n_bins_density = args
    return run_mist_levy(l_0, alpha_0, n_steps_per_pixel, box_width, mode=mode, \
                         pixel_size=pixel_size, smoothing_scale_per_pix=smoothing_scale_per_pix, \
                         n_bins_density=n_bins_density)

def init_worker():
    # use a unique random seed for each process
    np.random.seed(os.getpid())

def run_mist_levys(l_0, alpha_0, n_steps_per_pixel, box_width, mode='2D', \
                   pixel_size=1, smoothing_scale_per_pix=1, n_bins_density=100, num_sims=50, \
                   bmf=0, lmd_max=1e3, dl=0.025, h=1e-10, ncpus=cpu_count(), \
                   save_data=False, fname='my_levy'):
    
    # Prepare the input dictionary
    input_dict = {
        'l_0': l_0, \
        'alpha_0': alpha_0, \
        'n_steps_per_pixel': n_steps_per_pixel, \
        'box_width': box_width, \
        'mode': mode, \
        'pixel_size': pixel_size, \
        'smoothing_scale_per_pix': smoothing_scale_per_pix, \
        'n_bins_density': n_bins_density, \
        'num_sims': num_sims, \
        'bmf': bmf, \
        'lmd_max': lmd_max, \
        'dl': dl, \
        'h': h, \
        'ncpus': ncpus \
    }

    # prepare the arguments for parallel execution
    args = [(l_0, alpha_0, n_steps_per_pixel, box_width, mode, \
             pixel_size, smoothing_scale_per_pix, n_bins_density) for _ in range(num_sims)]

    # use multiprocessing Pool for parallel processing with tqdm progress bar
    with Pool(ncpus, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap(run_mist_levy_simulation, args), total=num_sims, desc="Simulations"))

    all_pdf_rho, all_bin_centers_rho, all_xi_0, all_s3 = [], [], [], []
    all_pdf_rho_mf = []
    all_pdf_rho_bmf = []
    
    for bins_rho, pdf_rho, xi_0, s3 in tqdm(results, total=len(results), desc="Processing Results"):
        bin_centers_rho = (bins_rho[:-1] + bins_rho[1:]) / 2
        all_pdf_rho.append(pdf_rho)
        all_bin_centers_rho.append(bin_centers_rho)
        all_xi_0.append(xi_0)
        all_s3.append(s3)

        pdf_rho_mf_xi_0=PDF_rho_mf(bin_centers_rho, xi_0)
        all_pdf_rho_mf.append(pdf_rho_mf_xi_0)

        if mode=='2D':
            pdf_rho_bmf_xi_0=PDF_rho_InvLaplaceMP(cgf=lambda x: cgf_bmf(x, alpha=alpha_0, bmf=bmf, dim=2), \
                                                  rhos=bin_centers_rho, xi_0=xi_0, \
                                                  lmd_max=lmd_max, dl=dl, h=h)
        if mode=='3D':
            pdf_rho_bmf_xi_0=PDF_rho_InvLaplaceMP(cgf=lambda x: cgf_bmf(x, alpha=alpha_0, bmf=bmf, dim=3), \
                                                  rhos=bin_centers_rho, xi_0=xi_0, \
                                                  lmd_max=lmd_max, dl=dl, h=h)
        all_pdf_rho_bmf.append(pdf_rho_bmf_xi_0)
    
    # Convert lists to numpy arrays for consistency
    all_pdf_rho = np.array(all_pdf_rho)
    all_bin_centers_rho = np.array(all_bin_centers_rho)
    all_xi_0 = np.array(all_xi_0)
    all_s3 = np.array(all_s3)

    all_pdf_rho_mf=np.array(all_pdf_rho_mf)
    all_pdf_rho_bmf=np.array(all_pdf_rho_bmf)

    # save data to a pickle file if save_data is True
    if save_data:
        data_dict = {
            'all_pdf_rho': all_pdf_rho, \
            'all_bin_centers_rho': all_bin_centers_rho, \
            'all_xi_0': all_xi_0, \
            'all_s3': all_s3, \
            'all_pdf_rho_mf': all_pdf_rho_mf, \
            'all_pdf_rho_bmf': all_pdf_rho_bmf, \
            'input': input_dict \
        }
        with open(f"data/{fname}.pkl", "wb") as f:
            pickle.dump(data_dict, f)

    return (all_pdf_rho, all_bin_centers_rho, all_xi_0, all_s3, \
            all_pdf_rho_mf, all_pdf_rho_bmf)
            
