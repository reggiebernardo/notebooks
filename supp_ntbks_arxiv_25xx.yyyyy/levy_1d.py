import os, pickle
import numpy as np
from scipy.special import hyp0f1, gamma, hyp2f1
from scipy.ndimage import gaussian_filter

from multiprocess import Pool, cpu_count
from tqdm import tqdm


# 1 levy mean field theory

def rhyp0f1(a, z):
    return hyp0f1(a, z)/gamma(a)

def PDF_rho_mf(rho, xi_0):
    '''Returns the mean field theoretical PDF of the density'''
    f1 = 4/(xi_0**2)
    f2 = np.exp(-2*(rho + 1)/xi_0)
    f3 = rhyp0f1(2, 4*rho/(xi_0**2))
    return f1*f2*f3


# 2 levy beyond mean field theory

# 1d bmf alpha=1/2 n=7 b&p2024
def cgf_bmf_7(x):
    lmd,xi=x
    num=-3*lmd*(975*lmd**3*xi**3 - 892590*lmd**2*xi**2 + 43255808*lmd*xi - 268435456)
    den=960*lmd**4*xi**4 - 1020557*lmd**3*xi**3 + 60312586*lmd**2*xi**2 - 532420608*lmd*xi + 805306368
    return num/den

# 1d bmf alpha=1/2 n=5 b&p2024
def cgf_bmf_5(x):
    lmd,xi=x
    return -3*lmd*(5*lmd*xi*(15*lmd*xi - 1648) + 65536)/(lmd*xi*(lmd*xi*(80*lmd*xi - 10849) + 123024) - 196608)

# 1d bmf alpha=1/2 n=3 b&p2024
def cgf_bmf_3(x):
    lmd,xi=x
    return -lmd*(5*lmd*xi - 64)/(2*lmd**2*xi**2 - 37*lmd*xi + 64)

def cgf_mf(x):
    lmd,xi=x
    return 2*lmd/(2 - lmd*xi)

def finite_diff(f, x, h=1e-10):
    x1=np.array(x, dtype=np.complex128); x2=np.array(x, dtype=np.complex128)
    x1[0]+=h; x2[0]-=h
    return (f(x1) - f(x2))/(2*h)

def PDF_rho_InvLaplace(cgf, rho, xi_0, lmd_max=1e3, dl=0.025, h=0.025):
    lmd_vals = np.arange(-lmd_max - dl/2, lmd_max + dl/2 + 1, dl)*1j
    phi_vals = np.array([cgf([lmd_val, xi_0]) for lmd_val in lmd_vals])
    dL_phi_vals = np.array([finite_diff(cgf, [lmd_val, xi_0], h=h) for lmd_val in lmd_vals])
    rho_PDF = np.sum(1j*dl*np.exp(-lmd_vals*rho + phi_vals)*dL_phi_vals) / (2j * np.pi * rho)
    return np.real(rho_PDF)

def PDF_rho_InvLaplaceMP(cgf, rhos, xi_0, num_workers=None, lmd_max=1e3, dl=0.025, h=0.025):
    if num_workers is None:
        num_workers = cpu_count()  # use all available CPU cores by default

    # create a pool of workers
    with Pool(processes=num_workers) as pool:
        # map the function over the rhos array in parallel
        pdf_density = pool.starmap(PDF_rho_InvLaplace, \
                                   [(cgf, rho, xi_0, lmd_max, dl, h) for rho in rhos])

    return np.array(pdf_density)

def s3_exact(alpha):
    ns=-alpha
    return (3/2)*hyp2f1((1+ns)/2,(1+ns)/2, 1/2, 1/4)


# 3 levy sims

def random_walk_fast_1d(step_size, box_size, x_start, periodic, length):
    x = np.zeros(length + 1)  # Initialize array to hold coordinates
    x[0] = x_start
    x_current = x_start
    for i in range(1, length + 1):
        delta_x = step_size[i - 1]
        x_current += delta_x
        if periodic == 1:
            x_current = x_current % box_size  # Use modulo for periodic boundary
        x[i] = x_current
    return x

def get_random_flight_1d(steps, box_size=75., periodic=True):
    _size = len(steps)
    if periodic is False:
        box_size = None
    if box_size is None:
        _x_start = 0.
        _periodic = 0
        box_size = 0.
    else:
        _x_start = np.floor(box_size/2) # np.random.uniform(0., box_size, 1)[0]
        _periodic = 1
    x = random_walk_fast_1d(steps, box_size, _x_start, _periodic, _size)
    return x

def get_levy_flight_1d(size, periodic=True, box_size=75., t_0=0.2, alpha=1.5):
    _u = np.random.uniform(0., 1., size - 1)
    # _steps = t_0 / (1. - _u) ** (1. / alpha)
    _steps=t_0*(np.random.uniform(0., 2., size-1)-1)/(_u**(1./alpha))
    if periodic == False:
        box_size = None
    x = get_random_flight_1d(_steps, box_size=box_size)
    return x

def run_levy_1d(l_0, alpha_0, n_steps_per_pixel, box_width, \
                smoothing_scale_per_pix=1, n_bins_density=100, return_rho_1d=False):
    grid_size = box_width
    pixel_size = box_width/grid_size
    n_steps = n_steps_per_pixel*int(box_width/pixel_size)
    
    # generate Levy flight using mistree
    x = get_levy_flight_1d(n_steps, t_0=l_0, alpha=alpha_0, box_size=box_width, periodic=True)
    density, x_edges = np.histogram(x, bins=grid_size, density=True)

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
    if return_rho_1d==False:
        return bins_rho_density, pdf_rho_density, xi_0, s3
    else:
        return bins_rho_density, pdf_rho_density, xi_0, s3, rho_density

def run_levy_simulation(args):
    l_0, alpha_0, n_steps_per_pixel, box_width, \
    smoothing_scale_per_pix, n_bins_density = args
    return run_levy_1d(l_0, alpha_0, n_steps_per_pixel, box_width, \
                       smoothing_scale_per_pix=smoothing_scale_per_pix, \
                       n_bins_density=n_bins_density)

def run_levys_1d(l_0, alpha_0, n_steps_per_pixel, box_width, \
                 smoothing_scale_per_pix=1, n_bins_density=100, num_sims=50):
    # Prepare the arguments for parallel execution
    args = [(l_0, alpha_0, n_steps_per_pixel, box_width, \
             smoothing_scale_per_pix, n_bins_density) for _ in range(num_sims)]

    # Use multiprocessing Pool for parallel processing with tqdm progress bar
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(run_levy_simulation, args), total=num_sims, desc="Simulations"))

    all_pdf_rho, all_bin_centers_rho, all_xi_0, all_s3 = [], [], [], []
    all_pdf_rho_minus_mf, all_pdf_rho_over_mf = [], []
    
    for bins_rho, pdf_rho, xi_0, s3 in results:
        bin_centers_rho = (bins_rho[:-1] + bins_rho[1:]) / 2
        all_pdf_rho.append(pdf_rho)
        all_bin_centers_rho.append(bin_centers_rho)
        all_xi_0.append(xi_0)
        all_s3.append(s3)
        all_pdf_rho_minus_mf.append(pdf_rho - PDF_rho_mf(bin_centers_rho, xi_0))
        all_pdf_rho_over_mf.append(pdf_rho/PDF_rho_mf(bin_centers_rho, xi_0))
    
    # Convert lists to numpy arrays for consistency
    all_pdf_rho = np.array(all_pdf_rho)
    all_bin_centers_rho = np.array(all_bin_centers_rho)
    all_xi_0 = np.array(all_xi_0)
    all_s3 = np.array(all_s3)
    all_pdf_rho_minus_mf = np.array(all_pdf_rho_minus_mf)
    all_pdf_rho_over_mf = np.array(all_pdf_rho_over_mf)
    
    return all_pdf_rho, all_bin_centers_rho, all_xi_0, all_s3, all_pdf_rho_minus_mf, all_pdf_rho_over_mf


# 4 compute box size systematics

def compute_s3_systematics(box_widths=np.arange(50000, 1000000+1, 50000), \
                           input_dict={'l_0': 0.0004, 'alpha_0': 1/2, 'n_steps_per_pixel': 3, \
                                       'smoothing_scale_per_pix': 4, \
                                       'n_bins_density': 300, 'num_sims': 100}, \
                           save_pickle=False, fname='s3_systematics_test'):

    # # b&p parameters
    l_0 = input_dict['l_0']; alpha_0 = input_dict['alpha_0']
    n_steps_per_pixel=input_dict['n_steps_per_pixel']
    smoothing_scale_per_pix=input_dict['smoothing_scale_per_pix']
    n_bins_density=input_dict['n_bins_density']; num_sims=input_dict['num_sims']

    xi_0_vs_box_width=[]
    s3_vs_box_width=[]
    for box_width in box_widths:
        _, _, all_xi_0, all_s3, _, _ = \
        run_levys_1d(l_0 = l_0, alpha_0 = alpha_0, n_steps_per_pixel = n_steps_per_pixel, \
                     box_width = box_width, smoothing_scale_per_pix=smoothing_scale_per_pix, \
                     n_bins_density=n_bins_density, num_sims = num_sims)
        xi_0_vs_box_width.append([ np.mean(all_xi_0), np.std(all_xi_0) ])
        s3_vs_box_width.append([ np.mean(all_s3), np.std(all_s3) ])

    xi_0_vs_box_width=np.array(xi_0_vs_box_width)
    s3_vs_box_width=np.array(s3_vs_box_width)

    combined_array = np.column_stack((box_widths, \
                                      xi_0_vs_box_width[:, 0], xi_0_vs_box_width[:, 1], \
                                      s3_vs_box_width[:, 0], s3_vs_box_width[:, 1] ))

    s3_box_systematics = {'s3_box_systematics': combined_array, 'input': input_dict}

    print("Run complete with input")
    print(input_dict)
    print()

    if save_pickle:
        # save to pickle file
        dir='./data'
        os.makedirs(dir, exist_ok=True)
        filepath = os.path.join(dir, fname + '.pkl')

        with open(filepath, 'wb') as f:
            pickle.dump(s3_box_systematics, f)

        print(f"Data saved at {filepath}")

    return s3_box_systematics


