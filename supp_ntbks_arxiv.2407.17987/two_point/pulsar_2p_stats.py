import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis, moment
from tqdm import tqdm
from multiprocess import Pool, cpu_count
import sys, os, pickle
from make_fpta_sims import *


def get_fft(t, r, Tspan_yr=30, n_bins=30, xx=10):
    # Mask for finite (non-NaN) values
    mask = np.isfinite(r)
    
    # Interpolation using scipy's interp1d
    ts=t-np.min(t) # corrects for phase in fft calculation
    interp_func = interp1d(ts[mask], r[mask], kind='cubic', fill_value="extrapolate")
    t_interp = np.linspace(0, (Tspan_yr*365.25), len(ts) * xx)
    r_interp = interp_func(t_interp)
    
    # Perform FFT-like calculation
    N = len(t_interp)
    a0_fft = np.sum(r_interp) * 2 / N
    a_fft = np.array([np.sum(r_interp * np.sin(2 * np.pi * k * np.arange(N) / N)) for k in np.arange(1, n_bins + 1)]) * 2 / N
    b_fft = np.array([np.sum(r_interp * np.cos(2 * np.pi * k * np.arange(N) / N)) for k in np.arange(1, n_bins + 1)]) * 2 / N

    # Return frequencies, 1 yr^-1 = 31.7 nHz
    yrinv_to_nhz = 31.7
    f0 = (1 / Tspan_yr) * yrinv_to_nhz
    freqs = np.arange(1, n_bins + 1) * f0
    
    return a0_fft, a_fft, b_fft, freqs


def compute_eaeb(psr_a, psr_b, eta_eaeb=1e-10):
    pos_a=psr_b.pos
    pos_b=psr_a.pos
    angle = np.arccos(np.dot(pos_a, pos_b)-eta_eaeb)
    return angle


def bin_res_pairs_in_k(psrs, na_bins=15, n_bins=14, tmin_mjd=0):
    # Get the minimum and maximum TOAs for all pulsars
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)
    Tspan_yr = np.round(Tspan / 86400 / 365.25, 1)

    # Define angular bins
    angular_bin_edges = np.linspace(0, np.pi, na_bins + 1)
    angular_bins = [[] for _ in range(na_bins + 1)]  # Include extra bin for zeroth angular bin

    # Compute angles between each pair of pulsars and bin them
    for i, psr_a in enumerate(psrs):
        for j, psr_b in enumerate(psrs):
            if i == j:
                angular_bins[0].append((psr_a, psr_b))  # Add to zeroth angular bin
            elif i < j:
                angle = compute_eaeb(psr_a, psr_b)
                bin_index = np.digitize(angle, angular_bin_edges) - 1 + 1  # Adjust index for zeroth bin
                angular_bins[bin_index].append((psr_a, psr_b))

    # Prepare the containers for the results
    a0a0_bins = [[] for _ in range(na_bins + 1)]
    akak_bins = [[] for _ in range(na_bins + 1)]
    bkbk_bins = [[] for _ in range(na_bins + 1)]
    zeta_bins = [[] for _ in range(na_bins + 1)]

    # Bin each pulsar in frequency within each angular bin
    for bin_idx, angular_bin in enumerate(angular_bins):
        for psr_a, psr_b in angular_bin:
            # Frequency binning for psr_a
            a0_psr_a, a_psr_a, b_psr_a, _ = \
                get_fft(tmin_mjd + psr_a.toas / 86400, psr_a.residuals, Tspan_yr=Tspan_yr, n_bins=n_bins)
            # Frequency binning for psr_b
            a0_psr_b, a_psr_b, b_psr_b, _ = \
                get_fft(tmin_mjd + psr_b.toas / 86400, psr_b.residuals, Tspan_yr=Tspan_yr, n_bins=n_bins)

            # Collect the results
            a0a0_bins[bin_idx].append(a0_psr_a * a0_psr_b)
            akak_bins[bin_idx].append(a_psr_a * a_psr_b)
            bkbk_bins[bin_idx].append(b_psr_a * b_psr_b)
            zeta_bins[bin_idx].append(compute_eaeb(psr_a, psr_b))

    # Convert results to numpy arrays
    a0a0_bins = [np.array(a0a0_bin) for a0a0_bin in a0a0_bins]
    akak_bins = [np.array(akak_bin) for akak_bin in akak_bins]
    bkbk_bins = [np.array(bkbk_bin) for bkbk_bin in bkbk_bins]
    zeta_bins = [np.array(zeta_bin) for zeta_bin in zeta_bins]

    # Return frequencies, 1 yr^-1 = 31.7 nHz
    yrinv_to_nhz = 31.7
    f0 = (1 / Tspan_yr) * yrinv_to_nhz
    freqs_bins = np.arange(1, n_bins + 1) * f0

    return freqs_bins, zeta_bins, a0a0_bins, akak_bins, bkbk_bins

def initialize_stats_dict(n_angular_bins, n_freq_bins):
    stats_dict = {'E1': [[] for _ in range(n_angular_bins)],
                  'V2': [[] for _ in range(n_angular_bins)],
                  'S3': [[] for _ in range(n_angular_bins)],
                  'K4': [[] for _ in range(n_angular_bins)]}
    for key in stats_dict:
        stats_dict[key] = [[[] for _ in range(n_freq_bins)] for _ in range(n_angular_bins)]
    return stats_dict

# def update_stats(stats_dict, akak_bins):
#     for i, akak_bin in enumerate(akak_bins):  # loop over angular separation
#         for k in range(akak_bin.shape[1]):  # loop over frequency bins
#             E1 = np.mean(akak_bin[:, k])
#             V2 = np.var(akak_bin[:, k])
#             S3 = moment(akak_bin[:, k], moment=3)
#             K4 = moment(akak_bin[:, k], moment=4)
#             stats_dict['E1'][i][k].append(E1)
#             stats_dict['V2'][i][k].append(V2)
#             stats_dict['S3'][i][k].append(S3)
#             stats_dict['K4'][i][k].append(K4)

def update_stats(stats_dict, akak_bins):
    for i, akak_bin in enumerate(akak_bins):  # loop over angular separation
        if akak_bin.size == 0:
            continue  # Skip empty bins
        for k in range(akak_bin.shape[1]):  # loop over frequency bins
            E1 = np.mean(akak_bin[:, k])
            V2 = np.var(akak_bin[:, k])
            S3 = moment(akak_bin[:, k], moment=3)
            K4 = moment(akak_bin[:, k], moment=4)
            stats_dict['E1'][i][k].append(E1)
            stats_dict['V2'][i][k].append(V2)
            stats_dict['S3'][i][k].append(S3)
            stats_dict['K4'][i][k].append(K4)


# parallelized version of pulsar pair angular and frequency binning
def bin_pairs(pulsar_indices, psrs, angular_bin_edges):
    i, j = pulsar_indices
    psr_a, psr_b = psrs[i], psrs[j]
    if i == j:
        return (0, (psr_a, psr_b))  # Add to zeroth angular bin
    elif i < j:
        angle = compute_eaeb(psr_a, psr_b)
        bin_index = np.digitize(angle, angular_bin_edges) - 1 + 1  # Adjust index for zeroth bin
        return (bin_index, (psr_a, psr_b))
    else:
        return None

def process_angular_bin(args):
    bin_idx, angular_bin, Tspan_yr, n_bins, tmin_mjd = args
    a0a0_bin, akak_bin, bkbk_bin, zeta_bin = [], [], [], []

    for psr_a, psr_b in angular_bin:
        # Frequency binning for psr_a
        a0_psr_a, a_psr_a, b_psr_a, _ = get_fft(tmin_mjd + psr_a.toas / 86400, psr_a.residuals, Tspan_yr=Tspan_yr, n_bins=n_bins)
        # Frequency binning for psr_b
        a0_psr_b, a_psr_b, b_psr_b, _ = get_fft(tmin_mjd + psr_b.toas / 86400, psr_b.residuals, Tspan_yr=Tspan_yr, n_bins=n_bins)

        # Collect the results
        a0a0_bin.append(a0_psr_a * a0_psr_b)
        akak_bin.append(a_psr_a * a_psr_b)
        bkbk_bin.append(b_psr_a * b_psr_b)
        zeta_bin.append(compute_eaeb(psr_a, psr_b))

    return bin_idx, np.array(a0a0_bin), np.array(akak_bin), np.array(bkbk_bin), np.array(zeta_bin)

def bin_res_pairs_in_k_mp(psrs, na_bins=15, n_bins=14, tmin_mjd=0, show_progress=False):
    # Get the minimum and maximum TOAs for all pulsars
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)
    Tspan_yr = np.round(Tspan / 86400 / 365.25, 1)

    # Define angular bins
    angular_bin_edges = np.linspace(0, np.pi, na_bins + 1)
    angular_bins = [[] for _ in range(na_bins + 1)]  # Include extra bin for zeroth angular bin

    # Compute angles between each pair of pulsars and bin them
    pulsar_indices = [(i, j) for i in range(len(psrs)) for j in range(len(psrs)) if i <= j]
    
    if show_progress:
        with Pool(cpu_count()) as pool:
            results = list(tqdm(pool.starmap(bin_pairs, \
                                             [(indices, psrs, angular_bin_edges) for indices in pulsar_indices]), \
                                total=len(pulsar_indices), desc="binning pairs"))
    else:
        with Pool(cpu_count()) as pool:
            results = pool.starmap(bin_pairs, [(indices, psrs, angular_bin_edges) for indices in pulsar_indices])

    for result in results:
        if result is not None:
            bin_index, psr_pair = result
            angular_bins[bin_index].append(psr_pair)

    # Prepare the containers for the results
    a0a0_bins = [[] for _ in range(na_bins + 1)]
    akak_bins = [[] for _ in range(na_bins + 1)]
    bkbk_bins = [[] for _ in range(na_bins + 1)]
    zeta_bins = [[] for _ in range(na_bins + 1)]

    # Bin each pulsar in frequency within each angular bin
    args = [(bin_idx, angular_bin, Tspan_yr, n_bins, tmin_mjd) for bin_idx, angular_bin in enumerate(angular_bins)]
    
    if show_progress:
        with Pool(cpu_count()) as pool:
            results = list(tqdm(pool.map(process_angular_bin, args), \
                                total=len(args), desc="processing ak dot ak, bk dot bk in angular bins"))
    else:
        with Pool(cpu_count()) as pool:
            results = pool.map(process_angular_bin, args)

    for bin_idx, a0a0_bin, akak_bin, bkbk_bin, zeta_bin in results:
        a0a0_bins[bin_idx] = a0a0_bin
        akak_bins[bin_idx] = akak_bin
        bkbk_bins[bin_idx] = bkbk_bin
        zeta_bins[bin_idx] = zeta_bin

    # Return frequencies, 1 yr^-1 = 31.7 nHz
    yrinv_to_nhz = 31.7
    f0 = (1 / Tspan_yr) * yrinv_to_nhz
    freqs_bins = np.arange(1, n_bins + 1) * f0

    return freqs_bins, zeta_bins, a0a0_bins, akak_bins, bkbk_bins


# master function
def ppkstats_master(npsrs=100, Tspan_yr=30, ntoas=100, toaerr=1e-6, \
                    add_gwb=False, gamma_gw=13/3, log10_A_gw=np.log10(2.4e-15), \
                    add_red=False, rn_input_dict={'log10_A_min':-16, 'log10_A_max': -13, \
                                                  'gamma_min': 2, 'gamma_max': 6}, \
                    add_white=False, n_bins=14, na_bins=15, n_sims=10, \
                    tmin_mjd=0, save_pickle=False, pkl_name='my_pickle', show_progress=True):
    
    n_angular_bins = na_bins+1  # na_bins + 1 for the zeroth angular bin (auto-power)
    n_freq_bins = n_bins

    akak_ensemble_stats = initialize_stats_dict(n_angular_bins, n_freq_bins)
    bkbk_ensemble_stats = initialize_stats_dict(n_angular_bins, n_freq_bins)

    angles_ensemble=[]
    freqs_ensemble=[]

    # loop over simulations
    for i in range(n_sims):
        # Redirect stdout and stderr to suppress fakepta output
        devnull = open(os.devnull, 'w')
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            psrs=make_fpta_sims(npsrs=npsrs, Tobs=Tspan_yr, ntoas=ntoas, toaerr=toaerr, \
                                add_gwb=add_gwb, gamma_gw=gamma_gw, log10_A_gw=log10_A_gw, \
                                add_red=add_red, rn_input_dict=rn_input_dict, \
                                add_white=add_white, save_pickle=False)
        finally:
            # Reset stdout and stderr to their original values
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        print(f'sim-{i} angular and frequency binning')
        freqs_bins, zeta_bins, a0a0_bins, akak_bins, bkbk_bins=\
        bin_res_pairs_in_k_mp(psrs, na_bins=na_bins, n_bins=n_bins, \
                              tmin_mjd=tmin_mjd, show_progress=show_progress)
        print()
        angles_ensemble.append(np.array([np.mean(zeta_bin) for zeta_bin in zeta_bins]))
        freqs_ensemble.append(freqs_bins)
        update_stats(akak_ensemble_stats, akak_bins)
        update_stats(bkbk_ensemble_stats, bkbk_bins)

    for key in akak_ensemble_stats:
        akak_ensemble_vals=np.array(akak_ensemble_stats[key])
        akak_ensemble_stats[key] = akak_ensemble_vals

    for key in bkbk_ensemble_stats:
        bkbk_ensemble_vals=np.array(bkbk_ensemble_stats[key])
        bkbk_ensemble_stats[key] = bkbk_ensemble_vals

    angles_ensemble=np.array(angles_ensemble)
    freqs_ensemble=np.array(freqs_ensemble)

    # input dictionary
    input_dict={'npsrs': npsrs, 'Tspan_yr': Tspan_yr, 'ntoas': ntoas, 'toaerr': toaerr, \
                'add_gwb': add_gwb, 'gamma_gw': gamma_gw, 'log10_A_gw': log10_A_gw, \
                'add_red': add_red, 'rn_input_dict': rn_input_dict, 'add_white': add_white, \
                'n_bins': n_bins, 'na_bins': na_bins, 'n_sims': n_sims, 'tmin_mjd': tmin_mjd}

    ppkstats_dict={'akak_ensemble_stats': akak_ensemble_stats, \
                   'bkbk_ensemble_stats': bkbk_ensemble_stats, \
                   'angles_ensemble': angles_ensemble, \
                   'freqs_ensemble': freqs_ensemble, 'input': input_dict}

    if save_pickle:
        save_ppkstats(ppkstats_dict, pkl_name=pkl_name)
    print(f'computed pair-stats for {Tspan_yr} yr-fpta with {npsrs} MSPs and {n_sims} sims')    
    return ppkstats_dict

def save_ppkstats(ppkstats_dict, pkl_name='my_pickle'):
    dir_path = './data'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    input_dict=ppkstats_dict['input']
    if input_dict['add_gwb']:
        gwb_binary = 1
    else:
        gwb_binary = 0

    if input_dict['add_red']:
        rn_binary = 1
    else:
        rn_binary = 0
    npsrs=input_dict['npsrs']
    n_sims=input_dict['n_sims']
    
    # Save the pickle file
    pkl_file = pkl_name + f'_npsrs{npsrs}_nsims{n_sims}_gwb{gwb_binary}_rn{rn_binary}'
    with open(f'{dir_path}/{pkl_file}.pkl', 'wb') as f:
        pickle.dump(ppkstats_dict, f)

def load_ppkstats(pkl_name, npsrs, n_sims, gwb_binary, rn_binary, print_input=True):
    dir_path = './data'
    pkl_file = pkl_name + f'_npsrs{npsrs}_nsims{n_sims}_gwb{gwb_binary}_rn{rn_binary}'
    pickle_loc = os.path.join(dir_path, pkl_file + '.pkl')
    if os.path.exists(pickle_loc):
        with open(pickle_loc, 'rb') as f:
            ppkstats_dict = pickle.load(f)
        if print_input:
            input_dict=ppkstats_dict['input']
            Tspan_yr=input_dict['Tspan_yr']
            print(f'loaded pre-computed pair-stats for {Tspan_yr} yr-fpta with {npsrs} MSPs and {n_sims} sims')
            print(ppkstats_dict['input'])
        return ppkstats_dict
    else:
        print('pre-computed stats do not exist')


# testing gaussianity codes

def C_hd(zeta):
    eps=1e-50
    x=np.cos(zeta)
    y=(1-x)/2
    return (1/2)-(y/4)+(3/2)*y*np.log(y + eps)

def Erarb_gwb(zeta):
    cab = C_hd(zeta)
    caa = 1

    if isinstance(zeta, np.ndarray):
        result = np.where(zeta < 1e-10, caa, cab)
    else:
        if zeta < 1e-2:
            result = caa
        else:
            result = cab
            
    return result

def Vrarb_gwb(zeta):
    cab=Erarb_gwb(zeta)
    return 1 + cab**2

# non variance normalized moments
def S3rarb_gwb(zeta):
    cab=Erarb_gwb(zeta)
    return 2*cab*(3 + cab**2)

def K4rarb_gwb(zeta):
    cab=Erarb_gwb(zeta)
    return 3*(3 + 14*(cab**2) + 3*(cab**4))

# gaussian noise model (uncorrelated)

def Erarb_noise(zeta):
    cab = 0
    caa = 1
    eps_zeta=1e-10
    if isinstance(zeta, np.ndarray):
        result = np.where(zeta < eps_zeta, caa, cab)
    else:
        if zeta < eps_zeta:
            result = caa
        else:
            result = cab
    return result

def Vrarb_noise(zeta):
    Eab=Erarb_noise(zeta)
    return 1 + Eab**2

# non variance normalized moments
def S3rarb_noise(zeta):
    Eab=Erarb_noise(zeta)
    return 2*Eab*(3 + Eab**2)

def K4rarb_noise(zeta):
    Eab=Erarb_noise(zeta)
    return 3*(3 + 14*(Eab**2) + 3*(Eab**4))
