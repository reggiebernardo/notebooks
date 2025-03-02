import numpy as np
from scipy.special import hyp0f1, gamma, hyp2f1
from multiprocess import Pool, cpu_count


# # quick usage with full input to get PDF of rho
# dim=3 # dimension
# bmf=3 # beyond mean field order
# alpha=5/2 # correlation power ansatz
# xi_0=1 # local variance    

# # rho values
# rho_min=0.05 # minimum rho
# rho_max=20 # maximum rho
# nrho=100 # number of rho values

# # evaluate PDF rho with given input parameters
# rhos=np.linspace(rho_min, rho_max, nrho) # array of rho values
# pdf_3dnumerical=PDF_rho_InvLaplaceMP(cgf=lambda p: cgf_bmf(p, alpha=alpha, dim=dim, bmf=bmf), \
#                                      rhos=rhos, xi_0=xi_0, num_workers=None)


# 1 levy mean field theory

def rhyp0f1(a, z):
    return hyp0f1(a, z)/gamma(a)

def PDF_rho_mf(rho, xi_0):
    '''Returns the mean field theoretical PDF of the density'''
    f1 = 4/(xi_0**2)
    f2 = np.exp(-2*(rho + 1)/xi_0)
    f3 = rhyp0f1(2, 4*rho/(xi_0**2))
    return f1*f2*f3

def cgf_mf(x):
    lmd,xi=x
    return 2*lmd/(2 - lmd*xi)


# 2 levy beyond mean field theory

def x2_exact_gaussian_filter(alpha, dim=2):
    ns=-alpha
    
    if dim==1:
        return gamma((1+ns)/2)
    if dim==2:
        return (1/2)*gamma( 1 + (ns/2) )
    if dim==3:
        return (1/2)*gamma((3+ns)/2)
    return None

def s3_exact_gaussian_filter(alpha, dim=2):
    ns=-alpha

    if dim==1:
        return (3/2)*hyp2f1((1+ns)/2,(1+ns)/2, 1/2, 1/4)
    if dim==2:
        return (3/2)*hyp2f1((2+ns)/2,(2+ns)/2, 1, 1/4)
    if dim==3:
        prefact = -3/(2*( (2+ns)**2 ))
        t1 = 3*hyp2f1((3+ns)/2,(3+ns)/2,-1/2,1/4)
        t2 = 2*(1+ns)*hyp2f1((3+ns)/2,(3+ns)/2,1/2,1/4)
        return prefact*( t1 + t2 )

    return None

def s3_bmf(alpha, dim=2, bmf=0):
    # derived by expanding cgfs directly
    b=dim-alpha
    
    if bmf==0:
        return 3/2
    
    if dim==2:
        if bmf==1:
            return 3*b**2/32 + 3/2
        if bmf==2:
            return 3*b**4/2048 + 3*b**3/512 + 51*b**2/512 + 3/2
        if bmf==3:
            return b**6/98304 + b**5/8192 + 49*b**4/24576 + 7*b**3/1024 + 77*b**2/768 + 3/2
        if bmf==4:
            return b**8/25165824 + b**7/1048576 + 61*b**6/3145728 + 11*b**5/65536 + 3329*b**4/1572864 + 459*b**3/65536 + 9865*b**2/98304 + 3/2
    
    if dim==3:
        if bmf==1:
            return b**2/16 + 3/2
        if bmf==2:
            return b**4/1280 + b**3/320 + 21*b**2/320 + 3/2
        if bmf==3:
            return b**6/215040 + b**5/17920 + 11*b**4/10752 + b**3/280 + 443*b**2/6720 + 3/2
        if bmf==4:
            return b**8/61931520 + b**7/2580480 + 13*b**6/1548288 + b**5/13440 + 4153*b**4/3870720 + 587*b**3/161280 + 591*b**2/8960 + 3/2
        
    return None

def cgf_bmf(p, alpha=1/2, dim=2, bmf=0):
    l,x=p
    b=dim-alpha

    if bmf==0:
        return cgf_mf(p)

    if dim==2:
        if bmf==1:
            return -l*(b*l*x*(b + 2) - 32)/(l*x*(-b*(b - l*x + 2) - 16) + 32)
        if bmf==2:
            return l*(b*l*x*(b + 2)*(-b*l*x*(b + 2)*(b + 4) + 16*b*(b + 10) + 1408) - 32768)/(l*x*(b*(4*b*l**2*x**2*(b + 2) - l*x*(b*(b*(b*(b + 8) + 84) + 336) + 1408) + 16*(b + 2)*(b*(b + 10) + 88)) + 16384) - 32768)
        if bmf==3:
            return -l*(b*l*x*(b + 2)*(b**2*l**2*x**2*(b + 2)**2*(b + 4)**2*(b + 6) - 4*b*l*x*(b + 2)*(b + 4)*(b*(b*(b*(b + 24) + 468) + 4144) + 21120) + 4096*b*(b*(b*(b + 28) + 428) + 2672) + 59768832) - 1207959552)/(l*x*(24*b**3*l**3*x**3*(b + 2)**2*(b + 4) - b**2*l**2*x**2*(b + 2)*(b*(b*(b*(b*(b*(b + 18) + 268) + 2520) + 20800) + 104064) + 337920) + 4*b*l*x*(b*(b*(b*(b*(b*(b*(b*(b + 32) + 680) + 8384) + 73232) + 406784) + 1950976) + 5498880) + 14942208) - 4096*b*(b + 2)*(b*(b*(b*(b + 28) + 428) + 2672) + 14592) - 603979776) + 1207959552)

    if dim==3:
        if bmf==1:
            return -l*(b*l*x*(b + 2) - 48)/(l*x*(-b*(b - l*x + 2) - 24) + 48)
        if bmf==2:
            return l*(b*l*x*(b + 2)*(-b*l*x*(b + 2)*(b + 4) + 24*b*(b + 10) + 2496) - 92160)/(l*x*(b*(4*b*l**2*x**2*(b + 2) - l*x*(b*(b*(b*(b + 8) + 116) + 496) + 2496) + 24*(b + 2)*(b*(b + 10) + 104)) + 46080) - 92160)
        if bmf==3:
            return l*(b*l*x*(b + 2)*(-b**2*l**2*x**2*(b + 2)**2*(b + 4)**2*(b + 6) + 6*b*l*x*(b + 2)*(b + 4)*(b*(b*(b*(b + 24) + 532) + 4976) + 28032) - 11520*b*(b*(b*(b + 28) + 452) + 2912) - 223395840) + 7431782400)/(l*x*(24*b**3*l**3*x**3*(b + 2)**2*(b + 4) - b**2*l**2*x**2*(b + 2)*(b*(b*(b*(b*(b*(b + 18) + 340) + 3576) + 34336) + 185856) + 672768) + 6*b*l*x*(b*(b*(b*(b*(b*(b*(b*(b + 32) + 744) + 9728) + 96144) + 608768) + 3566336) + 11046912) + 37232640) - 11520*b*(b + 2)*(b*(b*(b*(b + 28) + 452) + 2912) + 19392) - 3715891200) + 7431782400)

    return None


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
