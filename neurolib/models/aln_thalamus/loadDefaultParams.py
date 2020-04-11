import os
import numpy as np
import h5py

from ...utils.collections import dotdict


def loadDefaultParams(Cmat=None, Dmat=None, lookupTableFileName=None, seed=None):
    """Load default parameters for a network of aLN nodes.
    :param Cmat: Structural connectivity matrix (adjacency matrix) of coupling strengths, will be normalized to 1. If not given, then a single node simulation will be assumed, defaults to None
    :type Cmat: numpy.ndarray, optional
    :param Dmat: Fiber length matrix, will be used for computing the delay matrix together with the signal transmission speed parameter `signalV`, defaults to None
    :type Dmat: numpy.ndarray, optional
    :param lookUpTableFileName: Filename of lookup table with aln non-linear transfer functions and other precomputed quantities., defaults to aln-precalc/quantities_cascade.h
    :type lookUpTableFileName: str, optional
    :param seed: Seed for the random number generator, defaults to None
    :type seed: int, optional
    
    :return: A dictionary with the default parameters of the model
    :rtype: dict
    """

    params = dotdict({})

    # Todo: Model metadata
    # recently added for easier simulation of aln and brian in pypet
    params.model = "aln_thalamus"
    params.name = "aln_thalamus"
    params.description = "AdEx mean-field cortical model + mass model for thalamus"

    # runtime parameters
    params.dt = 0.1  # ms 0.1ms is reasonable
    params.duration = 2000  # Simulation duration (ms)
    np.random.seed(seed)  # seed for RNG of noise and ICs
    # set seed to 0, pypet will complain otherwise
    if seed is None:
        seed = 0
    params.seed = seed

    # options
    params.warn = 0  # warn if limits of lookup tables are exceeded
    params.dosc_version = 0  # if 0, use exponential fit to linear response function
    params.distr_delay = 0  # if 1, use distributed delays instead of fixed
    params.filter_sigma = 0  # if 1, filter sigmae/sigmai
    params.fast_interp = 1  # if 1, Interpolate the value from the look-up table instead of taking the closest value

    # ------------------------------------------------------------------------
    # global whole-brain network parameters
    # ------------------------------------------------------------------------

    if Cmat is None:
        params.N = 1
        params.Cmat = np.zeros((params.N, params.N))
        params.lengthMat = np.zeros((params.N, params.N))

    else:
        params.Cmat = Cmat.copy()  # coupling matrix
        np.fill_diagonal(params.Cmat, 0)  # no self connections
        params.N = len(params.Cmat)  # number of nodes
        params.lengthMat = Dmat  # delay matrix

    # Signal transmission speed in mm/ms
    params.signalV = 20.0

    # PSP current amplitude in (mV/ms) (or nA/[C]) for global coupling
    # connections between areas
    params.c_gl = 0.3
    # number of incoming E connections (to E population) from each area
    params.Ke_gl = 250.0

    # ------------------------------------------------------------------------
    # local E-I node parameters
    # ------------------------------------------------------------------------

    # external input parameters:
    params.tau_ou = 5.0  # ms timescale of ornstein-uhlenbeck (OU) noise
    params.sigma_ou = 0.0  # mV/ms/sqrt(ms) intensity of OU oise
    params.mue_ext_mean = 0.4  # mV/ms mean external input current to E
    params.mui_ext_mean = 0.3  # mV/ms mean external input current to I

    # Ornstein-Uhlenbeck noise state variables, set to mean input
    # mue_ou will fluctuate around mue_ext_mean (mean of the OU process)
    params.mue_ou = params.mue_ext_mean * np.ones((params.N,))  # np.zeros((params.N,))
    params.mui_ou = params.mui_ext_mean * np.ones((params.N,))  # np.zeros((params.N,))

    # external neuronal firing rate input
    params.ext_exc_rate = 0.0  # kHz external excitatory rate drive
    params.ext_inh_rate = 0.0  # kHz external inhibiroty rate drive

    # externaln input currents, same as mue_ext_mean but can be time-dependent!
    params.ext_exc_current = 0.0  # external excitatory input current [mV/ms], C*[]V/s=[]nA
    params.ext_inh_current = 0.0  # external inhibiroty input current [mV/ms]

    # Fokker Planck noise (for N->inf)
    params.sigmae_ext = 1.5  # mV/sqrt(ms) (fixed, for now) [1-5] (Internal noise due to random coupling)
    params.sigmai_ext = 1.5  # mV/sqrt(ms) (fixed, for now) [1-5]

    # recurrent coupling parameters
    params.Ke = 800.0  # Number of excitatory inputs per neuron
    params.Ki = 200.0  # Number of inhibitory inputs per neuron

    # synaptic delays
    params.de = 4.0  # ms local constant delay "EE = IE"
    params.di = 2.0  # ms local constant delay "EI = II"

    # synaptic time constants
    params.tau_se = 2.0  # ms  "EE = IE", for fixed delays
    params.tau_si = 5.0  # ms  "EI = II"

    # time constant for distributed delays (untested)
    params.tau_de = 1.0  # ms  "EE = IE"
    params.tau_di = 1.0  # ms  "EI = II"

    # PSC amplitudes
    params.cee = 0.3  # mV/ms
    params.cie = 0.3  # AMPA
    params.cei = 0.5  # GABA BrunelWang2003
    params.cii = 0.5

    # Coupling strengths used in Cakan2020
    params.Jee_max = 2.43  # mV/ms
    params.Jie_max = 2.60  # mV/ms
    params.Jei_max = -3.3  # mV/ms [0-(-10)]
    params.Jii_max = -1.64  # mV/ms

    # neuron model parameters
    params.a = 0.0  # nS, can be 15.0
    params.b = 0.0  # pA, can be 40.0
    params.EA = -80.0  # mV
    params.tauA = 200.0  # ms

    # single neuron paramters - if these are changed, new transfer functions must be precomputed!
    params.C = 200.0  # pF
    params.gL = 10.0  # nS
    params.EL = -65.0  # mV
    params.DeltaT = 1.5  # mV
    params.VT = -50.0  # mV
    params.Vr = -70.0  # mV
    params.Vs = -40.0  # mV
    params.Tref = 1.5  # ms

    params.N_pt = 5.0  # Costa et al. paper to their cortex

    # ------------------------------------------------------------------------
    # thalamus node parameters
    # ------------------------------------------------------------------------
    # local parameters for both populations
    params.tau_thlm = 20.0
    params.Q_max = 400.0e-3  # 1/ms
    params.theta = -58.5  # mV
    params.sigma = 6.0
    params.C1 = 1.8137993642
    params.C_m = 1.0  # muF/cm^2
    params.gamma_e = 70.0e-3  # 1/ms
    params.gamma_r = 100.0e-3  # 1/ms
    params.g_L = 1.0  # AU
    params.g_GABA = 1.0  # ms
    params.g_AMPA = 1.0  # ms
    params.g_LK = 0.018  # mS/cm^2
    params.E_AMPA = 0.0  # mV
    params.E_GABA = -70.0  # mV
    params.E_L = -70.0  # mV
    params.E_K = -100.0  # mV
    params.E_Ca = 120.0  # mV
    params.ctx_thlm_delay = 13.0  # ms
    # specific thalamo-cortical neurons population - TCR (excitatory)
    params.g_T_t = 3.0  # mS/cm^2
    params.g_h = 0.062  # mS/cm^2
    params.E_h = -40.0  # mV
    params.alpha_Ca = -51.8e-6  # nmol
    params.tau_Ca = 10.0  # ms
    params.Ca_0 = 2.4e-4
    params.k1 = 2.5e7
    params.k2 = 4.0e-4
    params.k3 = 1.0e-1
    params.k4 = 1.0e-3
    params.n_P = 4.0
    params.g_inc = 2.0
    # connectivity
    params.N_tr = 5.0
    params.N_tp = 2.6
    # noise
    params.d_phi = 0.0
    # specific thalamic reticular nuclei population - TRN (inhibitory)
    params.g_T_r = 2.3  # mS/cm^2
    # connectivity
    params.N_rt = 3.0
    params.N_rr = 25.0
    params.N_rp = 2.6
    # external input
    params.ext_current_t = 0.0
    params.ext_current_r = 0.0

    # ------------------------------------------------------------------------

    # Generate and set random initial conditions
    (
        mufe_init,
        mufi_init,
        IA_init,
        seem_init,
        seim_init,
        seev_init,
        seiv_init,
        siim_init,
        siem_init,
        siiv_init,
        siev_init,
        rates_exc_init,
        rates_inh_init,
        V_t_init,
        V_r_init,
        Ca_init,
        h_T_t_init,
        h_T_r_init,
        m_h1_init,
        m_h2_init,
        s_et_init,
        s_gt_init,
        s_er_init,
        s_gr_init,
        ds_et_init,
        ds_gt_init,
        ds_er_init,
        ds_gr_init,
    ) = generateRandomICs(params.N, seed)

    params.mufe_init = mufe_init  # (linear) filtered mean input
    params.mufi_init = mufi_init  #
    params.IA_init = IA_init  # adaptation current
    params.seem_init = seem_init  # mean of fraction of active synapses [0-1] (post-synaptic variable), chap. 4.2
    params.seim_init = seim_init  #
    params.seev_init = seev_init  # variance of fraction of active synapses [0-1]
    params.seiv_init = seiv_init  #
    params.siim_init = siim_init  #
    params.siem_init = siem_init  #
    params.siiv_init = siiv_init  #
    params.siev_init = siev_init  #
    params.rates_exc_init = rates_exc_init  #
    params.rates_inh_init = rates_inh_init  #
    params.V_t_init = V_t_init
    params.V_r_init = V_r_init
    params.Ca_init = Ca_init
    params.h_T_t_init = h_T_t_init
    params.h_T_r_init = h_T_r_init
    params.m_h1_init = m_h1_init
    params.m_h2_init = m_h2_init
    params.s_et_init = s_et_init
    params.s_gt_init = s_gt_init
    params.s_er_init = s_er_init
    params.s_gr_init = s_gr_init
    params.ds_et_init = ds_et_init
    params.ds_gt_init = ds_gt_init
    params.ds_er_init = ds_er_init
    params.ds_gr_init = ds_gr_init

    # load precomputed aLN transfer functions from hdfs
    if lookupTableFileName is None:
        lookupTableFileName = os.path.join(
            os.path.dirname(__file__), "..", "aln", "aln-precalc", "quantities_cascade.h5"
        )

    hf = h5py.File(lookupTableFileName, "r")
    params.Irange = hf.get("mu_vals")[()]
    params.sigmarange = hf.get("sigma_vals")[()]
    params.dI = params.Irange[1] - params.Irange[0]
    params.ds = params.sigmarange[1] - params.sigmarange[0]

    params.precalc_r = hf.get("r_ss")[()][()]
    params.precalc_V = hf.get("V_mean_ss")[()]
    params.precalc_tau_mu = hf.get("tau_mu_exp")[()]
    params.precalc_tau_sigma = hf.get("tau_sigma_exp")[()]

    return params


def computeDelayMatrix(lengthMat, signalV, segmentLength=1):
    """
    Compute the delay matrix from the fiber length matrix and the signal
    velocity

        :param lengthMat:       A matrix containing the connection length in
            segment
        :param signalV:         Signal velocity in m/s
        :param segmentLength:   Length of a single segment in mm

        :returns:    A matrix of connexion delay in ms
    """

    normalizedLenMat = lengthMat * segmentLength
    if signalV > 0:
        Dmat = normalizedLenMat / signalV  # Interareal delays in ms
    else:
        Dmat = lengthMat * 0.0
    return Dmat


def generateRandomICs(N, seed=None):
    """ Generates random Initial Conditions for the interareal network

        :params N:  Number of area in the large scale network

        :returns:   A tuple of 9 N-length numpy arrays representining:
                        mufe_init, IA_init, mufi_init, sem_init, sev_init,
                        sim_init, siv_init, rates_exc_init, rates_inh_init
    """
    np.random.seed(seed)
    # ALN
    mufe_init = 3 * np.random.uniform(0, 1, (N,))  # mV/ms
    mufi_init = 3 * np.random.uniform(0, 1, (N,))  # mV/ms
    IA_init = 200.0 * np.random.uniform(0, 1, (N,))  # pA
    seem_init = 0.5 * np.random.uniform(0, 1, (N,))
    seim_init = 0.5 * np.random.uniform(0, 1, (N,))
    seev_init = 0.001 * np.random.uniform(0, 1, (N,))
    seiv_init = 0.001 * np.random.uniform(0, 1, (N,))
    siim_init = 0.5 * np.random.uniform(0, 1, (N,))
    siem_init = 0.5 * np.random.uniform(0, 1, (N,))
    siiv_init = 0.01 * np.random.uniform(0, 1, (N,))
    siev_init = 0.01 * np.random.uniform(0, 1, (N,))
    rates_exc_init = 0.01 * np.random.uniform(0, 1, (N + 1, 1))  # + 1 for thalamus
    rates_inh_init = 0.01 * np.random.uniform(0, 1, (N + 1, 1))  # + 1 for thalamus

    # thalamus
    V_t_init = np.array([-68.0])
    V_r_init = np.array([-68.0])
    Ca_init = np.array([2.4e-4])
    h_T_t_init = np.array([0.0])
    h_T_r_init = np.array([0.0])
    m_h1_init = np.array([0.0])
    m_h2_init = np.array([0.0])
    s_et_init = np.array([0.0])
    s_gt_init = np.array([0.0])
    s_er_init = np.array([0.0])
    s_gr_init = np.array([0.0])
    ds_et_init = np.array([0.0])
    ds_gt_init = np.array([0.0])
    ds_er_init = np.array([0.0])
    ds_gr_init = np.array([0.0])

    return (
        mufe_init,
        mufi_init,
        IA_init,
        seem_init,
        seim_init,
        seev_init,
        seiv_init,
        siim_init,
        siem_init,
        siiv_init,
        siev_init,
        rates_exc_init,
        rates_inh_init,
        V_t_init,
        V_r_init,
        Ca_init,
        h_T_t_init,
        h_T_r_init,
        m_h1_init,
        m_h2_init,
        s_et_init,
        s_gt_init,
        s_er_init,
        s_gr_init,
        ds_et_init,
        ds_gt_init,
        ds_er_init,
        ds_gr_init,
    )
