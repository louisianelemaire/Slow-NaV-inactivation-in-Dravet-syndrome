# Hu model, with slow inactivation

# Hu model:
# DOI: 10.1016/j.neuron.2018.02.024
# based on the Wang-Buzsáki model
# gating functions were fitted to voltage-clamp date from the axons of fast-spiking GABAergic neurons

# modified to include slow inactivation of sodium channels, following Layer et al. (DOI: 10.3389/fncel.2021.754530)

import numpy as np

# parameter values

# default values
params_default = {
    "c": 0.9,
    "g_na_tot": 70,
    "p_mut": 0,
    "g_k": 15,
    "g_l": 0.1,
    "e_na": 55,
    "e_k": -90,
    "e_l": -65,
    "i_app": 0,
    "temp": 33,
    "q10_m": 2.2,
    "q10_h": 2.9,
    "q10_s": 2.9,
    "q10_n": 3,
    "shift_v": 20,
    "shift_m": 0,
    "shift_s": 0,
    "tau_s_wt": 30000,
    "tau_s_mut": 30000,
    "v_h": -60,
    "k": -10,
    "t_start": 0,
    "i_app_step_val": 0,
    "q10_s_off": False
}


params_wt = {
    "p_mut": 0
}


params_wt_high_temp = {
    "temp": 40
}


params_inact_shift_all_channels = {
    "shift_s": -15,
    "p_mut": 1
}


params_inact_shift_half_channels = {
    "shift_s": -15,
    "p_mut": 0.5
}


params_altered_inact_all_channels = {
    "shift_s": -15,
    "tau_s_mut": 3000,
    "p_mut": 1
}


params_altered_inact_half_channels = {
    "shift_s": -15,
    "tau_s_mut": 3000,
    "p_mut": 0.5
}

params_altered_inact_half_channels_high_temp = {
    "shift_s": -15,
    "tau_s_mut": 3000,
    "p_mut": 0.5,
    "temp": 40
}

params_altered_inact_half_channels_high_temp_s_off = {
    "shift_s": -15,
    "tau_s_mut": 3000,
    "p_mut": 0.5,
    "temp": 40,
    "q10_s_off": True
}

params_inact_shift_half_channels_high_temp = {
    "shift_s": -15,
    "p_mut": 0.5,
    "temp": 40
}


# helper functions

def v_shifted(v, params):
    return v - params["shift_v"]


def g_na_mut(params):
    # conductance from Na+ channels that may potentially be mutant
    g = params["p_mut"] * params["g_na_tot"]
    return g


def g_na_wt(params):
    g = params["g_na_tot"] * (1 - params["p_mut"])
    return g


def temp_diff(params):
    return params["temp"] - 24


def q_m(params):
    return params["q10_m"] ** (temp_diff(params) / 10)


def q_h(params):
    return params["q10_h"] ** (temp_diff(params) / 10)


def q_n(params):
    return params["q10_n"] ** (temp_diff(params) / 10)


def q_s(params):
    if params["q10_s_off"] == True:
        return 1
    else:
        return params["q10_s"] ** ((params["temp"] - 33) / 10)


######## Na+ current ########
def i_na_wt(v, h, s, params):
    return g_na_wt(params) * m_inf(v, params) ** 3 * h * s * (v - params["e_na"])


def i_na_mut(v, h, s, params):
    # component of the sodium current that might be affected by a mutation
    return g_na_mut(params) * m_inf(v - params["shift_m"], params) ** 3 * h * s * (v - params["e_na"])


def m_inf(v, params):
    return alpha_m(v, params) / (alpha_m(v, params) + beta_m(v, params))


def alpha_m(v, params):
    return 0.2567 * (-(v_shifted(v, params) + 60.84)) / (np.exp(-(v_shifted(v, params) + 60.84) / 9.722) - 1) * q_m(
        params)


def beta_m(v, params):
    return 0.1133 * (v_shifted(v, params) + 30.253) / (np.exp((v_shifted(v, params) + 30.253) / 2.848) - 1) * q_m(
        params)


def alpha_h(v, params):
    return 0.00105 * np.exp(-(v_shifted(v, params)) / 20) * q_h(params)


def beta_h(v, params):
    return 4.827 / (np.exp(-(v_shifted(v, params) + 18.646) / 12.452) + 1) * q_h(params)


# slow inactivation
def s_inf(v, params):
    above = 1
    below = 1 + np.exp(-(v - params["v_h"]) / params["k"])
    return above / below


def i_app_step(t, params):
    return params["i_app_step_val"] * np.heaviside(t-params["t_start"], 0.5)


######## K+ current ########

def i_k(v, n, n_bis, params):
    return params["g_k"] * n ** 3 * n_bis * (v - params["e_k"])


def alpha_n(v, params):
    return 0.0610 * (-(v - 29.991)) / (np.exp(-(v - 29.991) / 27.502) - 1) * q_n(params)


def beta_n(v, params):
    return 0.001504 * (np.exp(-v / 17.177)) * q_n(params)


def alpha_n_bis(v, params):
    return 0.0993 * (-(v - 33.720)) / (np.exp(-(v - 33.720) / 12.742) - 1) * q_n(params)


def beta_n_bis(v, params):
    return 0.1379 * (np.exp(-v / 500)) * q_n(params)


def i_leak(v, params):
    return params["g_l"] * (v - params["e_l"])


def rhs(t, y, params):
    """
    v: voltage
    h: sodium fast inactivation
    n: potassium activation
    n_bis: potassium different activation (justified in paper Hu et al.)
    """

    v, h, n, n_bis, s_wt, s_mut = y

    d_v_dt = (1 / params["c"]) * (
        - i_na_wt(v, h, s_wt, params) - i_na_mut(v, h, s_mut, params) - i_k(v, n, n_bis, params) - i_leak(v, params) +
        params["i_app"] + i_app_step(t, params))
    d_h_dt = (alpha_h(v, params) * (1 - h) - beta_h(v, params) * h)
    d_n_dt = (alpha_n(v, params) * (1 - n) - beta_n(v, params) * n)
    d_n_bis_dt = (alpha_n_bis(v, params) * (1 - n_bis) -
                  beta_n_bis(v, params) * n_bis)
    d_s_wt_dt = (s_inf(v, params) - s_wt) / params["tau_s_wt"] * q_s(params)
    d_s_mut_dt = (s_inf(v - params["shift_s"],
                  params) - s_mut) / params["tau_s_mut"] * q_s(params)

    return [d_v_dt, d_h_dt, d_n_dt, d_n_bis_dt, d_s_wt_dt, d_s_mut_dt]


rhs.var_labels = ["v (mV)", "h", "n",
                  r"$\tilde{n}$", r"$s_{\rm wt}$", "s_mut"]
rhs.dim = 6
