# Hu model
# DOI: 10.1016/j.neuron.2018.02.024
# based on the Wang-Buzsáki model
# gating functions were fitted to voltage-clamp date from the axons of fast-spiking GABAergic neurons


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
    "q10_n": 3,
    "shift_v": 20,
    "shift_m": 0
}


params_wt = {
    "p_mut": 0
}

params_wt_temp_28 = {
    "temp": 28
}

params_wt_temp_28_no_shift = {
    "temp": 28,
    "shift_v": 0
}

params_wt_high_temp = {
    "temp": 40
}

params_act_shift = {
    "shift_m": 10
}

params_act_shift_pure_nav11 = {
    "shift_m": 10,
    "p_mut": 0.5
}

params_act_shift_pure_nav11_high_temp = {
    "shift_m": 10,
    "p_mut": 0.5,
    "temp": 40
}

params_act_shift_pure_nav11_temp_28 = {
    "shift_m": 10,
    "p_mut": 0.5,
    "temp": 28
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


######## Na+ current ########
def i_na_wt(v, h, params):
    return g_na_wt(params) * m_inf(v, params) ** 3 * h * (v - params["e_na"])


def i_na_mut(v, h, params):
    # component of the sodium current that might be affected by a mutation
    return g_na_mut(params) * m_inf(v - params["shift_m"], params) ** 3 * h * (v - params["e_na"])


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


######## K+ currents ########

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


######## leak current ########

def i_leak(v, params):
    return params["g_l"] * (v - params["e_l"])


def rhs(t, y, params):
    """
    v: voltage
    h: sodium fast inactivation
    n: potassium activation
    n_bis: potassium different activation (justified in paper Hu et al.)
    """

    v, h, n, n_bis = y

    d_v_dt = (1 / params["c"]) * (
        - i_na_wt(v, h, params) - i_na_mut(v, h, params) - i_k(v, n, n_bis, params) - i_leak(v, params) +
        params["i_app"])
    d_h_dt = (alpha_h(v, params) * (1 - h) - beta_h(v, params) * h)
    d_n_dt = (alpha_n(v, params) * (1 - n) - beta_n(v, params) * n)
    d_n_bis_dt = (alpha_n_bis(v, params) * (1 - n_bis) -
                  beta_n_bis(v, params) * n_bis)

    return [d_v_dt, d_h_dt, d_n_dt, d_n_bis_dt]


rhs.var_labels = ["v (mV)", "h", "n", r"$n_{\rm bis}$"]
rhs.dim = 4
