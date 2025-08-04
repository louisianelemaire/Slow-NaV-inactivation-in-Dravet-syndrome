from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import scipy.signal as sig
import importlib
import pickle
import os
import copy
import numpy as np


class Simulation:
    def __init__(self, rhs, model_params, y_0, t_end, id=None):
        self.model_params = copy.deepcopy(model_params)  # model parameters
        self.y_0 = y_0  # initial condition
        self.t_end = t_end

        self.sol = None  # solution
        self.rhs = rhs
        self.dim = rhs.dim
        self.id = id

    def run(self, rtol=1e-13, atol=1e-19, method="LSODA"):
        sol = solve_ivp(self.rhs, [0, self.t_end], self.y_0, args=[
                        self.model_params], rtol=rtol, atol=atol, method=method)
        self.sol = sol

    def plot_time_traces(self, axes=None, time_shift=0, idxs=None):
        if axes is None:
            fig, axes = plt.subplots(self.dim, sharex=True)
        if idxs is None:
            idxs = range(self.dim)
        for idx_ax, idx in enumerate(idxs):
            ax = axes[idx_ax]
            ax.plot(self.sol.t-time_shift, self.sol.y[idx], color="k")
            ax.set_ylabel(self.rhs.var_labels[idx])
        axes[-1].set_xlabel("Time (ms)")

    def plot_voltage_trace(self, ax, time_shift=0, unit="ms", color="k", **kwargs):
        if unit == "ms":
            ax.plot(self.sol.t-time_shift,
                    self.sol.y[0], color=color, **kwargs)
        if unit == "second":
            ax.plot(self.sol.t/1000-time_shift,
                    self.sol.y[0], color=color, **kwargs)

    def plot_voltage_wrt_s(self, ax, v_offset=0, t_min=-np.inf, t_max=np.inf, **kwargs):
        t_vals = self.sol.t
        condition = (t_vals > t_min) & (t_vals < t_max)
        ax.plot(self.sol.y[-1][condition], self.sol.y[0]
                [condition]+v_offset, **kwargs)

    def plot_voltage_wrt_s_scaled(self, ax, v_offset=0, t_min=-np.inf, t_max=np.inf, **kwargs):
        t_vals = self.sol.t
        condition = (t_vals > t_min) & (t_vals < t_max)
        s_scaled = self.model_params["p_mut"]*self.sol.y[-1] + \
            (1-self.model_params["p_mut"])*self.sol.y[-2]
        ax.plot(s_scaled[condition], self.sol.y[0]
                [condition]+v_offset, **kwargs)

    def plot_one_trace(self, ax, time_shift=0, var_idx=0, unit="ms", color="k", **kwargs):
        if unit == "ms":
            ax.plot(self.sol.t-time_shift,
                    self.sol.y[var_idx], color=color, **kwargs)
        if unit == "second":
            ax.plot(self.sol.t/1000-time_shift,
                    self.sol.y[var_idx], color=color, **kwargs)

    def plot_s_tot(self, ax, time_shift=0, var_idx=0, unit="ms", color="k", **kwargs):
        p_mut = self.model_params["p_mut"]
        s_tot = p_mut * self.sol.y[-1] + (1-p_mut)*self.sol.y[-2]
        if unit == "ms":
            ax.plot(self.sol.t-time_shift,
                    s_tot, color=color, **kwargs)
        if unit == "second":
            ax.plot(self.sol.t/1000-time_shift,
                    s_tot, color=color, **kwargs)

    def save_fixed_point(self, model_name, id, do_print=False):
        last_point = self.sol.y.T[-1]
        last_point_dic = dict(enumerate(last_point.flatten(), 1))
        auto_dir = f"utils/models/{model_name}/auto/{id}"
        os.system(f"mkdir -p {auto_dir}")

        with open(f"{auto_dir}/eq.pickle", 'wb') as f:
            pickle.dump(last_point_dic, f)
        if do_print:
            print(last_point_dic)
            print(f"directory: {auto_dir}")

    def save_sol(self, model_name, i_app, id=None):
        to_save = self.sol
        dir = f"output/{model_name}/{self.id}"
        if id:
            dir = dir + "/" + id
        os.system(f"mkdir -p {dir}")
        path = f"{dir}/sol_i_app_{i_app}.pickle"
        with open(path, 'wb') as f:
            pickle.dump(to_save, f)

    def load_sol(self, model_name, i_app, id=None):
        dir = f"output/{model_name}/{self.id}"
        if id:
            dir = dir + "/" + id
        path = f"{dir}/sol_i_app_{i_app}.pickle"
        print(path)
        self.sol = pickle.load(open(path, 'rb'))

    # instantaneous firing frequency
    def plot_inst_freq(self, ax_main, unit="second", ax_freq=None, time_shift=0, linewidth=1, **kwargs):

        t_vals = self.sol.t
        v_vals = self.sol.y[0]

        spike_idx = sig.argrelmax(v_vals)[0]
        spike_times = t_vals[spike_idx]

        spike_times = spike_times[v_vals[spike_idx] > -40]

        isi_s = spike_times[1:] - spike_times[:-1]  # interspike intervals
        inst_freq = 1/isi_s * 1000  # instantaneous frequency in Hz

        if not ax_freq:
            ax_freq = ax_main.twinx()
        if unit == "second":
            ax_freq.plot(spike_times[1:]/1000-time_shift,
                         inst_freq[0:], zorder=10, linewidth=linewidth, **kwargs)
        if unit == "ms":
            ax_freq.plot((spike_times[1:]+spike_times[:-1])/2-time_shift,
                         inst_freq[0:], zorder=10, linewidth=linewidth, **kwargs)
        return ax_freq


def compute_transient_fI(model_name, param_set_id, i_app_vals, t_end, accuracy_scaling=100000):
    print(f"Computing fI curve for params: {param_set_id}")
    model = import_model(model_name)
    dim = model.rhs.dim
    y_0 = [-70] + [0.5] * (dim-1)
    t_end_find_init = 100000

    my_params = model.params_default | getattr(model, f"params_{param_set_id}")
    my_simu = Simulation(model.rhs, my_params, y_0, t_end, id=param_set_id)
    simu_all = []
    nb_spikes_all = [None] * len(i_app_vals)

    # find resting point
    my_simu.model_params["i_app"] = 0
    my_simu.t_end = t_end_find_init
    my_simu.run()
    y_0_rest_wt = my_simu.sol.y[:, -1]
    print(f"y_0_rest_wt: {y_0_rest_wt}")
    my_simu.y_0 = y_0_rest_wt
    my_simu.t_end = t_end

    # run simulations for all i_app
    for idx, i_app in enumerate(i_app_vals):
        # wild type
        my_simu.model_params["i_app"] = i_app
        my_simu.run(rtol=accuracy_scaling*1e-13, atol=accuracy_scaling*1e-19)
        my_simu.save_sol(model_name, i_app, id=f"fI_{t_end}")
        simu_all.append(copy.deepcopy(my_simu))

        # count spikes
        peaks, _ = find_peaks(my_simu.sol.y[0], height=-60)
        nb_spikes = len(peaks)
        my_simu.nb_spikes = nb_spikes
        nb_spikes_all[idx] = nb_spikes

    # save
    to_save = {"i_app_vals": i_app_vals,
               "nb_spikes": nb_spikes_all, "duration": t_end}
    path = f"output/{model_name}/{my_simu.id}/fI_{t_end}/fI.pickle"
    with open(path, 'wb') as f:
        pickle.dump(to_save, f)


def plot_fI_curve(ax, model_name, param_set_id, t_end, **kwargs):
    path = f"output/{model_name}/{param_set_id}/fI_{t_end}/fI.pickle"
    saved = pickle.load(open(path, 'rb'))
    i_app_vals = saved["i_app_vals"]
    freq_vals = [nb_spikes/(saved["duration"]/1000)
                 for nb_spikes in saved["nb_spikes"]]
    ax.plot(i_app_vals, freq_vals, **kwargs)


def import_model(model_name):
    return importlib.import_module(f'utils.models.{model_name}.{model_name}')


def plot_scaled_sodium_conductance(ax, simu, unit="ms"):
    params = simu.model_params
    g_Na_mut = params["mut_functional"] * \
        params["p_mut"] * params["g_Na_tot"]
    g_Na_wt = params["g_Na_tot"] * (1 - params["p_mut"])
    g = simu.sol.y[-1] * g_Na_mut + simu.sol.y[-2] * g_Na_wt
    if unit == "ms":
        t_vals = simu.sol.t
    if unit == "second":
        t_vals = simu.sol.t/1000
    ax.plot(t_vals, g, color="k")
