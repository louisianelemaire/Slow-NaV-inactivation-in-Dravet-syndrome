import auto
import os
from contextlib import contextmanager
import pickle
import numpy as np

colors_bif = {
    "PD": '#17becf',
    "HB": '#ff7f0e',
    'LP': '#d62728',
    "LPC": '#d62728',
    "SNIC": '#2ca02c',
    "HOM": "#8c564b",
    "FP": "k",
    "LP": "#800080",
}


hopf_color = "grey"
hopf_edge_color = "white"
snp_color = "k"
snp_edge_color = "white"
pd_color = "blue"
pd_edge_color = "white"
hopf_plot_args = {"marker": "^",
                  "edgecolors": hopf_edge_color, "color": hopf_color, "s": 37}
snp_plot_args = {"edgecolors": snp_edge_color, "color": snp_color, "s": 30}
pd_plot_args = {"marker":"*", "edgecolors": pd_edge_color, "color": pd_color, "s": 77}

@contextmanager
def temp_chdir(where):
    old = os.getcwd()
    try:
        os.chdir(where)
        yield
    finally:
        os.chdir(old)


def compute_auto_dir(model_name):
    auto_dir = f"utils/models/{model_name}/auto"
    os.system(f"mkdir -p {auto_dir}")
    return auto_dir


def convert_params_dict(params_simu):
    return {key.upper().replace("_", ""): value for key, value in params_simu.items()}


def load_fixed_point(auto_dir, variant_cont):
    filename = f"{auto_dir}/{variant_cont}/eq.pickle"
    with open(filename, 'rb') as f:
        inits = pickle.load(f)
    return inits


def run(filename, auto_dir, output_dir="", **kwargs):
    with temp_chdir(auto_dir):
        if "s" in kwargs:
            os.system(f"cp {output_dir}/s.{kwargs['s']} ./s.{kwargs['s']}")
        if "dat" in kwargs:
            os.system(f"cp {output_dir}/{kwargs['dat']} ./{kwargs['dat']}")
        branch = auto.run(**kwargs)
        os.system(f"mkdir -p {output_dir}")
        with temp_chdir(output_dir):
            os.system(f"pwd")
            auto.save(branch, f"{filename}")
            auto.clean()
        auto.clean()
        if "s" in kwargs:
            os.system(f"rm ./s.{kwargs['s']}")
        if "dat" in kwargs:
            os.system(f"rm ./{kwargs['dat']}")
    auto.clean()
    return branch
    


def load_run(directory, run_name):
    current_dir = os.getcwd()
    os.chdir(directory)
    run = auto.loadbd(run_name)
    os.chdir(current_dir)
    return run


def plot_bd(ax, auto_dir, output_dir="", run_names=None, var_idx=0, linestyle="-", linestyle_unstable=':', x_scaling=1, colors=None, **kwargs):
    if colors is None:
        colors = {1: colors_bif["FP"], 2: colors_bif["LP"]}
    var_col = var_idx + 2  # column index of the variable we want to plot
    l_eq_stable, l_eq_unstable, l_lc_stable, l_lc_unstable = None, None, None, None

    for idx, run_name in enumerate(run_names):
        run = load_run(f"{auto_dir}/{output_dir}", run_name)
        for br in run:
            stability_changes = br.stability()
            constants = br.c
            # equilibria (ips=1) or limit cycles (ips=2)
            ips = constants["IPS"]
            color = colors[ips]
            bd = br.toarray()  # bifurcation diagram as np array

            param = bd[0, :]
            var = bd[var_col, :]
            ind_prev = 0

            for ind in stability_changes:
                ind_abs = abs(ind)
                if ips == 1:  # fixed point
                    if ind < 0:
                        l_eq_stable, = ax.plot(x_scaling * param[ind_prev:ind_abs], var[ind_prev:ind_abs], color=color,
                                               linestyle=linestyle, **kwargs)
                    else:
                        l_eq_unstable, = ax.plot(x_scaling * param[ind_prev:ind_abs], var[ind_prev:ind_abs], color=color,
                                                 linestyle=linestyle_unstable, **kwargs)
                if ips == 2:  # limit cycles
                    if ind < 0:
                        ax.plot(x_scaling * param[ind_prev:ind_abs], var[ind_prev:ind_abs], color=color,
                                linestyle=linestyle, **kwargs)

                    else:
                        ax.plot(x_scaling * param[ind_prev:ind_abs], var[ind_prev:ind_abs], color=color,
                                linestyle=linestyle_unstable, **kwargs)
                    var_low = bd[1, :]
                    if ind < 0:
                        l_lc_stable, = ax.plot(x_scaling * param[ind_prev:ind_abs], var_low[ind_prev:ind_abs],
                                               color=color,
                                               linestyle=linestyle, **kwargs)
                    else:
                        l_lc_unstable, = ax.plot(x_scaling * param[ind_prev:ind_abs], var_low[ind_prev:ind_abs],
                                                 color=color,
                                                 linestyle=linestyle_unstable, **kwargs)

                ind_prev = ind_abs - 1
    return l_eq_stable, l_eq_unstable, l_lc_stable, l_lc_unstable


def plot_freq(ax, directory, output_dir, run_names, color="k", plot_unstable=True, x_scaling=1, linestyle=None, **kwargs):

    if linestyle is None:
        linestyle = "-"

    for idx, run_name in enumerate(run_names):
        # load bifurcation diagram
        run = load_run(f"{directory}/{output_dir}", run_name)
        br = run[0]
        stability_changes = br.stability()
        constants = br.c
        ips = constants["IPS"]  # steady states or limit cycles
        bd = br.toarray()  # bifurcation diagram as np array

        param = bd[0, :]
        var = bd[2, :]
        ind_prev = 0

        for ind in stability_changes:
            ind_abs = abs(ind)
            if ips == 2:
                var_low = bd[1, :]
                T = bd[-1, :]  # plot also frequency?
                f = 1000 / T
                if ind < 0:
                    ax.plot(param[ind_prev:ind_abs]*x_scaling, f[ind_prev:ind_abs],
                            color=color,
                            linestyle=linestyle, **kwargs)
                else:
                    if plot_unstable:
                        ax.plot(param[ind_prev:ind_abs]*x_scaling, f[ind_prev:ind_abs],
                                color=color,
                                linestyle=':', **kwargs)

            ind_prev = ind_abs - 1


def plot_special_point(ax, auto_dir, output_dir, run_name, label_auto, label, var_idx=0, label_legend=None, x_scaling=1, **kwargs):
    var_col = var_idx + 2
    # load bifurcation diagram
    run = load_run(f"{auto_dir}/{output_dir}", run_name)
    point = run(label_auto)
    par_val = point.b['data'][0] * x_scaling
    var_val = point.b['data'][var_col]

    line = ax.scatter(par_val, var_val, label=label_legend,
                      zorder=1000, **kwargs)
    return line


def plot_codim_2_bd_diagram(ax, auto_dir, output_dir, run_name, color="k", **kwargs):
    run = load_run(f"{auto_dir}/{output_dir}", run_name)
    for br in run:
        bd = br.toarray()
        par_1 = bd[0, :]
        par_2 = bd[-1, :]
        l = ax.plot(par_1, par_2, color=color, **kwargs)
    return l


def plot_slow_flow_hu(ax, directory, output_dir, run_name, params, param_str="GNATOT", x_scaling=1, y_scaling=1, y_shift=0, labels=[None, None], **kwargs):
    run = load_run(f"{directory}/{output_dir}", run_name)

    int_of_s_inf = []
    s_vals = []

    br = run[0]

    v_h = params["v_h"]
    k = params["k"]
    shift_s = params["shift_s"]
    tau_s_mut = params["tau_s_mut"]

    q_s = params["q10_s"] ** ((params["temp"] - 33) / 10)

    for row in br:
        label = row["LAB"]
        point = run(label)
        v_vals = point["V"] - shift_s
        t_vals = point["t"]

        s = point[param_str]

        pt = np.sign(row["PT"])

        if pt < 0:
            s_vals.append(s*x_scaling)

            above = 1
            below = 1 + np.exp(-(v_vals - v_h) / k)
            s_inf_vals = above/below
            int_val = np.trapz(s_inf_vals, t_vals)
            int_of_s_inf.append(int_val)

    ds_dt_av = (np.array(int_of_s_inf) -
                np.array([s for s in s_vals]))*q_s/tau_s_mut

    l, = ax.plot(s_vals, y_scaling*ds_dt_av-y_shift,
                 label=labels[1], **kwargs)
    return l


def plot_slow_flow_hu_scaled_s(ax, directory, output_dir, run_name, params, param_str="GNATOT", x_scaling=1, y_scaling=1, y_shift=0, labels=[None, None], **kwargs):
    run = load_run(f"{directory}/{output_dir}", run_name)

    int_of_s_inf = []
    int_of_s_inf_shift = []

    s_vals = []

    br = run[0]

    v_h = params["v_h"]
    k = params["k"]
    shift_s = params["shift_s"]
    tau_s_wt = params["tau_s_wt"]
    tau_s_mut = params["tau_s_mut"]
    p_mut = params["p_mut"]

    q_s = params["q10_s"] ** ((params["temp"] - 33) / 10)

    for row in br:
        label = row["LAB"]
        point = run(label)
        v_vals = point["V"]
        v_vals_shift = point["V"] - shift_s
        t_vals = point["t"]

        s = point[param_str]

        pt = np.sign(row["PT"])

        if pt < 0:
            s_vals.append(s*x_scaling)

            above = 1
            below = 1 + np.exp(-(v_vals - v_h) / k)
            s_inf_vals = above/below
            int_val = np.trapz(s_inf_vals, t_vals)
            int_of_s_inf.append(int_val)
            below_shift = 1 + np.exp(-(v_vals_shift - v_h) / k)
            s_inf_vals_shift = above/below_shift
            int_vals_shift = np.trapz(s_inf_vals_shift, t_vals)
            int_of_s_inf_shift.append(int_vals_shift)

    ds_tot_dt_av = ((1-p_mut)*np.array(int_of_s_inf) + p_mut * np.array(int_of_s_inf_shift) -
                    np.array([s for s in s_vals]))*q_s/tau_s_wt
    # ds_mut_dt_av =

    l, = ax.plot(s_vals, y_scaling*ds_tot_dt_av-y_shift,
                 label=labels[1], **kwargs)
    return l


def plot_ds_dt_fp_hu(ax, directory, output_dir, run_name, s_inf, params, xscaling=1, y_scaling=1, shift=0, **kwargs):
    run = load_run(f"{directory}/{output_dir}", run_name)
    br = run[0]

    bd = br.toarray()

    param = bd[0, :]
    var = bd[2, :]

    tau_s_mut = params["tau_s_mut"]
    q_s = params["q10_s"] ** ((params["temp"] - 33) / 10)

    # get stability
    points = [row["PT"] < 0 for row in br]

    param_stable = param[points]*xscaling
    var_stable = var[points]

    ds_dt = (s_inf(var_stable-shift, params) - param_stable)*q_s/tau_s_mut

    l, = ax.plot(param_stable, y_scaling*ds_dt, **kwargs)
    return l


def plot_ds_tot_dt_fp_hu(ax, directory, output_dir, run_name, s_inf, params, xscaling=1, y_scaling=1, **kwargs):
    run = load_run(f"{directory}/{output_dir}", run_name)
    br = run[0]

    bd = br.toarray()

    param = bd[0, :]
    var = bd[2, :]

    tau_s_mut = params["tau_s_mut"]
    q_s = params["q10_s"] ** ((params["temp"] - 33) / 10)
    p_mut = params["p_mut"]
    shift_s = params["shift_s"]

    # get stability
    points = [row["PT"] < 0 for row in br]

    param_stable = param[points]*xscaling
    var_stable = var[points]

    ds_dt = (p_mut*s_inf(var_stable-shift_s, params) + (1-p_mut) *
             s_inf(var_stable, params) - param_stable)*q_s/tau_s_mut

    l, = ax.plot(param_stable, y_scaling*ds_dt, **kwargs)
    return l
