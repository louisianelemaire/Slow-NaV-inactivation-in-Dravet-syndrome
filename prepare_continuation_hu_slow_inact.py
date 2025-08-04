# compute time traces of the Hu model with slow inactivation
# extract and save fixed point or limit cycle, to be used as starting point by auto

import utils.simu_helper as simu_helper

import numpy as np
import scipy.signal as sig


def compute_fp_and_lc(param_set_id, i_app_val, compute_lc=True):
    
    print(f"parameter set: {param_set_id}")

    ########################## preparation ##########################

    # model
    model_name = "hu_slow_inact"
    model = simu_helper.import_model(model_name)

    # parameter values
    my_params = model.params_default | getattr(model, f"params_{param_set_id}")


    ########################## resting state ##########################
    y_0 = [-60, 0.5, 0.5, 0.5, 0.5, 0.5]  # initial condition
    t_end = 1000000  # duration (in ms)

    my_simu_fixed_point = simu_helper.Simulation(model.rhs, my_params, y_0, t_end)  # define simulation
    my_simu_fixed_point.model_params["i_app"] = 0  # in uA/cm2

    my_simu_fixed_point.run()  # run simulation 

    my_simu_fixed_point.save_fixed_point(model_name, param_set_id, do_print=False) # save fixed point 


    ########################## limit cycle ##########################
    if compute_lc:
        t_end_lc = 200000 # duration (in ms) 

        # define simulation
        my_simu_tonic_spiking = simu_helper.Simulation(model.rhs, my_params, y_0, t_end_lc, id=param_set_id)

        # step current
        my_simu_tonic_spiking.model_params["t_start"] = 0
        my_simu_tonic_spiking.model_params["i_app_step_val"] = i_app_val 

        accuracy_scaling = 100000  # reduce accuracy
        y_0 = my_simu_fixed_point.sol.y.T[-1]  # initial condition

        my_simu_tonic_spiking.run(rtol=1e-13*accuracy_scaling, atol=1e-19*accuracy_scaling)  # run simulation

        #### select solution over one limit cycle ####

        #  spike times of last two spikes
        my_sol = my_simu_tonic_spiking.sol
        spike_idx = sig.argrelmax(my_sol.y[0].squeeze())[0] 
        my_isi_lim = spike_idx[-2:]
        period = my_sol['t'][my_isi_lim[1]] - my_sol['t'][my_isi_lim[0]]
        print(f"Period: {period}")  # for auto

        # solution between last two spikes
        t_vals = my_sol.t[my_isi_lim[0]:my_isi_lim[1]]
        v_vals = my_sol.y[0][my_isi_lim[0]:my_isi_lim[1]]
        h_vals = my_sol.y[1][my_isi_lim[0]:my_isi_lim[1]]
        n_vals = my_sol.y[2][my_isi_lim[0]:my_isi_lim[1]]
        n_bis_vals = my_sol.y[3][my_isi_lim[0]:my_isi_lim[1]]
        s_wt_vals = my_sol.y[4][my_isi_lim[0]:my_isi_lim[1]]
        s_mut_vals = my_sol.y[5][my_isi_lim[0]:my_isi_lim[1]]

        # # plot it
        # fig, axes = plt.subplots(6, 1, sharex=True)
        # axes[0].plot(t_vals, v_vals)
        # axes[1].plot(t_vals, h_vals)
        # axes[2].plot(t_vals, n_vals)
        # axes[3].plot(t_vals, n_bis_vals)
        # axes[4].plot(t_vals, s_wt_vals)
        # axes[5].plot(t_vals, s_mut_vals)

        # save it
        to_save = np.transpose(np.array([(t_vals-t_vals[0]), v_vals.squeeze(), h_vals.squeeze(), n_vals.squeeze(), n_bis_vals.squeeze(), s_wt_vals.squeeze(), s_mut_vals.squeeze()]))
        auto_dir = f"utils/models/{model_name}/auto/{param_set_id}"
        filename = f"{auto_dir}/lc_I_app_{i_app_val}.dat"
        np.savetxt(filename, to_save)