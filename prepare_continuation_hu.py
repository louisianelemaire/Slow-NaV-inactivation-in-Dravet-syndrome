# compute time traces of the Hu model (fast subsystem: s is a parameter)
# extract and save fixed point to be used as starting point by auto

import utils.simu_helper as simu_helper

# model
model_name = "hu"
model = simu_helper.import_model(model_name)

# initial condition
y_0 = [-60, 0.5, 0.5, 0.5]

def compute_fp(param_set_id):
    t_end = 100000  # duration

    # parameter values
    my_params = model.params_default | getattr(model, f"params_{param_set_id}")

    my_simu_fixed_point = simu_helper.Simulation(model.rhs, my_params, y_0, t_end)  # define simulation
    my_simu_fixed_point.model_params["i_app"] = 0  # applied current in uA/cm2

    # run simulation
    my_simu_fixed_point.run()

    # save fixed point
    my_simu_fixed_point.save_fixed_point(model_name, param_set_id, do_print=True)