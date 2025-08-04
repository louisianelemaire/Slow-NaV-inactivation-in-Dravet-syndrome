# script to compute the bifurcation diagram of the hu model (fast subsystem: s is a parameter)

import utils.auto_helper as auto_helper
import utils.simu_helper as simu_helper
import prepare_continuation_hu


def compute_bd(i_app_val):

    # fast subsystem
    model_name = "hu"
    hu = simu_helper.import_model(model_name)

    # parameter set
    param_set_id = "wt"
    g_na_max = hu.params_default["g_na_tot"]

    # fixed point, will be used as starting point of the continuation
    prepare_continuation_hu.compute_fp(param_set_id)
    inits = auto_helper.load_fixed_point(auto_dir, param_set_id)

    # directory with the auto files
    auto_dir = auto_helper.compute_auto_dir(model_name)


    ########################## continuation ##########################
    auto_helper.run(filename=f"branch_eq_wrt_i_app_ends_{i_app_val}", auto_dir=auto_dir, output_dir="wt", e=model_name, c=model_name,
                    ICP="IAPP", U=inits, UZSTOP={"IAPP": [i_app_val]})

    #### branch of fixed points ####
    # dummy run to start from g_na_tot=0
    auto_helper.run(filename=f"branch_eq_wrt_gnatot_i_app_{i_app_val}_dummy", auto_dir=auto_dir, output_dir="wt", e=model_name, c=model_name,
                s=f"branch_eq_wrt_i_app_ends_{i_app_val}", IRS="UZ1", ICP="GNATOT", UZSTOP={"GNATOT": [0]}, DS="-")

    # actual run
    auto_helper.run(filename=f"branch_eq_wrt_gnatot_i_app_{i_app_val}", auto_dir=auto_dir, output_dir="wt", e=model_name, c=model_name,
                s=f"branch_eq_wrt_gnatot_i_app_{i_app_val}_dummy", IRS="UZ1", ICP="GNATOT", UZSTOP={"GNATOT": [0, g_na_max]})

    #### branch of limit cycles ####
    # dummy (to remove first few points)
    auto_helper.run(filename=f"branch_lc_wrt_gnatot_i_app_{i_app_val}_dummy", auto_dir=auto_dir, output_dir="wt", e=model_name, c=model_name,
                s=f"branch_eq_wrt_gnatot_i_app_{i_app_val}", IRS="HB1", ICP=["GNATOT", "PERIOD"], UZSTOP={"GNATOT": [0, g_na_max], "PERIOD": 400}, IPLT=-1, IPS=2, STOP=["BP1", "LP10"], NPR=1, NMX=5)

    # actual run
    auto_helper.run(filename=f"branch_lc_wrt_gnatot_i_app_{i_app_val}", auto_dir=auto_dir, output_dir="wt", e=model_name, c=model_name,
                s=f"branch_lc_wrt_gnatot_i_app_{i_app_val}_dummy", IRS="EP1", ICP=["GNATOT", "PERIOD"], UZSTOP={"GNATOT": [0, g_na_max], "PERIOD": 400}, IPLT=-1, IPS=2, STOP=["BP1", "LP10"], NPR=1)
