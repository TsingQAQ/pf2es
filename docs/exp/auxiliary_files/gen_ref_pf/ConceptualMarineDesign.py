import numpy as np
import tensorflow as tf
import numpy as np
from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo
from trieste.objectives.multi_objectives import ConceptualMarineDesign


class CRE32():
    def __init__(self):
        self.problem_name = 'CRE32'
        self.n_objectives = 3
        self.n_variables = 6
        self.n_constraints = 9

        self.lbound = np.zeros(self.n_variables)
        self.ubound = np.zeros(self.n_variables)
        self.lbound[0] = 150.0
        self.lbound[1] = 20.0
        self.lbound[2] = 13.0
        self.lbound[3] = 10.0
        self.lbound[4] = 14.0
        self.lbound[5] = 0.63
        self.ubound[0] = 274.32
        self.ubound[1] = 32.31
        self.ubound[2] = 25.0
        self.ubound[3] = 11.71
        self.ubound[4] = 18.0
        self.ubound[5] = 0.75

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        # NOT g
        constraintFuncs = np.zeros(self.n_constraints)

        x_L = x[0]
        x_B = x[1]
        x_D = x[2]
        x_T = x[3]
        x_Vk = x[4]
        x_CB = x[5]

        displacement = 1.025 * x_L * x_B * x_T * x_CB
        V = 0.5144 * x_Vk
        g = 9.8065
        Fn = V / np.power(g * x_L, 0.5)
        a = (4977.06 * x_CB * x_CB) - (8105.61 * x_CB) + 4456.51
        b = (-10847.2 * x_CB * x_CB) + (12817.0 * x_CB) - 6960.32

        power = (np.power(displacement, 2.0 / 3.0) * np.power(x_Vk, 3.0)) / (a + (b * Fn))
        outfit_weight = 1.0 * np.power(x_L, 0.8) * np.power(x_B, 0.6) * np.power(x_D, 0.3) * np.power(x_CB, 0.1)
        steel_weight = 0.034 * np.power(x_L, 1.7) * np.power(x_B, 0.7) * np.power(x_D, 0.4) * np.power(x_CB, 0.5)
        machinery_weight = 0.17 * np.power(power, 0.9)
        light_ship_weight = steel_weight + outfit_weight + machinery_weight

        ship_cost = 1.3 * ((2000.0 * np.power(steel_weight, 0.85)) + (3500.0 * outfit_weight) + (
                    2400.0 * np.power(power, 0.8)))
        capital_costs = 0.2 * ship_cost

        DWT = displacement - light_ship_weight

        running_costs = 40000.0 * np.power(DWT, 0.3)

        round_trip_miles = 5000.0
        sea_days = (round_trip_miles / 24.0) * x_Vk
        handling_rate = 8000.0

        daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
        fuel_price = 100.0
        fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price
        port_cost = 6.3 * np.power(DWT, 0.8)

        fuel_carried = daily_consumption * (sea_days + 5.0)
        miscellaneous_DWT = 2.0 * np.power(DWT, 0.5)

        cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
        port_days = 2.0 * ((cargo_DWT / handling_rate) + 0.5)
        RTPA = 350.0 / (sea_days + port_days)

        voyage_costs = (fuel_cost + port_cost) * RTPA
        annual_costs = capital_costs + running_costs + voyage_costs
        annual_cargo = cargo_DWT * RTPA

        f[0] = annual_costs / annual_cargo
        f[1] = light_ship_weight
        # f_2 is dealt as a minimization problem
        f[2] = -annual_cargo

        # Reformulated objective functions
        constraintFuncs[0] = (x_L / x_B) - 6.0
        constraintFuncs[1] = -(x_L / x_D) + 15.0
        constraintFuncs[2] = -(x_L / x_T) + 19.0
        constraintFuncs[3] = 0.45 * np.power(DWT, 0.31) - x_T
        constraintFuncs[4] = 0.7 * x_D + 0.7 - x_T
        constraintFuncs[5] = 500000.0 - DWT
        constraintFuncs[6] = DWT - 3000.0
        constraintFuncs[7] = 0.32 - Fn

        KB = 0.53 * x_T
        BMT = ((0.085 * x_CB - 0.002) * x_B * x_B) / (x_T * x_CB)
        KG = 1.0 + 0.52 * x_D
        constraintFuncs[8] = (KB + BMT - KG) - (0.07 * x_B)
        constraintFuncs = np.where(constraintFuncs < 0, -constraintFuncs, 0)

        return f, constraintFuncs

# print(CRE32().evaluate(np.array([150.0,  20.0, 13.0, 10.0, 14.0, 0.63])))
# from trieste.space import Box
# xs = Box(ConceptualMarineDesign().bounds[0], ConceptualMarineDesign().bounds[1]).sample(1000000)
# cons_obs = ConceptualMarineDesign().constraint()(xs)
# print(tf.reduce_sum(tf.cast(tf.reduce_all(cons_obs >= 0 , axis=-1), dtype=tf.float64)) / 1000000)

# feasible region too small
# 0.025897

# res = moo_nsga2_pymoo(ConceptualMarineDesign().objective(), cons_num=8, cons=ConceptualMarineDesign().constraint(),
#                       input_dim= 6, obj_num= 3,
#                       bounds= tf.convert_to_tensor(ConceptualMarineDesign().bounds),
#                       popsize= 100,
#                       num_generation=1000)
# from matplotlib import pyplot as plt
# # res = DTLZ2(input_dim=4, num_objective=3).gen_pareto_optimal_points(100)
# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(projection='3d')
#
# ax.scatter(res.fronts[:, 0], res.fronts[:, 1], res.fronts[:, 2])
# plt.show()

# print(tf.reduce_max(res.fronts, axis=0))
#
# np.savetxt('ConceptualMarineDesign_6I_3O_PF_F.txt', res.fronts)
# np.savetxt('ConceptualMarineDesign_6I_3O_PF_X.txt', res.inputs)

pf = np.loadtxt(R'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\constraint_exp\cfg\ref_opts\ConceptualMarineDesign_6I_3O_PF_F.txt')
from trieste.acquisition.multi_objective.pareto import Pareto
print(Pareto(pf).hypervolume_indicator(tf.constant([-700, 13000, 3500], dtype=tf.float64)))
# print(np.max(pf, axis=0))