from trieste.objectives.multi_objectives import EE6
import  tensorflow as tf
from pymoo.core.problem import Problem
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from trieste.acquisition.multi_objective.utils import MOOResult

input_dim = 4

fun = EE6(input_dim=input_dim, exp_id=None).joint_objective_con()
lb = np.array([0.0] * input_dim)
ub = np.array([1.0] * input_dim)
obj_num = 2
popsize = 30
num_generation = 400

# fun(tf.constant([[7.84676117, 2.57460561,     0.2  ,      9.0000197]]))
# raise ValueError

class MyProblem(Problem):
    def __init__(self, n_var: int, n_obj: int, n_constr: int = 1):
        """
        :param n_var input variables
        :param n_obj number of objective functions
        :param n_constr number of constraint numbers
        """
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=lb,
            xu=ub,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        objs, vals = fun(tf.convert_to_tensor(x))
        out["F"] = objs
        out["G"] = -vals  # in pymoo, by default <0 is feasible


problem = MyProblem(n_var=input_dim, n_obj=obj_num)
algorithm = NSGA2(  # we keep the hyperparameter for NSGA2 fixed here
    pop_size=popsize,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True,
)

try:
    res = minimize(problem, algorithm, ("n_gen", num_generation), save_history=False, verbose=True)
except:
    raise

# np.savetxt('EE6_4D_front.txt', MOOResult(res).fronts)
# np.savetxt('EE6_4D_input.txt', MOOResult(res).inputs)
# np.savetxt('EE6_4D_constraint.txt', MOOResult(res).constraint)