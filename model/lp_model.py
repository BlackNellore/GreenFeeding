""" Mathematical model """
from optimizer import optimizer
import pandas
from model import data_handler
from model.nrc_equations import NRC_eq as nrc
import logging
from model.data_handler import Data

cnem_lb, cnem_ub = 0.8, 3

bigM = 100000


def model_factory(ds, parameters, lca=-1):
    if lca <= 0:
        return Model(ds, parameters)
    else:
        return ModelLCA(ds, parameters)


class Model:
    ds: data_handler.Data = None
    headers_feed_lib: data_handler.Data.IngredientProperties = None  # Feed Library
    data_feed_lib: pandas.DataFrame = None  # Feed Library
    data_feed_scenario: pandas.DataFrame = None  # Feeds
    headers_feed_scenario: data_handler.Data.ScenarioFeedProperties = None  # Feeds
    data_scenario: pandas.DataFrame = None  # Scenario
    headers_scenario: data_handler.Data.ScenarioParameters = None  # Scenario

    p_id, p_feed_scenario, p_breed, p_sbw, p_bcs, p_be, p_l, p_sex, p_a2, p_ph, p_selling_price, p_linearization_factor, \
    p_algorithm, p_identifier, p_lb, p_ub, p_tol, p_lca_id, p_obj = [None for i in range(19)]

    _diet = None
    _p_mpm = None
    _p_dmi = None
    _p_nem = None
    _p_pe_ndf = None
    _p_cnem = None
    _var_names_x = None

    _print_model_lp = False
    _print_model_lp_infeasible = False
    _print_solution_xml = False

    opt_sol = None
    prefix_id = ""

    def __init__(self, out_ds, parameters):
        self._cast_data(out_ds, parameters)

    def run(self, p_id, p_cnem):
        """Either build or update model, solve ir and return solution = {dict xor None}"""
        logging.info("Populating and running model")
        try:
            self.opt_sol = None
            self._p_cnem = p_cnem
            self._compute_parameters()
            if self._diet is None:
                self._build_model()
            else:
                self._update_model()
            return self._solve(p_id)
        except Exception as e:
            logging.error("An error occurred:\n{}".format(str(e)))
            return None

    def _solve(self, problem_id):
        """Return None if solution is infeasible or Solution dict otherwise"""
        diet = self._diet
        # diet.write_lp(name="CNEm_{}.lp".format(str(self._p_cnem)))
        diet.solve()
        status = diet.get_solution_status()
        logging.info("Solution status: {}".format(status))
        if status.__contains__("infeasible"):
            sol_id = {"Problem_ID": self.prefix_id + str(problem_id)}
            params = dict(zip(["CNEm", "MPm", "DMI", "NEm", "peNDF"],
                              [self._p_cnem, self._p_mpm * 0.001, self._p_dmi, self._p_nem, self._p_pe_ndf]))
            sol = {**sol_id, **params}
            self.opt_sol = None
            logging.warning("Infeasible parameters:{}".format(sol))
            return None

        sol_id = {"Problem_ID": problem_id}
        sol = dict(zip(diet.get_variable_names(), diet.get_solution_vec()))
        sol["obj_func"] = diet.get_solution_obj()
        sol["obj_cost"] = 0
        sol["factor"] = (self._p_dmi - self._p_nem / self._p_cnem)
        sol["CNEg"] = 0
        sol["obj_revenue"] = 0
        for i in range(len(self.cost_vector)):
            sol["CNEg"] += self.neg_vector[i] * diet.get_solution_vec()[i]
            sol["obj_cost"] += diet.get_solution_vec()[i] * self.expenditure_obj_vector[i]
            sol["obj_revenue"] += diet.get_solution_vec()[i] * self.revenue_obj_vector[i]

        p_swg = nrc.swg(sol["CNEg"], self._p_dmi, self._p_cnem, self._p_nem, self.p_sbw, self.p_linearization_factor)
        params = dict(zip(["CNEm", "MPm", "DMI", "NEm", "SWG", "peNDF"],
                          [self._p_cnem, self._p_mpm * 0.001, self._p_dmi, self._p_nem, p_swg, self._p_pe_ndf]))
        sol_activity = dict(zip(["{}_act".format(constraint) for constraint in self.constraints_names],
                                diet.get_solution_activity_levels(self.constraints_names)))
        sol_rhs = dict(zip(["{}_rhs".format(constraint) for constraint in self.constraints_names],
                           diet.get_constraints_rhs(self.constraints_names)))
        sol_red_cost = dict(zip(["{}_red_cost".format(var) for var in diet.get_variable_names()],
                                diet.get_dual_values()))
        sol_dual = dict(zip(["{}_dual".format(const) for const in diet.get_constraints_names()],
                            diet.get_dual_reduced_costs()))
        sol_slack = dict(zip(["{}_slack".format(const) for const in diet.get_constraints_names()],
                             diet.get_dual_linear_slacks()))
        sol_obj_cost = dict(zip(["{}_obj_cneg".format(var) for var in diet.get_variable_names()],
                                self.neg_vector))
        sol = {**sol_id, **params, **sol, **sol_rhs, **sol_activity,
               **sol, **sol_dual, **sol_red_cost, **sol_slack, **sol_obj_cost}
        self.opt_sol = diet.get_solution_obj()

        return sol

    # Parameters filled by inner method ._cast_data()
    n_ingredients = None
    cost_vector = None
    neg_vector = None
    cost_obj_vector = None
    constraints_names = None
    revenue_obj_vector = None
    expenditure_obj_vector = None

    def _cast_data(self, out_ds, parameters):
        """Retrieve parameters data from table. See data_handler.py for more"""
        self.ds = out_ds

        self.data_feed_lib = self.ds.data_feed_lib
        self.headers_feed_lib = self.ds.headers_feed_lib
        self.data_feed_scenario = self.ds.data_feed_scenario
        self.headers_feed_scenario = self.ds.headers_feed_scenario

        self.n_ingredients = self.data_feed_scenario.last_valid_index() + 1
        self.cost_vector = self.ds.get_column_data(self.data_feed_scenario, self.headers_feed_scenario.s_feed_cost)
        self.neg_vector = self.ds.get_column_data(self.data_feed_lib, self.headers_feed_lib.s_NEga)
        [self.p_id, self.p_feed_scenario, self.p_breed, self.p_sbw, self.p_bcs, self.p_be, self.p_l, self.p_sex, self.p_a2, self.p_ph,
         self.p_selling_price, self.p_linearization_factor,
         self.p_algorithm, self.p_identifier, self.p_lb, self.p_ub, self.p_tol, self.p_lca_id, self.p_obj] = parameters.values()

        ingredients = self.ds.data_feed_lib
        h_ingredients = self.ds.headers_feed_lib
        headers_feed_scenario = self.ds.headers_feed_scenario
        self.data_feed_scenario = self.ds.filter_column(self.ds.data_feed_scenario,
                                                        self.ds.headers_feed_scenario.s_feed_scenario,
                                                        self.p_feed_scenario)

        self.n_ingredients = self.data_feed_scenario.shape[0]
        self.cost_vector = self.ds.get_column_data(self.data_feed_scenario, headers_feed_scenario.s_feed_cost)
        dm_af_coversion = self.ds.get_column_data(ingredients, h_ingredients.s_DM)
        for i in range(len(self.cost_vector)):
            self.cost_vector[i] /= dm_af_coversion[i]
        self.neg_vector = self.ds.get_column_data(ingredients, h_ingredients.s_NEga)

    def _compute_parameters(self):
        """Compute parameters variable with CNEm"""
        self._p_mpm, self._p_dmi, self._p_nem, self._p_pe_ndf = \
            nrc.get_all_parameters(self._p_cnem, self.p_sbw, self.p_bcs,
                                   self.p_be, self.p_l, self.p_sex, self.p_a2, self.p_ph)

        self.cost_obj_vector = self.cost_vector.copy()
        self.revenue_obj_vector = self.cost_vector.copy()
        self.expenditure_obj_vector = self.cost_vector.copy()
        swg = []
        for i in range(len(self.cost_vector)):
            swg.append(nrc.swg(self.neg_vector[i], self._p_dmi, self._p_cnem,
                               self._p_nem, self.p_sbw, self.p_linearization_factor))
            self.revenue_obj_vector[i] = \
                self.p_selling_price * swg[i]
            self.expenditure_obj_vector[i] = self.cost_vector[i] * self._p_dmi
        r = [self.revenue_obj_vector[i] - self.expenditure_obj_vector[i] for i in range(len(self.revenue_obj_vector))]
        if self.p_obj == "MaxProfit":
            for i in range(len(self.cost_vector)):
                self.cost_obj_vector[i] = self.revenue_obj_vector[i] - self.expenditure_obj_vector[i]
        elif self.p_obj == "MinCost":
            for i in range(len(self.cost_vector)):
                self.cost_obj_vector[i] = - self.expenditure_obj_vector[i]
        elif self.p_obj == "MaxProfitSWG":
            for i in range(len(self.cost_vector)):
                if swg[i] == 0:
                    swg[i] = 1/bigM
                self.cost_obj_vector[i] = (self.revenue_obj_vector[i] - self.expenditure_obj_vector[i])/swg[i]

    def _build_model(self):
        """Build model (initially based on CPLEX 12.8.1)"""
        self._diet = optimizer.Optimizer()
        self._var_names_x = ["x" + str(j) for j in range(self.n_ingredients)]

        diet = self._diet
        diet.set_sense(sense="max")

        x_vars = list(diet.add_variables(obj=self.cost_obj_vector,
                                         lb=self.ds.get_column_data(self.data_feed_scenario, self.headers_feed_scenario.s_min),
                                         ub=self.ds.get_column_data(self.data_feed_scenario, self.headers_feed_scenario.s_max),
                                         names=self._var_names_x))

        "Constraint: sum(x a) == CNEm"
        diet.add_constraint(names=["CNEm GE"],
                            lin_expr=[[x_vars, self.ds.get_column_data(self.data_feed_lib, self.headers_feed_lib.s_NEma)]],
                            rhs=[self._p_cnem * 0.999],
                            senses=["G"]
                            )
        diet.add_constraint(names=["CNEm LE"],
                            lin_expr=[[x_vars, self.ds.get_column_data(self.data_feed_lib, self.headers_feed_lib.s_NEma)]],
                            rhs=[self._p_cnem * 1.001],
                            senses=["L"]
                            )
        "Constraint: sum(x) == 1"
        diet.add_constraint(names=["SUM 1"],
                            lin_expr=[[x_vars, [1] * len(x_vars)]],
                            rhs=[1],
                            senses=["E"]
                            )
        "Constraint: sum(x a)>= MPm"
        mpm_list = [nrc.mp(*self.ds.get_column_data(
            self.ds.filter_column(self.data_feed_lib, self.headers_feed_lib.s_ID, val_col),
            [self.headers_feed_lib.s_DM,
             self.headers_feed_lib.s_TDN,
             self.headers_feed_lib.s_CP,
             self.headers_feed_lib.s_RUP,
             self.headers_feed_lib.s_Forage,
             self.headers_feed_lib.s_Fat]))
                    for val_col in self.ds.get_column_data(self.data_feed_lib, self.headers_feed_lib.s_ID, int)]

        for i, v in enumerate(mpm_list):
            mpm_list[i] = v - self.neg_vector[i] * (nrc.swg_const(self._p_dmi, self._p_cnem, self._p_nem,
                                                                  self.p_sbw, self.p_linearization_factor) * 268
                                                    - self.neg_vector[i] * 29.4) * 0.001 / self._p_dmi

        diet.add_constraint(names=["MPm"],
                            lin_expr=[[x_vars, mpm_list]],
                            rhs=[self._p_mpm * 0.001 / self._p_dmi],
                            senses=["G"]
                            )

        rdp_data = [(1 - self.ds.get_column_data(self.data_feed_lib, self.headers_feed_lib.s_RUP)[x_index])
                    * self.ds.get_column_data(self.data_feed_lib, self.headers_feed_lib.s_CP)[x_index]
                    for x_index in range(len(x_vars))]

        "Constraint: RUP: sum(x a) >= 0.125 CNEm"
        diet.add_constraint(names=["RDP"],
                            lin_expr=[[x_vars, rdp_data]],
                            rhs=[0.125 * self._p_cnem],
                            senses=["G"]
                            )

        "Constraint: Fat: sum(x a) <= 0.06 DMI"
        diet.add_constraint(names=["Fat"],
                            lin_expr=[[x_vars, self.ds.get_column_data(self.data_feed_lib, self.headers_feed_lib.s_Fat)]],
                            rhs=[0.06],
                            senses=["L"]
                            )

        "Constraint: peNDF: sum(x a) <= peNDF DMI"
        pendf_data = [self.ds.get_column_data(self.data_feed_lib, self.headers_feed_lib.s_NDF)[x_index]
                      * self.ds.get_column_data(self.data_feed_lib, self.headers_feed_lib.s_pef)[x_index]
                      for x_index in range(len(x_vars))]
        diet.add_constraint(names=["peNDF"],
                            lin_expr=[[x_vars, pendf_data]],
                            rhs=[self._p_pe_ndf],
                            senses=["G"]
                            )

        # TODO: Put constraint to limit Urea in the diet: sum(x) * DMI <= up_limit

        self.constraints_names = diet.get_constraints_names()

    def _update_model(self):
        """Update RHS values on the model based on the new CNEm and updated parameters"""
        new_rhs = {
            "CNEm GE": self._p_cnem * 0.999,
            "CNEm LE": self._p_cnem * 1.001,
            "SUM 1": 1,
            "MPm": self._p_mpm * 0.001 / self._p_dmi,
            "RDP": 0.125 * self._p_cnem,
            "Fat": 0.06,
            "peNDF": self._p_pe_ndf}

        seq_of_pairs = tuple(zip(new_rhs.keys(), new_rhs.values()))
        self._diet.set_constraint_rhs(seq_of_pairs)
        self._diet.set_objective_function(list(zip(self._var_names_x, self.cost_obj_vector)))


class ModelLCA(Model):
    headers_lca_scenario: data_handler.Data.LCAScenario = None  # LCA
    data_lca_scenario: pandas.DataFrame = None  # LCA
    headers_lca_lib: data_handler.Data.LCALib = None  # LCA
    data_lca_lib: pandas.DataFrame = None  # LCA Library

    lca_weights = None  # type: dict
    lca_values = None  # type: dict
    lca_cost, methane_eq = None, None  # type: float
    lca_vector = None  # type: list
    weight_profit = 1
    weight_lca = 1

    def _cast_data(self, out_ds, parameters):
        Model._cast_data(self, out_ds, parameters)
        self.headers_lca_scenario = self.ds.headers_lca_scenario
        self.data_lca_scenario = self.ds.filter_column(self.ds.data_lca_scenario, self.ds.headers_lca_scenario.s_ID,
                                                  self.p_lca_id)
        self.data_lca_lib = self.ds.filter_column(self.ds.data_lca_lib,
                                                  self.ds.headers_lca_lib.s_ing_id,
                                                  list(self.data_feed_scenario[self.headers_feed_scenario.s_ID]))

        self.lca_weights = {}
        for h in self.headers_lca_scenario:
            # TODO: maybe parametrize trhis in config.py?
            if "LCA_" in h:
                new_h = h.replace("_weight", "")
                self.lca_weights[new_h] = list(self.data_lca_scenario[h])[0]

        self.lca_cost = list(self.data_lca_scenario[self.ds.headers_lca_scenario.s_LCA_cost])[0]
        if list(self.data_lca_scenario[self.ds.headers_lca_scenario.s_Methane])[0]:
            self.methane_eq = list(self.data_lca_scenario[self.ds.headers_lca_scenario.s_Methane_Equation])[0]

        self.lca_vector = [0 for i in range(self.data_feed_scenario.shape[0])]
        for i in range(len(self.lca_vector)):
            for lca in self.lca_weights.keys():
                self.lca_vector[i] += list(self.data_lca_lib[lca])[i] * self.lca_weights[lca]
        pass

    def _compute_parameters(self):
        Model._compute_parameters(self)
        for i in range(len(self.cost_vector)):
            self.cost_obj_vector[i] = self.cost_obj_vector[i] * self.weight_profit + \
                                      (-1) * (self._p_dmi * self.lca_vector[i]) * self.lca_cost * self.weight_lca
        pass
