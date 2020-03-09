""" Mathematical model """
from optimizer import optimizer
import pandas
from model import data_handler
from model.nrc_equations import NRC_eq as nrc
import logging
import math

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

    p_id, p_feed_scenario, p_breed, p_sbw, p_bcs, p_be, p_l, p_sex, p_a2, p_ph, p_selling_price, \
    p_linearization_factor, p_algorithm, p_identifier, p_lb, p_ub, p_tol, p_lca_id, p_multiobjective, \
    p_obj = [None for i in range(20)]

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

    def _get_params(self, p_swg):
        if p_swg is None:
            return dict(zip(["CNEm", "MPm", "DMI", "NEm", "peNDF"],
                            [self._p_cnem, self._p_mpm * 0.001, self._p_dmi, self._p_nem, self._p_pe_ndf]))
        else:
            return dict(zip(["CNEm", "MPm", "DMI", "NEm", "SWG", "peNDF"],
                            [self._p_cnem, self._p_mpm * 0.001, self._p_dmi, self._p_nem, p_swg, self._p_pe_ndf]))

    def _solve(self, problem_id):
        """Return None if solution is infeasible or Solution dict otherwise"""
        diet = self._diet
        # diet.write_lp(name="CNEm_{}.lp".format(str(self._p_cnem)))
        diet.solve()
        status = diet.get_solution_status()
        logging.info("Solution status: {}".format(status))
        if status.__contains__("infeasible"):
            self._infeasible_output(problem_id)
            return None

        sol_id = {"Problem_ID": problem_id}
        sol = dict(zip(diet.get_variable_names(), diet.get_solution_vec()))
        sol["obj_func"] = diet.get_solution_obj()
        sol["obj_cost"] = 0
        sol["factor"] = (self._p_dmi - self._p_nem / self._p_cnem)
        sol["CNEg"] = 0
        sol["obj_revenue"] = 0
        for i in range(len(self._var_names_x)):
            sol["CNEg"] += self.neg_vector[i] * diet.get_solution_vec()[i]
            sol["obj_cost"] += diet.get_solution_vec()[i] * self.expenditure_obj_vector[i]
            sol["obj_revenue"] += diet.get_solution_vec()[i] * self.revenue_obj_vector[i]

        p_swg = nrc.swg(sol["CNEg"], self._p_dmi, self._p_cnem, self._p_nem, self.p_sbw, self.p_linearization_factor)
        params = self._get_params(p_swg)
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

    def _infeasible_output(self, problem_id):
        sol_id = {"Problem_ID": self.prefix_id + str(problem_id)}
        params = self._get_params(p_swg=None)
        sol = {**sol_id, **params}
        self.opt_sol = None
        # diet.write_lp(f"lp_infeasible_{str(problem_id)}.lp")
        logging.warning("Infeasible parameters:{}".format(sol))

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

        self.data_feed_scenario = self.ds.data_feed_scenario
        self.headers_feed_scenario = self.ds.headers_feed_scenario

        [self.p_id, self.p_feed_scenario, self.p_breed, self.p_sbw, self.p_bcs, self.p_be, self.p_l, self.p_sex,
         self.p_a2, self.p_ph,self.p_selling_price, self.p_linearization_factor, self.p_algorithm, self.p_identifier,
         self.p_lb, self.p_ub, self.p_tol, self.p_lca_id, self.p_multiobjective, self.p_obj] = parameters.values()

        headers_feed_scenario = self.ds.headers_feed_scenario
        self.data_feed_scenario = self.ds.filter_column(self.ds.data_feed_scenario,
                                                        self.ds.headers_feed_scenario.s_feed_scenario,
                                                        self.p_feed_scenario)
        self.data_feed_scenario = self.ds.sort_df(self.data_feed_scenario, self.headers_feed_scenario.s_ID)

        self.ingredient_ids = list(
            self.ds.get_column_data(self.data_feed_scenario, self.headers_feed_scenario.s_ID, int))

        self.headers_feed_lib = self.ds.headers_feed_lib
        self.data_feed_lib = self.ds.filter_column(self.ds.data_feed_lib, self.headers_feed_lib.s_ID,
                                                   self.ingredient_ids)

        self.cost_vector = self.ds.sorted_column(self.data_feed_scenario, self.headers_feed_scenario.s_feed_cost,
                                                 self.ingredient_ids,
                                                 self.headers_feed_scenario.s_ID)
        self.neg_vector = self.ds.sorted_column(self.data_feed_lib, self.headers_feed_lib.s_NEga,
                                                self.ingredient_ids,
                                                self.headers_feed_lib.s_ID)
        self.n_ingredients = self.data_feed_scenario.shape[0]
        self.cost_vector = self.ds.sorted_column(self.data_feed_scenario, headers_feed_scenario.s_feed_cost,
                                                 self.ingredient_ids,
                                                 self.headers_feed_scenario.s_ID)
        dm_af_coversion = self.ds.sorted_column(self.data_feed_lib, self.headers_feed_lib.s_DM,
                                                self.ingredient_ids,
                                                self.headers_feed_lib.s_ID)
        for i in range(len(self.cost_vector)):
            self.cost_vector[i] /= dm_af_coversion[i]
        self.neg_vector = self.ds.sorted_column(self.data_feed_lib, self.headers_feed_lib.s_NEga,
                                                self.ingredient_ids,
                                                self.headers_feed_lib.s_ID)

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
            self.revenue_obj_vector[i] = self.p_selling_price * swg[i]
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
        pass

    def _build_model(self):
        """Build model (initially based on CPLEX 12.8.1)"""
        self._diet = optimizer.Optimizer()
        self._var_names_x = ["x" + str(f_id)
                             for f_id in self.ingredient_ids]

        diet = self._diet
        diet.set_sense(sense="max")

        x_vars = list(diet.add_variables(obj=self.cost_obj_vector,
                                         lb=self.ds.sorted_column(self.data_feed_scenario,
                                                                  self.headers_feed_scenario.s_min,
                                                                  self.ingredient_ids,
                                                                  self.headers_feed_scenario.s_ID),
                                         ub=self.ds.sorted_column(self.data_feed_scenario,
                                                                  self.headers_feed_scenario.s_max,
                                                                  self.ingredient_ids,
                                                                  self.headers_feed_scenario.s_ID),
                                         names=self._var_names_x))

        "Constraint: sum(x a) == CNEm"
        diet.add_constraint(names=["CNEm GE"],
                            lin_expr=[[x_vars, self.ds.sorted_column(self.data_feed_lib,
                                                                     self.headers_feed_lib.s_NEma,
                                                                     self.ingredient_ids,
                                                                     self.headers_feed_lib.s_ID)]],
                            rhs=[self._p_cnem * 0.999],
                            senses=["G"]
                            )
        diet.add_constraint(names=["CNEm LE"],
                            lin_expr=[[x_vars, self.ds.sorted_column(self.data_feed_lib,
                                                                     self.headers_feed_lib.s_NEma,
                                                                     self.ingredient_ids,
                                                                     self.headers_feed_lib.s_ID)]],
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
        mp_properties = self.ds.sorted_column(self.data_feed_lib,
                                              [self.headers_feed_lib.s_DM,
                                               self.headers_feed_lib.s_TDN,
                                               self.headers_feed_lib.s_CP,
                                               self.headers_feed_lib.s_RUP,
                                               self.headers_feed_lib.s_Forage,
                                               self.headers_feed_lib.s_Fat],
                                              self.ingredient_ids,
                                              self.headers_feed_lib.s_ID)
        mpm_list = [nrc.mp(*row) for row in mp_properties]

        for i, v in enumerate(mpm_list):
            mpm_list[i] = v - self.neg_vector[i] * (nrc.swg_const(self._p_dmi, self._p_cnem, self._p_nem,
                                                                  self.p_sbw, self.p_linearization_factor) * 268
                                                    - self.neg_vector[i] * 29.4) * 0.001 / self._p_dmi

        diet.add_constraint(names=["MPm"],
                            lin_expr=[[x_vars, mpm_list]],
                            rhs=[self._p_mpm * 0.001 / self._p_dmi],
                            senses=["G"]
                            )

        rdp_data = [(1 - self.ds.sorted_column(self.data_feed_lib,
                                               self.headers_feed_lib.s_RUP,
                                               self.ingredient_ids,
                                               self.headers_feed_lib.s_ID)[x_index])
                    * self.ds.sorted_column(self.data_feed_lib,
                                            self.headers_feed_lib.s_CP,
                                            self.ingredient_ids,
                                            self.headers_feed_lib.s_ID)[x_index]
                    for x_index in range(len(x_vars))]

        "Constraint: RUP: sum(x a) >= 0.125 CNEm"
        diet.add_constraint(names=["RDP"],
                            lin_expr=[[x_vars, rdp_data]],
                            rhs=[0.125 * self._p_cnem],
                            senses=["G"]
                            )

        "Constraint: Fat: sum(x a) <= 0.06 DMI"
        diet.add_constraint(names=["Fat"],
                            lin_expr=[[x_vars, self.ds.sorted_column(self.data_feed_lib,
                                                                     self.headers_feed_lib.s_Fat,
                                                                     self.ingredient_ids,
                                                                     self.headers_feed_lib.s_ID)]],
                            rhs=[0.06],
                            senses=["L"]
                            )

        "Constraint: peNDF: sum(x a) <= peNDF DMI"
        pendf_data = [self.ds.sorted_column(self.data_feed_lib,
                                            self.headers_feed_lib.s_NDF,
                                            self.ingredient_ids,
                                            self.headers_feed_lib.s_ID)[x_index]
                      * self.ds.sorted_column(self.data_feed_lib,
                                              self.headers_feed_lib.s_pef,
                                              self.ingredient_ids,
                                              self.headers_feed_lib.s_ID)[x_index]
                      for x_index in range(len(x_vars))]
        diet.add_constraint(names=["peNDF"],
                            lin_expr=[[x_vars, pendf_data]],
                            rhs=[self._p_pe_ndf],
                            senses=["G"]
                            )

        self.constraints_names = diet.get_constraints_names()
        # diet.write_lp(name="file.lp")
        pass

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

    def _rebuild_model(self, p_cnem):
        """
        DO NOT CALL THIS THING OUTSIDE THIS CLASS
        """
        try:
            self.opt_sol = None
            self._p_cnem = p_cnem
            self._compute_parameters()
            if self._diet is None:
                self._build_model()
            else:
                self._update_model()
            return self._diet
        except Exception as e:
            logging.error("An error occurred:\n{}".format(str(e)))
            return None


class ModelLCA(Model):
    headers_lca_scenario: data_handler.Data.LCAScenario = None  # LCA
    data_lca_scenario: pandas.DataFrame = None  # LCA
    headers_lca_lib: data_handler.Data.LCALib = None  # LCA
    data_lca_lib: pandas.DataFrame = None  # LCA Library

    lca_weights:dict = None
    ghg_cost, methane_eq = None, None  # type: float
    lca_vector:list = None
    methane_vector_GE20: list = None
    methane_vector_LE20: list = None
    methane_obj_vector: list = None
    lca_obj_vector:list = None
    weight_profit = None
    weight_lca = None
    lca_rhs = None
    forage_sense = None

    def _cast_data(self, out_ds, parameters):
        Model._cast_data(self, out_ds, parameters)
        self.headers_lca_scenario = self.ds.headers_lca_scenario
        self.data_lca_scenario = self.ds.filter_column(self.ds.data_lca_scenario, self.ds.headers_lca_scenario.s_ID,
                                                       self.p_lca_id)

        self.weight_lca = list(self.data_lca_scenario[self.ds.headers_lca_scenario.s_LCA_weight])[0]
        if math.isnan(self.weight_lca):
            self.weight_lca = 0.0

        self.weight_profit = 1.0 - self.weight_lca
        self.data_lca_lib = self.ds.filter_column(self.ds.data_lca_lib,
                                                  self.ds.headers_lca_lib.s_ing_id,
                                                  self.ingredient_ids)

        self.data_lca_lib = self.ds.sort_df(self.data_lca_lib, self.ds.headers_lca_lib.s_ing_id)
        if self.ingredient_ids != list(self.data_lca_lib[self.ds.headers_lca_lib.s_ing_id]):
            raise IndexError("LCA Library does not match all ingredients in Feeds")

        self.lca_weights = {}
        for h in self.ds.headers_lca_lib:
            if "LCA" in h:
                new_h = f'{h}_weight'
                self.lca_weights[h] = list(self.data_lca_scenario[new_h])[0]

        self.ghg_cost = list(self.data_lca_scenario[self.ds.headers_lca_scenario.s_LCA_cost])[0]
        if list(self.data_lca_scenario[self.ds.headers_lca_scenario.s_Methane])[0]:
            self.methane_eq = list(self.data_lca_scenario[self.ds.headers_lca_scenario.s_Methane_Equation])[0]

        self.lca_vector = []
        self.methane_vector_GE20 = []
        self.methane_vector_LE20 = []
        for i, x in enumerate(self.data_feed_lib.values):
            feed_properties = self.ds.map_values(list(self.data_feed_lib), x)
            methane_ge20, methane_le20 = nrc.ch4_diet(feed_properties[self.headers_feed_lib.s_Fat],
                                                      feed_properties[self.headers_feed_lib.s_CP],
                                                      feed_properties[self.headers_feed_lib.s_NDF],
                                                      feed_properties[self.headers_feed_lib.s_Starch],
                                                      feed_properties[self.headers_feed_lib.s_Sugars],
                                                      feed_properties[self.headers_feed_lib.s_OA])
            self.methane_vector_GE20.append(methane_ge20)
            self.methane_vector_LE20.append(methane_le20)

        for id in self.ingredient_ids:
            row = self.ds.filter_column(self.data_lca_lib, self.ds.headers_lca_lib.s_ing_id, id)
            sum_lca = 0
            for lca in self.lca_weights.keys():
                sum_lca += float(list(row[lca])[0]) * self.lca_weights[lca]
            self.lca_vector.append(sum_lca)
        pass

    def _compute_parameters(self):
        Model._compute_parameters(self)

        # Set multi-objective weighted function
        self.lca_obj_vector = [(-1) * (self._p_dmi * lca) * self.ghg_cost for lca in self.lca_vector]
        if self.forage_sense == 'L':
            self.methane_obj_vector = [(-1) * (self._p_dmi * ch4) * self.ghg_cost for ch4 in self.methane_vector_LE20]
        else:
            self.methane_obj_vector = [(-1) * (self._p_dmi * ch4) * self.ghg_cost for ch4 in self.methane_vector_GE20]

        for i in range(len(self.cost_vector)):
            self.cost_obj_vector[i] = self.cost_obj_vector[i] * self.weight_profit + \
                                      + (self.lca_obj_vector[i] + self.methane_obj_vector[i]) * self.weight_lca
        pass

    def get_obj_sol(self, solution):
        # Extract lca and profit(cost, profit/swg) components of the objective function
        lca_value = 0
        profit = 0
        if solution is None:
            raise Exception
        model = self._rebuild_model(solution['CNEm'])
        if model is not None:
            for i, x in enumerate(self._var_names_x):
                lca_value += solution[x] * (self.lca_obj_vector[i] + self.methane_obj_vector[i])
                profit += solution[x] * self.cost_obj_vector[i]
            return [profit, -lca_value / (self.ghg_cost * self._p_dmi)]
        else:
            raise Exception(f"Could not recreate model:{solution}")

    def set_LCA_rhs(self, val):
        # Set rhs and removes LCA from objective function
        self.lca_rhs = val
        self.set_obj_weights(1.0, 0.0)

    def set_obj_weights(self, f1_w, lca_w):
        self.weight_profit = f1_w
        self.weight_lca = lca_w

    def set_lca_weight(self):
        self.weight_lca = list(self.data_lca_scenario[self.ds.headers_lca_scenario.s_LCA_weight])[0]
        self.weight_profit = 1 - self.weight_lca

    def set_forage(self, ge_or_le):
        self.forage_sense = ge_or_le

    def _build_model(self):
        Model._build_model(self)
        if self.lca_rhs is None:
            self.lca_rhs = sum(self.lca_vector)
            constraint_vector = self.lca_vector
        elif self.lca_rhs == 'L':
            constraint_vector = [self.lca_vector[i] + self.methane_vector_LE20[i] for i in range(len(self.lca_vector))]
        elif self.lca_rhs == 'G':
            constraint_vector = [self.lca_vector[i] + self.methane_vector_GE20[i] for i in range(len(self.lca_vector))]

        "Constraint: sum(x lca) <= LCA_rhs"
        self._diet.add_constraint(names=["LCA epsilon"],
                                  lin_expr=[[self._var_names_x, constraint_vector]],
                                  rhs=[self.lca_rhs],
                                  senses=["L"]
                                  )

        "Constraint: sum(x forage) [<= >=] 20%"
        self._diet.add_constraint(names=["Forage Content"],
                                  lin_expr=[[self._var_names_x, self.ds.sorted_column(self.data_feed_lib,
                                                                                      self.headers_feed_lib.s_Forage,
                                                                                      self.ingredient_ids,
                                                                                      self.headers_feed_lib.s_ID)]],
                                  rhs=[10],
                                  senses=['L']
                                  )
        self.constraints_names = self._diet.get_constraints_names()

    def write_lp_inside(self, id):
        self._diet.write_lp(name=f"file_{id}.lp")

    def _update_model(self):
        """Update RHS values on the model based on the new CNEm and updated parameters"""
        new_rhs = {
            "CNEm GE": self._p_cnem * 0.999,
            "CNEm LE": self._p_cnem * 1.001,
            "SUM 1": 1,
            "MPm": self._p_mpm * 0.001 / self._p_dmi,
            "RDP": 0.125 * self._p_cnem,
            "Fat": 0.06,
            "peNDF": self._p_pe_ndf,
            "LCA epsilon": self.lca_rhs}

        seq_of_pairs = tuple(zip(new_rhs.keys(), new_rhs.values()))
        self._diet.set_constraint_rhs(seq_of_pairs)
        if self.forage_sense is not None:
            self._diet.set_constraint_sense("Forage Content", self.forage_sense)
            new_rhs["Forage Content"] = 0.2
            lca_cst = []
            if self.forage_sense == 'L':
                lca_cst = [self.lca_vector[i] + self.methane_vector_LE20[i] for i in range(len(self.lca_vector))]

            elif self.forage_sense == 'G':
                lca_cst = [self.lca_vector[i] + self.methane_vector_GE20[i] for i in range(len(self.lca_vector))]

            seq_of_triplets = tuple(zip(["LCA epsilon" for i in range(len(self._var_names_x))],
                                        self._var_names_x,
                                        lca_cst))
            self._diet.set_constraint_coefficients(seq_of_triplets)

        self._diet.set_objective_function(list(zip(self._var_names_x, self.cost_obj_vector)))

    def _get_params(self, p_swg):
        if p_swg is None:
            return dict(zip(["CNEm", "MPm", "DMI", "NEm", "peNDF", "LCA"],
                            [self._p_cnem, self._p_mpm * 0.001, self._p_dmi, self._p_nem, self._p_pe_ndf, self.lca_rhs]))
        else:
            return Model._get_params(self, p_swg)

    def _solve(self, problem_id):
        sol = Model._solve(self, problem_id)
        if sol is None:
            return None
        diet = self._diet
        lca_cost = 0
        for i, x in enumerate(self._var_names_x):
            lca_cost += diet.get_solution_vec()[i] * self.lca_obj_vector[i]

        sol["Forage Sense"] = self.forage_sense
        sol_methane_feed = {}
        if self.forage_sense == "L":
            sol_methane_feed = dict(zip([f"{var}_methane_share x dmi" for var in diet.get_variable_names()],
                                        [var * self._p_dmi for var in self.methane_vector_LE20]))
        elif self.forage_sense == 'G':
            sol_methane_feed = dict(zip([f"{var}_methane_share x dmi" for var in diet.get_variable_names()],
                                        [var * self._p_dmi for var in self.methane_vector_GE20]))
        sol["Methane Cost Obj"] = 0
        for i in range(len(self._var_names_x)):
            sol["Methane Cost Obj"] += self.methane_obj_vector[i] * diet.get_solution_vec()[i]
        sol["Enteric Emission"] = sol["Methane Cost Obj"] / self.ghg_cost
        sol["LCA Cost Obj"] = lca_cost
        sol["LCA Emission"] = -(lca_cost / (self.ghg_cost * self._p_dmi))
        sol["LCA weight"] = self.weight_lca
        sol["Profit weight"] = self.weight_profit
        sol["Real Obj Profit - (LCA Cost + Mehtane Cost)"] = sol["obj_revenue"] - sol["obj_cost"] \
                                                             + (lca_cost + sol["Methane Cost Obj"])  # values are negative already

        sol = {**sol, **sol_methane_feed}
        return sol
