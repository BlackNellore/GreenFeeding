""" Mathematical model """
from config import SOLVER, RNS_FEED_PARAMETERS
from model import data_handler
from model.nrc_equations import NRC_eq
import logging
import math
import pyomo.environ as pyo
from pyomo.opt.results import SolverResults
import pandas as pd

default_special_cost = 10.0

nrc = NRC_eq(**RNS_FEED_PARAMETERS)


def model_factory(ds, parameters, special_product=-1, multi_objective_id=-1):
    if special_product > 0:
        if multi_objective_id <= 0:
            return ModelReducedCost(ds, parameters, special_product)
        else:
            # return ModelLCAReducedCost(ds, parameters, special_product)
            return None
    else:
        if multi_objective_id <= 0:
            return Model(ds, parameters)
        else:
            return ModelLCA(ds, parameters)


class Model:
    _diet: pyo.ConcreteModel = None

    _print_model_lp = False
    _print_model_lp_infeasible = False
    _print_solution_xml = False

    prefix_id = ""

    data = None
    parameters = None
    computed = None

    def __init__(self, out_ds, parameters):
        if out_ds is None or parameters is None:
            # skip initialization for inherited classes
            return
        self.parameters = self.Parameters(parameters)
        self.data = self.Data(out_ds, self.parameters)
        self.computed = self.ComputedArrays()

    def run(self, p_id, p_cnem):
        """Either build or update model, solve it and return solution = {dict xor None}"""
        logging.info("Populating and running model")
        try:
            self.parameters.cnem = p_cnem
            if self.parameters.p_batch > 0:
                self._setup_batch()
            if not self._compute_parameters():
                self._infeasible_output(p_id)
                return None
            if self._diet is None:
                self._build_model()
            self._update_model()
            return self._solve(p_id)
        except Exception as e:
            logging.error("An error occurred in lp_model.py L86:\n{}".format(str(e)))
            return None

    def _get_params(self, p_swg):
        if p_swg is None:
            return dict(zip(["CNEm", "CNEg", "NEm", "NEg", "DMI", "MPm", "MPg", "MPr", "peNDF"],
                            [self.parameters.cnem, self.parameters.cneg, self.parameters.nem, self.parameters.neg,
                             self.parameters.dmi, self.parameters.mpmr * 0.001, self.parameters.mpgr * 0.001,
                             self.parameters.mpr * 0.001, self.parameters.pe_ndf]))
        else:
            return dict(zip(["CNEm", "CNEg", "NEm", "NEg", "SWG", "DMI", "MPm", "MPg", "MPr","peNDF"],
                            [self.parameters.cnem, self.parameters.cneg, self.parameters.nem, self.parameters.neg,
                             p_swg, self.parameters.dmi, self.parameters.mpmr * 0.001, self.parameters.mpgr * 0.001,
                             self.parameters.mpr * 0.001, self.parameters.pe_ndf]))

    def _solve(self, problem_id):
        """Return None if solution is infeasible or Solution dict otherwise"""
        if SOLVER == "" or SOLVER is None:
            if pyo.SolverFactory('cplex').available():
                slv = pyo.SolverFactory('cplex')
            elif pyo.SolverFactory('glpk').available():
                slv = pyo.SolverFactory('glpk')
            else:
                raise Exception("Solver not available")
        else:
            if pyo.SolverFactory(SOLVER).available():
                slv = pyo.SolverFactory(SOLVER)
            else:
                raise Exception("Solver not available")

        results = SolverResults()
        r = slv.solve(self._diet)
        results.load(r)
        if not (results.solver.status == pyo.SolverStatus.ok and
                results.solver.termination_condition == pyo.TerminationCondition.optimal):
            logging.info("Solution status: {}".format(results.solver.termination_condition))
            self._infeasible_output(problem_id)
            return None

        sol_id = {"Problem_ID": problem_id,
                  "Feeding Time": self.parameters.c_model_feeding_time,
                  "Initial weight": self.parameters.p_sbw,
                  "Final weight": self.parameters.c_model_final_weight}

        params = self._get_params(self.parameters.c_swg)

        sol = dict(zip([f"x{i} - {self.data.d_name_ing_map[i]}" for i in self._diet.v_x],
                       [self._diet.v_x[i].value for i in self._diet.v_x]))
        sol["obj_func"] = self._diet.f_obj.expr()
        sol["obj_cost"] = - self._diet.f_obj.expr() + self.computed.cst_obj
        if self.parameters.p_obj == "MaxProfitSWG" or self.parameters.p_obj == "MinCostSWG":
            sol["obj_cost"] *= self.parameters.c_swg
        sol["obj_revenue"] = self.computed.revenue

        sol["MP diet"] = self._diet.c_mpm.body()

        is_active_constraints = []
        l_slack = {}
        u_slack = {}
        lower = {}
        upper = {}
        duals = {}

        for c in self._diet.component_objects(pyo.Constraint):
            is_active_constraints.append(c.active)
            if c.active:
                duals["{}_dual".format(c)] = self._diet.dual[c]
                l_slack["{}_lslack".format(c)] = c.lslack()
                u_slack["{}_uslack".format(c)] = c.uslack()
                if c.has_lb():
                    lower["{}_lower".format(c)] = c.lower()
                    upper["{}_upper".format(c)] = "None"
                else:
                    lower["{}_lower".format(c)] = "None"
                    upper["{}_upper".format(c)] = c.upper()
            else:
                duals["{}_dual".format(c)] = "None"
                l_slack["{}_lslack".format(c)] = "None"
                u_slack["{}_uslack".format(c)] = "None"
                lower["{}_lower".format(c)] = "None"
                upper["{}_upper".format(c)] = "None"

        sol_red_cost = dict(zip(["x{}_red_cost".format(i) for i in self._diet.v_x],
                                [self._diet.rc[self._diet.v_x[i]] for i in self._diet.v_x]))

        sol_fat_orient = {"fat orient": self.parameters.p_fat_orient}

        sol = {**sol_id, **params, **sol, **sol_red_cost, **duals,
               **sol_fat_orient, **l_slack, **u_slack, **lower, **upper}

        return sol

    def _infeasible_output(self, problem_id):
        sol_id = {"Problem_ID": self.prefix_id + str(problem_id),
                  "Feeding Time": self.parameters.c_model_feeding_time,
                  "Initial weight": self.parameters.p_sbw,
                  "Final weight": self.parameters.c_model_final_weight}
        params = self._get_params(p_swg=None)
        sol = {**sol_id, **params}
        logging.warning("Infeasible parameters:{}".format(sol))

    class ComputedArrays:
        # Initialized in Model
        # Populated in method _compute_parameters
        revenue = None
        cst_obj = None
        dc_expenditure = None
        dc_obj_func = None
        dc_mp = None

        def __init__(self):
            pass

    class Parameters:
        # Initialized and Populated in Model
        mpmr = None
        mpgr = None
        mpr = None
        dmi = None
        nem = None
        neg = None
        pe_ndf = None
        cnem = None
        cneg = None

        # External assignment
        p_batch_execution_id = None
        p_fat_orient = None

        # Computed in Model
        c_swg = None
        c_model_feeding_time = None
        c_model_final_weight = None
        c_batch_map: dict = None

        # From outer scope
        [p_id, p_feed_scenario, p_batch, p_breed, p_sbw, p_feed_time, p_target_weight, p_bcs, p_be, p_l, p_sex, p_a2,
         p_ph, p_selling_price, p_algorithm, p_identifier, p_lb, p_ub, p_tol, p_dmi_eq, p_obj, p_find_reduced_cost,
         p_ing_level, p_lca_id] = [None for i in range(24)]

        init_parameters = None

        def __init__(self, parameters):
            self.init_parameters = parameters
            self.set_parameters(parameters)

        def set_parameters(self, parameters):
            if isinstance(parameters, dict):
                [self.p_id, self.p_feed_scenario, self.p_batch, self.p_breed, self.p_sbw, self.p_feed_time,
                 self.p_target_weight, self.p_bcs, self.p_be, self.p_l, self.p_sex, self.p_a2, self.p_ph,
                 self.p_selling_price, self.p_algorithm, self.p_identifier, self.p_lb, self.p_ub, self.p_tol,
                 self.p_dmi_eq, self.p_obj, self.p_find_reduced_cost, self.p_ing_level, self.p_lca_id] = \
                    parameters.values()
            elif isinstance(parameters, list):
                [self.p_id, self.p_feed_scenario, self.p_batch, self.p_breed, self.p_sbw, self.p_feed_time,
                 self.p_target_weight, self.p_bcs, self.p_be, self.p_l, self.p_sex, self.p_a2, self.p_ph,
                 self.p_selling_price, self.p_algorithm, self.p_identifier, self.p_lb, self.p_ub, self.p_tol,
                 self.p_dmi_eq, self.p_obj, self.p_find_reduced_cost, self.p_ing_level, self.p_lca_id] = \
                    parameters

        def compute_nrc_parameters(self):
            self.mpmr, self.dmi, self.nem, self.pe_ndf = \
                nrc.get_all_parameters(self.cnem, self.p_sbw, self.p_bcs, self.p_be, self.p_l, self.p_sex, self.p_a2,
                                       self.p_ph, self.p_target_weight, self.p_dmi_eq)

            self.cneg = nrc.cneg(self.cnem)
            self.neg = nrc.neg(self.cneg, self.dmi, self.cnem, self.nem)

        def set_batch_scenario(self, batch_dict):
            self.c_batch_map = batch_dict

    class Data:
        ds: data_handler.Data = None
        headers_scenario: data_handler.Data.ScenarioParameters = None  # Scenario
        headers_feed_scenario: data_handler.Data.headers_scenario = None

        ingredient_ids: list = None
        dc_mp_properties: dict = None
        dc_cost: dict = None
        dc_ub: dict = None
        dc_lb: dict = None
        dc_dm_af_conversion: dict = None
        dc_nem: float = None
        dc_npn: dict = None
        dc_fat: float = None
        d_name_ing_map: dict = None

        dc_rdp = None
        dc_pendf = None

        def __init__(self, out_ds, parameters):
            self.cast_data(out_ds, parameters)

        def cast_data(self, out_ds, parameters):
            """Retrieve parameters data from table. See data_handler.py for more"""
            self.ds = out_ds

            self.headers_feed_scenario = self.ds.headers_feed_scenario

            headers_feed_scenario = self.ds.headers_feed_scenario
            data_feed_scenario = self.ds.filter_column(self.ds.data_feed_scenario,
                                                       self.ds.headers_feed_scenario.s_feed_scenario,
                                                       parameters.p_feed_scenario)
            data_feed_scenario = self.ds.sort_df(data_feed_scenario, self.headers_feed_scenario.s_ID)

            self.ingredient_ids = list(
                self.ds.get_column_data(data_feed_scenario, self.headers_feed_scenario.s_ID, int))

            headers_feed_lib = self.ds.headers_feed_lib
            data_feed_lib = self.ds.filter_column(self.ds.data_feed_lib, headers_feed_lib.s_ID,
                                                  self.ingredient_ids)

            [self.dc_cost,
             self.dc_ub,
             self.dc_lb] = self.ds.multi_sorted_column(data_feed_scenario,
                                                       [headers_feed_scenario.s_feed_cost,
                                                        self.headers_feed_scenario.s_max,
                                                        self.headers_feed_scenario.s_min],
                                                       self.ingredient_ids,
                                                       self.headers_feed_scenario.s_ID,
                                                       return_dict=True
                                                       )
            [self.dc_dm_af_conversion,
             self.dc_nem,
             self.dc_npn,
             self.dc_fat,
             self.dc_mp_properties,
             rup,
             cp,
             ndf,
             pef,
             self.d_name_ing_map] = self.ds.multi_sorted_column(data_feed_lib,
                                                                [headers_feed_lib.s_DM,
                                                                 headers_feed_lib.s_NEma,
                                                                 headers_feed_lib.s_NPN,
                                                                 headers_feed_lib.s_Fat,
                                                                 [headers_feed_lib.s_DM,
                                                                  headers_feed_lib.s_TDN,
                                                                  headers_feed_lib.s_CP,
                                                                  headers_feed_lib.s_RUP,
                                                                  headers_feed_lib.s_Forage,
                                                                  headers_feed_lib.s_Fat],
                                                                 headers_feed_lib.s_RUP,
                                                                 headers_feed_lib.s_CP,
                                                                 headers_feed_lib.s_NDF,
                                                                 headers_feed_lib.s_pef,
                                                                 headers_feed_lib.s_FEED
                                                                 ],
                                                                self.ingredient_ids,
                                                                self.headers_feed_scenario.s_ID,
                                                                return_dict=True
                                                                )
            for k, v in self.dc_npn.items():
                self.dc_npn[k] = nrc.npn(k, v)

            self.dc_rdp = {}
            self.dc_pendf = {}
            for ids in self.ingredient_ids:
                self.dc_rdp[ids] = (1 - rup[ids]) * cp[ids]
                self.dc_pendf[ids] = ndf[ids] * pef[ids]

            if parameters.p_batch > 0:
                try:
                    batch_feed_scenario = self.ds.batch_map[parameters.p_id]["data_feed_scenario"][
                        parameters.p_feed_scenario]
                except KeyError:
                    logging.warning(f"No Feed_scenario batch for scenario {parameters.p_id},"
                                    f" batch {parameters.p_batch}, feed_scenario{parameters.p_feed_scenario}")
                    batch_feed_scenario = {}
                try:
                    batch_scenario = self.ds.batch_map[parameters.p_id]["data_scenario"][parameters.p_id]
                except KeyError:
                    logging.warning(f"No Scenario batch for scenario {parameters.p_id},"
                                    f" batch {parameters.p_batch}, scenario{parameters.p_feed_scenario}")
                    batch_scenario = {}

                parameters.set_batch_scenario({"data_feed_scenario": batch_feed_scenario,
                                               "data_scenario": batch_scenario})

        def setup_batch(self, parameters):
            for ing_id, data in parameters.c_batch_map["data_feed_scenario"].items():
                for col_name, vector in data.items():
                    if col_name == self.headers_feed_scenario.s_feed_cost:
                        self.dc_cost[ing_id] = \
                            vector[parameters.p_batch_execution_id]
                    elif col_name == self.headers_feed_scenario.s_min:
                        self.dc_lb[ing_id] = vector[parameters.p_batch_execution_id]
                    elif col_name == self.headers_feed_scenario.s_max:
                        self.dc_ub[ing_id] = vector[parameters.p_batch_execution_id]

    def _compute_parameters(self):
        """Compute parameters variable with CNEm"""
        self.parameters.compute_nrc_parameters()

        if self.parameters.neg is None:
            return False
        if math.isnan(self.parameters.p_feed_time) or self.parameters.p_feed_time == 0:
            self.parameters.c_model_final_weight = self.parameters.p_target_weight
            self.parameters.c_swg = nrc.swg(self.parameters.neg, self.parameters.p_sbw,
                                            self.parameters.c_model_final_weight)
            self.parameters.c_model_feeding_time = \
                (self.parameters.p_target_weight - self.parameters.p_sbw) / self.parameters.c_swg
        elif math.isnan(self.parameters.p_target_weight) or self.parameters.p_target_weight == 0:
            self.parameters.c_model_feeding_time = self.parameters.p_feed_time
            self.parameters.c_swg = nrc.swg_time(self.parameters.neg, self.parameters.p_sbw,
                                                 self.parameters.c_model_feeding_time)
            self.parameters.c_model_final_weight = \
                self.parameters.c_model_feeding_time * self.parameters.c_swg + self.parameters.p_sbw
        else:
            raise Exception("target weight and feeding time cannot be defined at the same time")

        self.parameters.mpgr = nrc.mpg(self.parameters.c_swg, self.parameters.neg, self.parameters.p_sbw,
                                       self.parameters.c_model_final_weight, self.parameters.c_model_feeding_time)
        self.parameters.mpr = self.parameters.mpgr + self.parameters.mpmr
        self.computed.dc_mp = {}
        for ing_id in self.data.ingredient_ids:
            self.computed.dc_mp[ing_id] = nrc.mp(ing_id,
                                                 *self.data.dc_mp_properties[ing_id],
                                                 self.parameters.p_fat_orient)

        self.computed.revenue = self.parameters.p_selling_price * (
                self.parameters.p_sbw + self.parameters.c_swg * self.parameters.c_model_feeding_time)
        self.computed.dc_expenditure = self.data.dc_cost.copy()

        if self.parameters.p_obj == "MaxProfit" or self.parameters.p_obj == "MinCost":
            for i in self.data.ingredient_ids:
                self.computed.dc_expenditure[i] =\
                    - self.data.dc_cost[i] * self.parameters.dmi * self.parameters.c_model_feeding_time / \
                                                  self.data.dc_dm_af_conversion[i]
        elif self.parameters.p_obj == "MaxProfitSWG" or self.parameters.p_obj == "MinCostSWG":
            for i in self.data.ingredient_ids:
                self.computed.dc_expenditure[i] = \
                    - self.data.dc_cost[i] * self.parameters.dmi * self.parameters.c_model_feeding_time / \
                    (self.data.dc_dm_af_conversion[i] * self.parameters.c_swg)

        self.computed.dc_obj_func = self.computed.dc_expenditure.copy()

        if self.parameters.p_obj == "MaxProfit":
            self.computed.cst_obj = self.computed.revenue
        elif self.parameters.p_obj == "MaxProfitSWG":
            self.computed.cst_obj = self.computed.revenue / self.parameters.c_swg
        elif self.parameters.p_obj == "MinCost" or self.parameters.p_obj == "MinCostSWG":
            self.computed.cst_obj = 0

        return True

    def _build_model(self):
        """Build model"""
        self._diet = pyo.ConcreteModel()
        self._diet.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
        self._diet.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

        # Set
        self._diet.s_var_set = pyo.Set(initialize=self.data.ingredient_ids)

        # Parameters
        self._diet.p_model_offset = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_model_cost = pyo.Param(self._diet.s_var_set, within=pyo.Any, mutable=True)
        self._diet.p_model_lb = pyo.Param(self._diet.s_var_set, within=pyo.Any, mutable=True)
        self._diet.p_model_ub = pyo.Param(self._diet.s_var_set, within=pyo.Any, mutable=True)
        self._diet.p_model_nem = pyo.Param(self._diet.s_var_set, initialize=self.data.dc_nem)
        self._diet.p_model_npn = pyo.Param(self._diet.s_var_set, initialize=self.data.dc_npn)
        self._diet.p_model_mp = pyo.Param(self._diet.s_var_set, within=pyo.Any, mutable=True)
        self._diet.p_model_rdp = pyo.Param(self._diet.s_var_set, initialize=self.data.dc_rdp)
        self._diet.p_model_fat = pyo.Param(self._diet.s_var_set, initialize=self.data.dc_fat)
        self._diet.p_model_pendf = pyo.Param(self._diet.s_var_set, initialize=self.data.dc_pendf)
        self._diet.p_rhs_cnem_ge = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_cnem_le = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_sum_1 = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_mp = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_mp_ub = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_rdp_ub = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_rdp = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_fat = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_alt_fat_ge = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_alt_fat_le = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_pendf = pyo.Param(within=pyo.Any, mutable=True)

        # Functions
        def bound_function(model, i):
            return model.p_model_lb[i], model.p_model_ub[i]

        # Variables
        self._diet.v_x = pyo.Var(self._diet.s_var_set, bounds=bound_function)

        # Objective
        self._diet.f_obj = pyo.Objective(
            expr=(self._diet.p_model_offset + pyo.summation(self._diet.p_model_cost, self._diet.v_x)),
            sense=pyo.maximize)

        # Constraints
        self._diet.c_cnem_ge = pyo.Constraint(
            expr=pyo.summation(self._diet.p_model_nem, self._diet.v_x) >= self._diet.p_rhs_cnem_ge)
        self._diet.c_cnem_le = pyo.Constraint(
            expr=pyo.summation(self._diet.p_model_nem, self._diet.v_x) <= self._diet.p_rhs_cnem_le)
        self._diet.c_sum_1 = pyo.Constraint(expr=pyo.summation(self._diet.v_x) == self._diet.p_rhs_sum_1)
        self._diet.c_mpm = pyo.Constraint(
            expr=pyo.summation(self._diet.p_model_mp, self._diet.v_x) >= self._diet.p_rhs_mp)
        self._diet.c_rdp = pyo.Constraint(
            expr=pyo.summation(self._diet.p_model_rdp, self._diet.v_x) >= self._diet.p_rhs_rdp)
        self._diet.c_fat = pyo.Constraint(
            expr=pyo.summation(self._diet.p_model_fat, self._diet.v_x) <= self._diet.p_rhs_fat)
        self._diet.c_alt_fat_ge = pyo.Constraint(
            expr=pyo.summation(self._diet.p_model_fat, self._diet.v_x) >= self._diet.p_rhs_alt_fat_ge)
        self._diet.c_alt_fat_le = pyo.Constraint(
            expr=pyo.summation(self._diet.p_model_fat, self._diet.v_x) <= self._diet.p_rhs_alt_fat_le)
        self._diet.c_pendf = pyo.Constraint(
            expr=pyo.summation(self._diet.p_model_pendf, self._diet.v_x) >= self._diet.p_rhs_pendf)
        self._diet.c_npn = pyo.Constraint(
            expr=pyo.summation(self._diet.p_model_npn, self._diet.v_x) <= self._diet.p_rhs_rdp_ub)
        self._diet.c_mpr = pyo.Constraint(
            expr=pyo.summation(self._diet.p_model_mp, self._diet.v_x) <= self._diet.p_rhs_mp_ub)

    def _update_model(self):
        """Update RHS values on the model based on the new CNEm and updated parameters"""
        self._diet.p_model_offset = self.computed.cst_obj
        for i in self._diet.s_var_set:
            self._diet.p_model_lb[i] = self.data.dc_lb[i]
            self._diet.p_model_ub[i] = self.data.dc_ub[i]
            self._diet.p_model_cost[i] = self.computed.dc_obj_func[i]
            self._diet.p_model_mp[i] = self.computed.dc_mp[i]

        self._diet.p_rhs_cnem_ge = self.parameters.cnem * 0.999
        self._diet.p_rhs_cnem_le = self.parameters.cnem * 1.001
        self._diet.p_rhs_sum_1 = 1
        self._diet.p_rhs_mp = (self.parameters.mpmr + self.parameters.mpgr) * 0.001 / self.parameters.dmi
        self._diet.p_rhs_mp_ub = 1.15 * (self.parameters.mpmr + self.parameters.mpgr) * 0.001 / self.parameters.dmi
        self._diet.p_rhs_rdp = 0.125 * self.parameters.cnem
        self._diet.p_rhs_rdp_ub = 0.125 * self.parameters.cnem * 0.67 * 100
        self._diet.p_rhs_fat = 0.06
        self._diet.p_rhs_alt_fat_ge = 0.039
        self._diet.p_rhs_alt_fat_le = 0.039
        self._diet.p_rhs_pendf = self.parameters.pe_ndf / self.parameters.dmi

        if self.parameters.p_fat_orient == "G":
            self._diet.c_alt_fat_ge.activate()
            self._diet.c_alt_fat_le.deactivate()
        else:
            self._diet.c_alt_fat_ge.deactivate()
            self._diet.c_alt_fat_le.activate()

    def set_fat_orient(self, direction):
        self.parameters.p_fat_orient = direction

    def set_batch_params(self, i):
        self.parameters.p_batch_execution_id = i

    def _setup_batch(self):
        parameters = self.parameters.init_parameters.copy()
        for col_name, vector in self.parameters.c_batch_map["data_scenario"].items():
            parameters[col_name] = vector[self.parameters.p_batch_execution_id]
        self.parameters.set_parameters(parameters)
        self.data.setup_batch(self.parameters)


class ModelReducedCost(Model):
    _special_id = None
    _special_cost = None

    def __init__(self, out_ds, parameters, special_id, special_cost=default_special_cost):
        Model.__init__(self, out_ds, parameters)
        self._special_id = special_id
        self._special_cost = special_cost

    def _solve(self, problem_id):
        sol = Model._solve(self, problem_id)
        sol["x{0}_price_{1}".format(self._special_id, str(int(100 * self.parameters.p_ing_level)))] = self._special_cost
        return sol

    def _compute_parameters(self):
        if not Model._compute_parameters(self):
            return False
        else:
            self.data.dc_cost[self._special_id] = self._special_cost
            self.computed.dc_expenditure[self._special_id] = \
                - self.data.dc_cost[self._special_id] * self.parameters.dmi * \
                self.parameters.c_model_feeding_time / self.data.dc_dm_af_conversion[self._special_id]
            if self.parameters.p_obj == "MaxProfitSWG" or self.parameters.p_obj == "MinCostSWG":
                self.computed.dc_expenditure[self._special_id] /= self.parameters.c_swg
            return True

    def set_special_cost(self, cost=default_special_cost):
        self._special_cost = cost

    def get_special_cost(self):
        return self._special_cost

    def get_special_id(self):
        return self._special_id, self.data.d_name_ing_map[self._special_id]


class ModelLCA(Model):
    class Parameters(Model.Parameters):
        # Computed in Model
        c_env_impacts_weights: dict = None  # list of vectors with LCA weight for each LCA scenario
        c_methane_equation: str = None  # Currently only option "IPCC2006" supported
        c_n2o_equation: str = None  # Currently only option "IPCC2006" supported
        c_normalize: bool = None  # true or false to normalize c_lca_ing_map by column (LCA)

        v_lca_weight: float = 0.0
        v_obj_weight: float = 1.0

        v_lca_rhs: float = None
        v_stage: float = None

        e_forage_sense: str = None

        def set_env_impacts_properties(self, lca_dict, headers: data_handler.Data.LCAScenario):
            self.c_env_impacts_weights = {}
            if self.c_env_impacts_weights is None:
                self.c_env_impacts_weights = {}
            for h in lca_dict:
                if "LCA" in h:
                    self.c_env_impacts_weights[h] = lca_dict[h].values[0]

            self.c_n2o_equation = lca_dict[headers.s_N2O_Equation].values[0]
            self.c_methane_equation = lca_dict[headers.s_Methane_Equation].values[0]
            self.c_normalize = lca_dict[headers.s_Normalize].values[0]

    class Data(Model.Data):
        d_forage: dict = None
        d_lca_ing_map: dict = None  # Matrix with ingredients' LCA
        ing_fat: dict = None
        ing_ash: dict = None
        ing_cp: dict = None
        ing_ndf: dict = None
        ing_starch: dict = None
        ing_sugars: dict = None
        ing_oa: dict = None

        headers_lca_lib: data_handler.Data.LCALib = None

        def cast_data(self, out_ds, parameters):
            Model.Data.cast_data(self, out_ds, parameters)

            headers_feed_lib = self.ds.headers_feed_lib
            data_feed_lib = self.ds.filter_column(self.ds.data_feed_lib, headers_feed_lib.s_ID,
                                                  self.ingredient_ids)

            [self.d_forage] = self.ds.multi_sorted_column(data_feed_lib,
                                                          [headers_feed_lib.s_Forage],
                                                          self.ingredient_ids,
                                                          self.headers_feed_scenario.s_ID,
                                                          return_dict=True
                                                          )

            headers_lca_scenario = self.ds.headers_lca_scenario
            data_lca_scenario = self.ds.filter_column(self.ds.data_lca_scenario, headers_lca_scenario.s_ID,
                                                      parameters.p_lca_id)
            parameters.set_env_impacts_properties(data_lca_scenario, headers_lca_scenario)

            self.headers_lca_lib = self.ds.headers_lca_lib
            data_lca_lib = self.ds.filter_column(self.ds.data_lca_lib,
                                                 self.headers_lca_lib.s_ing_id,
                                                 self.ingredient_ids)
            self.d_lca_ing_map = self.ds.sort_df(data_lca_lib, self.headers_lca_lib.s_ing_id)
            if self.ingredient_ids != list(self.d_lca_ing_map[self.headers_lca_lib.s_ing_id]):
                raise IndexError("LCA Library does not match all ingredients in Feeds")
            self.d_lca_ing_map.pop(self.ds.headers_lca_lib.s_ing_id)
            self.d_lca_ing_map.pop(self.ds.headers_lca_lib.s_name)

            # Properties for methane emission calculation
            headers_feed_lib = self.ds.headers_feed_lib
            data_feed_lib = self.ds.filter_column(self.ds.data_feed_lib, headers_feed_lib.s_ID,
                                                  self.ingredient_ids)
            self.ing_fat = self.ds.sorted_column(data_feed_lib,
                                                 headers_feed_lib.s_Fat,
                                                 None,
                                                 headers_feed_lib.s_ID,
                                                 True)
            self.ing_ash = self.ds.sorted_column(data_feed_lib,
                                                 headers_feed_lib.s_Ash,
                                                 None,
                                                 headers_feed_lib.s_ID,
                                                 True)
            self.ing_cp = self.ds.sorted_column(data_feed_lib,
                                                headers_feed_lib.s_CP,
                                                None,
                                                headers_feed_lib.s_ID,
                                                True)
            self.ing_ndf = self.ds.sorted_column(data_feed_lib,
                                                 headers_feed_lib.s_NDF,
                                                 None,
                                                 headers_feed_lib.s_ID,
                                                 True)
            self.ing_starch = self.ds.sorted_column(data_feed_lib,
                                                    headers_feed_lib.s_Starch,
                                                    None,
                                                    headers_feed_lib.s_ID,
                                                    True)
            self.ing_sugars = self.ds.sorted_column(data_feed_lib,
                                                    headers_feed_lib.s_Sugars,
                                                    None,
                                                    headers_feed_lib.s_ID,
                                                    True)
            self.ing_oa = self.ds.sorted_column(data_feed_lib,
                                                headers_feed_lib.s_OA,
                                                None,
                                                headers_feed_lib.s_ID,
                                                True)

    class ComputedArrays(Model.ComputedArrays):
        d_methane_vector_ge20: dict = None
        d_methane_vector_le20: dict = None
        env_impact_array: list = None
        profit_array: list = None

        cst_n2o_emission: float = None

    def __init__(self, out_ds, parameters):
        Model.__init__(self, None, None)
        self.parameters = self.Parameters(parameters)
        self.data = self.Data(out_ds, self.parameters)
        self.computed = self.ComputedArrays()

    @staticmethod
    def __normalise(vector):
        if isinstance(vector, list):
            return [(vector[i] - min(vector)) / (max(vector) - min(vector)) for i in range(len(vector))]
        elif isinstance(vector, pd.DataFrame):
            for col in vector.columns:
                temp = vector[col]
                temp = (temp - temp.min()) / (temp.max() - temp.min())
                vector[col] = temp
            return vector
        raise Exception('Invalid format parsed to normaliser: lp_model l501')

    def _compute_parameters(self):
        if not Model._compute_parameters(self):
            return False

        # computing methane
        methane_vector_ge20 = []
        methane_vector_le20 = []
        if self.parameters.c_methane_equation is not None:
            for i in self.data.ingredient_ids:
                methane_ge20, methane_le20 = nrc.ch4_diet(self.data.ing_fat[i],
                                                          self.data.ing_cp[i],
                                                          self.data.ing_ash[i],
                                                          self.data.ing_ndf[i],
                                                          self.data.ing_starch[i],
                                                          self.data.ing_sugars[i],
                                                          self.data.ing_oa[i],
                                                          i)
                methane_vector_ge20.append(methane_ge20)
                methane_vector_le20.append(methane_le20)

        self.computed.d_methane_vector_ge20 = methane_vector_ge20
        self.computed.d_methane_vector_le20 = methane_vector_le20

        if self.parameters.c_n2o_equation is not None:
            n2o_emission = \
                nrc.n2o_diet(self.parameters.c_model_final_weight,
                             self.parameters.c_n2o_equation) / self.parameters.dmi
        else:
            n2o_emission = 0
        self.computed.cst_n2o_emission = n2o_emission

        # Climate change impact adding methane and N2O for each ingredient
        env_impact_matrix = self.data.d_lca_ing_map.copy()

        if self.parameters.c_methane_equation is not None:
            if self.parameters.e_forage_sense == 'L':
                env_impact_matrix[self.data.headers_lca_lib.s_LCA_GHG] = \
                    env_impact_matrix[self.data.headers_lca_lib.s_LCA_GHG] + self.computed.d_methane_vector_le20 \
                    + n2o_emission
            else:
                env_impact_matrix[self.data.headers_lca_lib.s_LCA_GHG] = \
                    env_impact_matrix[self.data.headers_lca_lib.s_LCA_GHG] + self.computed.d_methane_vector_ge20 \
                    + n2o_emission

        # Normalizing env impacts
        if self.parameters.c_normalize:
            normalized_temp = self.__normalise(env_impact_matrix)
            env_impact_matrix = normalized_temp

        # Dot product env impact matrix and weights vector
        s = pd.Series(self.parameters.c_env_impacts_weights)
        env_impact_vector = env_impact_matrix.dot(s.array)

        # Adjusting units, EI total per kg of animal
        units_coverter = \
            self.parameters.dmi * self.parameters.c_model_feeding_time \
            / self.parameters.c_model_final_weight

        # Reassembles dictionary with ingridients' IDs
        self.computed.env_impact_array = dict(zip(self.data.ingredient_ids,
                                                  env_impact_vector.array * units_coverter))
        return True

    def _build_model(self):
        Model._build_model(self)
        # Constraint: sum(x forage) [<= >=] 20%
        self._diet.p_forage = pyo.Param(self._diet.s_var_set, initialize=self.data.d_forage)
        self._diet.p_rhs_forage_ge = pyo.Param(within=pyo.Any, initialize=0.2)
        self._diet.p_rhs_forage_le = pyo.Param(within=pyo.Any, initialize=0.2)
        self._diet.c_forage_ge = pyo.Constraint(
            expr=pyo.summation(self._diet.p_forage, self._diet.v_x) >= self._diet.p_rhs_forage_ge)
        self._diet.c_forage_le = pyo.Constraint(
            expr=pyo.summation(self._diet.p_forage, self._diet.v_x) <= self._diet.p_rhs_forage_le)

        if self.parameters.e_forage_sense == "G":
            self._diet.c_forage_ge.activate()
            self._diet.c_forage_le.deactivate()
        elif self.parameters.e_forage_sense == "L":
            self._diet.c_forage_ge.deactivate()
            self._diet.c_forage_le.activate()
        else:
            self._diet.c_forage_ge.activate()
            self._diet.c_forage_le.deactivate()

        # Constraint: sum(x lca) <= LCA_rhs
        self._diet.p_lca = pyo.Param(self._diet.s_var_set, within=pyo.Any, mutable=True)
        self._diet.p_rhs_lca = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.c_lca = pyo.Constraint(
            expr=pyo.summation(self._diet.p_lca, self._diet.v_x) <= self._diet.p_rhs_lca)

    def _update_model(self):
        Model._update_model(self)
        self._diet.p_model_offset = self.computed.cst_obj * self.parameters.v_obj_weight
        for i in self._diet.s_var_set:
            self._diet.p_model_cost[i] = self.computed.dc_obj_func[i] * self.parameters.v_obj_weight \
                                         + (-1.0) * self.computed.env_impact_array[i] * self.parameters.v_lca_weight

        if self.parameters.e_forage_sense == "G":
            self._diet.c_forage_ge.activate()
            self._diet.c_forage_le.deactivate()
        elif self.parameters.e_forage_sense == "L":
            self._diet.c_forage_ge.deactivate()
            self._diet.c_forage_le.activate()
        else:
            self._diet.c_forage_ge.deactivate()
            self._diet.c_forage_le.deactivate()

        if self.parameters.v_lca_rhs is None:
            self.parameters.v_lca_rhs = sum(self.computed.env_impact_array)
        self._diet.p_rhs_lca = self.parameters.v_lca_rhs

        for i in self._diet.s_var_set:
            self._diet.p_lca[i] = self.computed.env_impact_array[i]

    def set_obj_weights(self, f1_w, lca_w):
        self.parameters.v_obj_weight = f1_w
        self.parameters.v_lca_weight = lca_w

    def set_lca_rhs(self, val, stage):
        # Set rhs and removes LCA from objective function
        self.parameters.v_lca_rhs = val
        self.parameters.v_stage = stage
        self.set_obj_weights(1.0, 0.0)

    def set_forage(self, ge_or_le):
        """ ge_or_le = "L" or "G" """
        self.parameters.e_forage_sense = ge_or_le
        self.set_lca_rhs(100000, None)

    @staticmethod
    def get_obj_sol(solution):
        if solution is None:
            raise Exception
        lca_value = solution["EI Obj (weighted impacts)"]
        profit = solution["Profit"]
        cnem = solution['CNEm']
        return [profit, lca_value, cnem]

    def _solve(self, problem_id):
        try:
            sol = Model._solve(self, problem_id)
            if sol is None:
                return None

            sol["Forage Sense"] = self.parameters.e_forage_sense

            lca_value = 0
            for i in self._diet.v_x:
                lca_value += self.computed.env_impact_array[i] * self._diet.v_x[i].value

            sol["EI Obj (weighted impacts)"] = lca_value

            # Adjusting units, EI total per kg of animal
            units_coverter = \
                self.parameters.dmi * self.parameters.c_model_feeding_time \
                / self.parameters.c_model_final_weight

            sol['Converter (DMI * t)/(SBWf)'] = units_coverter
            sol_methane_feed = {}
            sol["Methane Total [kgCO2eq/kg Animal]"] = 0
            env_impact_matrix = self.data.d_lca_ing_map.copy()
            if self.parameters.c_methane_equation is not None:
                if self.parameters.e_forage_sense == "L":
                    methane_vector = dict(zip(self.data.ingredient_ids, self.computed.d_methane_vector_le20))
                else:
                    methane_vector = dict(zip(self.data.ingredient_ids, self.computed.d_methane_vector_ge20))
                for i in self._diet.v_x:
                    sol[f'Methane ing {i}'] = methane_vector[i]
                    sol["Methane Total [kgCO2eq/kg Animal]"] += methane_vector[i] * self._diet.v_x[i].value * \
                                                                units_coverter

                if self.parameters.e_forage_sense == 'L':
                    env_impact_matrix[self.data.headers_lca_lib.s_LCA_GHG] = \
                        env_impact_matrix[self.data.headers_lca_lib.s_LCA_GHG] \
                        + self.computed.d_methane_vector_le20 \
                        + self.computed.cst_n2o_emission
                else:
                    env_impact_matrix[self.data.headers_lca_lib.s_LCA_GHG] = \
                        env_impact_matrix[self.data.headers_lca_lib.s_LCA_GHG] \
                        + self.computed.d_methane_vector_ge20 \
                        + self.computed.cst_n2o_emission

            sol["N2O Total [kgCO2eq/kg Animal]"] = self.computed.cst_n2o_emission * units_coverter

            # # Normalizing env impacts
            # if self.parameters.c_normalize:
            #     normalized_temp = self.__normalise(env_impact_matrix)
            #     env_impact_matrix = normalized_temp

            # Dot product env impact matrix and weights vector
            s = pd.Series(self._diet.v_x.get_values())
            env_impact_matrix = pd.Series(env_impact_matrix).transpose()
            env_impact_vector = env_impact_matrix.dot(s.array)
            env_impact_vector = dict(zip(self.parameters.c_env_impacts_weights.keys(),
                                         env_impact_vector.array * units_coverter))

            for k, v in env_impact_vector.items():
                sol[f"LCA emited/kg Animal - {k}"] = v
            if self.parameters.v_stage is None:
                self.parameters.v_stage = self.parameters.v_lca_weight
            sol["Env Impact weight (Multi Objective)"] = self.parameters.v_stage
            sol["Profit weight (Multi Objective)"] = 1 - self.parameters.v_stage
            sol["Profit"] = sol["obj_revenue"] - sol["obj_cost"]
            sol["Obj Func [weighthed objs]"] = sol["Profit"] * sol["Profit weight (Multi Objective)"] + \
                                               (-1) * sol["Env Impact weight (Multi Objective)"] * sol[
                                                   "EI Obj (weighted impacts)"]

            # sol = {**sol, **sol_methane_feed, **sol_lca_feed}
            sol = {**sol, **sol_methane_feed}
            return sol
        except Exception as e:
            logging.error(f"An error occurred writing the solution on ln584:\n{str(e)}")
            raise e

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


class ModelLCAReducedCost(ModelLCA):
    # TODO: do
    pass
