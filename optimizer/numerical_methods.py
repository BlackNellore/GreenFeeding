import numpy as np
from aenum import Enum
import logging
from model.lp_model import Model, ModelReducedCost, ModelLCA

Status = Enum('Status', 'EMPTY READY SOLVED ERROR')

default_fat_list = ["G", "L"]


class Searcher:
    _model: Model = None
    _obj_func_key = None
    _msg = None
    _batch = False
    _fat_list = None

    _status = Status.EMPTY
    _solutions: list = None

    def __init__(self, model, batch=False, obj_func_key="obj_func"):

        self._model = model
        self._obj_func_key = obj_func_key
        self._solutions = []
        self._status = Status.READY
        self._batch = batch
        self._model.prefix_id = ""
        self._fat_list = default_fat_list

    def refine_bounds(self, lb=0.0, ub=1.0, tol=0.01, double_refinement=False, mode="BF"):
        if double_refinement:
            new_lb = {}
            new_ub = {}
            for i in range(len(self._fat_list)):
                self._model.set_fat_orient(self._fat_list[i])
                new_lb[self._fat_list[i]] = self._refine_bound(lb, ub, direction=1, tol=tol, mode=mode)
                if new_lb[self._fat_list[i]] is None:
                    return None, None
                new_ub[self._fat_list[i]] = self._refine_bound(lb, ub, direction=-1, tol=tol, mode=mode)
        else:
            new_lb = self._refine_bound(lb, ub, direction=1, tol=tol)
            if new_lb is None:
                return None, None
            new_ub = self._refine_bound(lb, ub, direction=-1, tol=tol)

        return new_lb, new_ub

    def _refine_bound(self, v0=0.0, vf=1.0, direction=1, tol=0.01, mode="BF"):
        """Return feasible parameter vi in S = [v0, vf] or solution dict of S """
        if mode == "BF":
            space = np.linspace(v0, vf, int(np.ceil((vf - v0 + tol) / tol)))
            if direction == -1:
                space = reversed(space)
            new_v: dict = self.__brute_force(self._model.run, space, first_feasible=True)
            if new_v is None:
                return new_v
            else:
                if 'CNEm' in new_v:
                    return new_v['CNEm']
                else:
                    raise RuntimeError('key not found in dict')
        elif mode == "GSS":
            space = np.linspace(v0, vf, int(np.ceil((vf - v0 + tol) / tol)))
            feasible = None
            for i in np.random.choice(space, len(space)):
                r = self._model.run(i, space[i])
                if r is not None:
                    feasible = space[i]
                    continue
            if feasible is not None:
                gss_results = []
                a, b = self.__feasible_golden_section_search_recursive(self._model.run,
                                                                       v0,
                                                                       feasible,
                                                                       gss_results,
                                                                       tol=tol)
        else:
            raise RuntimeError(f"wrong method parsed. Can not accept {mode}")

    def brute_force_search(self, lb, ub, p_tol, uncertain_bounds=False, mode="BF"):
        """Executes brute force search algorithm"""
        if self._status != Status.READY:
            self.__clear_searcher()
        if uncertain_bounds:
            lb, ub = self.refine_bounds(lb, ub, 0.01, double_refinement=False, mode="BF")
            if lb is None:
                self._status = Status.ERROR
                return
        cnem_space = np.linspace(lb, ub, int(np.ceil((ub - lb) / p_tol)))
        bf_results = self.__brute_force(self._model.run, cnem_space)
        if len(bf_results) == 0:
            self._status = Status.ERROR
        else:
            self._status = Status.SOLVED
        return bf_results

    @staticmethod
    def __brute_force(f, search_space, first_feasible=False) -> list or None:
        """
        Run Brute Force algorithm in a function f. In this model f is lp_model.Model.run()
        """
        try:
            if first_feasible:
                for i, val in enumerate(search_space):
                    r: dict = f(i, val)
                    logging.info("Brute force <iteration, cnem>: <{0}, {1}>".format(i, val))
                    if r is not None:
                        return r
            else:
                results = []
                for i, val in enumerate(search_space):
                    logging.info("ID: {}".format(i))
                    r: dict = f(i, val)
                    if r is not None:
                        results.append(r)
                        logging.info("Solution Appended")
                    else:
                        logging.info("Infeasible")
                return results

        except TypeError as e:
            logging.error("An error occurred in numerical_method.Searcher.__brute_force method: {}".format(e))
            return None

    def golden_section_search(self, lb, ub, p_tol, uncertain_bounds=True, mode="BF"):
        """Executes golden-section search algorithm"""
        if self._status != Status.READY:
            self.__clear_searcher()
        if uncertain_bounds:
            lb, ub = self.refine_bounds(lb, ub, 0.001, mode=mode)
            if lb is None:
                self._status = Status.ERROR
                return
        gss_results = []
        a, b = self.__golden_section_search_recursive(self._model.run, lb, ub, gss_results, tol=p_tol)
        if a is None:
            self._status = Status.ERROR
        else:
            self._status = Status.SOLVED
        return gss_results

    def __golden_section_search_recursive(
            self, f, a, b, results, p_id=0, tol=1e-3, h=None, c=None, d=None, fc=None, fd=None):
        """
        Run GSS algorithm in a function f. In this model f is lp_model.Model.run()
        To change evaluation function, rewrite _get_f(solution_element)
        _get_f must return a float or integer value to be compared with <, > operators
        """

        def _get_f(solution_element):
            """Extract evaluation value for method __golden_section_search_recursive"""
            return solution_element[self._obj_func_key]

        inv_phi = (np.sqrt(5) - 1) / 2
        inv_phi2 = (3 - np.sqrt(5)) / 2

        try:
            logging.info("\n{0}".format(p_id))
            (a, b) = (min(a, b), max(a, b))
            if h is None:
                h = b - a
            if h <= tol:
                if len(results) == 0:
                    solution = f(p_id, a)
                    results.append(solution)
                return a, b
            if c is None:
                c = a + inv_phi2 * h
            if d is None:
                d = a + inv_phi * h
            if fc is None:
                solution = f(p_id, c)
                fc = _get_f(solution)
                results.append(solution)
            if fd is None:
                solution = f(p_id, d)
                fd = _get_f(solution)
                results.append(solution)
            if fc > fd:
                return self.__golden_section_search_recursive(
                    f, a, d, results, p_id+1, tol, h * inv_phi, d=c, fd=fc)
            else:
                return self.__golden_section_search_recursive(
                    f, c, b, results, p_id+1, tol, h * inv_phi, c=d, fc=fd)
        except TypeError as e:
            logging.error("An error occurred in GSS method:\n{}".format(e))
            raise e

    def __feasible_golden_section_search_recursive(
            self, f, a, b, results, p_id=0, tol=1e-3, h=None, c=None, d=None, fc=None, fd=None):
        """
        Run GSS algorithm in a function f. In this model f is lp_model.Model.run()
        To change evaluation function, rewrite _get_f(solution_element)
        _get_f must return a float or integer value to be compared with <, > operators
        """

        def _get_f(solution_element, cst_x):
            """Extract evaluation value for method __golden_section_search_recursive"""
            if solution_element is None:
                return -100.0 * cst_x
            else:
                return -1000

        inv_phi = (np.sqrt(5) - 1) / 2
        inv_phi2 = (3 - np.sqrt(5)) / 2

        try:
            logging.info("\n{0}".format(p_id))
            (a, b) = (min(a, b), max(a, b))
            if h is None:
                h = b - a
            if h <= tol:
                if len(results) == 0:
                    solution = f(p_id, a)
                    results.append(solution)
                return a, b
            if c is None:
                c = a + inv_phi2 * h
            if d is None:
                d = a + inv_phi * h
            if fc is None:
                solution = f(p_id, c)
                fc = _get_f(solution, c)
                results.append(solution)
            if fd is None:
                solution = f(p_id, d)
                fd = _get_f(solution, d)
                results.append(solution)
            if fc > fd:
                return self.__feasible_golden_section_search_recursive(
                    f, a, d, results, p_id+1, tol, h * inv_phi, d=c, fd=fc)
            else:
                return self.__feasible_golden_section_search_recursive(
                    f, c, b, results, p_id+1, tol, h * inv_phi, c=d, fc=fd)
        except TypeError as e:
            logging.error("An error occurred in GSS method:\n{}".format(e))
            raise e

    def run_mono_objective(self, algorithm, lb, ub, tol, uncertain_bounds=True, find_red_cost=False):
        sol_vec = []
        for fat_sense in self._fat_list:
            if type(lb) == dict:
                self._msg = f"single objective lb={lb[fat_sense]}," \
                            f" ub={ub[fat_sense]}," \
                            f" algorithm={algorithm}"
            else:
                self._msg = f"single objective lb={lb}," \
                            f" ub={ub}," \
                            f" algorithm={algorithm}"
            self.__clear_searcher()
            self._model.set_fat_orient(fat_sense)
            if type(lb) == dict:
                sol_vec.append(getattr(self, algorithm)(lb[fat_sense],
                                                        ub[fat_sense],
                                                        tol,
                                                        uncertain_bounds))
            else:
                sol_vec.append(getattr(self, algorithm)(lb, ub, tol, uncertain_bounds=True))

        sol_vec = list(np.concatenate(sol_vec))
        status, solution = self.get_sol_results(-1, sol_vec, best=(self._batch or find_red_cost))
        if status == Status.SOLVED:
            if self._batch:
                self._solutions.append(solution)
            else:
                self._solutions = [solution]

    def run_scenario(self, algorithm, lb, ub, tol, lca_id, uncertain_bounds=True, find_red_cost=False):
        if lca_id > 0:
            self.multi_objective(algorithm, lb, ub, tol, lca_id)
        else:
            self.run_mono_objective(algorithm, lb, ub, tol, uncertain_bounds, find_red_cost)

    def clear_searcher(self):
        self.__clear_searcher()

    def set_batch_params(self, period):
        self._model.set_batch_params(period)

    def get_sol_results(self, lca_id, solution_vec, best):
        """
        Return list with results or optimal solution in a list
        return type is either list or float
        """
        if solution_vec is None:
            if self._batch or lca_id > 0:
                solution_vec = self._solutions
            else:
                solution_vec = self._solutions[0]
        if len(solution_vec) == 0 or solution_vec is None:
            return self._status, None
        if best:
            result = self.__extract_optimal(solution_vec)
        else:
            result = solution_vec.copy()
        return self._status, result

    def __extract_optimal(self, results, direction=1):
        """Extract optimal solution from list, Max: direction = 1; Min: direction = -1"""
        if direction != 1 and direction != -1:
            error_message = "The parsed value for \"direction\" is not acceptable." \
                            " Value must be 1 or -1, value parsed: {0}".format(direction)
            raise IOError(error_message)

        if results is None:
            logging.info("Result vector is None..{0}, {1}".format(self._status, self._msg))
            return None
        if len(results) <= 0:
            return None

        obj_vals = [p['obj_func']*direction for p in results]
        best_id = [i for i, j in enumerate(obj_vals) if j == max(obj_vals)]
        return results[best_id[0]]

    def __clear_searcher(self):
        # if force_clear or (not self._batch):
        self._solutions.clear()
        self._status = Status.READY
        self._model.prefix_id = self._msg

    def _get_last_solution(self):
        """
        :return: dict
        """
        if self._batch:
            sol = self._solutions.pop()
        else:
            sol = self._solutions[0]
        return sol

    def _search_reduced_cost_recursive(self, algorithm, lb, ub, tol, lb_cost, ub_cost, tol_cost, ing_level):
        # Pulling methods from subclass ModelReducedCost so the IDE can suggest auto-completion
        self._model: ModelReducedCost = self._model

        if ub_cost - lb_cost <= tol_cost:
            self._model.set_special_cost(lb_cost)
            self.run_scenario(algorithm, lb, ub, tol, lca_id=-1, uncertain_bounds=False)
        else:
            self.run_scenario(algorithm, lb, ub, tol, lca_id=-1, uncertain_bounds=False, find_red_cost=True)
            sol: dict = self._get_last_solution()
            var = sol["x{}".format(self._model.get_special_id())]

            if var > ing_level:
                new_lb_cost = self._model.get_special_cost()
                self._model.set_special_cost((new_lb_cost + ub_cost) / 2)
                self._search_reduced_cost_recursive(algorithm, lb, ub, tol, new_lb_cost, ub_cost, tol_cost, ing_level)
            elif var < ing_level:
                new_ub_cost = self._model.get_special_cost()
                self._model.set_special_cost((lb_cost + new_ub_cost) / 2)
                self._search_reduced_cost_recursive(algorithm, lb, ub, tol, lb_cost, new_ub_cost, tol_cost, ing_level)
            else:
                new_lb_cost = self._model.get_special_cost()
                new_ub_cost = new_lb_cost
                self._search_reduced_cost_recursive(algorithm,
                                                    lb,
                                                    ub,
                                                    tol,
                                                    new_lb_cost,
                                                    new_ub_cost,
                                                    tol_cost,
                                                    ing_level)

    def search_reduced_cost(self, algorithm, lb, ub, tol, ing_level, tol_cost=0.01):
        # Pulling methods from subclass ModelReducedCost so the IDE can suggest auto-completion
        self._model: ModelReducedCost = self._model

        self._model.set_special_cost(tol_cost)
        self.run_scenario(algorithm, lb, ub, tol, lca_id=-1, uncertain_bounds=False, find_red_cost=True)
        special_id = self._model.get_special_id()
        sol: dict = self._get_last_solution()
        var = sol["x{}".format(special_id)]
        if var >= ing_level:
            self._model.set_special_cost()
            lb_cost = tol_cost
            special_cost = self._model.get_special_cost()
            ub_cost = special_cost
            self.run_scenario(algorithm, lb, ub, tol, lca_id=-1, uncertain_bounds=False, find_red_cost=True)
            sol = self._get_last_solution()
            red_cost = sol["x{}_red_cost".format(special_id)] * self._model.data.dc_dm_af_conversion[special_id] / \
                       (sol["DMI"] * sol["Feeding Time"])

            if self._model.parameters.p_obj == "MaxProfitSWG" or self._model.parameters.p_obj == "MinCostSWG":
                red_cost *= sol["SWG"]

            if special_cost + red_cost < tol_cost:
                self._model.set_special_cost(2 * tol_cost)
            else:
                self._model.set_special_cost(special_cost + red_cost)

            self._search_reduced_cost_recursive(algorithm, lb, ub, tol, lb_cost, ub_cost, tol_cost, ing_level)
        else:
            if self._batch:
                self._solutions.append(sol)
            else:
                self._solutions = [self._solutions]

    def multi_objective(self, algorithm, lbs, ubs, tol, lca_id):
        if self._solutions is None:
            self._solutions = []
        _solutions_multiobjective = []
        for fat_sense in self._fat_list:
            self._model.set_fat_orient(fat_sense)
            self._model.set_lca_rhs(None, 0)
            lb, ub = None, None
            if type(lbs) is dict:
                self._msg = f"multi objective lb={lbs[fat_sense]}," \
                            f" ub={ubs[fat_sense]}," \
                            f" algorithm={algorithm}"
                lb = lbs[fat_sense]
                ub = ubs[fat_sense]
            if lca_id <= 0:
                logging.error(f"Multi-objective must have valid LCA ID. Check the input spreadsheet. LCA ID = {lca_id}")
                raise IndexError(f"Multi-objective must have valid LCA ID. Value parsed = {lca_id}")
            self._model: ModelLCA = self._model
            # forage = ['L', 'G']
            forage = ['L']
            for v in forage:
                self.__clear_searcher()
                self._model.set_obj_weights(1.0, 0.0)
                self._model.set_forage(v)
                msg = f"multi objective lb={lb}, ub={ub}, algorithm={algorithm}, max f1"
                msg += f", forage = {v}"

                sol_vec = getattr(self, algorithm)(lb, ub, tol, uncertain_bounds=True, mode="BF")  # max f1
                status, solution = self.get_sol_results(lca_id, sol_vec, best=True)  # get f1_ub, f2_lb
                if solution is None:
                    logging.error(f"Model is infeasible for forage concentration {v} 20%")
                    continue
                f1_ub, f2_ub, cnem_1 = self._model.get_obj_sol(solution)  # get f1_ub, f2_lb

                self._model.set_obj_weights(0.0, 1.0)
                self.__clear_searcher()
                msg = f"multi objective lb={lb}, ub={ub}, algorithm={algorithm}, max f2"
                msg += f", forage = {v}"

                sol_vec = getattr(self, algorithm)(lb, ub, tol, uncertain_bounds=True, mode="GSS")  # max f2
                status, solution = self.get_sol_results(lca_id, sol_vec, best=True)  # get f1_lb, f2_ub
                f1_lb, f2_lb, cnem_2 = self._model.get_obj_sol(solution)  # get f1_l, f2_ub

                # n_vals = int(np.ceil((1 + f2_ub - f2_lb) / tol))
                n_vals = 3
                lca_rhs_space = None
                if f2_lb + max(tol * (f2_ub - f2_lb) * 1.5, tol) >= f2_ub:
                    lca_rhs_space = np.linspace(f2_lb + tol * (f2_ub - f2_lb) * 1.5, f2_ub, 3)
                else:
                    if (f2_ub - f2_lb) < tol:
                        lca_rhs_space = np.linspace(f2_lb + max(tol * (f2_ub - f2_lb) * 1.5, tol), f2_ub, 3)
                    else:
                        lca_rhs_space = np.linspace(f2_lb + max(tol * (f2_ub - f2_lb) * 1.5, tol), f2_ub, n_vals)
                lca_rhs_space = list(reversed(lca_rhs_space))  # range[f2_ub, f2_lb]
                self._model.set_obj_weights(1.0, 0.0)
                print(f"Initializing: forage = {v}, fat constraint = {fat_sense}")
                for rhs in lca_rhs_space:
                    stage = 1 - (rhs - f2_lb + tol * 1.5)/(f2_ub - f2_lb + tol * 1.5)
                    print(f"stage = {stage*100:.4}%")
                    self._model.set_lca_rhs(rhs, stage)
                    self.__clear_searcher()
                    msg = f"multi objective lb={lb}, ub={ub}, algorithm={algorithm}, rhs={rhs}"
                    msg += f", forage = {v}"
                    sol_vec = getattr(self, algorithm)(lb, ub, tol, uncertain_bounds=True, mode="BF")
                    status, solution = self.get_sol_results(lca_id, sol_vec, best=True)
                    # self._model.write_lp_inside(rhs)
                    if status == Status.SOLVED:
                        _solutions_multiobjective.append(solution)
                    else:
                        # raise Exception("Multi-objective failure: Something wrong is going on. >numerical_methods.py")
                        print("Multi-objective failure: Something wrong is going on. >numerical_methods.py")
                        print(f2_ub - f2_lb)
                        print(status)
                        print(solution)
        self._solutions.extend(_solutions_multiobjective)


Algorithms = {'BF': 'brute_force_search', 'GSS': 'golden_section_search'}

if __name__ == "__main__":
    print("hello numerical_methods")
