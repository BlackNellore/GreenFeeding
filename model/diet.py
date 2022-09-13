from model import data_handler
import pandas

from model.output_handler import Output
from model.lp_model import model_factory
from optimizer.numerical_methods import Searcher, Status, Algorithms
import logging
from tqdm import tqdm

INPUT = {}
OUTPUT = None

_output: Output

ds: data_handler.Data

data_scenario: pandas.DataFrame  # Scenario
headers_scenario: data_handler.Data.ScenarioParameters  # Scenario
data_batch: pandas.DataFrame  # Scenario
headers_batch: data_handler.Data.BatchParameters  # Scenario

class Diet:

    @staticmethod
    def initialize(msg):
        global _output, ds, data_scenario, headers_scenario, data_batch, headers_batch
        _output = Output()
        ds = data_handler.Data(**INPUT)
        data_scenario = ds.data_scenario
        headers_scenario = ds.headers_scenario
        data_batch = ds.data_batch
        headers_batch = ds.headers_batch
        logging.info(msg)

    def run(self):


        def run_model():
            logging.info("Current Scenario:")
            logging.info("{}".format(parameters))

            logging.info("Initializing model")
            model = model_factory(ds,
                                  parameters,
                                  parameters[headers_scenario.s_find_reduced_cost],
                                  parameters[headers_scenario.s_lca_id])
            logging.info("Initializing numerical methods")
            optimizer = Searcher(model, batch)

            if parameters[headers_scenario.s_algorithm] == "GSS":
                msg = "Golden-Section Search algorithm"
            elif parameters[headers_scenario.s_algorithm] == "BF":
                msg = "Brute Force algorithm"
            else:
                logging.error("Algorithm {} not found, scenario skipped".format(
                    parameters[headers_scenario.s_algorithm]))
                return False

            tol = parameters[headers_scenario.s_tol]
            lb = parameters[headers_scenario.s_lb]
            ub = parameters[headers_scenario.s_ub]
            if not batch:
                lb, ub = self.refine_bounds(optimizer, parameters)
                if lb is None:
                    return False
                logging.info(f'Optimizing with {msg}')
                if parameters[headers_scenario.s_additive_id] > 0:
                    # LCA Multiobjective with additives
                    self.__multi_scenario(optimizer, parameters, lb, ub, tol, "additive")
                else:
                    # LCA Multiobjective simple
                    self.__single_scenario(optimizer, parameters, lb, ub, tol)
                    self.store_results(optimizer, parameters)
            else:
                # Batch scenario (feeds and prices)
                if list(ds.filter_column(data_batch,
                                         headers_batch.s_batch_id,
                                         parameters[headers_scenario.s_batch],
                                         int64=True)[headers_batch.s_only_costs_batch])[0]:
                    lb, ub = self.refine_bounds(optimizer, parameters, batch)
                    if lb is None:
                        return False
                logging.info(f"Optimizing with multiobjective epsilon-constrained based on {msg}")
                self.__multi_scenario(optimizer, parameters, lb, ub, tol, "batch")

            return True

        logging.info("Iterating through scenarios")
        run_scenarios = ds.data_scenario[ds.data_scenario[ds.headers_scenario.s_id] > 0]

        for scenario in tqdm(run_scenarios.iterrows(), total=run_scenarios.shape[0]):
            original_parameters = dict(zip(headers_scenario, scenario[1].values))
            parameters = dict(zip(headers_scenario, scenario[1].values))
            if parameters[headers_scenario.s_id] < 0:
                continue
            batch = False
            if parameters[headers_scenario.s_batch] > 0:
                batch = True

            if parameters[headers_scenario.s_sensitivity_analysis] is None:
                if not run_model():
                    continue
            else:
                df_sa = pandas.read_csv(parameters[headers_scenario.s_sensitivity_analysis])
                sa_map = ds.search_spreadsheet(df_sa.columns.to_list())
                dirName = ""
                for row in tqdm(df_sa.iterrows(), total=df_sa.shape[0]):
                    for k, v in sa_map.items():
                        ds.data_holder[v[0]].loc[v[1],v[2]] = row[1][k]
                    subscenario = ds.data_scenario[ds.data_scenario[ds.headers_scenario.s_id] == scenario[1][ds.headers_scenario.s_id]]
                    parameters = dict(zip(headers_scenario, subscenario.values[0]))
                    parameters[headers_scenario.s_identifier] = f"{parameters[headers_scenario.s_identifier]}_{int(row[1]['ID'])}"
                    if not run_model():
                        continue
                    dirName = _output.store_partial(row[1]['ID'], dirName)

        _output.store()

        logging.info("END")


    @staticmethod
    def refine_bounds(optimizer, parameters, batch=False):
        logging.info("Refining bounds")
        if batch:
            optimizer.set_batch_params(0)
        lb, ub = optimizer.refine_bounds(parameters[headers_scenario.s_lb],
                                         parameters[headers_scenario.s_ub],
                                         parameters[headers_scenario.s_tol],
                                         double_refinement=True
                                         )
        if lb is None:
            logging.warning(
                f"There is no feasible solution in the domain {parameters[headers_scenario.s_lb]}"
                f" <= CNEm <= {parameters[headers_scenario.s_ub]}")
            return None, None
        logging.info("Refinement completed")
        logging.info("Choosing optimization method")
        return lb, ub

    @staticmethod
    def __single_scenario(optimizer, parameters, lb, ub, tol):
        algorithm = Algorithms[parameters[headers_scenario.s_algorithm]]
        if parameters[headers_scenario.s_find_reduced_cost] > 0:
            optimizer.search_reduced_cost(algorithm, lb, ub, tol, parameters[headers_scenario.s_ing_level])
        else:
            optimizer.run_scenario(algorithm, lb, ub, tol, parameters[headers_scenario.s_lca_id])

    def __multi_scenario(self, optimizer, parameters, lb, ub, tol, multi_type):
        if multi_type == "batch":
            optimizer.clear_searcher()
            batch_id = parameters[headers_scenario.s_batch]
            batch_parameters = ds.filter_column(data_batch, headers_batch.s_batch_id, batch_id, int64=True)

            batch_space = range(list(batch_parameters[headers_batch.s_final_period])[0] -
                                list(batch_parameters[headers_batch.s_initial_period])[0] + 1)
            for i in tqdm(batch_space):
                optimizer.set_batch_params(i)
                self.__single_scenario(optimizer, parameters, lb, ub, tol)
                self.store_results(optimizer, parameters)
                optimizer.clear_searcher()
        elif multi_type == "additive":
            additive_id = parameters[headers_scenario.s_additive_id]
            additive_scenarios = ds.data_additive_scenario[
                ds.data_additive_scenario[ds.headers_additive_scenario.s_Additive_scn_id] == additive_id]
            for (index, row) in tqdm(additive_scenarios.iterrows(), desc='Running additive scenarios'):
                kwargs = dict(zip(['ing_id', 'inclusion', 'methane_reduction'],
                                  row[[ds.headers_additive_scenario.s_ID,
                                       ds.headers_additive_scenario.s_Inclusion,
                                       ds.headers_additive_scenario.s_Methane_reduction
                                       ]].array))
                optimizer.set_additives_params(**kwargs)
                self.__single_scenario(optimizer, parameters, lb, ub, tol)
                self.store_results(optimizer, parameters)

    @staticmethod
    def store_results(optimizer, parameters):
        logging.info("Saving solution locally")
        lca_id = parameters[headers_scenario.s_lca_id]
        status, solution = optimizer.get_sol_results(lca_id, None, False)
        if status == Status.SOLVED or lca_id > 0:
            _output.save_as_csv(name=str(parameters[headers_scenario.s_identifier]), solution=solution)
        else:
            logging.warning("Bad Status: {0}, {1}".format(status, parameters))


def config(input_info):
    global INPUT
    INPUT = input_info


if __name__ == "__main__":
    pass
