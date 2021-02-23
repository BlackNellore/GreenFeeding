from model import data_handler
import pandas

from model.output_handler import Output
from model.lp_model import model_factory
from optimizer.numerical_methods import Searcher, Status, Algorithms
import logging

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
        logging.info("Iterating through scenarios")
        for scenario in data_scenario.values:

            parameters = dict(zip(headers_scenario, scenario))
            if parameters[headers_scenario.s_id] < 0:
                continue
            batch = False
            if parameters[headers_scenario.s_batch] > 0:
                batch = True

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
                continue

            tol = parameters[headers_scenario.s_tol]
            lb = parameters[headers_scenario.s_lb]
            ub = parameters[headers_scenario.s_ub]
            if not batch:
                lb, ub = self.refine_bounds(optimizer, parameters)
                if lb is None:
                    continue
                logging.info(f'Optimizing with {msg}')
                self.__single_scenario(optimizer, parameters, lb, ub, tol)
                self.store_results(optimizer, parameters)
            else:
                if list(ds.filter_column(data_batch,
                                         headers_batch.s_batch_id,
                                         parameters[headers_scenario.s_batch],
                                         int64=True)[headers_batch.s_only_costs_batch])[0]:
                    lb, ub = self.refine_bounds(optimizer, parameters, batch)
                    if lb is None:
                        continue
                logging.info(f"Optimizing with multiobjective epsilon-constrained based on {msg}")
                self.__multi_scenario(optimizer, parameters, lb, ub, tol)

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

    def __multi_scenario(self, optimizer, parameters, lb, ub, tol):
        optimizer.clear_searcher()
        batch_id = parameters[headers_scenario.s_batch]
        batch_parameters = ds.filter_column(data_batch, headers_batch.s_batch_id, batch_id, int64=True)

        batch_space = range(list(batch_parameters[headers_batch.s_final_period])[0] -
                            list(batch_parameters[headers_batch.s_initial_period])[0] + 1)
        for i in batch_space:
            optimizer.set_batch_params(i)
            self.__single_scenario(optimizer, parameters, lb, ub, tol)
            self.store_results(optimizer, parameters)
            optimizer.clear_searcher()

    @staticmethod
    def store_results(optimizer, parameters):
        logging.info("Saving solution locally")
        lca_id = parameters[headers_scenario.s_lca_id]
        status, solution = optimizer.get_sol_results(lca_id, None, False)
        if status == Status.SOLVED or lca_id > 0:
            _output.save_as_csv(name=str(parameters[headers_scenario.s_identifier]), solution=solution)
        else:
            logging.warning("Bad Status: {0}, {1}".format(status, parameters))


def config(input_info, output_info):
    global INPUT, OUTPUT
    INPUT = input_info
    OUTPUT = output_info


if __name__ == "__main__":
    pass
