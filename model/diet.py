from model import data_handler
import pandas
from model.lp_model import model_factory
from optimizer.numerical_methods import Searcher, Status, Algorithms
import logging

INPUT = {}
OUTPUT = None

ds: data_handler.Data = None

data_scenario: pandas.DataFrame = None  # Scenario
headers_scenario: data_handler.Data.ScenarioParameters = None  # Scenario


class Diet:
    @staticmethod
    def initialize(msg):
        global ds, data_scenario, headers_scenario
        ds = data_handler.Data(**INPUT)
        data_scenario = ds.data_scenario
        headers_scenario = ds.headers_scenario
        logging.info(msg)

    @staticmethod
    def run():
        logging.info("Iterating through scenarios")
        results = {}
        for scenario in data_scenario.values:
            parameters = dict(zip(headers_scenario, scenario))
            multiobjective = parameters[headers_scenario.s_multiobjectve]

            logging.info("Current Scenario:")
            logging.info("{}".format(parameters))

            logging.info("Initializing simple model")
            model = model_factory(ds, parameters,
                                  parameters[headers_scenario.s_lca_id],
                                  parameters[headers_scenario.s_multiobjectve])
            logging.info("Initializing numerical methods")
            [lb, ub, tol] = [parameters[headers_scenario.s_lb],
                             parameters[headers_scenario.s_ub],
                             parameters[headers_scenario.s_tol]]

            optimizer = Searcher(model, tol)

            logging.info("Choosing optimization method")

            if parameters[headers_scenario.s_algorithm] == "GSS":
                msg = "Golden-Section Search algorithm"
            elif parameters[headers_scenario.s_algorithm] == "BF":
                msg = "Brute Force algorithm"
            else:
                logging.error("Algorithm {} not found, scenario skipped".format(
                    parameters[headers_scenario.s_algorithm]))
                continue
            algorithm = Algorithms[parameters[headers_scenario.s_algorithm]]
            if not multiobjective:
                logging.info(f'Optimizing with {msg}')
                optimizer.single_objective(algorithm, lb, ub, tol, parameters[headers_scenario.s_lca_id])
            else:
                logging.info(f"Optimizing with multiobjective epsilon-constrained based on {msg}")
                optimizer.multi_objective(algorithm, lb, ub, tol, parameters[headers_scenario.s_lca_id])

            logging.info("Saving solution locally")
            status, solution = optimizer.get_results()
            if status == Status.SOLVED:
                results[parameters[headers_scenario.s_identifier]] = solution
            else:
                logging.warning("Bad Status: {0}, {1}".format(status, parameters))

        logging.info("Exporting solution to {}".format(OUTPUT))
        ds.store_output(results, OUTPUT)

        logging.info("END")


def config(input_info, output_info):
    global INPUT, OUTPUT
    INPUT = input_info
    OUTPUT = output_info


if __name__ == "__main__":
    pass
