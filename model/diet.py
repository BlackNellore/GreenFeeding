from model import data_handler
from model.lp_model import model_factory
from optimizer.numerical_methods import Searcher, Status
import logging
import time

INPUT = {}
OUTPUT = None

[ds,
 scenarios,
 h_scenarios] = [None for i in range(3)]


def run():
    logging.info("Iterating through scenarios")
    results = {}
    for scenario in scenarios.values:
        parameters = dict(zip(h_scenarios, scenario))
        lca_id = parameters[h_scenarios.s_lca_id]

        logging.info("Current Scenario:")
        logging.info("{}".format(parameters))

        logging.info("Initializing model")
        model = model_factory(ds, parameters, lca_id)
        logging.info("Initializing numerical methods")
        optimizer = Searcher(model) # TODO: epsilon-constrained

        tol = parameters[h_scenarios.s_tol]
        logging.info("Refining bounds")
        lb, ub = optimizer.refine_bounds(parameters[h_scenarios.s_lb],
                                         parameters[h_scenarios.s_ub],
                                         tol
                                         )

        if lb is None or ub is None:
            logging.warning("There is no feasible solution in the domain {0} <= CNEm <= {1}"
                            .format(parameters[h_scenarios.s_lb], parameters[h_scenarios.s_ub]))
            continue
        logging.info("Refinement completed")
        logging.info("Choosing optimization method")
        if lca_id <= 0:
            if parameters[h_scenarios.s_algorithm] == "GSS":
                logging.info("Optimizing with Golden-Section Search algorithm")
                optimizer.golden_section_search(lb, ub, tol)
            elif parameters[h_scenarios.s_algorithm] == "BF":
                logging.info("Optimizing with Brute Force algorithm")
                optimizer.brute_force_search(lb, ub, tol)
            else:
                logging.error("Algorithm {} not found, scenario skipped".format(
                    parameters[h_scenarios.s_algorithm]))
                continue
        else:
            # TODO: implement epsilon-constrained
            pass
        logging.info("Saving solution locally")
        status, solution = optimizer.get_results()
        if status == Status.SOLVED:
            results[parameters[h_scenarios.s_identifier]] = solution
        else:
            logging.warning("Bad Status: {0}, {1}".format(status, parameters))

    logging.info("Exporting solution to {}".format(OUTPUT))
    ds.store_output(results, OUTPUT)

    logging.info("END")


def config(input_info, output_info):
    global INPUT, OUTPUT
    INPUT = input_info
    OUTPUT = output_info


def initialize(msg):
    global ds, feed_properties, h_feed_properties, scenarios, h_scenarios
    ds = data_handler.Data(**INPUT)

    feed_properties = ds.data_feed_properties
    h_feed_properties = ds.headers_feed_properties
    scenarios = ds.data_scenario
    h_scenarios = ds.headers_data_scenario

    logging.info(msg)


if __name__ == "__main__":
    start_time = time.time()
    initialize("Starting diet.py")
    run()
    elapsed_time = time.time() - start_time
    print(elapsed_time)
