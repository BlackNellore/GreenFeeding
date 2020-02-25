from typing import NamedTuple
import pandas
import logging


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def mask(df, key, value):
    """ Defines mask to filter row elements in panda's DataFrame """
    if type(value) is int:
        return df[df[key] == value]
    if type(value) is list:
        return df[df[key].isin(value)]


def unwrap_list(nested_list):
    new_list = []
    for sublist in nested_list:
        for item in sublist:
            new_list.append(item)
    return new_list


class Data:
    pandas.DataFrame.mask = mask


    # Sheet Cattle
    class ScenarioParameters(NamedTuple):
        s_id: str
        s_feed_scenario: str
        s_breed: str
        s_sbw: str
        s_bcs: str
        s_be: str
        s_l: str
        s_sex: str
        s_a2: str
        s_ph: str
        s_price: str
        s_linear_factor: str
        s_algorithm: str
        s_identifier: str
        s_lb: str
        s_ub: str
        s_tol: str
        s_lca_id: str
        s_obj: str

    # Sheet Feeds
    class ScenarioFeedProperties(NamedTuple):
        s_feed_scenario: str
        s_ID: str
        s_min: str
        s_max: str
        s_feed_cost: str
        s_name: str

    # Sheet Feed Library
    class IngredientProperties(NamedTuple):
        s_ID: str
        s_FEED: str
        s_Cost: str
        s_IFN: str
        s_Forage: str
        s_DM: str
        s_CP: str
        s_SP: str
        s_ADICP: str
        s_Sugars: str
        s_OA: str
        s_Fat: str
        s_Ash: str
        s_Starch: str
        s_NDF: str
        s_Lignin: str
        s_TDN: str
        s_ME: str
        s_NEma: str
        s_NEga: str
        s_RUP: str
        s_kd_PB: str
        s_kd_CB1: str
        s_kd_CB2: str
        s_kd_CB3: str
        s_PBID: str
        s_CB1ID: str
        s_CB2ID: str
        s_pef: str
        s_ARG: str
        s_HIS: str
        s_ILE: str
        s_LEU: str
        s_LYS: str
        s_MET: str
        s_CYS: str
        s_PHE: str
        s_TYR: str
        s_THR: str
        s_TRP: str
        s_VAL: str
        s_Ca: str
        s_P: str
        s_Mg: str
        s_Cl: str
        s_K: str
        s_Na: str
        s_S: str
        s_Co: str
        s_Cu: str
        s_I: str
        s_Fe: str
        s_Mn: str
        s_Se: str
        s_Zn: str
        s_Vit_A: str
        s_Vit_D: str
        s_Vit_E: str

    # Sheet LCA
    class LCALib(NamedTuple):
        s_ing_id: str
        s_name: str
        s_lca_ghg: str

    # Sheet LCA Library
    class LCAScenario(NamedTuple):
        s_ID: str
        s_LCA_cost: str
        s_Epislon: str
        s_LCA_weight: str
        s_LCA_GHG_weight: str
        s_Methane: str
        s_Methane_Equation: str

    headers_feed_lib: IngredientProperties = None  # Feed Library
    data_feed_lib: pandas.DataFrame = None  # Feed Library
    data_feed_scenario: pandas.DataFrame = None  # Feeds
    headers_feed_scenario: ScenarioFeedProperties = None  # Feeds
    data_scenario: pandas.DataFrame = None  # Scenario
    headers_scenario: ScenarioParameters = None  # Scenario
    headers_lca_scenario: LCAScenario = None # LCA
    data_lca_scenario: pandas.DataFrame = None  # LCA
    headers_lca_lib: LCALib = None  # LCA
    data_lca_lib: pandas.DataFrame = None  # LCA Library

    def __init__(self,
                 filename,
                 sheet_feed_lib,
                 sheet_feeds,
                 sheet_scenario,
                 sheet_lca,
                 sheet_lca_lib):
        """
        Read excel file
        :param filename : {'name'}
        :param sheet_* : {'name', 'headers'}
        """
        excel_file = pandas.ExcelFile(filename['name'])
        # TODO: Be sure that everything is on the same order

        # Feed Library Sheet
        data_feed_lib = pandas.read_excel(excel_file, sheet_feed_lib['name'])
        self.headers_feed_lib = self.IngredientProperties(*(list(data_feed_lib)))

        # Feeds scenarios
        self.data_feed_scenario = pandas.read_excel(excel_file, sheet_feeds['name'])
        self.headers_feed_scenario = self.ScenarioFeedProperties(*(list(self.data_feed_scenario)))

        # Filters feed library with the feeds on the scenario
        filter_ingredients_ids = \
            self.data_feed_scenario.filter(items=[self.headers_feed_scenario.s_ID]).values
        self.data_feed_lib = self.filter_column(data_feed_lib,
                                                self.headers_feed_lib.s_ID,
                                                unwrap_list(filter_ingredients_ids))

        # TODO Check if all ingredients exist in the library.

        # Sheet Scenario
        self.data_scenario = pandas.read_excel(excel_file, sheet_scenario['name'])
        self.headers_scenario = self.ScenarioParameters(*(list(self.data_scenario)))

        # LCA Sheet
        self.data_lca_scenario = pandas.read_excel(excel_file, sheet_lca['name'])
        self.headers_lca_scenario = self.LCAScenario(*(list(self.data_lca_scenario)))

        # LCA Library Sheet
        data_lca_lib = pandas.read_excel(excel_file, sheet_lca_lib['name'])
        self.headers_lca_lib = self.LCALib(*(list(data_lca_lib)))

        self.data_lca_lib = self.filter_column(data_lca_lib,
                                               self.headers_lca_lib.s_ing_id,
                                               unwrap_list(filter_ingredients_ids))

        # checking if config.py is consistent with Excel headers
        check_list = [(sheet_feed_lib, self.headers_feed_lib),
                      (sheet_feeds, self.headers_feed_scenario),
                      (sheet_scenario, self.headers_scenario),
                      (sheet_lca, self.headers_lca_scenario),
                      (sheet_lca_lib, self.headers_lca_lib)]
        try:
            for sheet in check_list:
                if sheet[0]['headers'] != [x for x in sheet[1]]:
                    raise IOError(sheet[0]['name'])
        except IOError as e:
            logging.error("Headers in config.py don't match header in Excel file:{}".format(e.args))
            [self.headers_feed_lib,
             self.headers_feed_scenario,
             self.headers_scenario,
             self.headers_lca_scenario,
             self.headers_lca_lib] = [None for i in range(5)]
            raise IOError(e)

        # Saving info in the log
        logging.info("\n\nAll data read")

    def datasets(self):
        """
        Return datasets
        :return list : [data_feed_lib, data_feed_scenario, data_scenario, data_lca_lib, data_lca_scenario]
        """
        return [self.data_feed_lib,
                self.data_feed_scenario,
                self.data_scenario,
                self.data_lca_lib,
                self.data_lca_scenario]

    def headers(self):
        """
        Return datasets' headers
        :return list : [headers_feed_lib, headers_feed_scenario, headers_scenario, headers_lca_lib, headers_lca_scenario]
        """
        return [self.headers_feed_lib,
                self.headers_feed_scenario,
                self.headers_scenario,
                self.headers_lca_lib,
                self.headers_lca_scenario]

    @staticmethod
    def filter_column(data_frame, col_name, val):
        """ Filter elements in data_frame where col_name == val or in [val]"""
        return data_frame.mask(col_name, val)

    @staticmethod
    def get_column_data(data_frame, col_name, func=None):
        """ Get all elements of a certain column"""
        if not isinstance(col_name, list):
            ds = data_frame.filter(items=[col_name]).values
            if func is None:
                resulting_list = [i for i in unwrap_list(ds)]
            else:
                resulting_list = [func(i) for i in unwrap_list(ds)]
            if "%" in col_name:
                resulting_list = [i*0.01 for i in unwrap_list(ds)]
            if len(resulting_list) == 1:
                return resulting_list[0]
            else:
                return resulting_list
        else:
            ds = data_frame.filter(items=col_name).get_values()
            if ds.shape[0] > 1:
                return [list(row) for row in list(ds)]
            else:
                return unwrap_list(ds)

    @staticmethod
    def map_values(headers, vals):
        """Map all column values in a dic based on the parsed header"""
        return dict(zip(list(headers), list(vals)))

    @staticmethod
    def match_by_column(data_frame1, data_frame2, col_name):
        """Return intersection of df1 with df2 filtered by column elements"""
        elements = data_frame2.filter(items=[col_name]).get_values()
        ds = data_frame1.mask(col_name, unwrap_list(elements))
        return ds

    @staticmethod
    def store_output(results_dict, filename="output1.xlsx"):
        writer = pandas.ExcelWriter(filename)
        if len(results_dict) == 0:
            logging.warning("NO SOLUTION FOUND, NOTHING TO BE SAVED")
            return
        for sheet_name in results_dict.keys():
            results = results_dict[sheet_name]
            if results is None or len(results) == 0:
                break
            df = pandas.DataFrame(results)
            df.to_excel(writer, sheet_name=sheet_name, columns=[*results[0].keys()])
        writer.save()

    @staticmethod
    def sort_df(dataframe, col):
        df: pandas.DataFrame = dataframe.copy()
        ids = list(df[col])
        ids.sort()
        mapping = dict(zip(ids, [i for i in range(len(ids))]))
        ids = [mapping[id] for id in df[col]]
        ids = pandas.Index(ids)
        df = df.set_index(ids).sort_index()
        return df

    def sorted_column(self, dataFrame, header, base_list, base_header):
        keys = list(self.get_column_data(dataFrame, base_header))
        vals = list(self.get_column_data(dataFrame, header))
        mapping = dict(zip(keys, vals))
        return [mapping[k] for k in base_list]



if __name__ == "__main__":
    print("hello data_handler")
    test_ds = Data(filename="../Input.xlsx",
                   sheet_feed_lib="Feed Library",
                   sheet_feeds="Feeds",
                   sheet_scenario="Scenario",
                   sheet_lca='LCA',
                   sheet_lca_lib='LCA Library'
                   )
