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
        s_batch: str
        s_breed: str
        s_sbw: str
        s_feeding_time: str
        s_target_weight: str
        s_bcs: str
        s_be: str
        s_l: str
        s_sex: str
        s_a2: str
        s_ph: str
        s_price: str
        s_algorithm: str
        s_identifier: str
        s_lb: str
        s_ub: str
        s_tol: str
        s_dmi_eq: str
        s_obj: str
        s_find_reduced_cost: str
        s_ing_level: str
        s_lca_id: str

    # Sheet Feeds
    class ScenarioFeedProperties(NamedTuple):
        s_feed_scenario: str
        s_ID: str
        s_min: str
        s_max: str
        s_feed_cost: str
        s_name: str

    # Sheet Batchs
    class BatchParameters(NamedTuple):
        s_batch_id: str
        s_file_name: str
        s_period_col: str
        s_initial_period: str
        s_final_period: str
        s_only_costs_batch: str

    # Sheet Feed Library
    class IngredientProperties(NamedTuple):
        s_ID: str
        s_FEED: str
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
        s_NEma: str
        s_NEga: str
        s_RUP: str
        s_pef: str
        s_NPN: str

    # Sheet LCA Library
    class LCALib(NamedTuple):
        s_ing_id: str
        s_name: str
        s_LCA_phosphorus: str
        s_LCA_renewable_fossil: str
        s_LCA_GHG: str
        s_LCA_acidification: str
        s_LCA_eutrophication: str
        s_LCA_land_competition: str

    # Sheet LCA Scenario
    class LCAScenario(NamedTuple):
        s_ID: str
        s_LCA_phosphorus_weight: str
        s_LCA_renewable_fossil_weight: str
        s_LCA_GHG_weight: str
        s_LCA_acidification_weight: str
        s_LCA_eutrophication_weight: str
        s_LCA_land_competition_weight: str
        s_Methane_Equation: str
        s_N2O_Equation: str
        s_Normalize: str

    # Sheet LCA Library
    class LCALib(NamedTuple):
        s_ing_id: str
        s_name: str
        s_LCA_phosphorus: str
        s_LCA_renewable_fossil: str
        s_LCA_GHG: str
        s_LCA_acidification: str
        s_LCA_eutrophication: str
        s_LCA_land_competition: str

    # Sheet LCA Scenario
    class LCAScenario(NamedTuple):
        s_ID: str
        s_LCA_phosphorus_weight: str
        s_LCA_renewable_fossil_weight: str
        s_LCA_GHG_weight: str
        s_LCA_acidification_weight: str
        s_LCA_eutrophication_weight: str
        s_LCA_land_competition_weight: str
        s_Methane_Equation: str
        s_N2O_Equation: str
        s_Normalize: str

    headers_feed_lib: IngredientProperties = None  # Feed Library
    headers_feed_scenario: ScenarioFeedProperties = None  # Feeds
    headers_scenario: ScenarioParameters = None  # Scenario
    headers_batch: BatchParameters = None  # Batch

    data_feed_lib: pandas.DataFrame = None  # Feed Library
    data_feed_scenario: pandas.DataFrame = None  # Feeds
    data_scenario: pandas.DataFrame = None  # Scenario
    data_batch: pandas.DataFrame = None  # Batch

    headers_lca_scenario: LCAScenario = None  # LCA
    data_lca_scenario: pandas.DataFrame = None  # LCA
    headers_lca_lib: LCALib = None  # LCA
    data_lca_lib: pandas.DataFrame = None  # LCA Library

    data_series = {}  # batch dictionary

    batchScenarioCandidate = None
    batchFeedScenarioCandidate = None

    batch_map = None

    def __init__(self,
                 filename,
                 sheet_feed_lib,
                 sheet_feeds,
                 sheet_scenario,
                 sheet_batch,
                 sheet_lca,
                 sheet_lca_lib):
        """
        Read excel file
        :param filename : {'name'}
        :param sheet_* : {'name', 'headers'}
        """
        excel_file = pandas.ExcelFile(filename['name'])

        # Feed Library Sheet
        data_feed_lib = pandas.read_excel(excel_file, sheet_feed_lib['name'])
        self.headers_feed_lib = self.IngredientProperties(*(list(data_feed_lib)))
        # data_feed_lib.astype({self.headers_feed_lib.s_ID: 'int64'}).dtypes

        # Feeds scenarios
        self.data_feed_scenario = pandas.read_excel(excel_file, sheet_feeds['name'])
        self.headers_feed_scenario = self.ScenarioFeedProperties(*(list(self.data_feed_scenario)))
        # self.data_feed_scenario.astype({self.headers_feed_scenario.s_ID: 'int64'}).dtypes

        # Filters feed library with the feeds on the scenario
        filter_ingredients_ids = \
            self.data_feed_scenario.filter(items=[self.headers_feed_scenario.s_ID]).values
        self.data_feed_lib = self.filter_column(data_feed_lib,
                                                self.headers_feed_lib.s_ID,
                                                unwrap_list(filter_ingredients_ids),
                                                int64=True)

        # Sheet Scenario
        self.data_scenario = pandas.read_excel(excel_file, sheet_scenario['name'])
        self.headers_scenario = self.ScenarioParameters(*(list(self.data_scenario)))
        # self.data_scenario.astype({self.headers_scenario.s_id: 'int64'}).dtypes

        # Sheet batch
        self.data_batch = pandas.read_excel(excel_file, sheet_batch['name'])
        self.headers_batch = self.BatchParameters(*(list(self.data_batch)))
        # self.data_batch.astype({self.headers_batch.s_batch_id: 'int64'}).dtypes

        # csv files
        csv_file_names = dict(zip(unwrap_list(self.data_batch.filter(items=[self.headers_batch.s_batch_id]).values),
                                  unwrap_list(self.data_batch.filter(items=[self.headers_batch.s_file_name]).values)))
        for name in csv_file_names.values():
            self.data_series[name] = pandas.read_csv(name, index_col=0)

        # filter batch executions from data_scenario
        batch_ids = unwrap_list(self.data_batch.filter(items=[self.headers_batch.s_batch_id]).values)
        batch_scenarios = self.filter_column(self.data_scenario,
                                             self.headers_scenario.s_batch,
                                             batch_ids,
                                             int64=True)

        # filter batch executions from data_feed_scenario
        feed_scenarios_id = unwrap_list(batch_scenarios.filter(items=[self.headers_scenario.s_feed_scenario]).values)
        batch_feed_scenarios = self.filter_column(self.data_feed_scenario,
                                                  self.headers_feed_scenario.s_feed_scenario,
                                                  feed_scenarios_id,
                                                  int64=True)

        self.batchScenarioCandidate = [self.headers_scenario.s_sbw, self.headers_scenario.s_bcs,
                                       self.headers_scenario.s_be, self.headers_scenario.s_l,
                                       self.headers_scenario.s_sex, self.headers_scenario.s_a2,
                                       self.headers_scenario.s_ph, self.headers_scenario.s_price]

        self.batchFeedScenarioCandidate = [self.headers_feed_scenario.s_min, self.headers_feed_scenario.s_max,
                                           self.headers_feed_scenario.s_feed_cost]

        list_batch_id = unwrap_list(batch_scenarios.filter(items=[self.headers_scenario.s_id]).values)

        self.batch_map = dict(zip(list_batch_id, [None] * batch_scenarios.shape[0]))

        for batch_id in self.batch_map.keys():
            if batch_id <= 0:
                continue
            self.batch_map[batch_id] = {"data_feed_scenario": {}, "data_scenario": {}}
            batch_data_feed_scenario = {}
            row = self.filter_column(batch_scenarios,
                                     self.headers_scenario.s_id,
                                     batch_id,
                                     int64=True)
            feed_scn = list(row[self.headers_scenario.s_feed_scenario])[0]
            batch_ids = list(row[self.headers_scenario.s_batch])[0]
            subset_feed = self.filter_column(batch_feed_scenarios,
                                             self.headers_feed_scenario.s_feed_scenario,
                                             feed_scn,
                                             int64=True)
            for h_feed_scn in self.batchFeedScenarioCandidate:
                for j, val in enumerate(list(subset_feed[h_feed_scn])):
                    if type(val) is str:
                        if feed_scn not in batch_data_feed_scenario:
                            batch_data_feed_scenario[feed_scn] = {}
                        feed_id = list(subset_feed[self.headers_feed_scenario.s_ID])[j]
                        if feed_id not in batch_data_feed_scenario[feed_scn]:
                            batch_data_feed_scenario[feed_scn][feed_id] = {}
                        batch_name = csv_file_names[batch_id]
                        initial = list(self.filter_column(self.data_batch,
                                                          self.headers_batch.s_batch_id,
                                                          batch_id,
                                                          int64=True)[self.headers_batch.s_initial_period])[0]
                        final = list(self.filter_column(self.data_batch,
                                                        self.headers_batch.s_batch_id,
                                                        batch_id,
                                                        int64=True)[self.headers_batch.s_final_period])[0]
                        batch_data_feed_scenario[feed_scn][feed_id][h_feed_scn] = \
                            list(self.get_series_from_batch(self.data_series[batch_name],
                                                            val,
                                                            [initial, final]))
            self.batch_map[batch_id]["data_feed_scenario"] = batch_data_feed_scenario

            batch_data_scenario = {}
            for h_scn in self.batchScenarioCandidate:
                for j, val in enumerate(list(row[h_scn])):
                    if type(val) is str:
                        if batch_id not in batch_data_scenario:
                            batch_data_scenario[batch_id] = {}
                        batch_name = csv_file_names[batch_id]
                        initial = list(self.filter_column(self.data_batch,
                                                          self.headers_batch.s_batch_id,
                                                          batch_id,
                                                          int64=True)[self.headers_batch.s_initial_period])[0]
                        final = list(self.filter_column(self.data_batch,
                                                        self.headers_batch.s_batch_id,
                                                        batch_id,
                                                        int64=True)[self.headers_batch.s_final_period])[0]
                        batch_data_scenario[batch_id][h_scn] = \
                            list(self.get_series_from_batch(self.data_series[batch_name],
                                                            val,
                                                            [initial, final]))
            self.batch_map[batch_id]["data_scenario"] = batch_data_scenario

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
                      (sheet_batch, self.headers_batch),
                      (sheet_lca, self.headers_lca_scenario),
                      (sheet_lca_lib, self.headers_lca_lib)
                      ]

        try:
            for sheet in check_list:
                sh_dict: dict = sheet[0]
                if sh_dict['headers'] != [x for x in sheet[1]]:
                    raise IOError(sh_dict['name'])
        except IOError as e:
            logging.error("Headers in config.py don't match header in Excel file:{}".format(e.args))
            # DO NOT DELETE - 20/10/2020
            # [self.headers_feed_lib,
            #  self.headers_feed_scenario,
            #  self.headers_scenario] = [None, None, None]
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
                self.data_scenario]

    def headers(self):
        """
        Return datasets' headers
        :return list : [headers_feed_lib,
                        headers_feed_scenario,
                        headers_scenario,
                        headers_lca_lib,
                        headers_lca_scenario]
        """
        return [self.headers_feed_lib,
                self.headers_feed_scenario,
                self.headers_scenario]

    @staticmethod
    def get_series_from_batch(batch, col_name, period):
        return batch[col_name].loc[period[0]:period[1]]

    @staticmethod
    def filter_column(data_frame, col_name, val, int64=True):
        """ Filter elements in data_frame where col_name == val or in [val]"""
        if int64:
            try:
                if isinstance(val, list):
                    val = list(map(int, val))
                    return data_frame.mask(col_name, val)
                else:
                    return data_frame.mask(col_name, int(val))
            except TypeError:
                return data_frame.mask(col_name, val)
        else:
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
                try:
                    resulting_list = [i * 0.01 for i in unwrap_list(ds)]
                except TypeError:
                    resulting_list = []
                    for i in unwrap_list(ds):
                        if isinstance(i, str):
                            resulting_list.append(i)
                        else:
                            resulting_list.append(i * 0.01)
            if len(resulting_list) == 1:
                return resulting_list[0]
            else:
                return resulting_list
        else:
            ds = data_frame.filter(items=col_name).to_numpy()
            if ds.shape[0] > 1:
                return [list(row) for row in list(ds)]
            else:
                return unwrap_list(ds)

    # def get_dict_data(self, dataframe, header, base_list, base_header):
    #     keys = list(self.get_column_data(dataframe, base_header))
    #     vals = list(self.get_column_data(dataframe, header))
    #     return dict(zip(keys, vals))

    @staticmethod
    def map_values(headers, vals):
        """Map all column values in a dic based on the parsed header"""
        return dict(zip(list(headers), list(vals)))

    @staticmethod
    def match_by_column(data_frame1, data_frame2, col_name):
        """Return intersection of df1 with df2 filtered by column elements"""
        elements = data_frame2.filter(items=[col_name]).to_numpy()
        ds = data_frame1.mask(col_name, unwrap_list(elements))
        return ds

    @staticmethod
    def store_output(results_dict, filename="output.xlsx"):
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
        ids = [mapping[col_id] for col_id in df[col]]
        ids = pandas.Index(ids)
        df = df.set_index(ids).sort_index()
        return df

    def sorted_column(self, data_frame, header, base_list, base_header, return_dict=False):
        """
        :param data_frame:
        :param header: Name of the column to extract data from
        :param base_list: Some list with an specific order of return (if return_dict=False)
        :param base_header: Column to get headers from
        :param return_dict: if False, return list
        :return:
        """
        keys = list(self.get_column_data(data_frame, base_header))
        vals = list(self.get_column_data(data_frame, header))
        mapping = dict(zip(keys, vals))
        if return_dict:
            return mapping
        else:
            return [mapping[k] for k in base_list]

    def multi_sorted_column(self, data_frame, header, base_list, base_header, return_dict=False):
        all_dicts = []
        for h in header:
            all_dicts.append(self.sorted_column(data_frame, h, base_list, base_header, return_dict))
        return all_dicts


if __name__ == "__main__":
    print("hello data_handler")
    test_ds = Data(filename="../Input.xlsx",
                   sheet_feed_lib="Feed Library",
                   sheet_feeds="Feeds",
                   sheet_scenario="Scenario",
                   sheet_batch="Batch",
                   sheet_lca='LCA',
                   sheet_lca_lib='LCA Library'
                   )
