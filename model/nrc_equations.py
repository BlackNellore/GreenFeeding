import numpy as np
import abc
import logging

try:
    import rpy2.robjects as robjects
    import rpy2.rinterface_lib.embedded as rinterface
    from rpy2.robjects import pandas2ri
except OSError as e:
    try:
        import os
        import platform

        if ('Windows', 'Microsoft') in platform.system():
            os.environ["R_HOME"] = 'C:/Program Files/R/R-4.0.3/bin/x64'
            os.environ["PATH"] = "C:/Program Files/R/R-4.0.3/bin/x64" + ";" + os.environ["PATH"]
        import rpy2.robjects as robjects
        import rpy2.rinterface_lib.embedded as rinterface
        from rpy2.robjects import pandas2ri
    except OSError as e2:
        print(f'{e}\n{e2}')
        str_1 = 'rpy2 lib raised error. You likely have to add some paths in Windows\' environment variables'
        str_2 = 'proceeding without RNS\' RData file'
        print(f'{e}\n{str_1}\n{str_2}\n')
        logging.warning(f'{e}\n{str_1}\n{str_2}\n')

feed_keys = ['f_fat', 'f_CP', 'f_NDF', 'f_starch', 'f_sugars', 'f_oa']


class NRC_abs(metaclass=abc.ABCMeta):
    # @abc.abstractmethod
    # def neg(self, *args):
    #     pass

    @abc.abstractmethod
    def dmi(self, *args):
        pass

    @abc.abstractmethod
    def mpm(self, *args):
        pass

    @abc.abstractmethod
    def mpg(self, *args):
        pass

    @abc.abstractmethod
    def nem(self, *args):
        pass

    @abc.abstractmethod
    def mp(self, *args):
        pass

    @abc.abstractmethod
    def npn(self, *args):
        pass

    @abc.abstractmethod
    def ttdn(self, *args):
        pass

    @abc.abstractmethod
    def pe_ndf(self, *args):
        pass

    @abc.abstractmethod
    def ch4_diet(self, *args):
        pass


class NRC_eq:
    outside_calc = False
    diff_report = False

    nrc_handler: NRC_abs = None
    comparison_Rdata: NRC_abs = None

    def __init__(self, **kwargs):
        if len(kwargs) == 0:
            self.nrc_handler = self.StaticHandler()
        else:
            source, report_diff, on_error = kwargs['source'], kwargs['report_diff'], kwargs['on_error']
            if source is not None:
                try:
                    fixed_str = source
                    if '.RData' not in source:
                        fixed_str = f'{source}.Rdata'
                    robjects.r['load'](fixed_str)  # handled except
                    self.outside_calc = True
                    self.diff_report = report_diff

                    if self.diff_report and not self.outside_calc:
                        raise RuntimeError(f'Inconsistent parameters configured. The code is wrong')
                    if self.outside_calc:
                        robjects.r['load'](fixed_str)
                        self.nrc_handler = self.RDataHandler()
                        self.comparison_Rdata = self.StaticHandler()
                    else:
                        self.nrc_handler = self.StaticHandler()
                except rinterface.RRuntimeError as exc:
                    # 0: quit; 1: report and continue with NRC; -1: silent continue
                    if on_error == 1:
                        logging.error(exc)
                        logging.error(f'RData file not found or error durring load.'
                                      f'Proceeding with NRC built-in equations')
                    elif on_error == -1:
                        logging.warning(exc)
                        logging.warning(f'RData file not found or error durring load.'
                                        f'Proceeding with NRC built-in equations')
                    else:
                        raise FileNotFoundError
            else:
                self.nrc_handler = self.StaticHandler()

    @staticmethod
    def swg(neg, sbw, final_weight=0):
        """ Shrunk Weight Gain """
        NRC_eq.StaticHandler.test_negative_values('swg', neg=neg, sbw=sbw, final_weight=final_weight)
        if final_weight == 0:
            final_weight = sbw
        p_sbw = (sbw + final_weight) / 2
        return 13.91 * np.power(neg, 0.9116) / np.power(p_sbw, 0.6836)

    def swg_time(self, neg, sbw, feeding_time):
        """ Shrunk Weight Gain """
        NRC_eq.StaticHandler.test_negative_values('swg', neg=neg, sbw=sbw, feeding_time=feeding_time)
        final_weight = 3.05463 * (-40.9215 + np.power(
            0.1 * (16745.7 + sbw * (267.93 + 1.07172 * sbw) + 260.259 * np.power(neg, (2279 / 2500))
                   * feeding_time), 0.5))

        # swg = NRC_eq.StaticHandler.swg(neg, sbw, final_weight)
        # estimated_feeding_time = (final_weight - sbw) / swg
        return self.swg(neg, sbw, final_weight)

    @staticmethod
    def cneg(cnem):
        """ Concentration energy for growth """
        if isinstance(cnem, dict):
            cnem = cnem['cnem']
        NRC_eq.StaticHandler.test_negative_values('cneg', cnem=cnem)
        return 0.8902 * cnem - 0.4359

    @staticmethod
    def report_diference(rdata, equations, name):
        diff = rdata - equations
        if equations == 0:
            logging.error(f'{name} Rdata diff eq: {diff:.2f}\t=\t{rdata:.2f}\t-\t{equations:.2f}%\t\t\t'
                          f'Rdata val({rdata:.2f})\tEquations val({equations:.2f})')
        else:
            diff_perc = rdata / equations - 1
            if abs(diff_perc) > 0.1:
                if abs(diff_perc) > 0.7:
                    logging.error(f'{name} Rdata diff eq: {diff:.2f}\t {diff_perc * 100:.2f}%\t\t\t'
                                  f'Rdata val({rdata:.2f})\tEquations val({equations:.2f})')
                else:
                    logging.warning(f'{name} Rdata diff eq: {diff:.2f}\t {diff_perc * 100:.2f}%\t\t\t'
                                    f'Rdata val({rdata:.2f})\tEquations val({equations:.2f})')
            else:
                logging.info(f'{name} Rdata diff eq: {diff:.2f}\t {diff_perc * 100:.2f}%\t\t\t'
                             f'Rdata val({rdata:.2f})\tEquations val({equations:.2f})')

    # def neg(self, *args):
    # if self.outside_calc:
    #     if self.diff_report:
    #         self.report_diference(self.nrc_handler.neg(), self.comparison_Rdata.neg(*args), 'NEg')
    #     return self.nrc_handler.neg()
    # else:
    #     return self.nrc_handler.neg(*args)
    # return self.nrc_handler.neg(*args)
    @staticmethod
    def neg(cneg, v_dmi, cnem, v_nem):
        """ Net energy for growth """
        NRC_eq.StaticHandler.test_negative_values('neg', cneg=cneg,
                                                  v_dmi=v_dmi,
                                                  cnem=cnem,
                                                  v_nem=v_nem)
        if (v_dmi - v_nem / cnem) < 0:
            return None
        return (v_dmi - v_nem / cnem) * cneg

    def dmi(self, *args):
        if self.outside_calc:
            if self.diff_report:
                self.report_diference(self.nrc_handler.dmi(), self.comparison_Rdata.dmi(*args), 'DMI')
            # vals = NRC_eq.StaticHandler.dmi(*args)
            return self.nrc_handler.dmi()
            # return vals
        else:
            return self.nrc_handler.dmi(*args)

    def mpm(self, *args):
        if self.outside_calc:
            if self.diff_report:
                self.report_diference(self.nrc_handler.mpm(), self.comparison_Rdata.mpm(*args), 'MPm')
            return self.nrc_handler.mpm()
        else:
            return self.nrc_handler.mpm(*args)

    def mpg(self, *args):
        if self.outside_calc:
            if self.diff_report:
                self.report_diference(self.nrc_handler.mpg(), self.comparison_Rdata.mpg(*args), 'MPg')
            # vals = NRC_eq.StaticHandler.mpg(*args)
            return self.nrc_handler.mpg()
            # return vals
        else:
            return self.nrc_handler.mpg(*args)

    def nem(self, *args):
        if self.outside_calc:
            if self.diff_report:
                self.report_diference(self.nrc_handler.nem(), self.comparison_Rdata.nem(*args), 'NEm')
            vals = NRC_eq.StaticHandler.nem(*args)
            # return self.nrc_handler.nem()
            return vals
        else:
            return self.nrc_handler.nem(*args)

    def get_all_parameters(self, cnem, sbw, bcs, be, lac, sex, a2, ph_val, target_weight, dmi_eq):
        """Easier way to get all parameters needed on the model at once"""
        return self.mpm((sbw+target_weight)/2), \
               self.dmi(cnem, sbw, target_weight, dmi_eq), \
               self.nem((sbw+target_weight)/2, bcs, be, lac, sex, a2), \
               self.pe_ndf(ph_val)

    def mp(self, ing_id, *args):
        if self.outside_calc:
            if self.diff_report:
                self.report_diference(self.nrc_handler.mp(ing_id), self.comparison_Rdata.mp(*args), 'MP')
            # vals = NRC_eq.StaticHandler.mp(*args)
            return self.nrc_handler.mp(ing_id)
            # return vals
        else:
            return self.nrc_handler.mp(*args)

    def npn(self, ing_id, *args):
        if self.outside_calc:
            if self.diff_report:
                self.report_diference(self.nrc_handler.npn(ing_id), self.comparison_Rdata.npn(*args), 'NPN')
            return self.nrc_handler.npn(ing_id)
        else:
            return args[0]

    def ttdn(self, ing_id, *args):
        if self.outside_calc:
            if self.diff_report:
                self.report_diference(self.nrc_handler.ttdn(ing_id), self.comparison_Rdata.ttdn(*args), 'TDN')
            return self.nrc_handler.ttdn(ing_id)
        else:
            return args[0]

    def pe_ndf(self, *args):
        if self.outside_calc:
            if self.diff_report:
                self.report_diference(self.nrc_handler.pe_ndf(), self.comparison_Rdata.pe_ndf(*args), 'peNDF')
            # return self.nrc_handler.pe_ndf()
            return NRC_eq.StaticHandler.pe_ndf(*args)
        else:
            return self.nrc_handler.pe_ndf(*args)

    def ch4_diet(self, *args):
        if self.outside_calc:
            if self.diff_report:
                self.report_diference(self.nrc_handler.mpg(),
                                      self.comparison_Rdata.mpg(*args), 'MPg')
            return self.nrc_handler.ch4_diet(*args)
        else:
            return self.nrc_handler.ch4_diet(*args)

    @staticmethod
    def n2o_diet(animal_final_weight, n2o_eq):
        # IPCC Tier 1
        # 300kg CO2eq/ kg N2O
        # 0.33 [kg Nex/Mg animal]
        # animal[kg]/1000 [Mg]
        # 0.02 [kg N2O/kg Nex]

        if n2o_eq == "IPCC2006":
            return 0.02 * 0.33 * animal_final_weight * 298 / 1000  # kg CO2eq/day
        else:
            return 0.02 * 0.33 * animal_final_weight * 298 / 1000

    class StaticHandler(NRC_abs):

        @staticmethod
        def dmi(cnem, sbw, final_weight, eq):
            """ Dry Matter Intake """
            NRC_eq.StaticHandler.test_negative_values('dmi', cnem=cnem, sbw=sbw)
            if final_weight == 0:
                final_weight = sbw
            p_sbw = (sbw + final_weight) / 2
            if eq == "NRC2016":
                return 0.007259 * p_sbw * (1.71167 + 2.64747 * cnem - np.power(cnem, 2))
            elif eq == "NRC1996":
                if cnem < 1:
                    return np.power(p_sbw, 0.75) * (-0.0869 + 0.2435 * cnem - 0.0466 * np.power(cnem, 2)) / 0.95
                else:
                    return np.power(p_sbw, 0.75) * (-0.0869 + 0.2435 * cnem - 0.0466 * np.power(cnem, 2)) / cnem

        @staticmethod
        def mpm(sbw):
            """ Metabolizable Protein for Maintenance """
            if isinstance(sbw, dict):
                sbw = sbw['sbw']
            NRC_eq.StaticHandler.test_negative_values('mpm', sbw=sbw)
            return 3.8 * np.power(sbw, 0.75)

        @staticmethod
        def mpg(swg, neg, sbw, target_weight, feeding_time):
            """Metabolizable protein for gain"""
            npg = swg * 268 - 29.4 * neg
            if sbw >= 300:
                return npg / 0.492
            else:
                return npg / (0.834 - (sbw * 0.00114))

        @staticmethod
        def nem(sbw, bcs, be, lac, sex, a2):
            """ Net Energy for Maintenance """
            NRC_eq.StaticHandler.test_negative_values('nem', sbw=sbw,
                                                      bcs=bcs,
                                                      be=be,
                                                      l=lac,
                                                      sex=sex,
                                                      a2=a2)
            return np.power(sbw, 0.75) * (0.077 * be * lac * (0.8 + 0.05 * (bcs - 1)) * sex + a2)

        @staticmethod
        def mp(p_dmi=0, p_tdn=0, p_cp=0, p_rup=0, p_forage=0, p_ee=0, fat_orient="L"):
            """Metabolizable Protein"""
            NRC_eq.StaticHandler.test_negative_values('mp', p_dmi=p_dmi,
                                                      p_tdn=p_tdn,
                                                      p_cp=p_cp,
                                                      p_rup=p_rup,
                                                      p_forage=p_forage,
                                                      p_ee=p_ee)
            if p_dmi > 1:
                percentage = 0.01
            else:
                percentage = 1

            # NRC 8th Ed. pg 95 and pg 366
            a, b, c, alpha = None, None, None, None
            if fat_orient == "L":
                a = 42.73
                b = 0.087
                c = p_tdn
                if p_forage < 1:
                    alpha = 0.8
                else:
                    alpha = 0.6
            elif fat_orient == "G":
                a = 53.33
                b = 0.096
                c = p_tdn - 2.55 * p_ee
                if p_forage < 1:
                    alpha = 0.8
                else:
                    alpha = 0.6

            protein = a * 1 / 1000 * 0 + \
                      0.64 * b * c * percentage * 1 / 1000 + \
                      p_rup * percentage * p_cp * percentage * alpha

            return protein

        def npn(self, *args):
            return args[0]

        def ttdn(self, *args):
            return args[0]

        @staticmethod
        def pe_ndf(ph_val):
            """Physically Effective Non-Detergent Fiber"""
            if isinstance(ph_val, dict):
                ph_val = ph_val['ph_val']
            NRC_eq.StaticHandler.test_negative_values('pe_ndf', ph_val=ph_val)
            return 0.01 * (ph_val - 5.46) / 0.038

        @staticmethod
        def test_negative_values(func_name, **kwargs):
            # print([v for k, v in kwargs.items()])
            aux_vals = []
            for k, v in kwargs.items():
                if isinstance(v, tuple) or isinstance(v, list):
                    aux_vals.append(v[0])
                else:
                    aux_vals.append(v)
            if any([v < 0.0 for v in aux_vals]):
                msg = f'negative values parsed into equation {func_name}: '
                for k, v in kwargs.items():
                    if isinstance(v, tuple):
                        v = v[0]
                    if v < 0:
                        msg = msg + f'<{k}, {v}>'
                raise ValueError(msg)

        @staticmethod
        def ch4_diet(fat, cp, ash, ndf, starch, sugars, oa, ing_id):
            """
            :params fat, cp, ndf, starch, sugars, oa: float
            :return [val_forage>=20%, val_forage<=20%]: list (kg CO2eq/day)
            """
            # Convert to kg CO2eq. {1/55.65} converts MJ to kg CH4 per head.
            # {25} conevrts kg CH4 to kg CO2eq (IPCC 4th assesment, Physical Science Basis, Ch2, pg 212)
            convert = 34 * 1 / 55.65
            cho = max(1 - (cp + fat + ash), 0)
            # cho2 = ndf + starch + sugars + oa
            feed_ge = (4.15 * cho + 9.4 * fat + 5.7 * cp)  # Mcal/Kg DM
            feed_ge = (4.73 * ndf + 3.82 * (cho - ndf) + 12.48 * fat + 6.29 * cp)  # Mcal/Kg DM Moraes et al 2014
            feed_ge *= 4.18  # Mcal to MJ
            feed_ge *= convert  # MJ/kg DM per day ===> Kg CO2e/kg DM per day

            return [(0.04 * feed_ge), (0.02 * feed_ge)]  # Output kg CO2eq/day per kg of feed

    class RDataHandler(NRC_abs):
        _feed_order: list = None
        _model_feed_order: list = None
        _staticHandler: NRC_abs = None

        def __init__(self):
            feeds3 = robjects.r['feeds3']
            py_feeds = pandas2ri.rpy2py_dataframe(feeds3)
            self._feed_order = list(map(int, py_feeds['FeedID'].to_list()))
            self._feed_aTDN = list(pandas2ri.ri2py_vector(robjects.r[f'anim.fd.aTDN.conc']))
            self._feed_tTDN = list(pandas2ri.ri2py_vector(robjects.r[f'anim.fd.tTDN.conc']))
            self._feed_RUP = list(map(float, py_feeds['RUP_1x'].to_list()))
            self._feed_NPN = list(map(float, py_feeds['NPN_SP'].to_list()))
            feed_SP = list(map(float, py_feeds['SP_CP'].to_list()))
            feed_CP = list(map(float, py_feeds['CP_DM'].to_list()))
            self._feed_NPN = [self._feed_NPN[i] * feed_SP[i] * feed_CP[i]/10000 for i in range(len(feed_SP))]
            self._feed_GE = list(pandas2ri.ri2py_vector(robjects.r[f'anim.fd.GE.frac']))

        @staticmethod
        def neg():
            return robjects.r['anim.NEgr.rate'][0]

        @staticmethod
        def dmi():
            return robjects.r['anim.DMI.rate_NRC2000'][0]

        @staticmethod
        def mpm():
            return robjects.r['anim.MPmr.rate'][0]

        @staticmethod
        def mpg():  # TODO Not use
            return robjects.r['anim.MPgr.rate'][0]

        @staticmethod
        def nem():
            return robjects.r['anim.NEmr.rate'][0]

        def mp(self, ing_id):
            try:
                index = self._feed_order.index(ing_id)
                # return self._feed_MP[index]
                return self._feed_aTDN[index] * 0.13 * 0.8 * 0.8 * 0.01 \
                       + 0.8 * self._feed_RUP[index] * 0.01 * 1.682 / self.dmi()
            except ValueError as err:
                logging.error(f'Ingredient index not found in image.Rdata file. ID = {ing_id},'
                              f' available  IDs = {self._feed_order}')
                raise err

        def pe_ndf(self):
            return robjects.r['anim.peNDF_required_acidosis.rate'][0]

        def ch4_diet(self, fat, cp, ash, ndf, starch, sugars, oa, ing_id):
            # Convert to kg CO2eq. {1/55.65} converts MJ to kg CH4 per head. {25} conevrts kg CH4 to kg CO2eq
            convert = 25 * 1 / 55.65
            try:
                index = self._feed_order.index(ing_id)
                co2perday = self._feed_GE[index] * 4.18  # Mcal/Kg DM => MJ/kg DM
                co2perday *= convert  # MJ/kg DM per day ===> Kg CO2e/kg DM per day
                return [(0.065 * co2perday), (0.033 * co2perday)]  # Output kg CO2eq/day per kg of feed
            except ValueError as err:
                logging.error(f'Ingredient index not found in image.Rdata file. ID = {ing_id},'
                              f' available  IDs = {self._feed_order}')
                raise err

        def npn(self, ing_id):
            try:
                index = self._feed_order.index(ing_id)
                return self._feed_NPN[index]
            except ValueError as err:
                logging.error(f'Ingredient index not found in image.Rdata file. ID = {ing_id},'
                              f' available  IDs = {self._feed_order}')
                raise err

        def ttdn(self, ing_id):
            try:
                index = self._feed_order.index(ing_id)
                return self._feed_tTDN[index] / 100
            except ValueError as err:
                logging.error(f'Ingredient index not found in image.Rdata file. ID = {ing_id},'
                              f' available  IDs = {self._feed_order}')
                raise err


if __name__ == "__main__":
    print("hello nrc_equations")
