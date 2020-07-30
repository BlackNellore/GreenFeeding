import numpy as np


class NRC_eq:

    @staticmethod
    def swg(neg, sbw, final_weight=0):
        """ Shrunk Weight Gain """
        NRC_eq.test_negative_values('swg', neg=neg, sbw=sbw, final_weight=final_weight)
        if final_weight == 0:
            final_weight = sbw
        p_sbw = (sbw + final_weight)/2
        return 13.91 * np.power(neg, 0.9116) / np.power(p_sbw, 0.6836)

    @staticmethod
    def swg_time(neg, sbw, feeding_time):
        """ Shrunk Weight Gain """
        NRC_eq.test_negative_values('swg', neg=neg, sbw=sbw, feeding_time=feeding_time)
        final_weight = 3.05463 * (-40.9215 + np.power(
            0.1 * (16745.7 + sbw * (267.93 + 1.07172 * sbw) + 260.259 * np.power(neg, (2279/2500))
                   * feeding_time), 0.5))

        # swg = NRC_eq.swg(neg, sbw, final_weight)
        # estimated_feeding_time = (final_weight - sbw) / swg
        return NRC_eq.swg(neg, sbw, final_weight)

    @staticmethod
    def cneg(cnem):
        """ Concentration energy for growth """
        if isinstance(cnem, dict):
            cnem = cnem['cnem']
        NRC_eq.test_negative_values('cneg', cnem=cnem)
        return 0.8902 * cnem - 0.4359

    @staticmethod
    def neg(cneg, v_dmi, cnem, v_nem):
        """ Net energy for growth """
        NRC_eq.test_negative_values('neg', cneg=cneg,
                                    v_dmi=v_dmi,
                                    cnem=cnem,
                                    v_nem=v_nem)
        if (v_dmi - v_nem/cnem) < 0:
            return None
        return (v_dmi - v_nem/cnem) * cneg

    # @staticmethod
    # def swg_const(v_dmi, cnem, v_nem, sbw, linear_factor):
    #     """
    #     DEBUG PURPOSES:
    #     Constant parameter of SWG equation
    #     """
    #     return 13.91 * linear_factor * (v_dmi - v_nem / cnem) / np.power(sbw, 0.6836)

    @staticmethod
    def dmi(cnem, sbw):
        """ Dry Matter Intake """
        NRC_eq.test_negative_values('dmi', cnem=cnem, sbw=sbw)
        return 0.007259 * sbw * (1.71167 + 2.64747 * cnem - np.power(cnem, 2))

    @staticmethod
    def mpm(sbw):
        """ Metabolizable Protein for Maintenance """
        if isinstance(sbw, dict):
            sbw = sbw['sbw']
        NRC_eq.test_negative_values('mpm', sbw=sbw)
        return 3.8 * np.power(sbw, 0.75)

    @staticmethod
    def nem(sbw, bcs, be, l, sex, a2):
        """ Net Energy for Maintenance """
        NRC_eq.test_negative_values('nem', sbw=sbw,
                                    bcs=bcs,
                                    be=be,
                                    l=l,
                                    sex=sex,
                                    a2=a2)
        return np.power(sbw, 0.75) * (0.077 * be * l * (0.8 + 0.05 * (bcs-1) * sex + a2))

    @staticmethod
    def get_all_parameters(cnem, sbw, bcs, be, l, sex, a2, ph_val):
        """Easier way to get all parameters needed on the model at once"""
        return NRC_eq.mpm(sbw), NRC_eq.dmi(cnem, sbw), NRC_eq.nem(sbw, bcs, be, l, sex, a2), NRC_eq.pe_ndf(ph_val)

    @staticmethod
    def mp(p_dmi=0, p_tdn=0, p_cp=0, p_rup=0, p_forage=0, p_ee=0):
        """Metabolizable Protein"""
        NRC_eq.test_negative_values('mp', p_dmi=p_dmi,
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
        if p_ee < 0.039:
            a = 42.73
            b = 0.087
            c = p_tdn
            if p_forage < 1:
                alpha = 0.8
            else:
                alpha = 0.8
        else:
            a = 53.33
            b = 0.096
            c = p_tdn - 2.55 * p_ee
            if p_forage < 1:
                alpha = 0.8
            else:
                alpha = 0.8

        protein = a * 1/1000 * 0 + 0.64 * b * c * percentage * 1/1000 + p_rup * percentage * p_cp * percentage * alpha

        return protein

    @staticmethod
    def pe_ndf(ph_val):
        """Physically Effective Non-Detergent Fiber"""
        if isinstance(ph_val, dict):
            ph_val = ph_val['ph_val']
        NRC_eq.test_negative_values('pe_ndf', ph_val=ph_val)
        return 0.01 * (ph_val - 5.46)/0.038

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
    def ch4_diet(fat, cp, NDF, starch, sugars, oa, dmi):
        """
        :params fat, cp, NDF, starch, sugars, oa: float
        :return [val_forage>=20%, val_forage<=20%]: list
        """
        # Convert to kg CO2eq. 1/55.65 convert to kg CH4 per head. 84 conevrts kg CH4 to kg CO2eq
        convert = 1/55.65 * 84
        feed_ge = dmi * 0.2389 * (4.15 * (NDF + starch + sugars + oa) +
                                  9.4 * fat +
                                  5.7 * cp) * convert
        return [(0.065 * feed_ge), (0.03 * feed_ge)]

    @staticmethod
    def n2o_diet(animal_final_weight):
        # IPCC Tier 1
        # 300kg CO2eq/ kg N2O
        # 0.33 [kg Nex/Mg animal]
        # animal[kg]/1000 [Mg]
        # 0.02 [kg N2O/kg Nex]
        return 0.02 * 0.33 * animal_final_weight * 300 / 1000


if __name__ == "__main__":
    pass
