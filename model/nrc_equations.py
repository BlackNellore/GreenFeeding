import numpy as np


def swg(v_cneg, v_dmi, cnem, v_nem, sbw, linear_factor):
    """ Shrunk Weight Gain """
    return linear_factor * 0.87 * (v_dmi - v_nem/cnem) * v_cneg / np.power(sbw, 0.6836)


def swg_const(v_dmi, cnem, v_nem, sbw, linear_factor):
    """
    DEBUG PURPOSES:
    Constant parameter of SWG equation
    """
    return linear_factor * 0.87 * (v_dmi - v_nem / cnem) / np.power(sbw, 0.6836)


def dmi(cnem, sbw):
    """ Dry Matter Intake """
    return 0.007259 * sbw * (1.71167 + 2.64747 * cnem - np.power(cnem, 2))


def mpm(sbw):
    """ Metabolizable Protein for Maintenance """
    return 3.8 * np.power(sbw, 0.75)


def nem(sbw, bcs, be, l, sex, a2):
    """ Net Energy for Maintenance """
    return np.power(sbw, 0.75) * (0.077 * be * l * (0.8 + 0.05 * (bcs-1) * sex + a2))


def get_all_parameters(cnem, sbw, bcs, be, l, sex, a2, ph_val):
    """Easier way to get all parameters needed on the model at once"""
    return mpm(sbw), dmi(cnem, sbw), nem(sbw, bcs, be, l, sex, a2), pe_ndf(ph_val)


def mp(p_dmi, p_tdn, p_cp, p_rup, p_forage, p_ee):
    """Metabolizable Protein"""
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


def pe_ndf(ph_val):
    """Physically Effective Non-Detergent Fiber"""
    return 0.01 * (ph_val - 5.46)/0.038


feed_keys = ['f_fat', 'f_CP', 'f_NDF', 'f_starch', 'f_sugars', 'f_oa']


# TODO: ch4 emissions
def ch4_diet(feed_properties):
    test_percentage = 0
    for i in feed_keys:
        test_percentage += feed_properties[i]
    if test_percentage > 1:
        test_percentage = 0.01
    else:
        test_percentage = 1

    feed_ge = 0.2389 * (4.15 * (feed_properties['f_NDF'] +
                                feed_properties['f_starch'] +
                                feed_properties['f_sugars'] +
                                feed_properties['f_oa']) +
                        9.4 * feed_properties['f_fat'] +
                        5.7 * feed_properties['f_CP']) * test_percentage
    return {'GE20': (0.065 * feed_ge), 'LE20': (0.03 * feed_ge)}


if __name__ == "__main__":
    print("hello nrc_equations")
