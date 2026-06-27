import numpy as np
from scipy.optimize import curve_fit as cf
from scipy.optimize import least_squares as ls


# curve_fitting
# outdated
def fitting(x_data, y_data_r, y_data_i):
    # bounds for parameters
    # rewrite to account for alpha*beta as parameter
    bounds_e_inf = [0, 0.7]
    bounds_slope = [0, 1]
    bounds_tau1 = [-3.5, 1]
    bounds_tau2 = [-7, -3]
    bound_tau3 = [-10, -6.3]
    bounds_epsilon = [-np.inf, np.inf]
    bounds = np.array([[0.5, 1], bounds_epsilon, bounds_tau1, bounds_slope, bounds_slope, bounds_epsilon,
                       bounds_tau2, bounds_slope, bounds_epsilon, bound_tau3, bounds_e_inf]).T

    # combine y_data
    y_data = np.concatenate([y_data_r, y_data_i])

    # initial guesses
    epsilon_inf = y_data_r[0]
    beta_1 = 1
    delta_epsilon_1 = y_data_i[-1]
    tau_1 = - x_data[-1]
    alpha_2 = 0.8
    beta_2 = 1
    delta_epsilon_2 = np.max(y_data_i)
    tau_2 = fc_tauhn(x_data[np.argmax(y_data_i[:-6])], alpha_2, beta_2)  # account for interpolation messing with y_max
    alpha_3 = 1
    delta_epsilon_3 = y_data_i[0]
    tau_3 = - x_data[0]
    ini_par = [beta_1, delta_epsilon_1, tau_1, alpha_2, alpha_2 * beta_2, delta_epsilon_2, tau_2, alpha_3,
               delta_epsilon_3, tau_3, epsilon_inf]

    try:
        fit_par, fit_cov = cf(composition, x_data, y_data, p0=ini_par, bounds=bounds)
    except RuntimeError:
        fit_par = ini_par
    return fit_par, ini_par


# iterative curve fitting
def fitting_seq(x_data, y_data_r, y_data_i, start_values, fit_func):
    # alpha_beta1 - 0, height1 - 1, fc1 - 2, alpha2 - 3, alpha_beta2 - 4, height2 - 5, fc2 - 6, alpha3 - 7, height3 - 8,
    # fc3 - 9, e_inf - 10, alpha_h2o - 11, height_h2o - 12, fc_h2o - 13

    # combine y_data
    y_data = np.concatenate([y_data_r, y_data_i])

    # limiting range for new parameters
    r = 0.1  # relative change
    r_ab = 0.1  # absolute change
    r_h = 0.05  # maximum height deviation for main HN function from experimental data
    ab_cols = [0, 3, 4, 7, 11]
    ab_cols_sides = [0, 7]

    # set up bounds
    bounds = np.array([
        [np.maximum(start_values[i] - r_ab, 0), np.minimum(start_values[i] + r_ab, 1)] if i in ab_cols
        else np.sort([(1 - r) * start_values[i], (1 + r) * start_values[i]])
        for i in range(len(start_values))
    ]).T

    bounds[0, ab_cols_sides] = np.maximum(bounds[0, ab_cols_sides], 0.3)  # restrict side functions

    bounds[0, -3] = np.maximum(bounds[0, -3], np.log10(1.4))  # epsilon infinity

    bounds[0, 5] = np.maximum(bounds[0, 5], np.max(y_data_i) * (1 + r_h))  # height restriction

    bounds[0, 2] = np.maximum(bounds[0, 2], 0)  # low frequency HN peak frequency lower limit

    bounds[1, 9] = np.minimum(bounds[1, 9], 8)  # high frequency HN peak frequency upper limit

    if start_values[5] < bounds[0, 5]:
        start_values[5] = bounds[0, 5]

    result = ls(residues, start_values, bounds=bounds, args=(x_data, y_data, fit_func))
    fit_par = result.x
    print(result.message, result.success, result.cost)
    return fit_par


# HN function (modified with -pi/2)
def hn_function_imaginary(frequency_w, alpha, alpha_beta, height, fc):
    # frequency_w, fc and delta_epsilon in log10

    beta = alpha_beta / alpha
    tau = fc_tauhn(fc, alpha, beta)
    delta_epsilon = height_delta(alpha, alpha_beta, height, fc)

    n = np.pi * alpha / 2
    u = 2 * np.pi * 10 ** (tau + frequency_w)
    h = np.pi / 2 * 10 ** delta_epsilon

    theta = np.arctan(np.sin(n) / ((u ** (-alpha)) + np.cos(n)))

    numerator = alpha_beta * h * u ** alpha * np.cos(n - (1 + beta) * theta)

    denominator = (1 + 2 * u ** alpha * np.cos(n) + u ** (2 * alpha)) ** ((1 + beta) / 2)

    return np.log10(numerator / denominator)


# HN function for real permittivity
def hn_function_real(frequency_w, alpha, alpha_beta, height, fc):
    # frequency_w, fc and both epsilons in log10

    beta = alpha_beta / alpha
    tau = fc_tauhn(fc, alpha, beta)
    delta_epsilon = height_delta(alpha, alpha_beta, height, fc)

    n = np.pi * alpha / 2
    u = 2 * np.pi * 10 ** (tau + frequency_w)
    h = 10 ** delta_epsilon

    theta = np.arctan(np.sin(n) / ((u ** (-alpha)) + np.cos(n)))

    numerator = h * np.cos(beta * theta)

    denominator = (1 + 2 * u ** alpha * np.cos(n) + u ** (2 * alpha)) ** (beta / 2)

    return np.log10(numerator / denominator)


# combined functions
def composition(frequency_w, alpha_beta1, delta1, tau1, alpha2, alpha_beta2, delta2, tau2, alpha3, delta3, tau3, e_inf):
    e_inf = 10 ** e_inf

    hn_1_r = 10 ** hn_function_real(frequency_w, 1, alpha_beta1, delta1, tau1)
    hn_2_r = 10 ** hn_function_real(frequency_w, alpha2, alpha_beta2, delta2, tau2)
    hn_3_r = 10 ** hn_function_real(frequency_w, alpha3, alpha3, delta3, tau3)
    e_real = np.log10(hn_1_r + hn_2_r + hn_3_r + e_inf)

    hn_1_i = 10 ** hn_function_imaginary(frequency_w, 1, alpha_beta1, delta1, tau1)
    hn_2_i = 10 ** hn_function_imaginary(frequency_w, alpha2, alpha_beta2, delta2, tau2)
    hn_3_i = 10 ** hn_function_imaginary(frequency_w, alpha3, alpha3, delta3, tau3)
    e_imaginary = np.log10(hn_1_i + hn_2_i + hn_3_i)

    return np.concatenate([e_real, e_imaginary])


# include hydrogen bonding dissociation
def composition_h2o(f, ab1, h1, fc1, a2, ab2, h2, fc2, a3, h3, fc3, e_inf, a_h20, h_h2o, fc_h2o):
    e_inf = 10 ** e_inf

    hn_1_r = 10 ** hn_function_real(f, 1, ab1, h1, fc1)
    hn_2_r = 10 ** hn_function_real(f, a2, ab2, h2, fc2)
    hn_3_r = 10 ** hn_function_real(f, a3, a3, h3, fc3)
    hn_h2o_r = 10 ** hn_function_real(f, a_h20, a_h20, h_h2o, fc_h2o)
    e_real = np.log10(hn_1_r + hn_2_r + hn_3_r + hn_h2o_r + e_inf)

    hn_1_i = 10 ** hn_function_imaginary(f, 1, ab1, h1, fc1)
    hn_2_i = 10 ** hn_function_imaginary(f, a2, ab2, h2, fc2)
    hn_3_i = 10 ** hn_function_imaginary(f, a3, a3, h3, fc3)
    hn_h2o_i = 10 ** hn_function_imaginary(f, a_h20, a_h20, h_h2o, fc_h2o)
    e_imaginary = np.log10(hn_1_i + hn_2_i + hn_3_i + hn_h2o_i)

    return np.concatenate([e_real, e_imaginary])


# residuals for given data set and function
def residues(parameters, x, y, fit_function):
    return fit_function(x, *parameters) - y


def delta_height(alpha, alpha_beta, delta_epsilon, fc):
    height = hn_function_imaginary(fc, alpha, alpha_beta, delta_epsilon, fc)
    return height


def height_delta(alpha, alpha_beta, height, fc):
    beta = alpha_beta / alpha
    tau = fc_tauhn(fc, alpha, beta)

    n = np.pi * alpha / 2
    u = 2 * np.pi * 10 ** (tau + fc)
    h = 2 / np.pi * 10 ** height

    theta = np.arctan(np.sin(n) / ((u ** (-alpha)) + np.cos(n)))

    numerator = (1 + 2 * u ** alpha * np.cos(n) + u ** (2 * alpha)) ** ((1 + beta) / 2) * h

    denominator = alpha_beta * u ** alpha * np.cos(n - (1 + beta) * theta)

    return np.log10(numerator / denominator)


# tau_HN to f_critical
def tauhn_fc(tau_hn, alpha, beta):
    tau_hn = 10 ** tau_hn
    m = np.pi * alpha / (2 + 2 * beta)
    f_c = 1 / (2 * np.pi * tau_hn) * np.sin(m) ** (1 / alpha) * np.sin(m * beta) ** (-1 / alpha)
    return np.log10(f_c)


# f_c to tau_hn
def fc_tauhn(f_c, alpha, beta):
    f_c = 10 ** f_c
    m = np.pi * alpha / (2 + 2 * beta)
    tau_hn = 1 / (2 * np.pi * f_c) * np.sin(m) ** (1 / alpha) * np.sin(m * beta) ** (-1 / alpha)
    return np.log10(tau_hn)


# lin fit for two data points
def lin_fit2(x, y):
    m = (y[1] - y[0]) / (x[1] - x[0])
    n = y[0] - m * x[0]
    return m, n


# linear fit
def f_lin(x, m, n):
    return m * x + n
