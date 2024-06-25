import lmfit
import numpy as np


def exp_func(x: np.ndarray, b: float):
    return np.exp(-b * x)


def exp_func_neg(x: np.ndarray, b: float):
    return -np.exp(-b * x)


def exp_func_spk(x: np.ndarray, a: float, b: float):
    return a * np.exp(-b * x)


def exp_func_spk_neg(x: np.ndarray, a: float, b: float):
    return -a * np.exp(-b * x)


def temporal_correlation(
    x: np.ndarray, y: np.ndarray, lag: int, delta_t: int
) -> np.ndarray:
    """Get the temporal correlation between two time series.

    The temporal correlation is a function of three parameters, t, lag and delta_t, where t is the starting time, lag is the lag time and delta_t is the time interval.

    :param x: Time series.
    :type x: np.ndarray
    :param y: Time series.
    :type y: np.ndarray
    :param lag: The lag time.
    :type lag: int
    :param delta_t: The time interval that temp correlation will be computed.
    :type delta_t: int
    :return: The temporal correlatiom between x and y.
    :rtype: np.ndarray
    """

    T = x.shape[0]
    lag = lag
    delta_t = delta_t
    corr = np.zeros(T)

    # One-dimensional time series.
    if x.ndim == 1:
        for t in range(T - lag - delta_t):
            fluctuations_x = x[t + lag : t + lag + delta_t] - np.mean(
                x[t + lag : t + lag + delta_t]
            )
            fluctuations_y = y[t : t + delta_t] - np.mean(y[t : t + delta_t])
            prod_fluc = np.mean(fluctuations_x * fluctuations_y)

            # Avoid NaN values.
            e = 1e-9
            norm_x = np.sqrt(np.mean(fluctuations_x**2) + e)
            norm_y = np.sqrt(np.mean(fluctuations_y**2) + e)
            prod_norm = norm_x * norm_y

            corr[t + lag + delta_t] = prod_fluc / prod_norm

        return corr

    # Two-dimensional time series.
    else:
        for t in range(T - lag - delta_t):
            fluctuations_x = x[t + lag : t + lag + delta_t, :] - np.mean(
                x[t + lag : t + lag + delta_t, :], axis=0
            )
            fluctuations_y = y[t : t + delta_t, :] - np.mean(
                y[t : t + delta_t, :], axis=0
            )
            prod_fluc = np.mean(np.sum(fluctuations_x * fluctuations_y, axis=1))

            # Avoid NaN values.
            e = 1e-9
            norm_x = np.sqrt(np.mean(np.sum(fluctuations_x**2, axis=1)) + e)
            norm_y = np.sqrt(np.mean(np.sum(fluctuations_y**2, axis=1)) + e)
            prod_norm = norm_x * norm_y

            corr[t + lag + delta_t] = prod_fluc / prod_norm

        return corr


def temporal_correlation_tau(
    x: np.ndarray, y: np.ndarray, lag_max: int, delta_t: int
) -> np.ndarray:
    """Get the temporal correlation in function of lag.

    :param x: Time series.
    :type x: np.ndarray
    :param y: Time series.
    :type y: np.ndarray
    :param lag_max: The highest value of lag time.
    :type lag_max: int
    :param delta_t: The time interval that temp correlation will be computed.
    :type delta_t: int
    :return: The tmeporal correlation in function of lag for each time.
    :rtype: np.ndarray
    """

    T = x.shape[0]
    corr_tau = np.zeros((T, lag_max))

    for lag in range(lag_max):
        corr_tau[:, lag] = temporal_correlation(x, y, lag, delta_t)

    return corr_tau[lag_max + delta_t :, :]


def correlation_time(
    x: np.ndarray, y: np.ndarray, lag_max: int, delta_t: int
) -> np.ndarray:
    """Get the correlation time for each time t in time series.

    :param x: Time series
    :type x: np.ndarray
    :param y: Time series
    :type y: np.ndarray
    :param lag_max: The highest value of lag time.
    :type lag_max: int
    :param delta_t: The time interval that temp correlation will be computed.
    :type delta_t: int
    :return: Correlation time for each time t in time series.
    :rtype: np.ndarray
    """

    corr_tau = temporal_correlation_tau(x, y, lag_max, delta_t)
    num_corr_tau = corr_tau.shape[0]
    lags = np.arange(lag_max)

    xi = np.zeros(num_corr_tau)

    # One-dimensional time series.
    if x.ndim == 1:
        for t in range(num_corr_tau):
            if corr_tau[t, 0] != 0:
                fit_func = exp_func if corr_tau[t, 0] > 0 else exp_func_neg

                model = lmfit.Model(fit_func)
                result = model.fit(corr_tau[t, :], x=lags, b=1)
                b_optimized = result.params["b"].value

                # Avoid divison by zero.
                if b_optimized != 0:
                    xi[t] = 1 / b_optimized
                else:
                    xi[t] = 0
            else:
                xi[t] = 0

    # Two-dimensional time series.
    else:
        for t in range(num_corr_tau):
            if corr_tau[t, 0] != 0:
                fit_func = exp_func_spk if corr_tau[t, 0] > 0 else exp_func_spk_neg

                model = lmfit.Model(fit_func)
                result = model.fit(corr_tau[t, :], x=lags, a=1, b=0.1)
                b_optimized = result.params["b"].value

                # Avoid divison by zero.
                if b_optimized != 0:
                    xi[t] = 1 / b_optimized
                else:
                    xi[t] = 0
            else:
                xi[t] = 0

    return xi
