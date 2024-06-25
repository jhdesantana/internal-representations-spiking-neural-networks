import numpy as np
from scipy.optimize import curve_fit
from scipy.special import zeta


def power_func(x: np.ndarray, a: float, b: float, c: float):
    return a * x ** (b) + c


def power_law_func(x: np.ndarray, a: float, b: float):
    return a * x ** (b)


def lognormal_func(x, mu, sigma):
    xmin = x.min()
    return (1 / ((x - xmin + 1) * sigma * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(x - xmin + 1) - mu) ** 2) / (2 * sigma**2)
    )


def power_law_pdf(x, alpha):
    return (x**-alpha) / (zeta(alpha, x.min()) - zeta(alpha, x.max()))


def exponential_pdf(x, alpha):
    xmin = x.min()
    return alpha * np.exp(-alpha * (x - xmin))


def stretched_exponential_pdf(x, alpha, beta):
    xmin = x.min()
    return beta * alpha * x ** (beta - 1) * np.exp(-alpha * (x**beta - xmin**beta))


def log_normal_pdf(x, mu, sigma):
    # p(x) = norm*f(x)
    norm = 2


def avalanches_s_t(
    activity: np.ndarray, threshold_value: float
) -> (np.ndarray, np.ndarray):

    threshold = np.mean(activity) * threshold_value
    idx_above = np.argwhere(activity > threshold).squeeze()

    avalanches = np.split(idx_above, np.where(np.diff(idx_above) != 1)[0] + 1)
    num_avalanches = len(avalanches)

    sizes = np.zeros(num_avalanches)
    durations = np.zeros(num_avalanches)
    for n in range(num_avalanches):
        durations[n] = avalanches[n].size
        sizes[n] = np.sum(activity[avalanches[n]] - int(threshold))

    return sizes, durations


def sizes_avegare_durations(
    sizes: np.ndarray, durations: np.ndarray
) -> (np.ndarray, np.ndarray):

    possible_durations = np.unique(durations)
    num_possible_durations = possible_durations.size
    size_avegare = np.zeros(num_possible_durations)

    for idx, d in enumerate(possible_durations):
        size_avegare[idx] = sizes[durations == d].mean()

    return size_avegare, possible_durations


def freq_avalanches(
    x: np.ndarray, xmin=1, xmax=None, function="pdf"
) -> (np.ndarray, np.ndarray):
    xmax = np.inf if xmax == None else xmax

    avalanches = np.unique(x)
    avalanches_truncated = avalanches[(avalanches >= xmin) & (avalanches <= xmax)]
    count = np.zeros(avalanches_truncated.size)

    for idx, i in enumerate(avalanches_truncated):
        count[idx] = np.sum(x == i)

    if function == "pdf":
        freq = count / np.sum(count)
        return avalanches_truncated, freq

    if function == "ccdf":
        freq = count / np.sum(count)
        freq_cum = 1 - np.cumsum(freq)
        return avalanches_truncated, freq_cum

    # return avalanches_truncated, freq


def fit_avalanches(x: np.ndarray, xmin=1, xmax=None, function="pdf"):
    xmax = np.inf if xmax == None else xmax
    rank, freq = freq_avalanches(x, xmin, xmax, function)

    popt, pcov = curve_fit(power_law_pdf, rank, freq, p0=1.5)
    perr = np.sqrt(np.diag(pcov))

    return popt, perr
