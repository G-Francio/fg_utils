# Some functions have been adapted from Barak, as it cannot be installed anymore
# I will force astropy units to make sure things are consistent across the script
#  This WILL cause headaches, but should be better in the long run
import warnings

import astropy.units as au
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as aconst
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import norm
from tqdm import tqdm

# obvious...
debug = False

# Just for convenience
kms = au.km / au.s
adm = au.dimensionless_unscaled
AA = au.AA
c_kms = aconst.c.to(kms)

# Defaults, you might want to change this. Not that it matters too much, but still...
zem = 0.0
xem = (1 + zem) * 121.567 * au.nm
default_nm2kms = [
    (
        au.nm,
        au.km / au.s,
        lambda x: np.log(x / xem.value) * aconst.c.to(au.km / au.s),
        lambda x: np.exp(x / aconst.c.to(au.km / au.s).value) * xem.value,
    )
]


def enum_tqdm(iter, total, msg):
    return enumerate(
        tqdm(iter, ncols=120, total=total, leave=False, desc="[INFO] " + msg)
    )


def get_xmin_xmax(x):
    mean = 0.5 * (x[1:] + x[:-1])
    xmin = np.append(x[0], mean)
    xmax = np.append(mean, x[-1])
    return xmin, xmax


def get_binsize(x):
    xmin, xmax = get_xmin_xmax(x)
    return np.mean(xmax - xmin)


def plot_spec(ax, spec, **kwargs):
    xlim = kwargs.pop("xlim", False)
    ps = kwargs.pop("ps", None)

    if not isinstance(xlim, bool) or xlim:
        # try to convert xlim to the same units as the spectrum
        spec_unit = spec.x.unit
        if isinstance(xlim, au.quantity.Quantity):
            xlim = xlim.to(spec_unit, equivalencies=spec._nm2kms)
        else:
            xlim = xlim * spec.x.unit
        wave = spec.x[(spec.x.value > xlim[0].value) & (spec.x.value < xlim[1].value)]
        flux = spec.y[(spec.x.value > xlim[0].value) & (spec.x.value < xlim[1].value)]
        err = spec.dy[(spec.x.value > xlim[0].value) & (spec.x.value < xlim[1].value)]
    else:
        wave = spec.x
        flux = spec.y
        err = spec.dy

    if ps == "step":
        (p_flux,) = ax.step(wave, flux, where="mid", **kwargs)
        (p_err,) = ax.step(wave, err, where="mid", **kwargs)
        return p_flux
    else:
        (p_flux,) = ax.plot(wave, flux, **kwargs)
        (p_err,) = ax.plot(wave, err, **kwargs)
        return p_flux


def convert_to_common_unit(specs, attr):
    ulist = [getattr(s, attr) for s in specs]

    if not specs_in_same_units(ulist):
        target_unit = ulist[0]
        warnings.warn(
            "Spec were not in the same xunits, trying to the unit of the first provided..."
        )
        try:
            for s in specs:
                s.convert_x(target_unit)
        except au.UnitConversionError:
            raise NameError("Could not convert a spectrum to a common unit, aborting.")


def specs_in_same_units(ulist):
    if len(set(ulist)) == 1:
        return True
    return False


def plot_multi_spec(*specs, **kwargs):
    assert len(specs) > 0, "You need to provide at least one spectrum!"

    ax = kwargs.pop("ax", None)
    title = kwargs.pop("title", "")
    ylim = kwargs.pop("ylim", False)
    figsize = kwargs.pop("figsize", (12, 12 / 1.61))

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    if ylim:
        ax.set_ylim(ylim)

    # Check that everything shares the same units
    convert_to_common_unit(specs, "_xunit")
    convert_to_common_unit(specs, "_yunit")

    line_dict = {}
    for spec in specs:
        line_dict[spec.name] = plot_spec(ax, spec, **kwargs)
    # Generate labels - everything that has None gets deleted
    if set(line_dict.keys()) != {None}:
        plotted_lines, legend_labels = [], []
        for key in line_dict.keys():
            if key is not None:
                legend_labels.append(key)
                plotted_lines.append(line_dict[key])
        ax.legend(plotted_lines, legend_labels)
    ax.set_xlabel(spec.x.unit)
    ax.set_ylabel(spec.y.unit)

    ax.set_title(title)
    ax.grid(ls="-.", c="lightgrey")
    ax.set_axisbelow(True)
    # fig.show()


def gen_chunks(a, n):
    for i in range(0, len(a), n):
        yield a[i : i + n]


def close_all():
    plt.close("all")


def is_iterable(e):
    try:
        iter(e)
        return True
    except TypeError:
        return False


def pad_invert(arr, window):
    # used for convolution.
    # given an array, adds elments to the edges using the end of the
    # array to avoid edge effects
    # shamelessely stolen from Barak
    n = len(window) // 2
    return np.concatenate(
        (2 * arr[0] - arr[n:0:-1], arr, 2 * arr[-1] - arr[-2 : -n - 2 : -1])
    )


def compute_corrections(params, n_smooth, y):
    xi_sigma = np.exp(
        params["a"] * np.log(n_smooth) ** 2
        + params["b"] * np.log(n_smooth)
        + params["c"]
    )
    xi_fwhm = params["d_1"] * y + params["d_3"]
    return xi_sigma, xi_fwhm


def plot_simulation_results(val, err, xsq, n_rep=5000, nbins=30):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8 / 1.61))
    n, bins, _ = ax.hist(val, facecolor="none", edgecolor="blue", bins=nbins)

    # Fit histogram to the data
    (mu, sigma) = norm.fit(val)
    y = norm.pdf(bins, mu, sigma) * n_rep * np.abs(bins[0] - bins[-1]) / nbins
    ax.plot(bins, y, "r--", linewidth=2)

    ax.text(
        ax.get_xlim()[0] * 1.1 if ax.get_xlim()[0] > 0 else ax.get_xlim()[0] * 0.9,
        np.max(n),
        "{} pm {}".format(round(mu, 3), round(sigma, 3)),
    )

    # Compute the average error and compare it to the dispersion
    av_error_from_fit = np.mean(err)

    ax.hlines(
        np.max(n) * 1.25,
        mu + sigma / 2,
        mu - sigma / 2,
        label="Error from gaussian",
        color="green",
    )
    ax.hlines(
        np.max(n) * 1.2,
        mu + av_error_from_fit / 2,
        mu - av_error_from_fit / 2,
        label="Error from fit",
        color="orange",
    )
    ax.axvline(
        np.mean(val),
        np.max(n) / ax.get_ylim()[1] * 0.95,
        np.max(n) / ax.get_ylim()[1] * 1.05,
        c="red",
        ls="-.",
    )
    ax.legend()
    print("Average chi square for this round: {:.3}".format(np.mean(xsq)))
    return fig, ax


# Here ↑ goes everything that does not require quantity inputs
# Here ↓ goes everything that does require quantity inputs


@au.quantity_input
def gaussian(x, mu, sig):
    return np.exp(-0.5 * np.power((x - mu) / sig, 2.0)) / (
        sig.value * np.sqrt(2 * np.pi)
    )


# THIS WILL RETURN A SCALAR!!
@au.quantity_input
def safe(col):
    return col[~np.isnan(col.value)].value


@au.quantity_input
def interpolate_flux(wa: kms, fl: adm, efl: adm, kind=3):
    """
    Interpolates the flux and error vectors and returns the splines


    Parameters
    ----------

    Returns
    ----------
    """
    # Extrapolation seem to work decently well, maybe even better than the previous
    #  implementation with interp1d. The values inside the interpolation range
    #  *should* be equal tho!
    spline_fl = InterpolatedUnivariateSpline(wa.value, fl, k=kind, ext=0)
    spline_efl = InterpolatedUnivariateSpline(wa.value, efl, k=kind, ext=0)

    return spline_fl, spline_efl


def split_chunk_independently(spec, n_chunk=None, n_pix_chunk=None, n_pix_smooth=2):
    # this will split chunks independetly from each other
    # Will take either the length of the chunk or
    #  the number of pixels in a chunk and a starting wave
    # Will take both nm and kms, so make sure that the units are consistent
    #  it's probably better to just make everything as kms and convert nm
    # Small, but not too small chunks are better (see paper)
    # for now, we use a very simplified version and that will do
    assert (
        n_chunk is not None or n_pix_chunk is not None
    ), "No valid input, pass n_chunk or n_pix_chunk"

    # get wave, flux, err so that I can split them together -> easier (hopefully)!
    x = spec.x.value
    y = spec.y.value
    dy = spec.dy.value
    if spec.y_smooth is None:
        spec.smooth(n_pix_smooth)
    y_smooth = spec.y_smooth.value
    dy_smooth = spec.dy_smooth.value

    if n_pix_chunk is not None and n_chunk is None:
        n_chunk = len(x) // n_pix_chunk

    array = np.array_split(
        np.concatenate((x, y, dy, y_smooth, dy_smooth)).reshape(5, len(x)),
        n_chunk,
        axis=1,
    )
    for i in array:
        yield i


def split_chunk(array, n_chunk):
    # this will split chunks in an arbitrary way.
    #  still have to decide how to go about it
    # maybe pass an array of wavelength to mark start/end of each chunk?
    pass


def is_number(s):
    if s is None:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False
