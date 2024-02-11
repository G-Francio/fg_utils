# -------------------------------- ** -------------------------------- #
# Actual FITS parsing: retrieves information based on the instrument.  #
# Same structure, sometimes cards have different names thus different  #
# functions.                                                           #
# Works provided an instrument is given, otherwise it can't set the    #
# correct cards.                                                       #
# -------------------------------- ** -------------------------------- #

import numpy as np

from .spec import generic_spectrum as gspec
from .utilities import AA, adm


def _revIVar(x, m):
    """
    Helper function to calculate the inverse variance with a minimum value.

    Parameters:
        x (float): Input value.
        m (float): Minimum value.

    Returns:
        float: Inverse variance with a minimum value.
    """
    if x == 0 or x < 0:
        return m
    return np.sqrt(1 / x)


def parse_common_fits(hdul, name, has_err=False):
    """
    Parses information from a given HDU, for data produced at WFCCD.

    Parameters:
        hdul: Header Data Unit List representing the FITS file.
        name (str): The name of the spectrum.
        has_err (bool): Whether the FITS file has error information.

    Returns:
        gspec: An instance of the generic_spectrum class.
    """
    start = hdul[0].header["CRVAL1"]
    step = hdul[0].header["CDELT1"]
    total = hdul[0].header["NAXIS1"]
    corr = hdul[0].header["CRPIX1"]

    wave = (np.arange(1, total + 1) - corr) * step + start
    flux = hdul[0].data
    if has_err:
        err = hdul[0].data
    else:
        err = flux * 0.1

    return gspec(wave=wave * AA, flux=flux * adm, err=err * adm, name=name)


def parse_fire(hdul, name):
    """
    Parses information from a given HDU, for data produced at FIRE.

    Parameters:
        hdul: Header Data Unit List representing the FITS file.
        name (str): The name of the spectrum.

    Returns:
        gspec: An instance of the generic_spectrum class.
    """
    data = hdul[5].data

    wave = data.field("WAVE")
    flux = data.field("FLUX")
    err = data.field("SIG")

    return gspec(wave=wave * AA, flux=flux * adm, err=err * adm, name=name)


def parse_lrs(hdul, name):
    """
    Parses information from a given HDU, for data produced at TNG LRS.

    Parameters:
        hdul: Header Data Unit List representing the FITS file.
        name (str): The name of the spectrum.

    Returns:
        gspec: An instance of the generic_spectrum class.
    """
    start = hdul[0].header["CRVAL1"]
    step = hdul[0].header["CDELT1"]
    total = hdul[0].header["NAXIS1"]
    corr = hdul[0].header["CRPIX1"]

    wave = (np.arange(1, total + 1) - corr) * step + start
    r_wav = np.argwhere((wave >= 3700) & (wave <= 8000))

    wave = wave[r_wav][:, 0]
    flux = hdul[0].data[r_wav][:, 0]
    err = flux * 0.1

    return gspec(wave=wave * AA, flux=flux * adm, err=err * adm, name=name)


def parse_generic(hdul, name):
    """
    Parses information from a generic HDU. Will fail most of the time,
    for every fail I will try to improve the function. This handles
    calibrated Gaia spectra at the minimum.

    Parameters:
        hdul: Header Data Unit List representing the FITS file.
        name (str): The name of the spectrum.

    Returns:
        gspec: An instance of the generic_spectrum class.
    """
    wave = hdul[1].data["wave"]
    flux = hdul[1].data["flux"]
    err = hdul[1].data["err"]

    return gspec(wave=wave * AA, flux=flux * adm, err=err * adm, name=name)


def parse_sdss(hdul, name):
    """
    Parses information from SDSS spectra.

    Parameters:
        hdul: Header Data Unit List representing the FITS file.
        name (str): The name of the spectrum.

    Returns:
        gspec: An instance of the generic_spectrum class.
    """
    data = np.array([np.array(i) for i in hdul[1].data])

    flux = data[:, 0]
    wave = 10 ** data[:, 1]
    err = np.vectorize(_revIVar)(data[:, 2], max(flux))

    return gspec(wave=wave * AA, flux=flux * adm, err=err * adm, name=name)


def parse_lamost(hdul, name):
    """
    Parses information from LAMOST spectra.

    Parameters:
        hdul: Header Data Unit List representing the FITS file.
        name (str): The name of the spectrum.

    Returns:
        gspec: An instance of the generic_spectrum class.
    """
    data = np.array([np.array(i) for i in hdul[0].data])

    flux = data[0, :]
    wave = data[2, :]
    err = np.vectorize(_revIVar)(data[1, :], max(flux)).reshape(1, -1)

    return gspec(wave=wave * AA, flux=flux * adm, err=err * adm, name=name)


def parse_astrocook_fits(hdul, name):
    """
    Parses information from spectra produced by Astrocook in fits format.

    Parameters:
        hdul: Header Data Unit List representing the FITS file.
        name (str): The name of the spectrum.

    Returns:
        gspec: An instance of the generic_spectrum class.
    """
    data = hdul[1].data

    wave = data.x * 10  # Marz wants A, not nm
    flux = data.y
    error = flux * 0.01

    nanwave = np.isfinite(wave)
    flux[np.isnan(flux)] = np.nanmedian(flux)
    error[np.isnan(error)] = np.inf

    return gspec(
        wave=wave[nanwave] * AA,
        flux=flux[nanwave] * adm,
        err=error[nanwave] * adm,
        name=name,
    )


def alt_parse_6df(hdul, name):
    """
    Parses information from spectra downloaded by 6dfGS.

    Parameters:
        hdul: Header Data Unit List representing the FITS file.
        name (str): The name of the spectrum.

    Returns:
        gspec: An instance of the generic_spectrum class.
    """
    data = hdul[7].data

    wave = data[3]
    flux = data[0]
    err = flux * 0.1
    return gspec(wave=wave * AA, flux=flux * adm, err=err * adm, name=name)


def parse_2df_6df(hdul, name):
    """
    Parses information from spectra downloaded by 6dfGS.

    Parameters:
        hdul: Header Data Unit List representing the FITS file.
        name (str): The name of the spectrum.

    Returns:
        gspec: An instance of the generic_spectrum class.
    """
    try:
        hdul[7].data
        return alt_parse_6df(hdul, name)
    except IndexError:
        start = hdul[0].header["CRVAL1"]
        step = hdul[0].header["CDELT1"]
        total = hdul[0].header["NAXIS1"]
        corr = hdul[0].header["CRPIX1"]

        # Transform flux, should not matter for redshift identification but might
        # as well consider it
        BZERO = hdul[0].header["BZERO"]
        BSCALE = hdul[0].header["BSCALE"]

        wave = (np.arange(1, total + 1) - corr) * step + start
        flux = BSCALE * hdul[0].data + BZERO

        err = hdul[2].data

        return gspec(wave=wave * AA, flux=flux * adm, err=err * adm, name=name)
