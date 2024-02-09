"""
Utilities related to spectra manipulation.
We try to keep things clean and only leave the class and method definition in spec.py
"""

import tarfile
import warnings
from copy import deepcopy as dc
from glob import glob
from tempfile import TemporaryDirectory

import numpy as np
from astropy import units as au
from astropy.io import fits

from .. import spec as gspec
from .. import utilities as utils


def combine(specs, cliphi=None, cliplo=None, verbose=False):
    # I will need to adapt Astrocook function here!
    """Combine spectra pixel by pixel, weighting by the inverse variance
    of each pixel.  Clip high sigma values by sigma times clip values
    Returns the combined spectrum.

    If the wavelength scales of the input spectra differ, combine()
    will rebin the spectra to a common linear (not log-linear)
    wavelength scale, with pixel width equal to the largest pixel
    width in the input spectra. If this is not what you want, rebin
    the spectra by hand with rebin() before using combine().
    """

    nspectra = len(specs)
    if verbose:
        print("{} spectra to combine".format(nspectra))
    if nspectra < 2:
        raise Exception("Need at least 2 spectra to combine.")

    if cliphi is not None and nspectra < 3:
        cliphi = None
    if cliplo is not None and nspectra < 3:
        cliplo = None

    # Check if wavescales are the same:
    sp_0 = specs[0]
    x = sp_0.x
    npts = len(x)
    needrebin = True
    for sp in specs:
        if len(sp.x) != npts:
            if verbose:
                print("Rebin required")
            break
        if (np.abs(sp.x - x) / x[0]).max() > 1e-8:
            if verbose:
                print((np.abs(sp.x - x) / x[0]).max(), "Rebin required")
            break
    else:
        needrebin = False
        if verbose:
            print("No rebin required")

    # interpolate over 1 sigma error arrays

    if needrebin:
        wstart = min([sp.x[0] for sp in specs])
        wend = max([sp.x[-1] for sp in specs])
        # Assume spectra have constant bin in velocity space
        # Convert all spectra to velocity space
        if verbose:
            print("Finding new bin size")
        bin_size = max([utils.get_binsize(s.convert_x(utils.kms)) for s in specs])
        # Choose largest wavelength bin size of old spectra.
        if verbose:
            print("Rebinning spectra")
        s_rebinned = [
            s.rebin(dv=bin_size, xstart=wstart, xend=wend, in_place=False)
            for s in specs
        ]
    else:
        s_rebinned = dc(specs)

    # sigma clipping, if requested
    if cliphi is not None or cliplo is not None:
        clip(cliphi, cliplo, s_rebinned)

    # Common axis:
    common_x = dc(s_rebinned[0].x)
    common_y = np.zeros(common_x.shape)
    common_dy = np.zeros(common_x.shape)

    # Co-addition
    for _, i in utils.enum_tqdm(
        range(len(common_x)), len(common_x), "Co-adding spectra..."
    ):
        wtot = fltot = ertot = 0.0
        npix = 0  # num of old spectrum pixels contributing to new
        for s in s_rebinned:
            # if not a sensible flux value, skip to the next pixel
            if s.dy[i] > 0:
                npix += 1
                # Weighted mean (weight by inverse variance)
                variance = s.dy[i] ** 2
                w = 1.0 / variance
                fltot += s.y[i] * w
                ertot += (s.dy[i] * w) ** 2
                wtot += w
        if npix > 0:
            common_y[i] = fltot / wtot
            common_dy[i] = np.sqrt(ertot) / wtot
        else:
            common_y[i] = np.nan
            common_dy[i] = np.nan

    new_name = specs[0].name + "_CoAdd" if specs[0].name is not None else "CoAdd"
    return gspec.generic_spectrum(
        wave=common_x,
        flux=common_y * specs[0].y.unit,
        err=common_dy * specs[0].dy.unit,
        z_em=specs[0].z_em,
        name=new_name,
    )


def clip(cliphi, cliplo, *specs_in, in_place=True):
    # clip the rebinned input spectra
    if in_place:
        specs = specs_in
    else:
        specs = dc(specs_in)
    # find pixels where we can clip: where we have at least three
    # good contributing values.
    goodpix = np.zeros(len(specs[0].x))
    for s in specs:
        goodpix += (s.dy > 0).astype(int)
    canclip = goodpix > 2
    # find median values
    medfl = np.median([s.y[canclip] for s in specs], axis=0)
    nclipped = 0
    for i, s in enumerate(specs):
        fl = s.y[canclip]
        er = s.dy[canclip]
        diff = (fl - medfl) / er
        if cliphi is not None:
            badpix = diff > cliphi
            s.dy[canclip][badpix] = np.nan
            nclipped += len(badpix.nonzero()[0])
        if cliplo is not None:
            badpix = diff < -cliplo
            s.dy[canclip][badpix] = np.nan
            nclipped += len(badpix.nonzero()[0])
    return nclipped


def unpack_acs(path):
    with TemporaryDirectory() as tmpdirname:
        if tarfile.is_tarfile(path):
            with tarfile.open(path, "r") as tar:
                # extract file to temporary dir
                tar.extractall(tmpdirname)

            # get whatever ends with _spec.fits, and try to read it
            spec_fits = glob(tmpdirname + "/*_spec.fits")
            # check that we get a single spectrum
            if len(spec_fits) > 1:
                warnings.warn("Too many spectra found, skipping.")
                return None
            else:
                return fits.getdata(spec_fits[0])
        else:
            warnings.warn("Invalid archive! Skipping.")
            return None


def load_acs(path, name=None, norm=False):
    # loads an astrocook session directly
    # unpack in temporary directory
    data = unpack_acs(path)

    wave = data["x"] * au.nm
    flux = data["y"] * utils.adm
    err = data["dy"] * utils.adm
    if norm:
        cont = data["cont"] * utils.adm
        return gspec.generic_spectrum(
            wave=wave, flux=flux / cont, err=err / cont, name=name
        )
    else:
        return gspec.generic_spectrum(wave=wave, flux=flux, err=err, name=name)


@au.quantity_input(vstart=utils.kms, vend=utils.kms, dv=utils.kms)
def flux_ccf(sp_1, sp_2, vstart, vend, dv, weighted=False):
    # tested, and checks out with astrocook -> I did not make any mistake while
    #  makind adjustements to this
    # Gotta convert to nm, otherwise we have issues (I'll figure out the proper
    #  way to compute this in velocity space in a second moment, for now we good)
    if not sp_1._xunit.is_equivalent(au.nm):
        sp_1.convert_x(au.nm)
    if not sp_2._xunit.is_equivalent(au.nm):
        sp_2.convert_x(au.nm)

    if sp_1.xmin is None:
        sp_1.gen_xmin_xmax()
    if sp_2.xmin is None:
        sp_2.gen_xmin_xmax()

    vstart = vstart.to(au.km / au.s).value
    vend = vend.to(au.km / au.s).value
    dv = dv.to(au.km / au.s).value
    spec_x = sp_1.x.value[:]

    xmin = spec_x[~np.isnan(spec_x)][0]
    xmax = spec_x[~np.isnan(spec_x)][-1]
    dv_orig = (sp_1.xmax - sp_1.xmin).value / spec_x * utils.c_kms.value
    xmean = 0.5 * (xmin + xmax)
    v_shift = np.arange(vstart, vend + dv, dv)

    x_shift = xmean * v_shift / utils.c_kms.value
    xstart = xmean * vstart / utils.c_kms.value
    xend = xmean * vend / utils.c_kms.value
    dx = xmean * dv / utils.c_kms.value
    scale = int(np.rint(np.nanmedian(dv_orig) / dv))

    x_osampl = np.arange(xmin + xstart, xmax + xend, dx)
    y1_osampl = np.interp(x_osampl, spec_x, sp_1.y.value)

    y2_osampl = np.interp(x_osampl, spec_x, sp_2.y.value)
    dy1_osampl = np.interp(x_osampl, spec_x, sp_1.dy.value)

    pan_l, pan_r = (
        int(abs(len(x_shift) * xstart / np.abs(xend - xstart))),
        int(abs(len(x_shift) * xend / np.abs(xend - xstart))),
    )
    ccf = []
    chi2 = []
    chi2r = []
    for i, xs in enumerate(x_shift):
        y1 = y1_osampl[pan_l : -pan_r - 1]
        y2 = y2_osampl[i : -pan_l - pan_r + i - 1]

        dy = dy1_osampl[pan_l : -pan_r - 1]

        y1 = y1[::scale]
        y2 = y2[::scale]
        dy = dy[::scale]

        y1m = y1 - np.nanmedian(y1)
        y2m = y2 - np.nanmedian(y2)

        ccf.append(
            np.nanmean(y2m * y1m) / np.sqrt(np.nanmean(y2m**2) * np.nanmean(y1m**2))
        )
        chi2i = (y1 - y2) ** 2 / dy**2

        if weighted:
            bf = np.abs(np.gradient(y2))
            bf = bf * len(chi2i) / np.sum(bf)
            chi2i = chi2i * bf

        chi2i_sum = np.nansum(chi2i)
        chi2.append(chi2i_sum)
        chi2r.append(chi2i_sum / len(y1))

    return np.array(v_shift), np.array(ccf), np.array(chi2), np.array(chi2r)
