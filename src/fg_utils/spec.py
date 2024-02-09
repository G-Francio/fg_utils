import bisect
import warnings
from copy import deepcopy as dc

import astropy.units as au
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as aconst
from astropy.stats import sigma_clip
from astropy.table import Table

# from scipy.ndimage import gaussian_filter1d
from . import utilities as utils


class generic_spectrum:
    """
    Basic class to mimic Astrocook spectrum structure.
    Attributes are similar, but not really
    """

    @au.quantity_input
    def __init__(self, **kwargs):
        self.x = kwargs.get("wave", None)
        self.y = kwargs.get("flux", None)
        self.dy = kwargs.get("err", None)
        self.name = kwargs.get("name", None)

        self.z_em = kwargs.get("z_em", 0)
        # ^ this should possibly and more properly
        #  be set to NaN. However, for picklying this is much better...
        self.x_em = (self.z_em + 1) * kwargs.get("line", 121.567 * au.nm)

        self.xmin = kwargs.get("xmin", None)
        self.xmax = kwargs.get("xmax", None)

        if self.x is not None:
            self._xunit = self.x.unit
        else:
            self._xunit = kwargs.get("xunit", None)

        if self.y is not None:
            self._yunit = self.y.unit
        else:
            self._yunit = kwargs.get("yunit", utils.adm)

        self._old_xunit = None
        self._old_yunit = None
        self.set_nm2kms()

    def gen_xmin_xmax(self):
        if self.x is None:
            raise NameError("Generate a x grid!")
        elif self.xmin is not None and self.xmax is not None:
            return 0
        else:
            self.xmin, self.xmax = utils.get_xmin_xmax(self.x)
            return 0

    def set_nm2kms(self):
        self._nm2kms = [
            (
                au.nm,
                au.km / au.s,
                lambda x: np.log(x / self.x_em.value) * aconst.c.to(au.km / au.s),
                lambda x: np.exp(x / aconst.c.to(au.km / au.s).value) * self.x_em.value,
            )
        ]

    def convert_x(self, to, equiv=None):
        if equiv is None:
            equiv = self._nm2kms

        self.gen_xmin_xmax()

        self._old_xunit = self.x.unit

        self.x = self.x.to(to, equivalencies=equiv)
        self.xmin = self.xmin.to(to, equivalencies=equiv)
        self.xmax = self.xmax.to(to, equivalencies=equiv)
        self._xunit = to
        return self.x

    def convert_y(self, to):
        self._old_yunit = self.y.unit
        self.y = self.y.to(to)
        self.dy = self.dy.to(to)

    @au.quantity_input
    def region_extract(self, xmin, xmax, xunit=None, in_place=True):
        assert xmax.value > xmin.value
        if xunit is None:
            xunit = self._xunit

        self.convert_x(xunit)
        inds = np.where((self.x > xmin) & (self.x < xmax))

        new_wave = self.x[inds]
        new_flux = self.y[inds]
        new_err = self.dy[inds]

        if in_place:
            self.x = new_wave
            self.y = new_flux
            self.dy = new_err
            self.xmin, self.xmax = utils.get_xmin_xmax(new_wave)
        else:
            return generic_spectrum(
                wave=new_wave, flux=new_flux, err=new_err, z_em=self.z_em
            )

    @au.quantity_input
    def rebin(
        self, dv, xstart=None, xend=None, filling=np.nan, in_place=True, equiv=None
    ):
        if equiv is None:
            equiv = self._nm2kms

        # Wave array HAS to be sorted, otherwise we have issues (not only on the rebinning...)
        assert all(self.x == np.sort(self.x))
        self.gen_xmin_xmax()
        if dv.unit != utils.kms:
            warnings.warn("dv units not km/s, converting...")
            dv.to(utils.kms, equivalencies=self._nm2kms)

        if self.x.unit != utils.kms:
            warnings.warn("Spec x units not km/s, converting...")
            self.convert_x(utils.kms)

        # Always rebin in velocity space - we check this just above
        if not (xstart is None or xend is None):
            xstart = xstart.to(utils.kms, equivalencies=equiv)
            xend = xend.to(utils.kms, equivalencies=equiv)
        else:
            # Create x, xmin, and xmax for rebinning
            if xstart is None:
                xstart = np.nanmin(self.x)
            if xend is None:
                xend = np.nanmax(self.x)

        x_r = np.arange(xstart.value, xend.value, dv.value) * self._xunit
        xmin_r, xmax_r = utils.get_xmin_xmax(x_r)

        # Compute y and dy combining contributions
        im = 0
        iM = 1

        y_r = np.array([]) * self.y.unit
        dy_r = np.array([]) * self.y.unit

        printval = self.name if self.name is not None else "Spectrum"
        for _, (m, M) in utils.enum_tqdm(
            zip(xmin_r.value, xmax_r.value), len(x_r), printval + ": Rebinning"
        ):
            im = bisect.bisect_left(np.array(self.xmax.value), m)
            iM = bisect.bisect_right(np.array(self.xmin.value), M)

            ysel = self.y[im:iM]
            dysel = self.dy[im:iM]
            frac = (
                np.minimum(M, self.xmax.value[im:iM])
                - np.maximum(m, self.xmin.value[im:iM])
            ) / dv.value

            nw = np.where(~np.isnan(ysel))
            ysel = ysel[nw]
            dysel = dysel[nw]
            frac = frac[nw]

            w = np.where(frac > 0)

            if len(frac[w]) > 0:
                weights = (frac[w] / dysel[w] ** 2).value
                # and False:
                if (
                    np.any(np.isnan(dysel))
                    or np.any(dysel == 0.0)
                    or np.sum(weights) == 0.0
                ):
                    y_r = np.append(y_r, np.average(ysel[w], weights=frac[w]))
                else:
                    y_r = np.append(y_r, np.average(ysel[w], weights=weights))
                dy_r = np.append(
                    dy_r,
                    np.sqrt(np.nansum(weights**2 * dysel[w].value ** 2))
                    / np.nansum(weights)
                    * self.y.unit,
                )
            else:
                y_r = np.append(y_r, filling)
                dy_r = np.append(dy_r, filling)

        if in_place:
            self.x = x_r
            self.xmin, self.xmax = xmin_r, xmax_r
            self.convert_x(self._old_xunit)
            self.y = y_r
            self.dy = dy_r
            return 0
        else:
            return generic_spectrum(wave=x_r, flux=y_r, err=dy_r)

    @au.quantity_input
    def sigma_clip(self, window_length=250 * utils.kms, in_place=True, **kwargs):
        # Get indexes for each sublist
        def get_indexes(arr, window_in_pix):
            for i in range(len(arr) - window_in_pix + 1):
                yield i, i + window_in_pix

        # Work in velocity space, it's easier
        self.convert_x(utils.kms, equiv=self._nm2kms)
        binsize = np.mean(self.xmax - self.xmin)
        window_in_pix = int(np.round(window_length / binsize).value)

        # Mask for the clipping
        mask = np.full(self.y.size, False)
        # Start from beginning, iterate over till the end
        for start, end in get_indexes(self.y, window_in_pix):
            mask[start:end] = sigma_clip(self.y[start:end], **kwargs).mask

        if in_place:
            self.y[np.where(mask)] = np.nan
            self.dy[np.where(mask)] = np.nan
        else:
            outspec = dc(self)
            outspec.y[np.where(mask)] = np.nan
            outspec.dy[np.where(mask)] = np.nan
            return outspec

    def plot(self, **kwargs):
        ax = kwargs.pop("ax", None)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 12 / 1.61))
            utils.plot_spec(ax, self, **kwargs)
        else:
            utils.plot_spec(ax, self, **kwargs)
        return 0

    def save(self, path, format="fits", overwrite=False):
        t = Table(
            [self.x.value, self.y.value, self.dy.value], names=["wave", "flux", "err"]
        )
        t.write(path, format=format, overwrite=overwrite)
