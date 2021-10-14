# Kepler Workflow

Workflow to extract light curves of background target from Kepler's TPFs using
[`psfmachine`](https://github.com/SSDataLab/psfmachine).

This workflow produces PSF and Aperture photometry of +120,000 sources. Light curves are
saved as FITS Light Curve files that can be read and analyzed using
[`lightkurve`](https://github.com/lightkurve/lightkurve) library.

# TODO

* Fix directories (need to check NAS)
* automatize file batch creation
