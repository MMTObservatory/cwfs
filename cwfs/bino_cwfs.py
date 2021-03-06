#!/usr/bin/env python
##
# @package mmtwfs
# @file cwfs.py
# @brief main script to run cwfs
##
# @authors: Bo Xin & Chuck Claver (LSST); modified by T. E. Pickering (MMTO)

import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path

import astropy.units as u
from astropy.io import fits

from mmtwfs.zernike import ZernikeVector
from mmtwfs.wfs import WFSFactory

from cwfs.instrument import Instrument
from cwfs.algorithm import Algorithm
from cwfs.image import Image, readFile
from cwfs.tools import outParam, outZer4Up

import logging

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

log = logging.getLogger('CWFS')
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


def main():

    parser = argparse.ArgumentParser(
        description='----LLST cwfs (Curvature Wavefront Sensing) code modified for MMTO Binospec use----'
    )
    parser.add_argument(
        'images',
        metavar="<FITS image>",
        nargs=2,
        help='Intra and extra focal image file names (no path). If focus not known, first image is assumed to be intra-focal.'
    )
    parser.add_argument(
        '-imdir',
        metavar='<image dir>',
        dest='imgDir',
        default='.',
        help='Relative or absolute path for input images. Default "./".'
    )
    parser.add_argument(
        '-focoff',
        metavar='<M2 focus offset>',
        dest='focoff',
        type=float,
        default=1000.0,
        help='Magnitude of focus offset applied to M2 in microns. Default 1000.0 um.'
    )
    parser.add_argument(
        '-ixy',
        metavar=('<intra x>', '<intra y>'),
        dest='intra_xy',
        nargs=2,
        type=float,
        default=[0, 0],
        help='Intra focal field (x,y) in deg. Default [0 0].'
    )
    parser.add_argument(
        '-exy',
        metavar=('<extra x>', '<extra y>'),
        dest='extra_xy',
        nargs=2,
        type=float,
        default=[0, 0],
        help='Extra focal field (x,y) in deg. Default [0 0].'
    )
    parser.add_argument(
        '-i',
        '--instrument',
        metavar='<instrument>',
        dest='instruFile',
        default='mmto',
        help='Instrument parameter file. Default "mmto". ".param" is appended automatically.'
    )
    parser.add_argument(
        '-a',
        '--algorithm',
        metavar='<algorithm>',
        dest='algoFile',
        default='exp',
        help='Algorithm parameter file. Default "exp". ".algo" is appended automatically.'
    )
    parser.add_argument(
        '-m',
        '--model',
        metavar='<model>',
        dest='model',
        choices=('paraxial', 'onAxis', 'offAxis'),
        default='onAxis',
        help='Optical model to be used. Default "onAxis".'
    )
    parser.add_argument(
        '-o',
        '--output',
        metavar='<output file root>',
        dest='output',
        default='',
        help='Root of filenames to use for outputs. Default "cwfs".'
    )
    parser.add_argument(
        '-m1',
        '--sendm1',
        dest='sendm1',
        action='store_true',
        help='Send force corrections to the primary (M1)'
    )
    parser.add_argument(
        '-m2',
        '--sendm2',
        dest='sendm2',
        action='store_true',
        help='Send force corrections to the secondary (M2)'
    )
    parser.add_argument(
        '-thresh',
        '--threshold',
        metavar='<threshold>',
        dest='thresh',
        type=float,
        default=150.0,
        help='Amplitude threshold to determine which modes to use in corrective force calculations. Default 150 nm.'
    )
    parser.add_argument(
        '--m1gain',
        metavar='<M1 gain>',
        dest='m1gain',
        type=float,
        default=1.0,
        help='Gain factor for calculating M1 corrections. Default 1.0.'
    )
    parser.add_argument(
        '--m2gain',
        metavar='<M2 gain>',
        dest='m2gain',
        type=float,
        default=1.0,
        help='Gain factor for calculating M2 corrections. Default 1.0.'
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version='%(prog)s 1.0'
    )
    parser.add_argument(
        '-d',
        '--debug',
        metavar='<debug level>',
        dest='debugLevel',
        type=int,
        default=0,
        choices=(-1, 0, 1, 2, 3),
        help='debug level, -1=quiet, 0=Zernikes, 1=operator, 2=expert, 3=everything, default=0'
    )
    args = parser.parse_args()

    if args.debugLevel >= 1:
        log.info(args)

    # instatiate Binospec wfs object for calculating corrections from zernike solutions and sending them on
    wfs = WFSFactory(wfs="binospec", plot=False)
    if args.sendm1 or args.sendm2:
        wfs.connect()

    # configure gains for calculating corrections
    wfs.m1gain = args.m1gain
    wfs.m2gain = args.m2gain

    # get rotator and focus values from the headers, if available
    rots = []
    focusvals = []
    images = args.images
    for image in images:
        if args.imgDir:
            image = os.path.join(args.imgDir, image)
        h = fits.open(image)
        hdr = h[-1].header
        data = h[-1].data
        if len(h) > 1 or data.shape == (516, 532):
            log.info(f"Found raw Binospec SOG image. Trimming and rewriting {image} as single extension.")
            data = data[259-128:259+128, 295-128:295+128]
            data = np.flipud(data)

        f = fits.PrimaryHDU(data, hdr)
        f.writeto(image, overwrite=True)
        if 'ROT' in hdr:
            rots.append(hdr['ROT'])
        if 'FOCUS' in hdr:
            focusvals.append(hdr['FOCUS'])

    if args.output == '':
        args.output = image

    if len(rots) > 0:
        rot = np.array(rots).mean() * u.deg
        log.info(f"Using rotator angle of {rot.round(2)}.")
    else:
        log.warning("WARNING: No rotator information in headers. Assuming rotator angle of 0.0.")
        rot = 0.0 * u.deg

    if len(focusvals) == 2:
        focusvals = np.array(focusvals)
        intraFile = Path(images[np.argmin(focusvals)])
        extraFile = Path(images[np.argmax(focusvals)])
        focoff = focusvals.max() - focusvals.mean()
    else:
        focoff = args.focoff
        intraFile = Path(images[0])
        extraFile = Path(images[1])
        log.warning(f"WARNING: No focus information in image headers. Assuming M2 focus offset of {focoff} um.")

    log.info(f"\t Intra-focal image: {intraFile.name}")
    log.info(f"\t Extra-focal image: {extraFile.name}")

    # The pupil rotation in the single-object guider on binospec was determined to be 0 deg.
    rotation = 0 * u.deg - rot

    log.info(f"Total pupil rotation: {rotation.round(2)}")

    I1 = Image(readFile(str(intraFile)), args.intra_xy, Image.INTRA)
    I2 = Image(readFile(str(extraFile)), args.extra_xy, Image.EXTRA)

    # load instrument and algorithm parameters
    inst = Instrument(args.instruFile, I1.sizeinPix)

    # this is a MMTO hack. 0.0 doesn't work, but this will yield an annular zernike solution that is very close to circular.
    # the MMTO wfs code currently doesn't support annular zernikes for calculating corrections.
    inst.obscuration = 0.01

    # convert M2 focus offset in microns to meters of focus shift at the instrument focal plane
    inst.offset = focoff * 1.0e-6 * 18.8

    # set up fitting algorithm
    algo = Algorithm(args.algoFile, inst, args.debugLevel)

    # run it
    algo.runIt(inst, I1, I2, args.model)

    # output parameters
    outParam(args.output + ".param", algo, inst, I1, I2, args.model)

    # save wavefront image
    plt.imshow(algo.Wconverge, origin='lower')
    plt.savefig(args.output + ".pdf")
    # convert the LSST zernike array to a ZernikeVector that the mmtwfs code
    # can handle, then apply necessary derotation to it.
    zv = ZernikeVector()
    zv.from_array(algo.zer4UpNm, modestart=4, normalized=True)
    zv.denormalize()
    zv.rotate(angle=-rotation)
    if args.debugLevel >= 0:
        log.info("\n" + repr(zv))

    # output Zernikes 4 and up
    outZer4Up(algo.zer4UpNm, 'nm', args.output + ".raw.lsst.zernikes")
    zv.save(filename=args.output + ".rot.zernikes")
    zv.fringe_bar_chart().savefig(args.output + ".zernikes.pdf")

    # calculate the corrections
    focus = wfs.calculate_focus(zv)
    cc_x, cc_y = wfs.calculate_cc(zv)
    forces, m1focus, zv_masked = wfs.calculate_primary(zv, threshold=args.thresh * u.nm)
    log.info("Focus: {0:0.1f}  M1 Focus: {3:0.1f}  CC_X: {1:0.1f}  CC_Y: {2:0.1f}".format(
            focus,
            cc_x,
            cc_y,
            m1focus
        )
    )
    if args.debugLevel >= 1:
        log.info(forces)

    # send corrections to secondary, if applicable
    if args.sendm2:
        log.info("Sending {0:0.1f} focus to secondary...".format(focus))
        wfs.secondary.focus(focus)
        log.info("Sending {0:0.1f}/{1:0.1f} CC_X/CC_Y to secondary...".format(cc_x, cc_y))
        wfs.secondary.correct_coma(cc_x, cc_y)

    # send corrections to primary, if applicable
    if args.sendm1:
        log.info("Sending forces to cell and {0:0.1f} focus to secondary...".format(m1focus))
        wfs.telescope.correct_primary(forces, m1focus, filename=args.output + ".forces")
        time.sleep(10)


if __name__ == "__main__":
    main()
