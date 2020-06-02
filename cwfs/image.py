#!/usr/bin/env python
##
# @package cwfs
# @file image.py
##
# @authors: Bo Xin & Chuck Claver
# @       Large Synoptic Survey Telescope


import sys
import os

import numpy as np
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
from astropy.io import fits
from skimage import filters
from scipy.ndimage import center_of_mass
from scipy.signal import correlate

from . import tools
from lsst.cwfs.tools import ZernikeAnnularGrad, ZernikeGrad, \
    ZernikeAnnularJacobian, ZernikeJacobian


def readFile(filename):
    image = None
    if isinstance(filename, str):
        if (filename.endswith(".txt")):
            image = np.loadtxt(filename)
            # this assumes this txt file is in the format
            # I[0,0]   I[0,1]
            # I[1,0]   I[1,1]
            image = image[::-1, :]
        elif (filename.endswith(".fits")):
            IHDU = fits.open(filename)
            image = IHDU[0].data
            IHDU.close()

    if image is None:
        raise IOError("Unrecognised file type for %s" % filename)

    return image


class Image(object):
    INTRA = "intra"
    EXTRA = "extra"

    def __init__(self, image, fieldXY, type):
        """!Create a cwfs Image

        @param image   A numpy 2-d array
        @param fieldXY The position in the focal plane (degrees)
        @param type    Type of image (Image.INTRA, Image.EXTRA)
        """
        self.image = image

        self.fieldX, self.fieldY = fieldXY

        if type.lower() not in (Image.INTRA, Image.EXTRA):
            raise TypeError("Image must be intra or extra")
        self.type = type

        # we will need self.fldr to be on denominator
        self.fldr = np.max((np.hypot(self.fieldX, self.fieldY), 1e-8))

        if self.image.shape[0] != self.image.shape[1]:
            print('original %s image size = (%d, %d)' % (
                type, self.image.shape[0], self.image.shape[1]))
            d = int(abs(self.image.shape[0] -  self.image.shape[1])/2)
            if self.image.shape[0]<self.image.shape[1]:
                self.image = self.image[:,d:self.image.shape[0]+d]
            else:
                self.image = self.image[d:self.image.shape[1]+d,:]
            print('we cut it to (%d, %d)' % (
                self.image.shape[0], self.image.shape[1]))
        elif self.image.shape[0] % 2 == 1:
            print('%s image size = (%d, %d)' % (
                type, self.image.shape[0], self.image.shape[1]))
            print('Error: number of pixels cannot be odd numbers')
            sys.exit()

        self.sizeinPix = self.image.shape[0]

    # if we pass inst.maskParam, a try: catch: is needed in cwfs.py
    def makeMaskList(self, inst, model):
        if (model == 'paraxial' or model == 'onAxis'):
            if inst.obscuration == 0:
                self.masklist = np.array([[0, 0, 1, 1]])
            else:
                self.masklist = np.array([[0, 0, 1, 1],
                                          [0, 0, inst.obscuration, 0]])
        else:
            self.maskCa, self.maskRa, self.maskCb, self.maskRb = \
                interpMaskParam(self.fieldX, self.fieldY, inst.maskParam)
            cax, cay, cbx, cby = rotateMaskParam(  # only change the center
                self.maskCa, self.maskCb, self.fieldX, self.fieldY)
            self.masklist = np.array(
                [[0, 0, 1, 1], [0, 0, inst.obscuration, 0],
                 [cax, cay, self.maskRa, 1], [cbx, cby, self.maskRb, 0]])

    def makeMask(self, inst, boundaryT, maskScalingFactor):

        self.pMask = np.ones(inst.sensorSamples, dtype=int)
        self.cMask = self.pMask

        rMask = inst.apertureDiameter / (2 * inst.focalLength / inst.offset)\
            * maskScalingFactor

        for ii in range(self.masklist.shape[0]):

            r = np.sqrt((inst.xSensor - self.masklist[ii, 0])**2 +
                        (inst.ySensor - self.masklist[ii, 1])**2)

            # Initialize both mask elements to the opposite of the pass/block
            # boolean
            pMaskii = (1 - self.masklist[ii, 3]) * np.ones(
                (inst.sensorSamples, inst.sensorSamples), dtype=int)
            cMaskii = (1 - self.masklist[ii, 3]) * np.ones(
                (inst.sensorSamples, inst.sensorSamples), dtype=int)

            # Find the indices that correspond to the mask element, set them to
            # the pass/block boolean
            idx = r <= self.masklist[ii, 2]
            if (self.masklist[ii, 3] >= 1):
                # make a mask >r so that we can keep a larger area of S
                aidx = np.nonzero(r <= self.masklist[ii, 2] *
                                  (1 + boundaryT * inst.pixelSize / rMask))
            else:
                aidx = np.nonzero(r <= self.masklist[ii, 2] *
                                  (1 - boundaryT * inst.pixelSize / rMask))
            pMaskii[idx] = self.masklist[ii, 3]
            cMaskii[aidx] = self.masklist[ii, 3]

            # Multiplicatively add the current mask elements to the model masks
            # padded mask - for use at the offset planes
            self.pMask = self.pMask * pMaskii
            # non-padded mask corresponding to aperture
            self.cMask = self.cMask * cMaskii

    def getOffAxisCorr(self, instDir, order):
        self.offAxis_coeff = np.zeros((4, int((order + 1) * (order + 2) / 2)))
        self.offAxis_coeff[0, :], self.offAxisOffset = getOffAxisCorr_single(
            os.path.join(instDir, 'offAxis_cxin_poly%d.txt' % (order)),
            self.fldr)
        self.offAxis_coeff[1, :], _ = getOffAxisCorr_single(
            os.path.join(instDir, 'offAxis_cyin_poly%d.txt' % (order)),
            self.fldr)
        self.offAxis_coeff[2, :], _ = getOffAxisCorr_single(
            os.path.join(instDir, 'offAxis_cxex_poly%d.txt' % (order)),
            self.fldr)
        self.offAxis_coeff[3, :], _ = getOffAxisCorr_single(
            os.path.join(instDir, 'offAxis_cyex_poly%d.txt' % (order)),
            self.fldr)

    def upResolution(self, oversample, lm, ln):

        # lm and ln are dimensions after upResolution
        sm = lm / oversample
        sn = ln / oversample

        newI = np.zeros((lm, ln))
        for i in range(sm):
            for j in range(sn):
                for k in range(oversample):
                    for l in range(oversample):
                        newI[i * oversample + k, j * oversample + l] = \
                            self.image[i, j] / oversample / oversample
        self.image = newI

    def downResolution(self, oversample, sm, sn):

        # sm and sn are dimensions after downResolution

        newI = np.zeros((sm, sn))
        for i in range(sm):
            for j in range(sn):
                subI = self.image[i * oversample:(i + 1) * oversample,
                                  j * oversample:(j + 1) * oversample]
                idx = (~np.isnan(subI)).nonzero()
                if np.sum(np.sum(subI)) > 0:
                    newI[i, j] = np.sum(subI[idx])
                    newI[i, j] = newI[i, j] / np.sum(np.sum(idx))
                else:
                    newI[i, j] = np.nan

        self.image = newI

    def imageCoCenter(self, inst, algo):

        x1, y1 = getCenter(self.image)
        if algo.debugLevel >= 3:
            print('imageCoCenter: (x1,y1)=(%8.2f,%8.2f)\n' % (x1, y1))

        stampCenterx1 = inst.sensorSamples / 2. + 0.5
        stampCentery1 = inst.sensorSamples / 2. + 0.5
        radialShift = 3.5 * algo.upReso * \
            (inst.offset / 1e-3) * (10e-6 / inst.pixelSize)

        radialShift = radialShift * self.fldr / 1.75
        if (self.fldr > 1.75):
            radialShift = 0

        I1c = self.fieldX / self.fldr
        I1s = self.fieldY / self.fldr

        stampCenterx1 = stampCenterx1 + radialShift * I1c
        stampCentery1 = stampCentery1 + radialShift * I1s

        self.image = np.roll(self.image, int(
            np.round(stampCentery1 - y1)), axis=0)
        self.image = np.roll(self.image, int(
            np.round(stampCenterx1 - x1)), axis=1)

    def compensate(self, inst, algo, zcCol, oversample, model):

        if ((zcCol.ndim == 1) and (len(zcCol) != algo.numTerms)):
            raise Exception(
                'input:size', 'zcCol in compensate needs to be a %d \
                row column vector\n' % algo.numTerms)

        sm, sn = self.image.shape

        projSamples = sm * oversample

        # Let us create a look-up table for x -> xp first.
        luty, lutx = np.mgrid[
            -(projSamples / 2 - 0.5):(projSamples / 2 + 0.5),
            -(projSamples / 2 - 0.5):(projSamples / 2 + 0.5)]
        lutx = lutx / (projSamples / 2 / inst.sensorFactor)
        luty = luty / (projSamples / 2 / inst.sensorFactor)

        # set up the mapping
        lutxp, lutyp, J = aperture2image(
            self, inst, algo, zcCol, lutx, luty, projSamples, model)
        #    print "J",J.shape

        show_lutxyp = showProjection(
            lutxp, lutyp, inst.sensorFactor, projSamples, 0)
        if (np.all(show_lutxyp <= 0)):
            self.caustic = 1
            return

        show_lutxyp = tools.padArray(show_lutxyp, projSamples + 20)

        struct0 = ndimage.generate_binary_structure(2, 1)
        struct = ndimage.iterate_structure(struct0, 4)
        struct = ndimage.morphology.binary_dilation(struct, structure=struct0)
        struct = ndimage.morphology.binary_dilation(
            struct, structure=struct0).astype(int)
        show_lutxyp = ndimage.morphology.binary_dilation(
            show_lutxyp, structure=struct)
        show_lutxyp = ndimage.morphology.binary_erosion(
            show_lutxyp, structure=struct)
        show_lutxyp = tools.extractArray(show_lutxyp, projSamples)

        self.centerOnProjection(show_lutxyp.astype(float))


        # let's construct the interpolant,
        # to get the intensity on (x',p') plane
        # that corresponds to the grid points on (x,y)
        yp, xp = np.mgrid[-(sm / 2 - 0.5):(sm / 2 + 0.5), -
                          (sm / 2 - 0.5):(sm / 2 + 0.5)]
        xp = xp / (sm / 2 / inst.sensorFactor)
        yp = yp / (sm / 2 / inst.sensorFactor)

        # xp = reshape(xp,sm^2,1);
        # yp = reshape(yp,sm^2,1);
        # self.image = reshape(self.image,sm^2,1);
        #
        # FIp = TriScatteredInterp(xp,yp,self.image,'nearest');
        # lutIp = FIp(lutxp, lutyp);

        lutxp[np.isnan(lutxp)] = 0
        lutyp[np.isnan(lutyp)] = 0

        #    lutIp=interp2(xp,yp,self.image,lutxp,lutyp)
        #    print xp.shape, yp.shape, self.image.shape
        #    print lutxp.ravel()
        #    print xp[:,0],yp[0,:]
        ip = interpolate.RectBivariateSpline(
            yp[:, 0], xp[0, :], self.image, kx=1, ky=1)

        #    ip = interpolate.interp2d(xp, yp, self.image)
        #    ip = interpolate.interp2d(xp, yp, self.image)
        #    print lutxp.shape, lutyp.shape
        #    lutIp = ip(0.5, -0.5)
        #    print lutIp, 'lutIp1'
        #    lutIp = ip([-0.1],[-0.1])
        #    print lutIp, 'lutIp2'
        #    lutIp = ip(np.array(0.5,-0.1), np.array(-0.5, -0.1))
        #    print lutIp, 'lutIp12',lutxp.ravel()[0:10]
        lutIp = np.zeros(lutxp.shape[0] * lutxp.shape[1])
        for i, (xx, yy) in enumerate(zip(lutxp.ravel(), lutyp.ravel())):
            lutIp[i] = ip(yy, xx)
        lutIp = lutIp.reshape(lutxp.shape)

        self.image = lutIp * J

        if (self.type == 'extra'):
            self.image = np.rot90(self.image, k=2)

        # if we want the compensator to drive down tip-tilt
        # self.image = offsetImg(-shiftx, -shifty, self.image);
        # self.image=circshift(self.image,[round(-shifty) round(-shiftx)]);

        self.image[np.isnan(self.image)] = 0
        # self.image < 0 will not be physical, remove that region
        # x(self.image<0) = NaN;
        self.caustic = 0
        if (np.any(self.image < 0) and np.all(self.image0 >= 0)):
            print(
                'WARNING: negative scale parameter, \
            image is within caustic, zcCol (in um)=\n')

        #    for i in range(len(zcCol)):
        #        print zcCol[i]
        #        print('%5.2f '%(zcCol[i]*1.e6))
        #    print('\n');
            self.caustic = 1

        self.image[self.image < 0] = 0
        if (oversample > 1):
            self.downResolution(self, oversample, sm, sn)

    def normalizeI(self, outerR, obsR):
        xmax = self.image.shape[1]
        ymax = self.image.shape[0]
        yfull, xfull = np.mgrid[1:xmax + 1, 1:ymax + 1]

        rfull = np.sqrt((xfull - self.centerx)**2 + (yfull - self.centery)**2)
        idxsig = (rfull < 1.0 * outerR) & (rfull > obsR * outerR)

        self.image = self.image / sum(self.image[idxsig])

    def getSNR(self, outerR, obsR, saturation=1e10):
        xmax = self.image.shape[1]
        ymax = self.image.shape[0]
        yfull, xfull = np.mgrid[1:xmax + 1, 1:ymax + 1]

        rfull = np.sqrt((xfull - self.centerx)**2 + (yfull - self.centery)**2)
        idxsig = (rfull < 1.0 * outerR) & (rfull > obsR * outerR)
        idxbg = (rfull < 2.5 * outerR) & (rfull > 1.5 * outerR)

        self.SNRsig = np.mean(self.image[idxsig])
        self.SNRbg = np.std(self.image[idxbg] - np.mean(self.image[idxbg]))
        self.SNR = self.SNRsig / self.SNRbg
        # if saturated, set SNR to negative
        # if (np.sum(abs(self.image - np.max(self.image))<1e-5)>4):# or
        # np.max(self.image)>40000):
        if np.any(self.image > saturation):
            self.SNR = self.SNR * (-1)
            print('Saturation detected\n' % self.type)

    def centerOnProjection(self, template, window=20):
        length = self.image.shape[0]
        center = length // 2
        corr = correlate(self.image, template, mode='same')
        mask = np.zeros(corr.shape)
        r = window // 2
        mask[center-r:center+r,center-r:center+r] = 1
        idx = np.argmax(corr * mask)
        xmatch = (idx // length)
        ymatch = (idx % length)
        dx = center - xmatch
        dy = center - ymatch

        self.image = np.roll(np.roll(self.image, dx, axis=0), dy, axis=1)

def getCenter(image):
    cut = filters.threshold_otsu(image)
    mask = image > cut
    y, x = center_of_mass(mask)
    return x, y

def linear2D(xydata, c00, c10, c01):
    (x, y) = xydata
    f = c00 + c10 * x + c01 * y
    return f


def getOffAxisCorr_single(confFile, fldr):
    cwfsSrcDir = os.path.split(os.path.abspath(__file__))[0]
    cwfsBaseDir = '%s/../' % cwfsSrcDir
    cdata = np.loadtxt(os.path.join(cwfsBaseDir, confFile))
    c = cdata[:, 1:]
    offset = cdata[0, 0]

    ruler = np.sqrt(c[:, 0]**2 + c[:, 1]**2)
#    print ruler, fldr, (ruler >= fldr).argmax(), (ruler >= fldr).argmin()
    step = ruler[1] - ruler[0]

    p2 = (ruler >= fldr)
#    print "FINE",p2, p2.shape
    if (np.count_nonzero(p2) == 0):  # fldr is too large to be in range
        p2 = c.shape[0] - 1
        p1 = p2
        w1 = 1
        w2 = 0
    elif (p2[0]):  # fldr is too small to be in range
        p2 = 0  # this is going to be used as index
        p1 = 0  # this is going to be used as index
        w1 = 1
        w2 = 0
    else:
        p1 = p2.argmax() - 1
        p2 = p2.argmax()
        w1 = (ruler[p2] - fldr) / step
        w2 = (fldr - ruler[p1]) / step

#    print c[p1,2:], c[p2,2:], w1,p1, w2, p2
    corr_coeff = np.dot(w1, c[p1, 2:]) + np.dot(w2, c[p2, 2:])

    return corr_coeff, offset


def interpMaskParam(fieldX, fieldY, maskParam):
    fldr = np.sqrt(fieldX**2 + fieldY**2)

    cwfsSrcDir = os.path.split(os.path.abspath(__file__))[0]
    cwfsBaseDir = '%s/../' % cwfsSrcDir
    c = np.loadtxt(os.path.join(cwfsBaseDir, maskParam))
    ruler = np.sqrt(2 * c[:, 0]**2)
    step = ruler[1] - ruler[0]

    p2 = (ruler >= fldr)
    if (np.count_nonzero(p2) == 0):  # fldr is too large to be in range
        p2 = c.shape[0]
        p1 = p2
        w1 = 1
        w2 = 0
    elif (p2[0]):  # fldr is too small to be in range
        p2 = 0  # this is going to be used as index
        p1 = 0  # this is going to be used as index
        w1 = 1
        w2 = 0
    else:
        p1 = p2.argmax() - 1
        p2 = p2.argmax()
        w1 = (ruler[p2] - fldr) / step
        w2 = (fldr - ruler[p1]) / step

    param = np.dot(w1, c[p1, 1:]) + np.dot(w2, c[p2, 1:])
    ca = param[0]
    ra = param[1]
    cb = param[2]
    rb = param[3]

    return ca, ra, cb, rb


def rotateMaskParam(ca, cb, fieldX, fieldY):

    # so that fldr is never zero (see next line)
    fldr = np.max((np.sqrt(fieldX**2 + fieldY**2), 1e-8))
    c = fieldX / fldr
    s = fieldY / fldr

    cax = c * ca
    cay = s * ca
    cbx = c * cb
    cby = s * cb

    return cax, cay, cbx, cby

def createPupilGrid(lutx, luty, onepixel, ca, cb, ra, rb, fieldX, fieldY=None):
    """Create the pupil grid"""

    if (fieldY is None):
        fldr = fieldX
        fieldX = fldr / 1.4142
        fieldY = fieldX

    cax, cay, cbx, cby = rotateMaskParam(ca, cb, fieldX, fieldY)

    lutr = np.sqrt((lutx - cax)**2 + (luty - cay)**2)
    tmp = lutr.copy()
    tmp[np.isnan(tmp)] = -999
    idxout = (tmp > ra + onepixel)
    lutx[idxout] = np.nan
    luty[idxout] = np.nan
    idxbound = (tmp <= ra + onepixel) & (tmp > ra) & (~np.isnan(lutx))
    lutx[idxbound] = (lutx[idxbound] - cax) / lutr[idxbound] * ra + cax
    luty[idxbound] = (luty[idxbound] - cay) / lutr[idxbound] * ra + cay

    lutr = np.sqrt((lutx - cbx)**2 + (luty - cby)**2)
    tmp = lutr.copy()
    tmp[np.isnan(tmp)] = 999
    idxout = (tmp < rb - onepixel)
    lutx[idxout] = np.nan
    luty[idxout] = np.nan
    idxbound = (tmp >= rb - onepixel) & (tmp < rb) & (~np.isnan(lutx))
    lutx[idxbound] = (lutx[idxbound] - cbx) / lutr[idxbound] * rb + cbx
    luty[idxbound] = (luty[idxbound] - cby) / lutr[idxbound] * rb + cby

    return lutx, luty


def aperture2image(Im, inst, algo, zcCol, lutx, luty, projSamples, model):

    R = inst.apertureDiameter / 2.0
    if (Im.type == 'intra'):
        myC = - inst.focalLength * \
            (inst.focalLength - inst.offset) / inst.offset / R**2
    elif (Im.type == 'extra'):
        myC = inst.focalLength * (inst.focalLength / inst.offset + 1) / R**2

    polyFunc = tools.getFunction('poly%d_2D' % algo.offAxisPolyOrder)
    polyGradFunc = tools.getFunction('poly%dGrad' % algo.offAxisPolyOrder)

    lutr = np.sqrt(lutx**2 + luty**2)
    # 1 pixel larger than projected pupil. No need to be EF-like, anything
    # outside of this will be masked off by the computational mask
    onepixel = 1 / (projSamples / 2 / inst.sensorFactor)
    # print "LUTR",lutr.shape,((lutr >1 + onepixel) |
    # (lutr<inst.obscuration-onepixel)).shape
    idxout = ((lutr > 1 + onepixel) | (lutr < inst.obscuration - onepixel))
    lutx[idxout] = np.nan
    luty[idxout] = np.nan
    # outer boundary (1 pixel wide boundary)
    idxbound = ((lutr <= 1 + onepixel) & (lutr > 1))
    lutx[idxbound] = lutx[idxbound] / lutr[idxbound]
    luty[idxbound] = luty[idxbound] / lutr[idxbound]
    idxinbd = ((lutr < inst.obscuration) & (
        lutr > inst.obscuration - onepixel))  # inner boundary
    lutx[idxinbd] = lutx[idxinbd] / lutr[idxinbd] * inst.obscuration
    luty[idxinbd] = luty[idxinbd] / lutr[idxinbd] * inst.obscuration

    if (model == 'offAxis'):
        lutx, luty = createPupilGrid(
            lutx, luty, onepixel,
            Im.maskCa, Im.maskCb, Im.maskRa, Im.maskRb,
            Im.fieldX, Im.fieldY)

    if (model == 'paraxial'):
        lutxp = lutx
        lutyp = luty
    elif (model == 'onAxis'):
        myA2 = (inst.focalLength**2 - R**2) / \
            (inst.focalLength**2 - lutr**2 * R**2)
        idx = myA2 < 0
        myA = myA2.copy()
        myA[idx] = np.nan
        myA[~idx] = np.sqrt(myA2[~idx])
        lutxp = algo.maskScalingFactor * myA * lutx
        lutyp = algo.maskScalingFactor * myA * luty
    elif (model == 'offAxis'):
        # nothing is hard-coded here: (1e-3) is because the
        # offAxis correction param files are based on offset=1.0mm
        tt = Im.offAxisOffset
        if (Im.type == 'intra'):
            cx = (Im.offAxis_coeff[0, :] - Im.offAxis_coeff[2, :]) * \
                (tt + inst.offset) / (2 * tt) + Im.offAxis_coeff[2, :]
            cy = (Im.offAxis_coeff[1, :] - Im.offAxis_coeff[3, :]) * \
                (tt + inst.offset) / (2 * tt) + Im.offAxis_coeff[3, :]
        elif (Im.type == 'extra'):
            cx = (Im.offAxis_coeff[0, :] - Im.offAxis_coeff[2, :]) * \
                (tt - inst.offset) / (2 * tt) + Im.offAxis_coeff[2, :]
            cy = (Im.offAxis_coeff[1, :] - Im.offAxis_coeff[3, :]) * \
                (tt - inst.offset) / (2 * tt) + Im.offAxis_coeff[3, :]
            cx = -cx  # this will be inverted back by typesign later on.
            # we do the inversion here to make the (x,y)->(x',y') equations has
            # the same form as the paraxial case.
            cy = -cy

        costheta = (Im.fieldX + Im.fieldY) / Im.fldr / 1.4142
        if (costheta > 1):
            costheta = 1
        elif (costheta < -1):
            costheta = -1

        sintheta = np.sqrt(1 - costheta**2)
        if (Im.fieldY < Im.fieldX):
            sintheta = -sintheta

        # first rotate back to reference orientation
        lutx0 = lutx * costheta + luty * sintheta
        luty0 = -lutx * sintheta + luty * costheta
        # use mapping at reference orientation
        lutxp0 = polyFunc(cx, lutx0, y=luty0)
        lutyp0 = polyFunc(cy, lutx0, y=luty0)
        lutxp = lutxp0 * costheta - lutyp0 * sintheta  # rotate back
        lutyp = lutxp0 * sintheta + lutyp0 * costheta
        # Zemax data are in mm, therefore 1000
        reduced_coordi_factor = 1 / \
            (inst.sensorSamples / 2 * inst.pixelSize / inst.sensorFactor) \
            / 1000
        # reduced coordinates, so that this can be added with the dW/dz
        lutxp = lutxp * reduced_coordi_factor
        lutyp = lutyp * reduced_coordi_factor
    else:
        print('wrong model number in compensate\n')
        return

    if (zcCol.ndim == 1):
        if (algo.zobsR > 0):
            lutxp = lutxp + myC * \
                ZernikeAnnularGrad(zcCol, lutx, luty, algo.zobsR, 'dx')
            lutyp = lutyp + myC * \
                ZernikeAnnularGrad(zcCol, lutx, luty, algo.zobsR, 'dy')
        else:
            lutxp = lutxp + myC * ZernikeGrad(zcCol, lutx, luty, 'dx')
            lutyp = lutyp + myC * ZernikeGrad(zcCol, lutx, luty, 'dy')
    else:
        FX, FY = np.gradient(zcCol,
                             inst.sensorFactor / (inst.sensorSamples / 2))
        lutxp = lutxp + myC * FX
        lutyp = lutyp + myC * FY

    if (Im.type == 'extra'):
        lutxp = - lutxp
        lutyp = - lutyp

    # Below for calculation of the Jacobian

    if (zcCol.ndim == 1):
        if (model == 'paraxial'):
            if (algo.zobsR > 0):
                J = (1 +
                     myC * ZernikeAnnularJacobian(
                         zcCol, lutx, luty, algo.zobsR, '1st') +
                     myC**2 * ZernikeAnnularJacobian(
                         zcCol, lutx, luty, algo.zobsR, '2nd'))
            else:
                J = (1 + myC * ZernikeJacobian(zcCol, lutx, luty, '1st') +
                     myC**2 * ZernikeJacobian(zcCol, lutx, luty, '2nd'))

        elif (model == 'onAxis'):
            xpox = algo.maskScalingFactor * myA * (
                1 +
                lutx**2 * R**2. / (inst.focalLength**2 - R**2 * lutr**2)) + \
                myC * ZernikeAnnularGrad(zcCol, lutx, luty, algo.zobsR, 'dx2')
            ypoy = algo.maskScalingFactor * myA * (
                1 +
                luty**2 * R**2. / (inst.focalLength**2 - R**2 * lutr**2)) + \
                myC * ZernikeAnnularGrad(zcCol, lutx, luty, algo.zobsR, 'dy2')
            xpoy = algo.maskScalingFactor * myA * \
                lutx * luty * R**2 / (inst.focalLength**2 - R**2 * lutr**2) + \
                myC * ZernikeAnnularGrad(zcCol, lutx, luty, algo.zobsR, 'dxy')
            ypox = xpoy

            J = (xpox * ypoy - xpoy * ypox)
        elif (model == 'offAxis'):
            xp0ox = polyGradFunc(cx, lutx0, luty0, 'dx') * costheta - \
                polyGradFunc(cx, lutx0, luty0, 'dy') * sintheta
            yp0ox = polyGradFunc(cy, lutx0, luty0, 'dx') * costheta - \
                polyGradFunc(cy, lutx0, luty0, 'dy') * sintheta
            xp0oy = polyGradFunc(cx, lutx0, luty0, 'dx') * sintheta + \
                polyGradFunc(cx, lutx0, luty0, 'dy') * costheta
            yp0oy = polyGradFunc(cy, lutx0, luty0, 'dx') * sintheta + \
                polyGradFunc(cy, lutx0, luty0, 'dy') * costheta
            xpox = (xp0ox * costheta - yp0ox * sintheta) * \
                reduced_coordi_factor + \
                myC * ZernikeAnnularGrad(zcCol, lutx, luty, algo.zobsR, 'dx2')

            ypoy = (xp0oy * sintheta + yp0oy * costheta) * \
                reduced_coordi_factor + \
                myC * ZernikeAnnularGrad(zcCol, lutx, luty, algo.zobsR, 'dy2')

            temp = myC * \
                ZernikeAnnularGrad(zcCol, lutx, luty, algo.zobsR, 'dxy')
            # if temp==0,xpoy doesn't need to be symmetric about x=y
            xpoy = (xp0oy * costheta - yp0oy * sintheta) * \
                reduced_coordi_factor + temp
            # xpoy-flipud(rot90(ypox))==0 is true
            ypox = (xp0ox * sintheta + yp0ox * costheta) * \
                reduced_coordi_factor + temp
            J = (xpox * ypoy - xpoy * ypox)

    else:

        FXX, FXY = np.gradient(FX,
                               inst.sensorFactor / (inst.sensorSamples / 2))
        tmp, FYY = np.gradient(FY,
                               inst.sensorFactor / (inst.sensorSamples / 2))
        if (model == 'paraxial'):
            xpox = 1 + myC * FXX
            ypoy = 1 + myC * FYY
            xpoy = 1 + myC * FXY
            ypox = xpoy
        elif (model == 'onAxis'):
            xpox = algo.maskScalingFactor * myA * \
                (1 +
                 lutx**2 * R**2. / (inst.focalLength**2 - R**2 * lutr**2)) + \
                myC * FXX
            ypoy = algo.maskScalingFactor * myA * \
                (1 +
                 luty**2 * R**2 / (inst.focalLength**2 - R**2 * lutr**2)) + \
                myC * FYY
            xpoy = algo.maskScalingFactor * myA * lutx * luty * R**2. / \
                (inst.focalLength**2 - R**2 * lutr**2) + myC * FXY
            ypox = xpoy
        elif (model == 'offAxis'):
            xpox = polyGradFunc(cx, lutx, luty, 'dx') * \
                reduced_coordi_factor + myC * FXX
            ypoy = polyGradFunc(cy, lutx, luty, 'dy') * \
                reduced_coordi_factor + myC * FYY
            xpoy = polyGradFunc(cx, lutx, luty, 'dy') * \
                reduced_coordi_factor + myC * FXY
            ypox = polyGradFunc(cy, lutx, luty, 'dx') * \
                reduced_coordi_factor + myC * FXY

        J = (xpox * ypoy - xpoy * ypox)

    return lutxp, lutyp, J


def showProjection(lutxp, lutyp, sensorFactor, projSamples, raytrace):
    n1, n2 = lutxp.shape
    show_lutxyp = np.zeros((n1, n2))
    idx = (~np.isnan(lutxp))
    xR = np.zeros((n1, n2))
    yR = np.zeros((n1, n2))
    # x=0.5 is center of pixel#1
    xR[idx] = np.round((lutxp[idx] + sensorFactor) *
                  (projSamples / sensorFactor) / 2 + 0.5)
    yR[idx] = np.round((lutyp[idx] + sensorFactor) *
                  (projSamples / sensorFactor) / 2 + 0.5)
    mask = np.bitwise_and(
        np.bitwise_and(
            np.bitwise_and(xR > 0,
                           xR < n2),
            yR > 0),
        yR < n1
    )

    if raytrace:
        for i,j in zip(np.array(yR - 1, dtype=int)[mask],np.array(xR - 1, dtype=int)[mask]):
            show_lutxyp[i,j] += 1
    else:
        show_lutxyp[np.array(yR - 1, dtype=int)[mask],
                    np.array(xR - 1, dtype=int)[mask]] = 1
    return show_lutxyp
