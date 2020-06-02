import os

import numpy as np

from ..instrument import Instrument
from ..algorithm import Algorithm
from ..image import Image, readFile
from ..tools import getDataDir


def test_matlab_validation():
    """
    Test calculations against matlab predictions
    """
    rootdir = getDataDir()

    myinst = 'lsst'
    validationDir = os.path.join(str(rootdir), 'validation')

    tests = [
        ('testImages/F1.23_1mm_v61', 'z7_0.25_%s.txt', (0, 0),          ('fft',),       'paraxial'),
        ('testImages/LSST_C_SN26',   'z7_0.25_%s.txt', (0, 0),          ('fft', 'exp'), 'onAxis'),
        ('testImages/LSST_NE_SN25',  'z11_0.25_%s.txt', (1.185, 1.185), ('fft', 'exp'), 'offAxis'),
    ]
    # filenames with matlab results and tolerance on absolute discrepancy (in nm)
    #
    # N.b. these tolerances are set at 10nm because centering algorithm has changed.
    #      difference in the wavefront on the ~10nm is well below noise level.
    #
    matlabZFile_Tol = [
        ('F1.23_1mm_v61_z7_0.25_fft.txt', 10),
        ('LSST_C_SN26_z7_0.25_fft.txt',   10),
        ('LSST_C_SN26_z7_0.25_exp.txt',   10),
        ('LSST_NE_SN25_z11_0.25_fft.txt', 10),
        ('LSST_NE_SN25_z11_0.25_exp.txt', 10)
    ]
    #
    # Check that we have the right number of matlab files.  Not really a unit test, just consistency
    #
    nTest = 0
    for inDir, filenameFmt, fldxy, algorithms, model in tests:
        nTest += len(algorithms)
    assert(nTest == len(matlabZFile_Tol))

    j = 0                           # counter for matlab outputs, matlabZFile_Tol
    for imgDir, filenameFmt, fldxy, algorithms, model in tests:
        imgDir = os.path.join(str(rootdir), imgDir)
        intraFile = os.path.join(imgDir, filenameFmt % "intra")
        I1 = Image(readFile(intraFile), fldxy, Image.INTRA)

        extraFile = os.path.join(imgDir, filenameFmt % "extra")
        I2 = Image(readFile(extraFile), fldxy, Image.EXTRA)

        inst = Instrument(myinst, I1.sizeinPix)

        for algorithm in algorithms:
            matlabZFile, tol = matlabZFile_Tol[j]
            j += 1

            algo = Algorithm(algorithm, inst, 1)
            algo.runIt(inst, I1, I2, model)

            zer = algo.zer4UpNm
            matZ = np.loadtxt(os.path.join(validationDir, matlabZFile))

            aerr = np.abs(matZ - zer)

            assert(np.max(aerr) < tol)
