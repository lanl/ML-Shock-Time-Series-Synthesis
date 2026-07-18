import numpy as np
import scipy as sp
import scipy.interpolate


def interpft(x, ny):  # Python Version of MATLABs interpft function
    sz = np.size(x)
    m = sz
    n = 1

    if ny > m:
        incr = 1
    else:
        raise Exception("Haven't coded this section yet")

    a = np.fft.fft(np.squeeze(x))  # Collapse to Array
    nyqst = np.int64(np.ceil((m + 1) / 2))

    part1 = a[0:nyqst]
    part2 = np.zeros([ny - m])
    part3_idx = np.arange(nyqst + 0, m)
    part3 = a[part3_idx]

    b = np.concatenate([part1, part2, part3])

    if np.remainder(m, 2) == 0:
        b[nyqst - 1] = b[nyqst - 1] / 2
        b[nyqst + ny - m - 1] = b[nyqst - 1]

    y = np.fft.ifft(b)

    if np.all(np.isreal(x)):
        y = np.real(y)

    y = y * ny / m

    y_idx = np.arange(0, ny)
    y = y[y_idx]  # Skip over extra points when oldny <= m.

    return y


def SRS(
        t,
        data,
        freqLines=None,
        n_elements: int = 100,
        spectrumType="maximax",
        modelOutput="absolute acceleration",
        SRSDamping=0.03,
        sampleRate = 32768, 
        resampleInput=True,
        t_units="s",
        data_units="g's",
):
    # Input Units of Data are G and the outputs units of the SRS are also in G
    # Internally converts to m/s^2

    assert t_units == "s"  # Time in Seconds
    assert data_units == "g's"  # Acceleration in Gs

    # convert from units of g's to m/s^2
    G = 9.80665  # m/s2
    measuredSignal = data * G  # Convert from
    outputUnits = "m/s^2"

    # Perform Calculations

    # Set default frequency vector if not predefined, 100 uniform log
    # spaced frequency values covering the range of [0.1%,25%] of SOURCES sample rate
    dt = 1 / sampleRate

    if freqLines is None:
        freqLines = np.logspace(np.log10(sampleRate / 1e4), np.log10(sampleRate / 4), n_elements)
    else:  # Not None and argument was passed in
        assert type(freqLines) is np.ndarray
        assert np.size(freqLines) > 1

    freqUnits = "cyc / s"

    if resampleInput:
        sF = 8  # increase sample rate by this factor
        dt = dt / sF
        m = interpft(measuredSignal, len(measuredSignal) * sF)  # fft resample
        sR = sampleRate * sF  # define resampled sample rate
    else:
        m = measuredSignal
        sR = sampleRate

    # May Need to Re-write with If statement
    if spectrumType == "maximax":

        def spectrum(k, p, r):
            return np.max(np.abs(k))

        # spectrum = lambda k, p, r: np.max(np.abs(k))
    else:
        raise Exception("Haven't coded this section yet")

    # zero pad the resampled signal with one full cycle of the lowest frequency
    # SDOF.
    zp1 = np.transpose(m)
    zp2 = np.zeros(np.int64(np.ceil(sR / (np.min(freqLines) * np.sqrt(1 - SRSDamping ** 2)))))
    zpad = np.concatenate([zp1, zp2])

    pri = len(m)  # define the length of the primary response

    s = np.zeros(len(freqLines))  # preallocate srs data vector

    #  SRS filtering. Coefficients are ISO Standard r
    Q = 1 / (2 * SRSDamping)

    # zeropad with one full cycle of the current SDOF to capture
    # residual response. Note alternate approaches to analytically caclcualte residual
    # response have been proposed in literature and could be included in future
    # versions.

    for j in range(1, len(freqLines) + 1, 1):

        idx = j - 1

        res = np.int64(
            np.ceil(sR / (freqLines[idx] * np.sqrt(1 - SRSDamping ** 2)))
        )  # define the length of the residual response

        mPR = zpad[0: pri + (res)]  # include zeropadding for residual. One cycle of current SDOF

        # set up convenience variables
        omega = 2 * np.pi * freqLines[idx]
        A = omega * dt / (2 * Q)
        B = omega * dt * np.sqrt(1 - 1 / (4 * Q ** 2))

        # filter denominator
        a = np.zeros(3)

        a[1 - 1] = 1
        a[2 - 1] = -2 * np.exp(-A) * np.cos(B)
        a[3 - 1] = np.exp(-2 * A)

        # filter numerator
        b = np.zeros(3)

        if modelOutput == "absolute acceleration":  # absolute acceleration
            b[1 - 1] = 1 - np.exp(-A) * np.sin(B) / B
            b[2 - 1] = 2 * np.exp(-A) * (np.sin(B) / B - np.cos(B))
            b[3 - 1] = np.exp(-2 * A) - np.exp(-A) * np.sin(B) / B

            rS = sp.signal.lfilter(b, a, mPR)

        s[idx] = spectrum(rS, pri, res)  # Units of m/s^2

    SRS_data_G = s / G  # Units of G

    return freqLines, SRS_data_G



