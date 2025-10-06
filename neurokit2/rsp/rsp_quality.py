# - * - coding: utf-8 - * -

from .rsp_peaks import rsp_peaks
from ..signal.signal_templatequality import signal_templatequality


def rsp_quality(rsp_cleaned, peaks=None, sampling_rate=1000, method="templatematch"):
    """**RSP Signal Quality Assessment**

    Assess the quality of the RSP Signal. using template matching:

    * The ``"templatematch"`` method (loosely based on Charlton et al., 2021) computes a continuous
      index of quality of the RSP signal, by calculating the correlation coefficient between each
      individual breath and an average (template) breath shape. This index is therefore
      relative: 1 corresponds to breaths that are closest to the average breath shape (i.e.
      correlate exactly with it) and 0 corresponds to there being no correlation with the average
      breath shape. Note that for pre-processing, Charlton et al. cleaned the signal with a low-pass filter
      below 1 Hz (the "charlton2021" method in rsp_clean), and used the "bettermann1996" breath detection
      algorithm (the "bettermann1996" method in rsp_peaks).

    Parameters
    ----------
    rsp_cleaned : Union[list, np.array, pd.Series]
        The cleaned RSP signal in the form of a vector of values.
    peaks : tuple or list
        The list of RSP peak samples returned by ``rsp_peaks()``. If None, peaks is computed from
        the signal input.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    method : str
        The method for computing RSP signal quality. The only option is ``"templatematch"`` (default).

    Returns
    -------
    quality : array
        Vector containing the quality index ranging from 0 to 1.

    See Also
    --------
    signal_templatequality

    References
    ----------
    * Charlton, P.H, et al. (2021). An impedance pneumography signal quality index: Design, assessment
      and application to respiratory rate monitoring. Biomedical Signal Processing and Control, 65, 102339.

    Examples
    --------
    * **Example 1:** 'templatematch' method

    .. ipython:: python

      import neurokit2 as nk

      sampling_rate=50
      rsp = nk.rsp_simulate(duration=30, sampling_rate=sampling_rate, method="breathmetrics")
      rsp_cleaned = nk.rsp_clean(rsp, sampling_rate=sampling_rate, method="charlton")
      quality = nk.rsp_quality(rsp_cleaned, sampling_rate=sampling_rate, method="templatematch")

      @savefig p_rsp_quality.png scale=100%
      nk.signal_plot([rsp_cleaned, quality], standardize=True)
      @suppress
      plt.close()

    """

    method = method.lower()  # remove capitalised letters

    # Detect RSP peaks (if not done already)
    if peaks is None:
        _, peaks = rsp_peaks(rsp_cleaned, sampling_rate=sampling_rate, method="bettermann1996")
        peaks = peaks["RSP_Peaks"]

    # Sanitise method name
    if method.lower() in ["templatematch", "charlton2021", "charlton"]:
        method = "templatematch"
    else:
        raise ValueError(
            f"Method '{method}' not recognised. Please use 'templatematch'."
        )

    # Run
    quality = signal_templatequality(
        rsp_cleaned,
        beat_inds=peaks,
        signal_type="rsp",
        sampling_rate=sampling_rate,
        method=method,
    )

    return quality
