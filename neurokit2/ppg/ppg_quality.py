# - * - coding: utf-8 - * -

from .ppg_peaks import ppg_peaks
from ..signal.signal_quality import signal_quality


def ppg_quality(ppg_cleaned, peaks=None, sampling_rate=1000, method="templatematch"):
    """**PPG Signal Quality Assessment**

    Assess the quality of the PPG Signal using various methods:

    * The ``"templatematch"`` method (loosely based on Orphanidou et al., 2015) computes a continuous
      index of quality of the PPG signal, by calculating the correlation coefficient between each
      individual pulse wave and an average (template) pulse wave shape. This index is therefore
      relative: 1 corresponds to pulse waves that are closest to the average pulse wave shape (i.e.
      correlate exactly with it) and 0 corresponds to there being no correlation with the average
      pulse wave shape.

    * The ``"dissimilarity"`` method (loosely based on Sabeti et al., 2019) computes a continuous index
      of quality of the PPG signal, by calculating the level of dissimilarity between each individual
      pulse wave and an average (template) pulse wave shape (after they are normalised). A value of
      zero indicates no dissimilarity (i.e. equivalent pulse wave shapes), whereas values above or below
      indicate increasing dissimilarity. The original method used dynamic time-warping to align the pulse
      waves prior to calculating the level of dsimilarity, whereas this implementation does not currently
      include this step.

    * The ``"ho2025"` method (Ho et al., 2025) assesses PPG quality on a beat-by-beat basis by predicting
      whether each interbeat-interval (IBI) is accurate. To do so, beats are detected using a primary beat detector,
      and each IBI is predicted to be accurate only if a secondary beat detector detects beats
      in the same positions (within a tolerance). In this implementation, all signal samples within an
      IBI are rated as high quality (1) if that IBI is predicted to be accurate, or low
      quality (0) if that IBI is predicted to be inaccurate. Ho et al. proposed this approach for the ECG, and here
      it has been applied to the PPG. The general approach was derived by Ho et al from the previously proposed bSQI
      approach.

    Parameters
    ----------
    ppg_cleaned : Union[list, np.array, pd.Series]
        The cleaned PPG signal in the form of a vector of values.
    peaks : tuple or list
        The list of PPG pulse wave peak samples returned by ``ppg_peaks()``. If None, peaks is computed from
        the signal input.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    method : str
        The method for computing PPG signal quality, can be ``"templatematch"`` (default), ``"dissimilarity"``,
        or ``"ho2025"``.

    Returns
    -------
    quality : array
        Vector containing the quality index ranging from 0 to 1 for ``"templatematch"`` method,
        or an unbounded value (where 0 indicates high quality) for ``"dissimilarity"`` method,
        or zeros and ones (where 1 indicates high quality) for ``"ho2025"`` method.

    See Also
    --------
    signal_quality, ppg_clean

    References
    ----------
    * Orphanidou, C. et al. (2015). "Signal-quality indices for the electrocardiogram and photoplethysmogram:
      derivation and applications to wireless monitoring". IEEE Journal of Biomedical and Health Informatics, 19(3), 832-8.
    * Sabeti E. et al. (2019). Signal quality measure for pulsatile physiological signals using morphological features:
      Applications in reliability measure for pulse oximetry. Informatics in Medicine Unlocked, 16, 100222.
    * Ho, S.Y.S et al. (2025). "Accurate RR-interval extraction from single-lead, telehealth electrocardiogram signals.
      medRxiv, 2025.03.10.25323655. https://doi.org/10.1101/2025.03.10.25323655

    Examples
    --------
    * **Example 1:** 'templatematch' method

    .. ipython:: python

      import neurokit2 as nk

      sampling_rate = 100
      ppg = nk.ppg_simulate(duration=30, sampling_rate=sampling_rate, heart_rate=80)
      ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=sampling_rate)
      quality = nk.ppg_quality(ppg_cleaned, sampling_rate=sampling_rate, method="templatematch")

      @savefig p_ppg_quality.png scale=100%
      nk.signal_plot([ppg_cleaned, quality], standardize=True)
      @suppress
      plt.close()

    """

    method = method.lower()  # remove capitalised letters

    # Detect PPG peaks (if not done already)
    if peaks is None:
        _, peaks = ppg_peaks(ppg_cleaned, sampling_rate=sampling_rate)
        peaks = peaks["PPG_Peaks"]

    # Sanitise method name
    if method in ["templatematch", "orphanidou2015"]:
        method = "templatematch"
    elif method in ["dissimilarity", "sabeti2019"]:
        method = "dissimilarity"
    elif method in ["ho2025", "ho", "ibi", "ici"]:
        method = "ici"
    else:
        raise ValueError(f"Method '{method}' not recognised. Please use 'templatematch', 'dissimilarity', or 'ho2025'.")

    # Run 'templatematch' and 'dissimilarity' methods
    if method in ["templatematch", "dissimilarity"]:
        quality = signal_quality(
            ppg_cleaned,
            cycle_inds=peaks,
            signal_type="ppg",
            sampling_rate=sampling_rate,
            method=method,
        )
    elif method == "ici":
        # Assess quality using Ho2025 method (IBI accuracy prediction)
        quality = signal_quality(
            ppg_cleaned,
            signal_type="ppg",
            primary_detector="charlton",
            secondary_detector="elgendi",
            sampling_rate=sampling_rate,
            method="ici",
        )

    return quality
