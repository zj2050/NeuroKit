# - * - coding: utf-8 - * -
import numpy as np

from ..epochs import epochs_to_df
from ..signal import signal_interpolate, signal_cyclesegment


def signal_quality(
    signal, sampling_rate=1000, cycle_inds=None, signal_type=None, method="templatematch", primary_detector=None, secondary_detector=None
):
    """**Assess quality of signal by comparing individual cycle morphologies with a template**

    Assess the quality of a quasi-periodic signal (e.g. PPG, ECG or RSP) using the specified method. You can pass an
    unfiltered signal as an input, but typically a filtered signal (e.g. cleaned using ``ppg_clean()``, ``ecg_clean()`` or
    ``rsp_clean()``) will result in more reliable results. The following methods are available:

    * The ``"templatematch"`` method (loosely based on Orphanidou et al., 2015) computes a continuous
      index of quality of the PPG or ECG signal, by calculating the correlation coefficient between each
      individual beat's morphology and an average (template) beat morphology. This index is therefore
      relative: 1 corresponds to a signal where each individual beat's morphology is closest to the average beat morphology
      (i.e. correlate exactly with it) and 0 corresponds to there being no correlation with the average beat morphology.

    * The ``"disimilarity"`` method (loosely based on Sabeti et al., 2019) computes a continuous index
      of quality of the PPG or ECG signal, by calculating the level of disimilarity between each individual
      beat's morphology and an average (template) beat morpholoy (after they are normalised). A value of
      zero indicates no disimilarity (i.e. equivalent beat morphologies), whereas values above or below
      indicate increasing disimilarity. The original method used dynamic time-warping to align the pulse
      waves prior to calculating the level of dsimilarity, whereas this implementation does not currently
      include this step.

    * The ``"ibi"`` method (based on Ho et al., 2025) assesses signal quality on a beat-by-beat basis by predicting
      whether each interbeat-interval (IBI) is accurate. To do so, beats are detected using a primary detector,
      and each IBI is predicted to be accurate only if a secondary detector detects beats
      in the same positions (within a tolerance). In this implementation, all signal samples within an
      IBI are rated as high quality (1) if that IBI is predicted to be accurate, or low
      quality (0) if that IBI is predicted to be inaccurate. This approach was derived from the previously
      proposed bSQI approach.


    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The cleaned signal, such as that returned by ``ppg_clean()`` or ``ecg_clean()``.
    sampling_rate : int
        The sampling frequency of ``signal`` (in Hz, i.e., samples/second). Defaults to 1000.
    cycle_inds : tuple or list
        The list of beat or breath samples (e.g. PPG or ECG peaks returned by ``ppg_peaks()`` or ``ecg_peaks()``, or RSP peaks
        returned by ``rsp_peaks()``).
    signal_type : str
        The signal type (e.g. 'ppg', 'ecg', or 'rsp').
    method : str
        The processing pipeline to apply. Can be one of ``"disimilarity"``, ``"templatematch"``. The default is
        ``"templatematch"``.
    primary_detector : str
        The name of the primary cycle (i.e. beat or breath) detector (e.g. the defaults are ``"unsw"`` for the ECG, and
        ``"charlton"`` for the PPG).
    secondary_detector : str
        The name of the secondary cycle detector (e.g. the defaults are ``"neurokit"`` for the ECG, and ``"elgendi"``
        for the PPG).
    **kwargs
        Additional keyword arguments, usually specific for each method.

    Returns
    -------
    quality : array
        Vector containing the quality index ranging from 0 to 1 for ``"templatematch"`` method,
        or an unbounded value (where 0 indicates high quality) for ``"disimilarity"`` method.

    See Also
    --------
    ppg_quality, ecg_quality, rsp_quality

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
    * **Example 1:** Using IBI method to assess PPG signal quality

    .. ipython:: python

      import neurokit2 as nk
      
      sampling_rate = 100
      ppg = nk.ppg_simulate(
          duration=30, sampling_rate=sampling_rate, heart_rate=70, motion_amplitude=1, burst_number=10, random_state=12
      )
      quality = nk.ppg_quality(ppg, sampling_rate=sampling_rate, method="ibi")
      nk.signal_plot([ppg, quality], standardize=True)
      plt.close()
    
    * **Example 2:** Using template-matching method to assess ECG signal quality

    .. ipython:: python

      import neurokit2 as nk
      
      sampling_rate = 100
      duration = 20
      ecg = nk.ecg_simulate(
          duration=duration, sampling_rate=sampling_rate, heart_rate=70, noise=0.5
      )
      quality = nk.ecg_quality(ecg, sampling_rate=sampling_rate, method="templatematch")
      nk.signal_plot([ecg, quality], standardize=True)
      plt.close()
    
    * **Example 2:** Using template-matching method to assess RSP signal quality

    .. ipython:: python

      import neurokit2 as nk
      
      sampling_rate = 50
      duration = 30
      rsp = nk.rsp_simulate(duration=30, sampling_rate=sampling_rate, method="breathmetrics")
      rsp_cleaned = nk.rsp_clean(rsp, sampling_rate=sampling_rate, method="charlton")
      quality = nk.rsp_quality(rsp_cleaned, sampling_rate=sampling_rate, method="templatematch")
      nk.signal_plot([rsp_cleaned, quality], standardize=True)
      plt.close()
    
    """

    # Standardize inputs
    signal_type = signal_type.lower()  # remove capitalised letters
    method = method.lower()  # remove capitalised letters
    
    # Run selected quality assessment method
    if method in ["templatematch"]:  # Based on the approach in Orphanidou et al. (2015) and Charlton et al. (2021)
        quality = _quality_templatematch(
            signal, cycle_inds=cycle_inds, signal_type=signal_type, sampling_rate=sampling_rate,
        )
    elif method in ["disimilarity"]:  # Based on the approach in Sabeti et al. (2019)
        quality = _quality_disimilarity(
            signal, cycle_inds=cycle_inds, signal_type=signal_type, sampling_rate=sampling_rate
        )
    elif method in ["ibi", "ho2025"]:  # Based on the approach in Ho et al. (2025)
        if signal_type not in ["ecg", "ppg"]:
            raise Exception("IBI quality assessment is only compatible with cardiovascular signals")
        quality = _quality_ibi(
            signal, signal_type=signal_type, primary_detector=primary_detector, secondary_detector=secondary_detector,
            sampling_rate=sampling_rate
        )

    return quality


# =============================================================================
# Calculate template morphology
# =============================================================================
def _calc_template_morph(signal, cycle_inds, signal_type, sampling_rate=1000):
    
    # Segment to get individual beat morphologies
    cycles, average_cycle_rate = signal_cyclesegment(signal, cycle_inds, sampling_rate=sampling_rate)

    # convert these to dataframe
    ind_morph = epochs_to_df(cycles).pivot(
        index="Label", columns="Time", values="Signal"
    )
    ind_morph.index = ind_morph.index.astype(int)
    ind_morph = ind_morph.sort_index()

    # Filter Nans
    valid_beats_mask = ~ind_morph.isnull().any(axis=1)
    ind_morph = ind_morph[valid_beats_mask]
    cycle_inds = np.array(cycle_inds)[valid_beats_mask.values]

    # Find template pulse wave as the average pulse wave shape
    templ_pw = ind_morph.mean()

    return templ_pw, ind_morph, cycle_inds


# =============================================================================
# Quality assessment using template-matching method
# =============================================================================
def _quality_templatematch(
    signal, cycle_inds=None, signal_type="ppg", sampling_rate=1000
):
    
    # Obtain individual beat morphologies and template beat morphology
    templ_morph, ind_morph, cycle_inds = _calc_template_morph(
        signal,
        cycle_inds=cycle_inds,
        signal_type=signal_type,
        sampling_rate=sampling_rate,
    )

    # Find correlation coefficients (CCs) between individual beat morphologies and the template
    cc = np.zeros(len(cycle_inds) - 1)
    for beat_no in range(0, len(cycle_inds) - 1):
        temp = np.corrcoef(ind_morph.iloc[beat_no], templ_morph)
        cc[beat_no] = temp[0, 1]

    # Interpolate beat-by-beat CCs
    quality = signal_interpolate(
        cycle_inds[0:-1], cc, x_new=np.arange(len(signal)), method="previous"
    )

    return quality


# =============================================================================
# Disimilarity measure method
# =============================================================================
def _norm_sum_one(pw):

    # ensure all values are positive
    pw = pw - pw.min() + 1

    # normalise pulse wave to sum to one
    pw = pw / np.sum(pw)

    return pw


def _calc_dis(pw1, pw2):
    # following the methodology in https://doi.org/10.1016/j.imu.2019.100222 (Sec. 3.1.2.5)

    # convert to numpy arrays
    pw1 = np.array(pw1)
    pw2 = np.array(pw2)

    # normalise to sum to one
    pw1 = _norm_sum_one(pw1)
    pw2 = _norm_sum_one(pw2)

    # ignore any elements which are zero because log(0) is -inf
    rel_els = (pw1 != 0) & (pw2 != 0)

    # calculate disimilarity measure (using pw2 as the template)
    dis = np.sum(pw2[rel_els] * np.log(pw2[rel_els] / pw1[rel_els]))

    return dis


# =============================================================================
# Quality assessment using disimilarity method
# =============================================================================
def _quality_disimilarity(
    signal, cycle_inds=None, signal_type="ppg", sampling_rate=1000
):

    # Obtain individual beat morphologies and template beat morphology
    templ_morph, ind_morph, cycle_inds = _calc_template_morph(
        signal,
        cycle_inds=cycle_inds,
        signal_type=signal_type,
        sampling_rate=sampling_rate,
    )

    # Find individual disimilarity measures
    dis = np.zeros(len(cycle_inds) - 1)
    for beat_no in range(0, len(cycle_inds) - 1):
        dis[beat_no] = _calc_dis(ind_morph.iloc[beat_no], templ_morph)

    # Interpolate beat-by-beat dis's
    quality = signal_interpolate(
        cycle_inds[0:-1], dis, x_new=np.arange(len(signal)), method="previous"
    )

    return quality


# =============================================================================
# Quality assessment using IBI method
# =============================================================================
def _quality_ibi(
            signal, signal_type, primary_detector, secondary_detector, sampling_rate
        ):
    
    # Specify default beat detectors
    if primary_detector is None:
        if signal_type == "ecg":
            primary_detector = "unsw"
        elif signal_type == "ppg":
            primary_detector = "charlton"
    if secondary_detector is None:
        if signal_type == "ecg":
            secondary_detector = "neurokit"
        elif signal_type == "ppg":
            secondary_detector = "elgendi"

    # Sanitize inputs
    signal_type = signal_type.lower()  # remove capitalised letters
    primary_detector = primary_detector.lower()  # remove capitalised letters
    secondary_detector = secondary_detector.lower()  # remove capitalised letters
    signal = np.asarray(signal)

    # check that signal_type is either "ecg" or "ppg"
    if signal_type not in ["ecg", "ppg"]:
        raise ValueError("`signal_type` must be 'ecg' or 'ppg'.")

    # Specify constants
    tolerance_ms = 150 # tolerance window size, in milliseconds
    tolerance_samps = int((tolerance_ms / 1000) * sampling_rate)

    # Detect beats using each beat detector in turn
    beats_primary = _signal_beats(signal, signal_type=signal_type, beat_detector=primary_detector, sampling_rate=sampling_rate)
    beats_secondary = _signal_beats(signal, signal_type=signal_type, beat_detector=secondary_detector, sampling_rate=sampling_rate)
    
    # Filter closely spaced beats to keep only the highest amplitude beat
    beats_primary = _filter_close_beats(beats_primary, signal, tolerance_samps)
    beats_secondary = _filter_close_beats(beats_secondary, signal, tolerance_samps)
    
    # Build quality array
    quality = np.zeros(len(signal), dtype=int)
    for i in range(len(beats_primary) - 1):
        start = beats_primary[i]
        end = beats_primary[i + 1]

        match_start = any(abs(start - s) <= tolerance_samps for s in beats_secondary)
        match_end = any(abs(end - s) <= tolerance_samps for s in beats_secondary)

        if match_start and match_end:
            quality[start:end] = 1

    return quality


def _filter_close_beats(beats, signal, tolerance_samps):

    if len(beats) == 0:
        return []
    
    filtered = [beats[0]]
    for i in range(1, len(beats)):
        if beats[i] - filtered[-1] > tolerance_samps:
            filtered.append(beats[i])
        else:
            # Keep the higher amplitude peak
            if signal[beats[i]] > signal[filtered[-1]]:
                filtered[-1] = beats[i]
    
    return filtered


def _signal_beats(signal, signal_type, beat_detector, sampling_rate):

    # Import peak-detection functions (placed here to avoid circular imports)
    from ..ppg import ppg_peaks
    from ..ecg import ecg_peaks

    if signal_type=="ecg":
        
        # Detect beats in ECG signal
        signals, info = ecg_peaks(
            signal,
            sampling_rate=sampling_rate,
            method=beat_detector,
        )

        # Extract beats
        beats = info["ECG_R_Peaks"]

    elif signal_type=="ppg":
        
        # Detect beats in PPG signal
        signals, info = ppg_peaks(
            signal,
            sampling_rate=sampling_rate,
            method=beat_detector,
        )

        # Extract beats
        beats = info["PPG_Peaks"]

    return beats