# - * - coding: utf-8 - * -
import numpy as np

def signal_ibiquality(signal, signal_type, primary_detector=None, secondary_detector=None, sampling_rate=1000):
    """**Assess quality of cardiac / cardiovascular signal by comparing beat detections from two beat detectors**

    Assess the quality of a cardiac (e.g. ECG) / cardiovascular (e.g. PPG) signal. You can pass an unfiltered
    signal as an input, but typically a filtered signal (e.g. cleaned using ``ecg_clean()`` or ``ppg_clean()``)
    will result in more reliable results.

    To do so, the algorithm (proposed in Ho et al. 2025) assesses signal quality on a beat-by-beat basis by predicting
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
    signal_type : str
        The signal type (e.g. "ppg" or "ecg").
    primary_detector : str
        The name of the primary beat detector (e.g. the defaults are ``"unsw"`` for the ECG, and ``"charlton"``
        for the PPG).
    secondary_detector : str
        The name of the secondary beat detector (e.g. the defaults are ``"neurokit"`` for the ECG, and ``"elgendi"``
        for the PPG).
    sampling_rate : int
        The sampling frequency of ``signal`` (in Hz, i.e., samples/second). Defaults to 1000.

    Returns
    -------
    quality : array
        Vector containing the quality index values: 1 for high-quality samples, 0 for low-quality samples.

    See Also
    --------
    ecg_quality

    References
    ----------
    * Ho, S.Y.S et al. (2025). "Accurate RR-interval extraction from single-lead, telehealth electrocardiogram signals.
      medRxiv, 2025.03.10.25323655. https://doi.org/10.1101/2025.03.10.25323655
    
    Examples
    --------
    * **Example 1:** assessing PPG signal quality

    .. ipython:: python

      import neurokit2 as nk
      
      sampling_rate = 100
      ppg = nk.ppg_simulate(
          duration=30, sampling_rate=sampling_rate, heart_rate=70, motion_amplitude=1, burst_number=10, random_state=12
      )
      quality = nk.ppg_quality(ppg, sampling_rate=sampling_rate, method="ho2025")
      nk.signal_plot([ppg, quality], standardize=True)  # a period is deemed to be low quality from 1200-1500 samples
      plt.close()
    
    """

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
