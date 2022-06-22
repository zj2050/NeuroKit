# -*- coding: utf-8 -*-
import numpy as np
import scipy.interpolate


def signal_interpolate(
    x_values, y_values, x_new=None, method="quadratic", fill_value=None
):
    """**Interpolate a signal**
    Interpolate a signal using different methods.
    Parameters
    ----------
    x_values : Union[list, np.array, pd.Series]
        The samples corresponding to the values to be interpolated.
    y_values : Union[list, np.array, pd.Series]
        The values to be interpolated.
    x_new : Union[list, np.array, pd.Series] or int
        The samples at which to interpolate the y_values. Samples before the first value in x_values
        or after the last value in x_values will be extrapolated. If an integer is passed, nex_x
        will be considered as the desired length of the interpolated signal between the first and
        the last values of x_values. No extrapolation will be done for values before or after the
        first and the last values of x_values.
    method : str
        Method of interpolation. Can be ``"linear"``, ``"nearest"``, ``"zero"``, ``"slinear"``,
        ``"quadratic"``, ``"cubic"``, ``"previous"``, ``"next"`` or ``"monotone_cubic"``. The
        methods ``"zero"``, ``"slinear"``,``"quadratic"`` and ``"cubic"`` refer to a spline
        interpolation of zeroth, first, second or third order; whereas ``"previous"`` and
        ``"next"`` simply return the previous or next value of the point. An integer specifying the
        order of the spline interpolator to use.
        See `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.
        PchipInterpolator.html>`_ for details on the ``"monotone_cubic"`` method.
    fill_value : array-like or (array-like, array_like) or “extrapolate”
        If a ndarray (or float), this value will be used to fill in for 
        requested points outside of the data range.  
        If a two-element tuple, then the first element is used as a fill value 
        for x_new < x[0] and the second element is used for x_new > x[-1]. 
        If “extrapolate”, then points outside the data range will be extrapolated.
        If not provided, then the default is ([y_values[0]], [y_values[-1]]).
    Returns
    -------
    array
        Vector of interpolated samples.
    Examples
    --------
    .. ipython:: python
      import numpy as np
      import neurokit2 as nk
      import matplotlib.pyplot as plt
      # Generate Simulated Signal
      signal = nk.signal_simulate(duration=2, sampling_rate=10)
      # We want to interpolate to 2000 samples
      x_values = np.linspace(0, 2000, num=len(signal), endpoint=False)
      x_new = np.linspace(0, 2000, num=2000, endpoint=False)
      # Visualize all interpolation methods
      @savefig p_signal_interpolate1.png scale=100%
      nk.signal_plot([
          nk.signal_interpolate(x_values, signal, x_new=x_new, method="zero"),
          nk.signal_interpolate(x_values, signal, x_new=x_new, method="linear"),
          nk.signal_interpolate(x_values, signal, x_new=x_new, method="quadratic"),
          nk.signal_interpolate(x_values, signal, x_new=x_new, method="cubic"),
          nk.signal_interpolate(x_values, signal, x_new=x_new, method="previous"),
          nk.signal_interpolate(x_values, signal, x_new=x_new, method="next"),
          nk.signal_interpolate(x_values, signal, x_new=x_new, method="monotone_cubic")
      ], labels = ["Zero", "Linear", "Quadratic", "Cubic", "Previous", "Next", "Monotone Cubic"])
      # Add original data points
      plt.scatter(x_values, signal, label="original datapoints", zorder=3)
      @suppress
      plt.close()
    """
    # Sanity checks
    if len(x_values) != len(y_values):
        raise ValueError(
            "NeuroKit error: signal_interpolate(): x_values and y_values must be of the same length."
        )
    if isinstance(x_new, int):
        if len(x_values) == x_new:
            return y_values
    else:
        if len(x_values) == len(x_new):
            return y_values
    if method == "monotone_cubic":
        interpolation_function = scipy.interpolate.PchipInterpolator(
            x_values, y_values, extrapolate=True
        )
    else:
        if fill_value is None:
            fill_value = ([y_values[0]], [y_values[-1]])
        interpolation_function = scipy.interpolate.interp1d(
            x_values, y_values, kind=method, bounds_error=False, fill_value=fill_value,
        )
    if isinstance(x_new, int):
        x_new = np.linspace(x_values[0], x_values[-1], x_new)
    interpolated = interpolation_function(x_new)

    if method == "monotone_cubic":
        # Swap out the cubic extrapolation of out-of-bounds segments generated by
        # scipy.interpolate.PchipInterpolator for constant extrapolation akin to the behavior of
        # scipy.interpolate.interp1d with fill_value=([y_values[0]], [y_values[-1]].
        interpolated[: int(x_values[0])] = interpolated[int(x_values[0])]
        interpolated[int(x_values[-1]) :] = interpolated[int(x_values[-1])]
    return interpolated
