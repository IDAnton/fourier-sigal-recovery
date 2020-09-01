from ffilter import FFPlotter, FourierFilter

import numpy as np

def test_interface_check():
    signal = [
        [1, 1, 0],
        [2, 3, np.pi],
        [3, 5, np.pi/3]
    ]
    ff = FourierFilter(200, 5, signal)
    ffp = FFPlotter(ff)

    assert hasattr(ffp, 'plot')
