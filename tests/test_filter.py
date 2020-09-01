from ffilter import FourierFilter

import numpy as np

def test_ctor():
    signal = [
        [1, 1, 0],
        [2, 3, np.pi],
        [3, 5, np.pi/3]
    ]
    ff = FourierFilter(200, 5, signal)

    assert hasattr(ff, 'set')

    ff.set()
    expected_attributes = [
        'time', 'signal', 'freqs', 'power',
        'mask', 'recovered', 'transformed']

    for attr in expected_attributes:
        assert hasattr(ff, attr)
        assert isinstance(getattr(ff, attr), np.ndarray)

def test_time():
    signal = [
        [1, 1, 0],
        [2, 3, np.pi],
        [3, 5, np.pi/3]
    ]
    ff = FourierFilter(200, 5, signal)
    ff.set()

    assert np.allclose(ff.time, np.linspace(0, 1, 200))

def test_signals():
    for i in range(3):
        signal = [
            [int(i==0), 1, 0],
            [int(i==1), 3, np.pi],
            [int(i==2), 5, np.pi/3]
        ]
        ff = FourierFilter(200, 5, signal)
        ff.set()
        time = np.linspace(0, 1, 200)
        expected = signal[i][0] *\
            np.sin(2.*np.pi*signal[i][1]*time+ signal[i][2])
        assert ff.signal.size == 200
        assert np.allclose(ff.signal, expected)

def test_power():
    freqs = [1, 3, 5]
    for i, f in enumerate(freqs):
        signal = [
            [int(i==0), 1, 0],
            [int(i==1), 3, np.pi],
            [int(i==2), 5, np.pi/3]
        ]
        ff = FourierFilter(200, 5, signal)
        ff.set()
        assert ff.power.size == 101
        assert np.argmax(ff.power) == f
