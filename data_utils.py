"""Download, validate and split the UCI Airfoil Self-Noise data."""
from pathlib import Path
import hashlib
import io
import urllib.request
import zipfile
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

URL = 'https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip'
FEATURES = ['frequency_hz', 'angle_deg', 'chord_m', 'velocity_m_s', 'thickness_m']

def load_data(path):
    path = Path(path)
    if not path.exists():
        try:
            with urllib.request.urlopen(URL, timeout=30) as response:
                archive = zipfile.ZipFile(io.BytesIO(response.read()))
            names = [n for n in archive.namelist() if n.endswith('airfoil_self_noise.dat')]
            if len(names) != 1:
                raise ValueError('Expected one airfoil_self_noise.dat file')
            payload = archive.read(names[0])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        except Exception as exc:
            raise RuntimeError('Download failed. Download the UCI dataset manually, '
                               'extract airfoil_self_noise.dat into data/, and retry.') from exc
    values = np.loadtxt(path)
    if values.shape != (1503, 6) or not np.isfinite(values).all():
        raise ValueError('Expected 1503 finite rows and 6 numeric columns')
    return values[:, :5], values[:, 5], hashlib.sha256(path.read_bytes()).hexdigest()

def split_data(X, seed=42):
    # Frequency sweeps sharing all four other inputs stay in one partition.
    # These are inferred condition groups, NOT supplied experiment identifiers.
    _, groups = np.unique(X[:, 1:], axis=0, return_inverse=True)
    trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                         random_state=seed).split(X, groups=groups))
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.25,
                  random_state=seed + 1).split(X[trainval], groups=groups[trainval]))
    train, val = trainval[tr], trainval[va]
    parts = [train, val, test]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not set(groups[parts[i]]) & set(groups[parts[j]])
    assert len(np.unique(np.concatenate(parts))) == len(X)
    return train, val, test, groups
