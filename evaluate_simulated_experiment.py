import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pickle

from fake_crystals import FakeVoronoiCrystal

with open('positions_measured.pkl', 'rb') as f:
    positions_measured = pickle.load(f)
with open('cluster_label_history.pkl', 'rb') as f:
    cluster_label_history = pickle.load(f)

data = xr.open_dataset('data.nc')
data = data['ARPES'][:len(positions_measured)].values

crystal = FakeVoronoiCrystal()


