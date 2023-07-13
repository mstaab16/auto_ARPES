import zmq
from messages import *
import numpy as np
import matplotlib.pyplot as plt
from sim_ARPES import simulate_ARPES_measurement
from scipy.spatial import Voronoi

from time import perf_counter

context = zmq.Context()

class FakeVoronoiCrystal:
    def __init__(self):
        np.random.seed(42)
        num_crystalllites = 5
        vor_points = np.random.uniform(-2,2,(num_crystalllites,2))
        self.vor_azimuths = np.random.uniform(0,45, num_crystalllites)
        self.vor = Voronoi(vor_points)
        self.xcoords = np.linspace(-3,3,91)
        self.ycoords = np.linspace(-3,3,91)

    def measure(self, x, y):
        num_angles = 128
        num_energies = 128
        if np.abs(x) > 2 or np.abs(y) > 2:
            return x, y, np.random.exponential(0.001,(num_energies, num_angles))

        dx_sq = (self.vor.points[:,0] - x)**2
        dy_sq = (self.vor.points[:,1] - y)**2
        d_sq = np.sqrt(dx_sq + dy_sq)
        idx = np.argmax(d_sq)
        azimuth = self.vor_azimuths[idx]
        spectrum = simulate_ARPES_measurement(
                        polar=0, tilt=0, azimuthal=azimuth,
                        num_angles=num_angles, num_energies=num_energies)
        return x, y, spectrum

    def plot_measure(self, x, y):
        num_angles = 128
        num_energies = 128
        if np.abs(x) > 2 or np.abs(y) > 2:
            return -1

        dx_sq = (self.vor.points[:,0] - x)**2
        dy_sq = (self.vor.points[:,1] - y)**2
        d_sq = np.sqrt(dx_sq + dy_sq)
        idx = np.argmax(d_sq)
        azimuth = self.vor_azimuths[idx]
        return azimuth

    def get_boundaries(self):
        return np.min(self.xcoords), np.max(self.xcoords), np.min(self.ycoords), np.max(self.ycoords)

    def plot(self):
        xgrid, ygrid = np.meshgrid(self.xcoords, self.ycoords)
        z = np.vectorize(self.plot_measure)(xgrid.ravel(), ygrid.ravel()).reshape(xgrid.shape)
        plt.imshow(z, cmap='nipy_spectral_r', origin='lower')
        plt.colorbar()
        plt.show()

class FakeGrapheneCrystal:
    def __init__(self):
        import h5py
        self.filename =  r"20190915_01325_binned.h5"
        self.file = h5py.File(self.filename, 'r')
        self.data = self.file['2D_Data']['Fixed_Spectra1'][:]
        self.xcoords = self.file['0D_Data']['Scan X'][:]
        self.ycoords = self.file['0D_Data']['Scan Y'][:]


    def measure(self, x, y):
        dx = (self.xcoords - x)**2
        dy = (self.ycoords - y)**2
        d = np.sqrt(dx + dy)
        i = np.argmin(d)
        measured_x = self.xcoords[i]
        measured_y = self.ycoords[i]
        spectrum = self.data[:,:,i]
        return measured_x, measured_y, spectrum

    def get_boundaries(self):
        return np.min(self.xcoords), np.max(self.xcoords), np.min(self.ycoords), np.max(self.ycoords)

        

def main():
#     gr = FakeGrapheneCrystal()
#     xmin, xmax, ymin, ymax = gr.get_boundaries()
#     start = perf_counter()
#     for _ in range(10):
#         x_choice, y_choice = np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)
#         measured_x, measured_y, spectrum = gr.measure(x_choice, y_choice)
#         print(f"Measuring at {x_choice}, {y_choice}")
#         print(f"Measured at {measured_x}, {measured_y}")
# 
#     end = perf_counter()
#     print(f"Time per measurement: {(end-start)/10:.3f} s")
#     
#     plt.imshow(spectrum.T, origin='lower', cmap='Greys')
#     plt.show()
    vor = FakeVoronoiCrystal()
    vor.plot()

if __name__ == "__main__":
    main()
