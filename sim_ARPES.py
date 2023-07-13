import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time

def sph_to_xyz(sph_coords):
    def single_coord(coord):
        theta, phi = coord
        return (
                    np.cos(phi) * np.sin(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(theta)
                )
    return np.array(list(map(lambda x: single_coord(x), sph_coords)))

def xyz_to_sph(xyz_coords):
    def single_coord(coord):
        x, y, z = coord
        # print(x,y,z)
        # print(np.arccos(x/np.sqrt(x**2 + y**2 + 1e-6)))
        return (
                    np.arccos(z + 1e-10),
                    np.sign(y + 1.12941e-10)*np.arccos(x/np.sqrt(x**2 + y**2 + 1e-10))
                )
    return np.array(list(map(lambda x: single_coord(x), xyz_coords)))

def rotate_z(sph_coords, phi):
    xyz_coords = sph_to_xyz(sph_coords)
    rz = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1],
    ])
    return xyz_to_sph([np.dot(rz,coord) for coord in xyz_coords])

def rotate_y(sph_coords, theta):
    xyz_coords = sph_to_xyz(sph_coords)
    rz = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    return xyz_to_sph([np.dot(rz,coord) for coord in xyz_coords])

def rotate_x(sph_coords, theta):
    xyz_coords = sph_to_xyz(sph_coords)
    rz = np.array([
        [1,0,0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    return xyz_to_sph([np.dot(rz,coord) for coord in xyz_coords])

def get_normed_xyz_coords_from_orientation(acceptance_angle, num_angles, polar, azimuthal, tilt, angles_in_degrees = True):
    # All angle arguments are in degrees. To use radians use angles_in_degrees = False
    # Zero polar, azimuthal, and tilt corresponds to coordinates along the ky=0 plane (phi = 0) (physics convention)
    # Rotations are applied in this order: 
    #   - Azimuthal (rotate about kz axis)
    #   - Polar_y (rotate about ky axis)
    #   - Polar_x (rotate about kx axis)

    if angles_in_degrees:
        acceptance_angle = np.radians(acceptance_angle)
        polar = np.radians(polar)
        azimuthal = np.radians(azimuthal)
        tilt = np.radians(tilt)
    
    slit_coords = np.array([np.linspace(-acceptance_angle/2, acceptance_angle/2, num_angles), np.zeros(num_angles)]).T

    rotated_coords = sph_to_xyz(rotate_x(rotate_y(rotate_z(slit_coords, azimuthal), tilt), polar))

    kx, ky, kz = rotated_coords.T

    return kx, ky, kz

def fermi_function(temperature, energy):
    """
    This function returns the Fermi function for a given energy and temperature
    """
    beta = 1/(8.617e-5*temperature) #eV^-1
    return 1/(np.exp(beta*(energy)) + 1)


def cuprate_tight_binding(kx, ky, realistic_cuprate=False):
    t0 = 0.090555
    t1=-0.724474
    t2=0.0627473
    t3=-0.165944
    t4=-0.0150311
    t5=0.0731361
    energy1 = t0 + 0.5 * t1 * ( np.cos(kx) + np.cos(ky)) + t2 * np.cos(kx) * np.cos(ky) + 0.5* t3 * (np.cos(2 * kx) + np.cos(2*ky)) + 0.5 * t4 * (np.cos(2 * kx) * np.cos(ky) + np.cos(kx) * np.cos(2*ky)) + t5 * np.cos(2*kx) * np.cos(2*ky)
    if realistic_cuprate == True:
        return energy1
    
    t0 = 0.290555
    t1=-0.624474
    t2=0.0727473
    t3=-0.265944
    t4=-0.1150311
    t5=0.0731361
    energy2 = t0 + 0.5 * t1 * ( np.cos(kx) + np.cos(ky)) + t2 * np.cos(kx) * np.cos(ky) + 0.5* t3 * (np.cos(2 * kx) + np.cos(2*ky)) + 0.5 * t4 * (np.cos(2 * kx) * np.cos(ky) + np.cos(kx) * np.cos(2*ky)) + t5 * np.cos(2*kx) * np.cos(2*ky)
    return [energy1, energy2]

def pure_spectral_weight(kx, ky, temperature = 30,
                         k_resolution = 0.011, e_resolution = 0.025,
                         energy_range = (-0.7, 0.1), num_energies = 200,
                         noise_level=0.3):
    """
    This function computes the APRES intesnity after meshing the kx and ky.
    Returns a 3D array of the spectral weight at each kx, ky and energy.
    The spectral weight has the fermi function applied.
    The size of the gaussian convolution is determined by the e and k resolutions.
    The k and e units are in inverse Angstroms and eV respectively.
    """
    energy_bands = cuprate_tight_binding(kx, ky)
    energies = np.linspace(energy_range[0], energy_range[1], num_energies)
    # spectral_weight = np.abs(np.random.normal(0,0.005, size=(len(kx), len(energies))).astype(np.float32))
    spectral_weight = np.zeros((len(kx), len(energies)), dtype=np.float32)
    # Set spectral weight to 1 where the energy is for a given kx, ky
    for energy in energy_bands:
        for i in range(len(kx)):
            for j in range(len(energies)-1):
                #if energy[i] is within the energy range pixel, set the spectral weight to 1
                if energy[i] > energies[j] and energy[i] < energies[j+1]:
                    spectral_weight[i][j] += 1
    
    # calculate the fermi function in the shape of spectral weight
    fermi = fermi_function(temperature, energies)
    fermi = np.tile(fermi, (len(kx),1))
    # multiply the spectral weight by the fermi function
    spectral_weight = gaussian_filter(spectral_weight, sigma = 3)
    spectral_weight = spectral_weight*fermi
    #spectral_weight *= fermi_function(energies[j])

    # if K_RES and E_RES are not 0, use scipy gaussian_filter to apply resolution blur
    k_width = np.sqrt((kx[-1] - kx[0])**2 + (ky[-1] - ky[0])**2)
    k_resolution_for_scipy = (k_resolution/(k_width))*len(kx)
    e_resolution_for_scipy = (e_resolution/(np.max(energy_range) - np.min(energy_range)))*num_energies
    if k_resolution != 0 and e_resolution != 0:
        spectral_weight = gaussian_filter(spectral_weight, sigma = (k_resolution_for_scipy, e_resolution_for_scipy))
        spectral_weight *= np.random.uniform(1-noise_level, 1, size=spectral_weight.shape)

    return spectral_weight.transpose()

def simulate_ARPES_measurement(polar=0.0, tilt=0.0, azimuthal=0.0,
                               photon_energy=100.0, noise_level=0.3,
                               acceptance_angle=30.0, num_angles=250,
                               num_energies=200, temperature=30.0,
                               k_resolution=0.011, e_resolution=0.025, energy_range=(-0.7, 0.1)):
    """
    This function simulates an ARPES measurement for a given set of parameters.
    Returns the simulated ARPES spectrum.
    """
    inverse_hbar_times_sqrt_2me = 0.512316722 #eV^-1/2 Angstrom^-1
    r = inverse_hbar_times_sqrt_2me*np.sqrt(photon_energy) #(1/hbar) * sqrt(2 m_e KE)
    kx, ky, kz = get_normed_xyz_coords_from_orientation(acceptance_angle, num_angles, polar, azimuthal, tilt)
    kx *= r
    ky *= r
    kz *= r
    spectrum = pure_spectral_weight(kx, ky, temperature=temperature, k_resolution=k_resolution, e_resolution=e_resolution, energy_range=energy_range, num_energies=num_energies, noise_level=noise_level)
    return spectrum

if __name__ == "__main__":
    sender = imagezmq.ImageSender(connect_to='tcp://localhost:5432')
    while True:
        azimuth = np.random.uniform(0, 360)
        spectrum = simulate_ARPES_measurement(polar=np.random.uniform(-15,15), tilt=np.random.uniform(-15,15), azimuthal=azimuth)
        sender.send_image(f"{azimuth:.2f}", spectrum)
        time.sleep(0.01)

    # plt.imshow(spectrum, cmap='Greys', aspect='auto', origin='lower')
    # plt.show()
