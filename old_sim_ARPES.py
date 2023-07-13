import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from time import perf_counter

# for the following parameters, the code approximates a 25meV resolution and 0.09 inverse angstrom resolution.
# by eye, some noise is added.
# SAMPLING_RATE = 1000
# E_RES = 0.00714*SAMPLING_RATE
# K_RES = 0.0027*SAMPLING_RATE
# MIN_ENERGY = -3
# MAX_ENERGY = 0.5


SAMPLING_RATE = 750
NUM_ENE = 700
MIN_ENERGY = -0.7
MAX_ENERGY = 0.1
MIN_K = -np.pi
MAX_K = np.pi
E_RES = (0.025/(MAX_ENERGY-MIN_ENERGY))*SAMPLING_RATE
K_RES = (0.011/(MAX_K-MIN_K))*SAMPLING_RATE
SAVE = False
ANGLES = 900
TEMPERATURE = 30
FILE_NAME = f'simulate_polycrystal/output/all_spectra_ANGLES_{ANGLES}_SR_{SAMPLING_RATE}.npy'


def fermi_function(energy):
    """
    This function returns the Fermi function for a given energy and temperature
    """
    beta = 1/(8.617e-5*TEMPERATURE) #eV^-1
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

def pure_spectral_weight(kx, ky, energy_range = (-0.7, 0.1), num_energies = 700):
    """
    This function computes the APRES intesnity after meshing the kx and ky.
    Returns a 3D array of the spectral weight at each kx, ky and energy.
    The spectral weight has the fermi function applied.
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
    fermi = fermi_function(energies)
    fermi = np.tile(fermi, (len(kx),1))
    # multiply the spectral weight by the fermi function
    spectral_weight = spectral_weight*fermi
    #spectral_weight *= fermi_function(energies[j])

    # if K_RES and E_RES are not 0, use scipy gaussian_filter to apply resolution blur
    if K_RES != 0 and E_RES != 0:
        spectral_weight = gaussian_filter(spectral_weight, sigma = (K_RES*7, E_RES/2.5))

    return spectral_weight.transpose()

def calculate_spectral_weight(kx, ky):
    """
    This function computes the APRES intesnity after meshing the kx and ky.
    Returns a 3D array of the spectral weight at each kx, ky and energy.
    """
    energy = cuprate_tight_binding(kx, ky)
    energy_range = np.linspace(MIN_ENERGY, MAX_ENERGY, SAMPLING_RATE)
    spectral_weight = np.abs(np.random.normal(0,0.005, size=(len(kx), len(energy_range))).astype(np.float32))
    #spectral_weight = np.zeros((len(kx), len(energy_range)), dtype=np.float32)
    # Set spectral weight to 1 where the energy is for a given kx, ky
    for i in range(len(kx)):
        for j in range(len(energy_range)-1):
            #if energy[i] is within the energy range pixel, set the spectral weight to 1
            if energy[i] > energy_range[j] and energy[i] < energy_range[j+1]:
                spectral_weight[i][j] += 10
    
    # calculate the fermi function in the shape of spectral weight
    fermi = fermi_function(energy_range)
    fermi = np.tile(fermi, (len(kx),1))
    # multiply the spectral weight by the fermi function
    spectral_weight = spectral_weight*fermi
    #spectral_weight *= fermi_function(energy_range[j])

    # if K_RES and E_RES are not 0, use scipy gaussian_filter to apply resolution blur
    if K_RES != 0 and E_RES != 0:
        spectral_weight = gaussian_filter(spectral_weight, sigma = (1e-6, 1e-6))

    return spectral_weight.transpose()

def plot_ARPES(kx0, kx1, ky0, ky1,):
    """
    This function plots the ARPES speectral weight as a function of energy for the line between kx0, ky0 and kx1, ky1
    As an image plot.
    """
    t = np.linspace(0,1,SAMPLING_RATE)
    kx = kx0 + t*(kx1 - kx0)
    ky = ky0 + t*(ky1 - ky0)
    min_k = np.min(np.sqrt(kx**2 + ky**2))
    max_k = np.max(np.sqrt(kx**2 + ky**2))
    spectral_weight = calculate_spectral_weight(kx, ky)
    plt.figure()
    #flip image so that the energy axis is increasing
    plt.imshow(spectral_weight, cmap='Greys', extent = [min_k, max_k, MIN_ENERGY, MAX_ENERGY], aspect = 'auto', origin = 'lower')

    #Crpp the image above the fermi level (energy = 0)
    #plt.ylim(min_energy, max_energy)
    plt.ylim(-3,0.5)
    plt.axhline(y=0, color='k', linestyle='--')
    # add a title  showing the kx, ky values to 2 decimal places
    plt.title(f'kx0 = {kx0:.2f}, ky0 = {ky0:.2f}, kx1 = {kx1:.2f}, ky1 = {ky1:.2f}')

    plt.xlabel('k')
    plt.ylabel('E')
    plt.show()

def plot_many_dispersions():
    """
    This function plots the ARPES speectral weight as a function of energy for the line between kx0, ky0 and kx1, ky1
    As an image plot.
    """
    # time the function and print out estimated time to completion
    spectral_weight = None
    t0 = perf_counter()
    all_spectra = np.zeros((ANGLES, SAMPLING_RATE, SAMPLING_RATE), dtype=np.float32)
    for i, theta in enumerate(np.linspace(0,np.pi/4,ANGLES)):
        print(f"Angle {theta:.2f}: {100*theta/(np.pi/2):.0f}% Time elapsed: {perf_counter() - t0:.2f}s, estimated time remaining: {(perf_counter() - t0)*(np.pi/4 - theta)/(theta)/60:.2f}min")
        kr = np.linspace(MIN_K,MAX_K,SAMPLING_RATE)
        kx = kr*np.cos(theta)
        ky = kr*np.sin(theta)
        all_spectra[i] = calculate_spectral_weight(kx, ky)
    all_spectra *= np.abs(np.random.normal(1, 0.1, size = all_spectra.shape))
    spectral_weight = np.sum(all_spectra, axis = 0)
    plt.figure()
    #flip image so that the energy axis is increasing
    plt.imshow(np.log(spectral_weight+1), cmap='Greys', extent = [0, np.max(kr), MIN_ENERGY, MAX_ENERGY], aspect = 'auto', origin = 'lower')

    #Crpp the image above the fermi level (energy = 0)
    #plt.ylim(min_energy, max_energy)
    plt.axhline(y=0, color='k', linestyle='--')

    plt.xlabel('k')
    plt.ylabel('E')
    if SAVE:
        np.save(FILE_NAME, all_spectra)
    plt.show()

def plot_bandstructure(kx,ky,hamiltonian):
    """
    This function plots the bandstructure of the tight-binding Hamiltonian
    """
    plt.figure()
    plt.plot(kx,hamiltonian)
    plt.xlabel('kx')
    plt.ylabel('E')
    plt.show()

def plot_energy_surface(kx,ky,hamiltonian):
    """
    This function plots the energy as a function of kx and ky in a 3D plot
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(kx,ky,hamiltonian)
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('E')
    plt.show()

def plot_energy_contour(kx,ky,hamiltonian):
    """
    This function plots the energy as a function of kx and ky in a contour plot
    """
    plt.figure()
    plt.contourf(kx,ky,hamiltonian)
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.show()

def plot_dispersions_at_many_angles():
    for theta in np.linspace(0,np.pi/4,ANGLES):
        kr = 1*np.linspace(MIN_K, MAX_K, SAMPLING_RATE)
        kx = kr*np.cos(theta)
        ky = kr*np.sin(theta)
        hamiltonian = cuprate_tight_binding(kx,ky)
        plt.plot(kr,hamiltonian, color = 'black', alpha = 2/ANGLES)
    plt.xlabel('kr')
    plt.ylabel('E')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.ylim(MIN_ENERGY,MAX_ENERGY)
    plt.xlim(np.min(kr),np.max(kr))
    plt.show()

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

    # plt.plot(kx,ky)
    # plt.title(f'omega = {np.degrees(azimuthal):0.2f}, theta = {np.degrees(polar):0.2f}')
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    # plt.axhline(0, linestyle = ':', color='k')
    # plt.axvline(0, linestyle = ':', color='k')
    # plt.show()
    # exit()


def ARPES_intensity_for_given_rotation(omega, tilt, theta):
    # kx0 = np.zeros((SAMPLING_RATE))
    # ky0 = np.linspace(MIN_K,MAX_K,SAMPLING_RATE)
    # kz0 = np.zeros((SAMPLING_RATE))

    slit_coords = np.radians(np.array([np.linspace(-15, 15, SAMPLING_RATE), np.zeros(SAMPLING_RATE)])).T

    # coords = np.array([kx0, ky0, kz0]).T
    # rotated_coords = sph_to_xyz(rotate_x(xyz_to_sph(coords), theta))
    rotated_coords = sph_to_xyz(rotate_x(rotate_y(rotate_z(slit_coords, omega), tilt), theta))

    kx, ky, kz = rotated_coords.T

    plt.plot(kx,ky)
    plt.title(f'omega = {np.degrees(omega):0.2f}, theta = {np.degrees(theta):0.2f}')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axhline(0, linestyle = ':', color='k')
    plt.axvline(0, linestyle = ':', color='k')
    plt.show()
    exit()
    occupied_states_spectra_weight = calculate_spectral_weight(kx, ky)
    gaussian_noise = np.abs(np.random.normal(1, 0.1, size = occupied_states_spectra_weight.shape))
    noisy_spectral_weight = occupied_states_spectra_weight*gaussian_noise
    return noisy_spectral_weight

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

def main():
    # get_normed_xyz_coords_from_orientation(30, 100, 0, 90, 15)

    acceptance_angle = 30
    num_angles = 350
    num_energies = 350
    num_cuts = 20

    spectra = np.zeros((num_cuts, num_angles, num_energies))
    angular_coords = np.zeros((num_cuts, 2, num_angles)) # theta and phi for each individual edc
    k_coords = np.zeros((num_cuts, 3, num_angles))

    # TODO: Impliment correct constants here
    r =  0.5*np.sqrt(100) #(1/hbar) * sqrt(2 m_e KE)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_subplot(121)
    slit_angles = np.linspace(-acceptance_angle/2, acceptance_angle/2, num_angles)
    k_parallel = r * np.cos(slit_angles)
    

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    for i in range(num_cuts):
        # fig = plt.figure()
        # ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        # ax2 = fig.add_subplot(1, 2, 2)
        # polar = np.random.uniform(-15,15)
        # tilt = np.random.uniform(-15,15)
        polar = 0
        tilt = 0
        azimuthal = np.random.uniform(0, 360)
        kx, ky, kz = get_normed_xyz_coords_from_orientation(acceptance_angle, num_angles, polar, azimuthal, tilt)
        angular_coords[i] = xyz_to_sph(np.array([kx, ky, kz]).T).T

        kx, ky, kz = r * kx, r * ky, r * kz
        spectrum = pure_spectral_weight(kx, ky, num_energies=num_energies)

        spectra[i] = spectrum
        k_coords[i] = [kx, ky, kz]
        ax1.plot(kx,ky,kz)
    ax1.set_xlim(-r, r)
    ax1.set_ylim(-r, r)
    ax1.set_zlim(-r, r)
    # ax2.imshow(spectrum, cmap='Greys', aspect = 'auto', origin = 'lower', extent = [np.min(k_parallel), np.max(k_parallel), -0.7, 0.1])
    fig.tight_layout()
    plt.show()
    
    from datetime import datetime
    header = "/Users/matthewstaab/Documents/VishikLab/coding_projects/PolyARPES/simulate_polycrystal/output/"
    footer = f"_2D_double_band.npy"
    np.save(header + "spectra" + footer, spectra)
    np.save(header + "angular_coords" + footer, angular_coords)
    np.save(header + "k_coords" + footer, k_coords)
    
    # plt.axhline(0, linestyle = ':', color='k')
    # plt.axvline(0, linestyle = ':', color='k')
    # plt.show()

    # fake_arpes_data = ARPES_intensity_for_given_rotation(omega, tilt, theta)
    # plt.imshow(fake_arpes_data, cmap='Greys', aspect = 'auto', origin = 'lower')
    # plt.show()

    # plot_ARPES(0, 2*np.pi, 0, 2*np.pi)
    # plot_many_dispersions()
    # plot_dispersions_at_many_angles()
    

    # kx = np.linspace(0,2*np.pi,SAMPLING_RATE)
    # ky = 0
    # hamiltonian = cuprate_tight_binding(kx,ky)
    # plot_bandstructure(kx,ky,hamiltonian)
# 
    # kx = np.linspace(0,2*np.pi,SAMPLING_RATE)
    # ky = np.linspace(0,2*np.pi,SAMPLING_RATE)
    # kx, ky = np.meshgrid(kx,ky)
    # hamiltonian = cuprate_tight_binding(kx,ky)
    # plot_energy_surface(kx,ky,hamiltonian)
    # plot_energy_contour(kx,ky,hamiltonian)




if __name__ == "__main__":
    main()
