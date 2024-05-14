import json
import math
from io import StringIO
import h5py
from typing import OrderedDict
from datetime import datetime

import numpy as np
from numba import jit, prange

#https://triqs.github.io/tprf/latest/reference/python_reference.html#wannier90-tight-binding-parsers
def parse_hopping_from_wannier90_hr_dat(filename):

    with open(filename, 'r') as fd:
        lines = fd.readlines()

    lines.pop(0) # pop time header

    num_wann = int(lines.pop(0))
    nrpts = int(lines.pop(0))

    nlines = int(np.ceil(float(nrpts / 15.)))

    deg = np.array([])
    for line in lines[:nlines]:
        deg = np.concatenate((deg, np.loadtxt(StringIO(line), dtype=int, ndmin=1)))
    
    assert( deg.shape == (nrpts,) )

    hopp = "".join(lines[nlines:])
    hopp = np.loadtxt(StringIO(hopp))

    assert( hopp.shape == (num_wann**2 * nrpts, 7) )
    
    R = np.array(hopp[:, :3], dtype=int) # Lattice coordinates in multiples of lattice vectors
    nm = np.array(hopp[:, 3:5], dtype=int) - 1 # orbital index pairs, wannier90 counts from 1, fix by remove 1
    
    n_min, n_max = R.min(0), R.max(0)

    t_re = hopp[:, 5]
    t_im = hopp[:, 6]
    t = t_re + 1.j * t_im # complex hopping amplitudes for each R, mn (H(R)_{mn})

    # -- Dict with hopping matrices

    r_dict = OrderedDict()
    hopp_dict = {}
    for idx in range(R.shape[0]):
        r = tuple(R[idx])

        if r not in r_dict:
            r_dict[r] = 1
        else:
            r_dict[r] += 1

        if r not in hopp_dict:
            hopp_dict[r] = np.zeros((num_wann, num_wann), dtype=complex)

        n, m = nm[idx]
        hopp_dict[r][n, m] = t[idx]

    # -- Account for degeneracy of the Wigner-Seitz points
    
    for r, weight in zip(list(r_dict.keys()), deg):
        hopp_dict[r] /= weight

    return hopp_dict, num_wann, n_min, n_max


@jit(nopython=True) 
def parse_self_energy_file(filename, ncol, nrow, num_orb, num_kpoints, ham_K):

    with h5py.File(filename, "r") as f:
        data = np.asarray(f['data'])

    selfen = np.zeros((num_orb, num_kpoints, ncol), dtype=np.complex128)
    for orbital_index in range(num_orb):
        mask = data[:, 1] == orbital_index + 1
        if np.any(mask):
            selfen[orbital_index, :, :] =  1e-3 * (data[mask, 4] + 1j * data[mask, 5]).reshape(num_kpoints, ncol) # from meV to eV

    selfen = np.transpose(selfen) 

    #new selfen in WF basis
    selfen_transformed = np.zeros((ncol, num_kpoints, num_orb, num_orb), dtype=np.complex128)
    for e in range(num_kpoints):
        _ , evec = np.linalg.eigh(ham_K[e])
        evec_herm = evec.conj().T

        for num in range(ncol):
            selfen_transformed[num, e] = np.matmul(evec_herm, np.matmul(np.diag(selfen[num, e]), evec))

    selfen_transformed = np.pad(selfen_transformed, [(nrow, nrow), (0, 0), (0, 0), (0, 0)])
    return selfen_transformed


@jit(nopython=True) 
def kmesh_preparation(cell_vec):
    #reciprocal vectors and and kmesh for integration
    rec_vec = np.zeros((3,3))
    k_vec  = np.zeros((num_kpoints,3))

    rec_vec[0]  = (2 * np.pi / np.linalg.det(cell_vec)) * np.cross(cell_vec[1], cell_vec[2])
    rec_vec[1]  = (2 * np.pi / np.linalg.det(cell_vec)) * np.cross(cell_vec[2], cell_vec[0])
    rec_vec[2]  = (2 * np.pi / np.linalg.det(cell_vec)) * np.cross(cell_vec[0], cell_vec[1])

    idx = 0
    for i in range(kmesh[0]):
        for j in range(kmesh[1]):
            for k in range(kmesh[2]):
                k_vec[idx] = (rec_vec[0] * i/ kmesh[0]) + (rec_vec[1] * j / kmesh[1]) + (rec_vec[2] * k / kmesh[2])
                idx+=1

    return k_vec 



@jit(nopython=True) 
def energy_contour_preparation(ncol, nrow, e_fermi, e_low, smearing):
    # prepare the energy contour for integration
    num_freq = ncol + 2 * nrow
    de = complex((e_fermi - e_low) / ncol, smearing / nrow)

    freq = np.zeros(num_freq, dtype=np.complex128)
    d_freq = np.zeros(num_freq, dtype=np.complex128)

    e_const = complex(e_low, 0)
    idx_x = 0
    idx_y = 1

    idx = 0
    for i in range(num_freq):
        if (i == nrow):
            idx_x = 1
            idx_y = 0
            idx = 0
            e_const = complex(e_low, smearing)
        elif (i == nrow + ncol):
            idx_x = 0
            idx_y = -1
            idx = 0
            e_const = complex(e_fermi, smearing)
        
        freq[i] = e_const + complex(idx * idx_x * (de).real, idx * idx_y * (de).imag)
        d_freq[i] = complex(idx_x * (de).real, idx_y * (de).imag)
        idx = idx + 1

    return freq, d_freq



def coordination_sort(central_atom, num_mag_atoms, n_min, n_max, cell_vec, positions):
    #This function sorts atoms depending on the radius from the central atom with index 'central_atom'
    n_size = n_max - n_min + 1  # Plus 1 for 0th
    num_points = num_mag_atoms * n_size.prod()

    # Index filling
    index_ = np.stack(np.unravel_index(np.arange(num_points),
                                       (*n_size, num_mag_atoms)),
                      axis=-1)
    index_ += [*n_min, 0]

    r = index_[:, :3] @ cell_vec + positions[index_[:, 3]] - positions[central_atom]
    radius_ = (r * r).sum(-1)**0.5

    # Sort radius descending
    idx = radius_.argsort()

    return radius_[idx], index_[idx]



@jit(nopython=True) 
def calc_hamK(num_orb, num_kpoints, n_min, n_max, cell_vec, k_vec, ham_R):
    #Fourier transformation  of Hamiltonian
    ham_K = np.zeros((2, num_kpoints, num_orb, num_orb), dtype=np.complex128)

    for i in range(n_min[0], n_max[0] + 1):
        for j in range(n_min[1], n_max[1] + 1):
            for k in range(n_min[2], n_max[2] + 1):
                for z in range(2):
                    r = i * cell_vec[0] + j * cell_vec[1] + k * cell_vec[2]
                    dot_product = (k_vec @ r).reshape(num_kpoints, 1, 1)
                    phase = np.exp(-1j * dot_product)
                    ham_K[z] += phase * np.ascontiguousarray(ham_R[z, i + n_max[0], j + n_max[1], k + n_max[2], :, :])

    return ham_K



@jit(nopython=True, parallel=True) 
def calc_exchange(central_atom, index_temp, num_orb, num_kpoints, num_freq, spin, cell_vec, k_vec, freq, d_freq, ham_K, selfen, mag_orbs):
    # This function calculates exchange coupling parameter between atoms with index 'central_atom' and 'index_temp'

    weight = 1/num_kpoints
    exchange = np.zeros((mag_orbs[central_atom], mag_orbs[central_atom]))

    shift_i = np.sum(mag_orbs[:central_atom])
    shift_j = np.sum(mag_orbs[:index_temp[3]])
    
    r = index_temp[0] * cell_vec[0] + index_temp[1] * cell_vec[1] + index_temp[2] * cell_vec[2]

    phase = np.zeros((num_kpoints), dtype=np.complex128)
    for e in range(num_kpoints):
        phase[e] = np.exp( 1j * np.dot(k_vec[e],r) )

    #frequency loop parallelization 
    for num in prange(num_freq):
        loc_exchange = np.zeros((mag_orbs[central_atom], mag_orbs[central_atom]))
        loc_greenK = np.zeros((2,num_orb, num_orb), dtype=np.complex128)
        corr_greenK = np.zeros((2,num_orb, num_orb), dtype=np.complex128)
        delta_i = np.zeros((mag_orbs[central_atom], mag_orbs[central_atom]), dtype=np.complex128)
        delta_j = np.zeros((mag_orbs[index_temp[3]], mag_orbs[index_temp[3]]), dtype=np.complex128)
        greenR_ij = np.zeros((mag_orbs[central_atom], mag_orbs[index_temp[3]]), dtype=np.complex128)
        greenR_ji = np.zeros((mag_orbs[index_temp[3]], mag_orbs[central_atom]), dtype=np.complex128)

        for  e in range(num_kpoints):
            for z in range(2):
                #G = 1/(E - H)
                loc_greenK[z] = np.linalg.inv(freq[num] * np.eye(num_orb) - ham_K[z, e])

                #Dyson  equation for correlated  Green's function
                corr_greenK[z] = np.linalg.inv(np.linalg.inv(loc_greenK[z]) - selfen[z, num, e])

            delta_i[:mag_orbs[central_atom],:mag_orbs[central_atom]] += weight * (ham_K[0, e, shift_i:mag_orbs[central_atom] + shift_i, shift_i:mag_orbs[central_atom] + shift_i] - 
            ham_K[1, e, shift_i:mag_orbs[central_atom] + shift_i, shift_i:mag_orbs[central_atom] + shift_i] + 
            selfen[0, num, e, shift_i:mag_orbs[central_atom] + shift_i, shift_i:mag_orbs[central_atom] + shift_i] - 
            selfen[1, num, e, shift_i:mag_orbs[central_atom] + shift_i, shift_i:mag_orbs[central_atom] + shift_i])
            

            delta_j[:mag_orbs[index_temp[3]],:mag_orbs[index_temp[3]]] += weight * (ham_K[0, e, shift_j:mag_orbs[index_temp[3]] + shift_j, shift_j:mag_orbs[index_temp[3]] + shift_j] - 
            ham_K[1, e, shift_j:mag_orbs[index_temp[3]] + shift_j, shift_j:mag_orbs[index_temp[3]] + shift_j] + 
            selfen[0, num, e, shift_j:mag_orbs[index_temp[3]] + shift_j, shift_j:mag_orbs[index_temp[3]] + shift_j] -
            selfen[1, num, e, shift_j:mag_orbs[index_temp[3]] + shift_j, shift_j:mag_orbs[index_temp[3]] + shift_j])

            greenR_ij += weight * phase[e] * corr_greenK[1, shift_i:mag_orbs[central_atom] + shift_i, shift_j:mag_orbs[index_temp[3]] + shift_j]
            greenR_ji += weight * np.conj(phase[e]) * corr_greenK[0, shift_j:mag_orbs[index_temp[3]] + shift_j, shift_i:mag_orbs[central_atom] + shift_i]

        dot_product = np.dot(np.dot(np.dot(delta_i, greenR_ij),delta_j),greenR_ji) 
        loc_exchange = (1 / (2 * np.pi * spin**2 )) * (dot_product * d_freq[num]).imag

        exchange -= loc_exchange # sum reduction 

    return exchange



@jit(nopython=True, parallel=True) 
def calc_occupation(central_atom, num_orb, num_kpoints, num_freq, ham_K, selfen, freq, d_freq, mag_orbs):
    #This function calculates occupation matrices for atom with index 'central_atom'

    weight = 1/num_kpoints
    occ = np.zeros((2, mag_orbs[central_atom], mag_orbs[central_atom]))
    shift_i = mag_orbs[:central_atom].sum()

    #frequency loop parallelization 
    for num in prange(num_freq):
        loc_occ = np.zeros((2, mag_orbs[central_atom], mag_orbs[central_atom]))
        loc_greenK = np.zeros((2,num_orb, num_orb), dtype=np.complex128)
        corr_greenK = np.zeros((2,num_orb, num_orb), dtype=np.complex128)
        greenR_ii = np.zeros((2, mag_orbs[central_atom], mag_orbs[central_atom]), dtype=np.complex128)

        for  e in range(num_kpoints):

            for z in range(2):
                #G = 1/(E - H)
                loc_greenK[z] = np.linalg.inv(freq[num] * np.eye(num_orb) - ham_K[z,e]) 

                #Dyson  equation for correlated  Green's function
                corr_greenK[z] = np.linalg.inv(np.linalg.inv(loc_greenK[z]) - selfen[z, num, e])

            greenR_ii += weight * corr_greenK[:, shift_i:mag_orbs[central_atom] + shift_i, shift_i:mag_orbs[central_atom] + shift_i]

        loc_occ = (1 / np.pi) * (greenR_ii * d_freq[num]).imag

        occ -= loc_occ # sum reduction 

    return occ



if __name__ == '__main__':
    print("Program xchange.x v.4.0 (python) starts on  ", datetime.now())
    print('=' * 69)


    hops_up, num_orb, n_min, n_max = parse_hopping_from_wannier90_hr_dat('spin_up.dat') 
    n_size = n_max - n_min + 1  # Plus 1 for 0th

    ham_R = np.zeros((2, *n_size, num_orb, num_orb), dtype='c16')
    
    for r in hops_up.keys():
        r_idx = np.array(r)
        ham_idx = hops_up.get(r)
        
        for m in range(num_orb):
            for n in range(num_orb):
                ham_R[0, r_idx[0] + n_max[0], r_idx[1] + n_max[1], r_idx[2] + n_max[2], m, n] = ham_idx[m, n]

    
    hops_dn, num_orb, n_min, n_max = parse_hopping_from_wannier90_hr_dat('spin_dn.dat') 
    
    for r in hops_dn.keys():
        r_idx = np.array(r)
        ham_idx = hops_dn.get(r)
        
        for m in range(num_orb):
            for n in range(num_orb):
                ham_R[1, r_idx[0] + n_max[0], r_idx[1] + n_max[1], r_idx[2] + n_max[2], m, n] = ham_idx[m, n]

    # =======================================================================
    # Read information from input file in json format
    with open('in.json') as fp:
        data = json.load(fp)

    num_mag_atoms = data['number_of_magnetic_atoms']  
    max_sphere_num = data['max_sphere_num'] 
    central_atom = data['central_atom']  
    spin = data['spin']  
    ncol = data['ncol'] 
    nrow = data['nrow']
    smearing = data['smearing']
    e_low = data['e_low']
    e_fermi = data['e_fermi']

    kmesh = np.array(data['kmesh'])  
    mag_orbs = np.array(data['orbitals_of_magnetic_atoms'])
    specific = np.array(data['exchange_for_specific_atoms'])  #i j k atom2
    cell_vec = np.array(data['cell_vectors'])
    positions = np.array(data['positions_of_magnetic_atoms'])

    num_kpoints = np.prod(kmesh)
    num_freq = ncol + 2 * nrow
    num_specific_pairs =  specific.shape[0]

    #check conditions
    if(np.all(specific == 0)):
        specific_true = False
    else:
        specific_true = True

    for i in range(num_specific_pairs):
        if (specific[i, 3] < 0  or  specific[i, 3] >= num_mag_atoms):
            print('ERROR! specific[:, 3] must be in the range 0 <= input < ', num_mag_atoms)
            exit()

    if (central_atom < 0  or  central_atom >= num_mag_atoms):
        print('ERROR! central_atom must be in the range 0 <= input < ', num_mag_atoms)
        exit()

    k_vec = kmesh_preparation(cell_vec)
    freq, d_freq  = energy_contour_preparation(ncol, nrow, e_fermi, e_low, smearing)

    #===============================================================================
    # print scanned data
    print('-' * 69)
    print('Parameters of integration:')
    print('Lowest energy of integration: ', e_low, 'Fermi energy: ', e_fermi)
    print('Smearing: ', smearing)
    print('Vertical steps:  ',  nrow, '; Horizontal steps: ', ncol)
    print('Number of k-points: ', num_kpoints)
    print('Total number of Wannier functions: ',  num_orb)
    print('-' * 69)
    print('Crystal structure of system:')

    print('crystal axes: (cart. coord. in units of alat)')
    for i in range(3):
        print('{:.2f}'.format(cell_vec[i,0]), '{:.2f}'.format(cell_vec[i,1]), '{:.2f}'.format(cell_vec[i,2]))  

    print('atomic positions and  orbitals')
    for i in range(num_mag_atoms):
        print('{:.2f}'.format(positions[i,0]), '{:.2f}'.format(positions[i,1]), '{:.2f}'.format(positions[i,2]), mag_orbs[i])

    if(specific_true):
        print('\n')
        print('Exchange coupling will be calculated only between the following pairs:')
        for i in range(num_specific_pairs):
            print(i, central_atom,'(000)<-->', specific[i, 3], "(" , specific[i, 0] , specific[i, 1] , specific[i, 2], ')')

        print('Set <exchange_for_specific_atoms>: [[0, 0, 0, 0]] to calculate for all of the pairs')

    else:
        print('\n')
        print('Exchange couplings will be calculated between all pairs around',central_atom)
        print('within coordination spheres smaller than #',max_sphere_num)
    
    print('-' * 69)


    ham_K = calc_hamK(num_orb, num_kpoints, n_min, n_max, cell_vec, k_vec, ham_R)
    print('Fourier transformation  of Hamiltonian is completed')


    #self-energy from EPW read and convert to Wannier function basis 
    selfen_up = parse_self_energy_file('selfen_up.h5', ncol, nrow, num_orb, num_kpoints, ham_K[0])
    selfen_dn = parse_self_energy_file('selfen_dn.h5', ncol, nrow, num_orb, num_kpoints, ham_K[1])
    selfen = np.stack((selfen_up, selfen_dn)) #spin up and down
 

    if(specific_true):

        for p in range(num_specific_pairs):

            r = np.zeros(3) 

            for x in range(3):
                r[x] = specific[p, 0] * cell_vec[0][x] + specific[p, 1] * cell_vec[1][x] + specific[p, 2] * cell_vec[2][x] + (positions[specific[p, 3]][x] - positions[central_atom][x])

            print('\n')
            print("Interaction of atom", central_atom, "(000)<-->atom", specific[p, 3], "(", specific[p, 0], specific[p, 1], specific[p, 2], ") in distance", '{:.4f}'.format(np.linalg.norm(r)))

            exchange = calc_exchange(central_atom, specific[p], num_orb, num_kpoints, num_freq, spin, cell_vec, k_vec, freq, d_freq, ham_K, selfen, mag_orbs)
            
            print('\n'.join('  '.join('{:.6f}'.format(item) for item in row) for row in exchange))
            print('# ', central_atom, specific[p, 3], specific[p, 0], specific[p, 1], specific[p, 2], '{:.6f}'.format(np.trace(exchange)), 'eV') #for post-processing

    else:

        num_points = num_mag_atoms * np.prod(n_size)

        radius, index = coordination_sort(central_atom, num_mag_atoms, n_min, n_max, cell_vec, positions)
        print('=' * 69)

        neighbor_num = 1
        sphere_num = 0

        for p in range(num_points):

            if(p == 0):
                occ = calc_occupation(central_atom, num_orb, num_kpoints, num_freq, ham_K, selfen, freq, d_freq, mag_orbs)
                print('Occupation matrix (N_up - N_dn) for atom ', central_atom)

                print('\n'.join('  '.join('{:.3f}'.format(item) for item in row) for row in (occ[0] - occ[1])))
                print('Trace equals to: ', '{:.3f}'.format(np.trace(occ[0] - occ[1])))

            else:
                if (math.fabs(radius[p - 1] - radius[p]) < 1E-4):
                    neighbor_num = neighbor_num + 1
                else:
                    neighbor_num = 1
                    sphere_num = sphere_num + 1

                if (sphere_num == max_sphere_num):
                    break

                print('\n')
                print("Interaction of atom", central_atom, "(000)<-->atom", index[p ,3], "(", index[p, 0], index[p,1], index[p, 2], ") in sphere", sphere_num ,"with radius", '{:.4f}'.format(radius[p]), " -- ", neighbor_num)

                exchange = calc_exchange(central_atom, index[p], num_orb, num_kpoints, num_freq, spin, cell_vec, k_vec, freq, d_freq, ham_K, selfen, mag_orbs)

                print('\n'.join('  '.join('{:.6f}'.format(item) for item in row) for row in exchange))
                print('# ', central_atom, index[p, 3], index[p, 0], index[p, 1], index[p, 2], '{:.6f}'.format(np.trace(exchange)), 'eV') #for post-processing

    print('\n')
    print(f'This run was terminated on: {datetime.now()}')
    print(f'JOB DONE')
    print('=' * 69)