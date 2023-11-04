import json
import math
from io import StringIO
from typing import OrderedDict
from datetime import date, datetime
import torch

import numpy as np
from numba import jit


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

    E = np.zeros(num_freq, dtype=np.complex128)
    dE = np.zeros(num_freq, dtype=np.complex128)
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
        
        E[i] = e_const + complex(idx * idx_x * (de).real, idx * idx_y * (de).imag)
        dE[i] = complex(idx_x * (de).real, idx_y * (de).imag)
        idx = idx+1

    return E, dE



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
def calc_hamK(num_orb, num_kpoints, n_min, n_max, cell_vec, k_vec, Ham_R):
    #Fourier transformation  of Hamiltonian
    Ham_K = np.zeros((2, num_kpoints, num_orb, num_orb), dtype=np.complex128)

    for i in range(n_min[0], n_max[0]+1):
        for j in range(n_min[1], n_max[1]+1):
            for k in range(n_min[2], n_max[2]+1):
                for z in range(2):
                    r = i * cell_vec[0] + j * cell_vec[1] + k * cell_vec[2]
                    dot_product = (k_vec @ r).reshape(num_kpoints, 1, 1)
                    phase = np.exp(-1j * dot_product)
                    Ham_K[z] += phase * np.ascontiguousarray(Ham_R[z,i + n_max[0], j + n_max[1], k + n_max[2],:,:])

    return Ham_K




def calc_greenK(num_freq, num_kpoints, num_orb, Ham_K, E):
    #Greens function in frequency and kmesh grid

    freq =  E.reshape(num_freq, 1, 1, 1)  *  np.tile(np.eye(num_orb), (num_kpoints, 1, 1))
    freq = np.stack((freq, freq)) # spin up and down

    Ham_K_up = Ham_K[0]
    Ham_K_dn = Ham_K[1]
    Ham_K_up = np.tile(Ham_K_up, (num_freq, 1, 1, 1))
    Ham_K_dn = np.tile(Ham_K_dn, (num_freq, 1, 1, 1))
    Ham_K = np.stack((Ham_K_up, Ham_K_dn))

    #batched inversion
    with torch.no_grad():
        greenK = (torch.linalg.inv(torch.from_numpy(freq - Ham_K))).numpy() 

    # greenK = np.linalg.inv(freq - Ham_K)
    
    return greenK




def cal_delta(num_freq, n_max, Ham_R):
    #On-site splitting
    delta = Ham_R[0, n_max[0], n_max[1], n_max[2]] - Ham_R[1, n_max[0], n_max[1], n_max[2]]

    delta  = np.tile(delta, (num_freq, 1, 1))

    return delta



def calc_exchange(central_atom, index_temp, num_kpoints, num_freq, spin, cell_vec, k_vec, dE, mag_orbs, greenK, delta):
    # This function calculates exchange coupling parameter between atoms with index 'central_atom' and 'index_temp'

    weight = 1/num_kpoints    

    r = index_temp[0] * cell_vec[0] + index_temp[1] * cell_vec[1] + index_temp[2] * cell_vec[2]
    phase = np.exp(1j * np.matmul(k_vec, r))


    shift_i = np.sum(mag_orbs[:central_atom])
    shift_j = np.sum(mag_orbs[:index_temp[3]])

    delta_i = np.zeros((num_freq, mag_orbs[central_atom], mag_orbs[central_atom]), dtype=np.complex128)
    delta_j = np.zeros((num_freq, mag_orbs[index_temp[3]], mag_orbs[index_temp[3]]), dtype=np.complex128)

    delta_i[:, :mag_orbs[central_atom],:mag_orbs[central_atom]] = delta[:, shift_i:mag_orbs[central_atom] + shift_i, shift_i:mag_orbs[central_atom] + shift_i]
    delta_j[:, :mag_orbs[index_temp[3]],:mag_orbs[index_temp[3]]] = delta[:, shift_j:mag_orbs[index_temp[3]] + shift_j, shift_j:mag_orbs[index_temp[3]] + shift_j]


    greenK_ij = np.zeros((num_freq, num_kpoints, mag_orbs[central_atom], mag_orbs[index_temp[3]]), dtype=np.complex128)
    greenK_ji = np.zeros((num_freq, num_kpoints, mag_orbs[central_atom], mag_orbs[index_temp[3]]), dtype=np.complex128)

    greenK_ij[:, :, :mag_orbs[central_atom], :mag_orbs[index_temp[3]]] = greenK[1, :, :, shift_i:mag_orbs[central_atom] + shift_i, shift_j:mag_orbs[index_temp[3]] + shift_j]
    greenK_ji[:, :, :mag_orbs[index_temp[3]], :mag_orbs[central_atom]] = greenK[0, :, :, shift_j:mag_orbs[index_temp[3]] + shift_j, shift_i:mag_orbs[central_atom] + shift_i]

    greenR_ij =  weight * np.tensordot(greenK_ij, phase, axes=(1, 0))
    greenR_ji =  weight * np.tensordot(greenK_ji, np.conj(phase), axes=(1, 0))

    dot_product =  np.matmul(np.matmul(np.matmul(delta_i, greenR_ij), delta_j),greenR_ji) 

    #exchange ~ Im \int_{-\inf Ef} delta_i * greenR_ij * delta_j * greenR_ji * dE

    exchange = np.zeros((mag_orbs[central_atom], mag_orbs[central_atom]))
    exchange -= (1 / (2 * np.pi * spin**2 )) * np.tensordot(dot_product, dE, axes=(0, 0)).imag

    return exchange




def calc_occupation(central_atom, num_kpoints, mag_orbs, dE, greenK):
    weight = 1 / num_kpoints
    shift_i = np.sum(mag_orbs[:central_atom])

    greenR = weight * np.sum(greenK, axis = 2)

    occ = np.tensordot(greenR, dE, axes=([1], 0)).imag
    occ_local = - (1 / np.pi) * occ[:, shift_i:mag_orbs[central_atom] + shift_i, shift_i:mag_orbs[central_atom] + shift_i]

    return occ_local



if __name__ == '__main__':
    print("Program xchange.x v.4.0 (vectorized version_test) starts on  ", datetime.now())
    print('=' * 69)


    hops_up, num_orb, n_min, n_max = parse_hopping_from_wannier90_hr_dat('spin_up.dat') 
    n_size = n_max - n_min + 1  # Plus 1 for 0th

    Ham_R = np.zeros((2, *n_size, num_orb, num_orb), dtype='c16')
    
    for r in hops_up.keys():
        r_idx = np.array(r)
        Ham_idx = hops_up.get(r)
        
        for m in range(num_orb):
            for n in range(num_orb):
                Ham_R[0, r_idx[0] + n_max[0], r_idx[1] + n_max[1], r_idx[2] + n_max[2], m, n] = Ham_idx[m,n]

    
    hops_dn, num_orb, n_min, n_max = parse_hopping_from_wannier90_hr_dat('spin_dn.dat') 
    
    for r in hops_dn.keys():
        r_idx = np.array(r)
        Ham_idx = hops_dn.get(r)
        
        for m in range(num_orb):
            for n in range(num_orb):
                Ham_R[1, r_idx[0] + n_max[0], r_idx[1] + n_max[1], r_idx[2] + n_max[2], m, n] = Ham_idx[m,n]

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


    if (central_atom < 0  or  central_atom >= num_mag_atoms):
        print('ERROR! central_atom must be in the range 0 <= input < ', num_mag_atoms)
        exit()


    specific_true = False
    for i in range(4):
        if(specific[i] != 0):
            specific_true = True
            break

    if (specific[3] < 0 or specific[3] >= num_mag_atoms):
        print('ERROR! <exchange_for_specific_atoms: [3]> must be in the range 0 <= input < ',num_mag_atoms)
        exit()



    k_vec = kmesh_preparation(cell_vec)

    E, dE  = energy_contour_preparation(ncol, nrow, e_fermi, e_low, smearing)

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
        print('Exchange coupling will be calculated only between 1tom',central_atom,'(000)<-->1tom', specific[3], "(" , specific[0] , specific[1] , specific[2], ').')
        print('To calculate exchange couplings between all atoms set <exchange_for_specific_atoms>: [0, 0, 0, 0] in <in.json> file')

    else:
        print('Exchange couplings will be calculated between all atoms around',central_atom)
        print('within coordination spheres smaller than #',max_sphere_num)
    
    print('-' * 69)


    Ham_K = calc_hamK(num_orb, num_kpoints, n_min, n_max, cell_vec, k_vec, Ham_R)

    print('Fourier transformation  of Hamiltonian is completed')

    greenK = calc_greenK(num_freq, num_kpoints, num_orb, Ham_K, E)

    print('Greens function in frequency and kmesh grid is calculated')

    delta = cal_delta(num_freq, n_max, Ham_R)


    if(specific_true):
        index_temp = np.zeros(4, dtype=int)
        r = np.zeros(3) 
        # specific[0] = i; specific[1] = j; specific[2] = k; specific[3] = atom2

        for x in range(3):
            r[x] = specific[0] * cell_vec[0][x] + specific[1] * cell_vec[1][x] + specific[2] * cell_vec[2][x] + (positions[specific[3]][x] - positions[central_atom][x])

        print('=' * 69)         
        print("Interaction of atom", central_atom, "(000)<-->atom", specific[3], "(", specific[0], specific[1], specific[2], ") in distance", '{:.4f}'.format(np.linalg.norm(r)))

        exchange = calc_exchange(central_atom, specific, num_kpoints, num_freq, spin, cell_vec, k_vec, dE, mag_orbs, greenK, delta)

        print('\n'.join('  '.join('{:.6f}'.format(item) for item in row) for row in exchange))
        print('Trace equals to: ', '{:.6f}'.format(np.trace(exchange)), 'eV')

    else:
        num_points = num_mag_atoms * np.prod(n_size)

        radius, index = coordination_sort(central_atom, num_mag_atoms, n_min, n_max, cell_vec, positions)
        print('=' * 69)

        neighbor_num = 1
        sphere_num = 0

        for p in range(num_points):

            if(p == 0):
                occ = calc_occupation(central_atom, num_kpoints, mag_orbs, dE, greenK)

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
                print("Interaction of atom", central_atom, "(000)<-->atom", index[p,3], "(", index[p,0], index[p,1], index[p,2], ") in sphere", sphere_num ,"with radius", '{:.4f}'.format(radius[p]), " -- ", neighbor_num)

                exchange = calc_exchange(central_atom, index[p], num_kpoints, num_freq, spin, cell_vec, k_vec, dE, mag_orbs, greenK, delta)

                print('\n'.join('  '.join('{:.6f}'.format(item) for item in row) for row in exchange))
                print('# ', central_atom, index[p,3], index[p,0], index[p,1], index[p,2], '{:.6f}'.format(np.trace(exchange)), 'eV') #for post-processing



    print('\n')
    print(f'This run was terminated on: {datetime.now()}')
    print(f'JOB DONE')
    print('=' * 69)