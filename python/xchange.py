import json
import math
from datetime import datetime

import numpy as np
from numba import jit


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
                for z in range(3):
                    k_vec[idx,z] = (rec_vec[0,z] * i/ kmesh[0]) + (rec_vec[1,z] * j / kmesh[1]) + (rec_vec[2,z] * k / kmesh[2])

                idx+=1

    return k_vec 



@jit(nopython=True) 
def energy_contour_preparation(ncol, nrow, e_fermi, e_low, smearing):
    # prepare the energy contour for integration
    ntot = ncol + 2 * nrow
    de = complex((e_fermi - e_low) / ncol, smearing / nrow)

    E = np.zeros(ntot, dtype=np.complex128)
    dE = np.zeros(ntot, dtype=np.complex128)
    e_const = complex(e_low, 0)

    idx_x = 0
    idx_y = 1

    idx = 0
    for i in range(ntot):
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

    for i in range(n_min[0], n_max[0]):
        for j in range(n_min[1], n_max[1]):
            for k in range(n_min[2], n_max[2]+1):
                for z in range(2):
                    r = i * cell_vec[0] + j * cell_vec[1] + k * cell_vec[2]
                    t = (k_vec @ r).reshape(num_kpoints, 1, 1)
                    t = np.exp(-1j * t)
                    rhs = np.ascontiguousarray(Ham_R[z,i + n_max[0], j + n_max[1], k + n_max[2],:,:])
                    Ham_K[z] += t * rhs

    return Ham_K



@jit(nopython=True) 
def calc_exchange(central_atom, index_temp, num_orb, num_kpoints, ntot, spin, cell_vec, k_vec, E, dE, Ham_K, mag_orbs):
    # This function calculates exchange coupling parameter between atoms with index 'central_atom' and 'index_temp'

    weight = 1/num_kpoints

    r =  np.zeros(3)

    loc_greenK = np.zeros((2,num_orb, num_orb), dtype=np.complex128)
    greenK_ij = np.zeros((mag_orbs[central_atom], mag_orbs[index_temp[3]]), dtype=np.complex128)
    greenK_ji = np.zeros((mag_orbs[index_temp[3]], mag_orbs[central_atom]), dtype=np.complex128)
    exchange = np.zeros((mag_orbs[central_atom], mag_orbs[central_atom]))

    shift_i = mag_orbs[:central_atom].sum()
    shift_j = mag_orbs[:index_temp[3]].sum()
    
    for z in range(3):
         r[z] = index_temp[0] * cell_vec[0][z] + index_temp[1] * cell_vec[1][z] + index_temp[2] * cell_vec[2][z]

    for num in range(ntot):
        delta_i = np.zeros((mag_orbs[central_atom], mag_orbs[central_atom]), dtype=np.complex128)
        delta_j = np.zeros((mag_orbs[index_temp[3]], mag_orbs[index_temp[3]]), dtype=np.complex128)
        greenR_ij = np.zeros((mag_orbs[central_atom], mag_orbs[index_temp[3]]), dtype=np.complex128)
        greenR_ji = np.zeros((mag_orbs[index_temp[3]], mag_orbs[central_atom]), dtype=np.complex128)


        for  e in range(num_kpoints):

            for z in range(2):
                #G = 1/(E - H)
                loc_greenK[z] = np.linalg.inv(E[num]*np.diag(np.ones(num_orb)) - Ham_K[z,e])
           
            # read the necessary block
            greenK_ij[:mag_orbs[central_atom], :mag_orbs[index_temp[3]]] = loc_greenK[1, shift_i:mag_orbs[central_atom], shift_j:mag_orbs[index_temp[3]]]
            greenK_ji[:mag_orbs[index_temp[3]], :mag_orbs[central_atom]] = loc_greenK[0, shift_j:mag_orbs[index_temp[3]], shift_i:mag_orbs[central_atom]]


            delta_i[:mag_orbs[central_atom],:mag_orbs[central_atom]] += weight * (Ham_K[0, e, shift_i:mag_orbs[central_atom], shift_i:mag_orbs[central_atom]] - 
            Ham_K[1, e, shift_i:mag_orbs[central_atom],shift_i:mag_orbs[central_atom]])

            delta_j[:mag_orbs[index_temp[3]],:mag_orbs[index_temp[3]]] += weight * (Ham_K[0, e, shift_j:mag_orbs[index_temp[3]], shift_j:mag_orbs[index_temp[3]]] - 
            Ham_K[1, e, shift_j:mag_orbs[index_temp[3]],shift_j:mag_orbs[index_temp[3]]])


            greenR_ij += weight * np.exp( 1j * np.dot(k_vec[e],r) ) * greenK_ij
            greenR_ji += weight * np.exp(-1j * np.dot(k_vec[e],r) ) * greenK_ji

        dot_product = np.dot(np.dot(np.dot(delta_i, greenR_ij),delta_j),greenR_ji) 

        exchange -= (1 / (2 * np.pi * spin**2 )) * (dot_product * dE[num]).imag
       
    return exchange


@jit(nopython=True) 
def calc_occupation(central_atom, num_orb, num_kpoints, ntot, Ham_K, E, dE, mag_orbs):
    #This function calculates occupation matrices for atom with index 'central_atom'

    weight = 1/num_kpoints

    loc_greenK = np.zeros((2,num_orb, num_orb), dtype=np.complex128)
    greenK_ii = np.zeros((2, mag_orbs[central_atom], mag_orbs[central_atom]), dtype=np.complex128)
    occ = np.zeros((2, mag_orbs[central_atom], mag_orbs[central_atom]))

    shift_i = mag_orbs[:central_atom].sum()

    for num in range(ntot):
        greenR_ii = np.zeros((2, mag_orbs[central_atom], mag_orbs[central_atom]), dtype=np.complex128)

        for  e in range(num_kpoints):

            for z in range(2):
                #G = 1/(E - H)
                loc_greenK[z] = np.linalg.inv(E[num]*np.diag(np.ones(num_orb)) - Ham_K[z,e]) 
           
            #read the necessary block
            greenK_ii[:, :mag_orbs[central_atom], :mag_orbs[central_atom]] = loc_greenK[:, shift_i:mag_orbs[central_atom], shift_i:mag_orbs[central_atom]]

            greenR_ii += weight * greenK_ii

        occ -= (1 / np.pi) * (greenR_ii * dE[num]).imag

    return occ




if __name__ == '__main__':
    print("Program xchange.x v.3.0 (python) starts on  ", datetime.now())
    print('=' * 69)


    with open('spin_up.dat') as fp:
        rows = (line.split() for line in fp)
        data = [([int(u) for u in row[:3]], [int(u) for u in row[3:5]],
                 [float(u) for u in row[5:]]) for row in rows]

    # [N, 3] vectors, [N, 2] orbitals, [N, 2] hamiltonian
    vecs, orbs, ham_values = map(np.array, zip(*data))
    ham_values = ham_values.astype('f8').view('c16').ravel()  # View as complex [N]

    num_orb = max(orbs[:,0])
    n_min, n_max = vecs.min(0), vecs.max(0)  # [3]
    n_size = n_max - n_min + 1  # Plus 1 for 0th

    Ham_R = np.zeros((2, *n_size, num_orb, num_orb), dtype='c16')
    Ham_R[(0, *(vecs + n_max).T, *(orbs.T - 1))] = ham_values



    with open('spin_dn.dat') as fp:
        rows = (line.split() for line in fp)
        data = [([int(u) for u in row[:3]], [int(u) for u in row[3:5]],
                 [float(u) for u in row[5:]]) for row in rows]

    # [N, 3] vectors, [N, 2] orbitals, [N, 2] hamiltonian
    vecs, orbs, ham_values = map(np.array, zip(*data))
    ham_values = ham_values.astype('f8').view('c16').ravel()  # View as complex [N]

    Ham_R[(1, *(vecs + n_max).T, *(orbs.T - 1))] = ham_values


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

    num_kpoints = kmesh[0] * kmesh[1] * kmesh[2]
    ntot = ncol + 2 * nrow


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
        print('Exchange coupling will be calculated only between Atom',central_atom,'(000)<-->Atom', specific[3], "(" , specific[0] , specific[1] , specific[2], ').')
        print('To calculate exchange couplings between all atoms set <exchange_for_specific_atoms>: [0, 0, 0, 0] in <in.json> file')

    else:
        print('Exchange couplings will be calculated between all atoms around',central_atom)
        print('within coordination spheres smaller than #',max_sphere_num)
    
    print('-' * 69)


    Ham_K = calc_hamK(num_orb, num_kpoints, n_min, n_max, cell_vec, k_vec, Ham_R)

    print('Fourier transformation  of Hamiltonian is completed')
 

    if(specific_true):
        index_temp = np.zeros(4, dtype=int)
        r = np.zeros(3) 
        # specific[0] = i; specific[1] = j; specific[2] = k; specific[3] = atom2

        for x in range(3):
            r[x] = specific[0] * cell_vec[0][x] + specific[1] * cell_vec[1][x] + specific[2] * cell_vec[2][x] + (positions[specific[3]][x] - positions[central_atom][x])

        print('=' * 69) 
        print("Atom", central_atom, "(000)<-->Atom", specific[0], "(", specific[1], specific[2], specific[3], ") with radius", '{:.4f}'.format(np.linalg.norm(r)))

        exchange = calc_exchange(central_atom, specific, num_orb, num_kpoints, ntot, spin, cell_vec, k_vec, E, dE, Ham_K, mag_orbs)

        print('\n'.join('  '.join('{:.6f}'.format(item) for item in row) for row in exchange))
        print('Trace equals to: ', '{:.6f}'.format(np.trace(exchange)), 'eV')

    else:
        num_points = num_mag_atoms * n_size.prod()

        radius, index = coordination_sort(central_atom, num_mag_atoms, n_min, n_max, cell_vec, positions)
        print('=' * 69)

        neighbor_num = 1
        sphere_num = 0

        for p in range(num_points):

            if(p == 0):
                occ = calc_occupation(central_atom, num_orb, num_kpoints, ntot, Ham_K, E, dE, mag_orbs)

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
                print("Atom", central_atom, "(000)<-->Atom", index[p,3], "(", index[p,0], index[p,1], index[p,2], ") with radius", '{:.4f}'.format(radius[p]), " is #", neighbor_num)

                exchange = calc_exchange(central_atom, index[p], num_orb, num_kpoints, ntot, spin, cell_vec, k_vec, E, dE, Ham_K, mag_orbs)

                print('\n'.join('  '.join('{:.6f}'.format(item) for item in row) for row in exchange))
                print('# ', central_atom, index[p,3], index[p,0], index[p,1], index[p,2], '{:.6f}'.format(np.trace(exchange)), 'eV')



    print('\n')
    print(f'This run was terminated on: {datetime.now()}')
    print(f'JOB DONE')
    print('=' * 69)