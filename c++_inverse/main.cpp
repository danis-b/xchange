#include <complex>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include "functions.h"
#include <nlohmann/json.hpp> //we use json format for input file

using json = nlohmann::json;

int main()
{
    time_t td;
    int i, j, k, x, y, z, e, p, central_atom, sphere_num, neighbor_num, max_sphere_num;
    int num_mag_atoms, num_orb, num_kpoints, num_points;
    int n_size[3], n_min[3], n_max[3], vecs[3], orbs[2], kmesh[3], specific[4], index_temp[4];

    double spin, vol, matrix_trace, rad_specific;
    double cell_vec[3][3], rec_vec[3][3], ham_values[2], r[3];

    bool specific_true, err;

    // integration parameters
    int ncol, nrow, ntot, num;
    double smearing, e_low, e_fermi;
    std::complex<double> de, e_const;

    td = time(NULL);
    std::cout << "Program Xchange.x v.3.0 starts on ";
    std::cout << ctime(&td);
    std::cout << "=====================================================================" << std::endl;

    std::ifstream up;
    up.open("spin_up.dat");
    if (!up)
    {
        std::cout << "ERROR!Cannot open file <spin_up.dat>!" << std::endl;
        return 0;
    }

    std::ifstream dn;
    dn.open("spin_dn.dat");
    if (!dn)
    {
        std::cout << "ERROR!Cannot open file <spin_dn.dat>!" << std::endl;
        return 0;
    }

    // define the possible ranges [n_min; n_max] and  num_orb from .dat files
    for (i = 0; i < 3; i++)
    {
        n_max[i] = n_min[i] = 0;
    }

    num_orb = 1;
    while (!up.eof())
    {
        up >> vecs[0] >> vecs[1] >> vecs[2] >> orbs[0] >> orbs[1] >> ham_values[0] >> ham_values[1];

        if (orbs[0] > num_orb)
            num_orb = orbs[0];

        for (i = 0; i < 3; i++)
        {
            if (n_max[i] < vecs[i])
                n_max[i] = vecs[i];
            else if (n_min[i] > vecs[i])
                n_min[i] = vecs[i];
        }
    }

    for (i = 0; i < 3; i++)
    {
        n_size[i] = n_max[i] - n_min[i] + 1; // Plus 1 for 0th
    }

    //===============================================================================
    // read information from input files and matrix initialization

    std::ifstream in;
    in.open("in.json");
    if (!in)
    {
        std::cout << "ERROR! Cannot open input file  in.json" << std::endl;
        return 0;
    }

    json json_file;
    in >> json_file;

    in.close();

    json_file.at("number_of_magnetic_atoms").get_to(num_mag_atoms);
    json_file.at("max_sphere_num").get_to(max_sphere_num);
    json_file.at("central_atom").get_to(central_atom);
    json_file.at("spin").get_to(spin);
    json_file.at("ncol").get_to(ncol);
    json_file.at("nrow").get_to(nrow);
    json_file.at("smearing").get_to(smearing);
    json_file.at("e_low").get_to(e_low);
    json_file.at("e_fermi").get_to(e_fermi);

    if (central_atom < 0 || central_atom >= num_mag_atoms)
    {
        std::cout
            << "ERROR! central_atom must be in the range 0 <= input < "
            << num_mag_atoms << std::endl;
        return 0;
    }

    std::vector<std::vector<double>> positions(
        num_mag_atoms, std::vector<double>(
                           3));

    std::vector<int> mag_orbs(num_mag_atoms);

    // read matrices from json file

    i = 0;
    for (auto &element : json_file["kmesh"])
    {
        kmesh[i] = element;
        i++;
    }

    i = 0;
    for (auto &element : json_file["orbitals_of_magnetic_atoms"])
    {
        mag_orbs[i] = element;
        i++;
    }

    i = 0;
    for (auto &element : json_file["exchange_for_specific_atoms"])
    {
        specific[i] = element;
        i++;
    }

    specific_true = false;
    for (i = 0; i < 4; i++)
    {
        if (specific[i] != 0)
        {
            specific_true = true;
            break;
        }
    }

    if (specific[0] < 0 || specific[0] >= num_mag_atoms)
    {
        std::cout
            << "ERROR! <exchange_for_specific_atoms: [0]> must be in the range 0 <= input < "
            << num_mag_atoms << std::endl;
        return 0;
    }

    num_kpoints = kmesh[0] * kmesh[1] * kmesh[2];

    i = 0;
    for (auto &element : json_file["cell_vectors"])
    {
        j = 0;
        for (auto &element1 : element)
        {
            cell_vec[i][j] = element1;
            j++;
        }
        i++;
    }

    i = 0;
    for (auto &element : json_file["positions_of_magnetic_atoms"])
    {
        j = 0;
        for (auto &element1 : element)
        {
            positions[i][j] = element1;
            j++;
        }
        i++;
    }

    // Hamiltonian in real space
    double ******Ham_R = new double *****[2];
    for (i = 0; i < 2; i++)
    {
        Ham_R[i] = new double ****[n_size[0]];
        for (j = 0; j < n_size[0]; j++)
        {
            Ham_R[i][j] = new double ***[n_size[1]];
            for (k = 0; k < n_size[1]; k++)
            {
                Ham_R[i][j][k] = new double **[n_size[2]];
                for (x = 0; x < n_size[2]; x++)
                {
                    Ham_R[i][j][k][x] = new double *[num_orb];
                    for (y = 0; y < num_orb; y++)
                    {
                        Ham_R[i][j][k][x][y] = new double[num_orb];
                    }
                }
            }
        }
    }

    // Hamiltonian in reciprocal space
    std::complex<double> ****Ham_K = new std::complex<double> ***[2];
    for (z = 0; z < 2; z++)
    {
        Ham_K[z] = new std::complex<double> **[num_kpoints];
        for (i = 0; i < num_kpoints; i++)
        {
            Ham_K[z][i] = new std::complex<double> *[num_orb];
            for (j = 0; j < num_orb; j++)
            {
                Ham_K[z][i][j] = new std::complex<double>[num_orb];
            }
        }
    }

    // k-vectors for BZ integration
    double **k_vec = new double *[num_kpoints];
    for (z = 0; z < num_kpoints; z++)
    {
        k_vec[z] = new double[3];
    }

    up.clear();
    up.seekg(0, std::ios::beg);

    while (!up.eof())
    {
        up >> vecs[0] >> vecs[1] >> vecs[2] >> orbs[0] >> orbs[1] >> ham_values[0] >> ham_values[1];

        Ham_R[0][vecs[0] - n_min[0]][vecs[1] - n_min[1]][vecs[2] - n_min[2]][orbs[0] - 1][orbs[1] - 1] = ham_values[0];
    }
    std::cout << "File spin_up.dat was  scanned  successfully" << std::endl;

    up.close();

    while (!dn.eof())
    {
        dn >> vecs[0] >> vecs[1] >> vecs[2] >> orbs[0] >> orbs[1] >> ham_values[0] >> ham_values[1];

        Ham_R[1][vecs[0] - n_min[0]][vecs[1] - n_min[1]][vecs[2] - n_min[2]][orbs[0] - 1][orbs[1] - 1] = ham_values[0];
    }
    std::cout << "File spin_dn.dat was  scanned  successfully" << std::endl;

    dn.close();

    //===============================================================================
    // reciprocal vectors and and kmesh for integration

    vol = cell_vec[0][0] * (cell_vec[1][1] * cell_vec[2][2] - cell_vec[1][2] * cell_vec[2][1]) -
          cell_vec[0][1] * (cell_vec[1][0] * cell_vec[2][2] - cell_vec[1][2] * cell_vec[2][0]) +
          cell_vec[0][2] * (cell_vec[1][0] * cell_vec[2][1] - cell_vec[1][1] * cell_vec[2][0]); // volume

    rec_vec[0][0] = (2 * M_PI / vol) * (cell_vec[1][1] * cell_vec[2][2] - cell_vec[1][2] * cell_vec[2][1]);
    rec_vec[0][1] = (2 * M_PI / vol) * (cell_vec[1][2] * cell_vec[2][0] - cell_vec[1][0] * cell_vec[2][2]);
    rec_vec[0][2] = (2 * M_PI / vol) * (cell_vec[1][0] * cell_vec[2][1] - cell_vec[1][1] * cell_vec[2][0]);

    rec_vec[1][0] = (2 * M_PI / vol) * (cell_vec[2][1] * cell_vec[0][2] - cell_vec[2][2] * cell_vec[0][1]);
    rec_vec[1][1] = (2 * M_PI / vol) * (cell_vec[2][2] * cell_vec[0][0] - cell_vec[2][0] * cell_vec[0][2]);
    rec_vec[1][2] = (2 * M_PI / vol) * (cell_vec[2][0] * cell_vec[0][1] - cell_vec[2][1] * cell_vec[0][0]);

    rec_vec[2][0] = (2 * M_PI / vol) * (cell_vec[0][1] * cell_vec[1][2] - cell_vec[0][2] * cell_vec[1][1]);
    rec_vec[2][1] = (2 * M_PI / vol) * (cell_vec[0][2] * cell_vec[1][0] - cell_vec[0][0] * cell_vec[1][2]);
    rec_vec[2][2] =
        (2 * M_PI / vol) * (cell_vec[0][0] * cell_vec[1][1] - cell_vec[0][1] * cell_vec[1][0]); // reciprocal vectors

    e = 0;
    for (i = 0; i < kmesh[0]; i++)
    {
        for (j = 0; j < kmesh[1]; j++)
        {
            for (k = 0; k < kmesh[2]; k++)
            {
                for (z = 0; z < 3; z++)
                {

                    k_vec[e][z] = (i * rec_vec[0][z]) / kmesh[0] + (j * rec_vec[1][z]) / kmesh[1] +
                                  (k * rec_vec[2][z]) / kmesh[2];
                }
                e++;
            }
        }
    }

    //===============================================================================
    // prepare the energy contour for integration

    ntot = ncol + 2 * nrow;
    de = std::complex<double>((e_fermi - e_low) / ncol, smearing / nrow);

    std::complex<double> *E = new std::complex<double>[ntot];
    std::complex<double> *dE = new std::complex<double>[ntot];

    x = 0;
    y = 1;

    num = 0;
    e_const = std::complex<double>(e_low, 0);

    for (i = 0; i < ntot; i++)
    {
        if (i == nrow)
        {
            x = 1;
            y = 0;
            num = 0;
            e_const = std::complex<double>(e_low, smearing);
        }
        if (i == nrow + ncol)
        {
            x = 0;
            y = -1;
            num = 0;
            e_const = std::complex<double>(e_fermi, smearing);
        }

        E[i] = e_const + std::complex<double>(num * x * (de).real(), num * y * (de).imag());
        dE[i] = std::complex<double>(x * (de).real(), y * (de).imag());
        num++;
    }

    //===============================================================================
    // print scanned data
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "Parameters of integration:" << std::endl;
    std::cout << "Lowest energy of integration: " << e_low << "; Fermi energy: " << e_fermi << std::endl;
    std::cout << "Smearing: " << smearing << std::endl;
    std::cout << "Vertical steps:  " << nrow << "; Horizontal steps: " << ncol << std::endl;
    std::cout << "Number of k-points: " << num_kpoints << std::endl;
    std::cout << "Total number of Wannier functions: " << num_orb << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "Crystal structure of system:" << std::endl;
    std::cout << "crystal axes: (cart. coord. in units of alat)" << std::endl;
    for (i = 0; i < 3; i++)
    {
        std::cout << std::showpoint << cell_vec[i][0] << "   " << cell_vec[i][1] << "   " << cell_vec[i][2]
                  << std::endl;
    }

    std::cout << std::endl;
    std::cout << "reciprocal axes: (cart. coord. in units 1/alat)" << std::endl;
    for (i = 0; i < 3; i++)
    {
        std::cout << std::showpoint << rec_vec[i][0] << "   " << rec_vec[i][1] << "   " << rec_vec[i][2] << std::endl;
    }

    std::cout << std::endl
              << "atomic positions"
              << "\t"
              << "orbitals" << std::endl;
    for (i = 0; i < num_mag_atoms; i++)
    {
        std::cout << std::showpoint << positions[i][0] << "   " << positions[i][1] << "   " << positions[i][2] << "\t" << mag_orbs[i]
                  << std::endl;
    }

    if (specific_true)
    {
        std::cout << std::endl
                  << "Exchange coupling will be calculated only between Atom " << central_atom
                  << "(000)<-->Atom " << specific[0] << "(" << specific[1] << specific[2] << specific[3]
                  << ")." << std::endl;
        std::cout << "To calculate exchange couplings between all atoms set <exchange_for_specific_atoms>: [0, 0, 0, 0] in <in.json> file"
                  << std::endl;
    }
    else
    {
        std::cout << std::endl
                  << "Exchange couplings will be calculated between all atoms around " << central_atom << std::endl;
        std::cout << "within coordination spheres smaller than # " << max_sphere_num << std::endl;
    }

    std::cout << "---------------------------------------------------------------------" << std::endl;

    //===============================================================================
    // Fourier transformation  of Hamiltonian

    calc_hamK(num_orb, num_kpoints, n_min, n_max, cell_vec, k_vec, Ham_R, Ham_K);

    std::cout << "Fourier transformation  of Hamiltonian is completed" << std::endl;

    //===============================================================================
    // Exchange couplings and occupation matrices

    if (specific_true)
    {
        index_temp[0] = specific[1]; // i
        index_temp[1] = specific[2]; // j
        index_temp[2] = specific[3]; // k
        index_temp[3] = specific[0]; // atom 2

        matrix_trace = 0;

        std::vector<std::vector<double>> exchange(
            mag_orbs[central_atom], std::vector<double>(
                                        mag_orbs[central_atom]));

        for (x = 0; x < 3; x++)
        {
            r[x] = specific[1] * cell_vec[0][x] + specific[2] * cell_vec[1][x] + specific[3] * cell_vec[2][x] +
                   (positions[specific[0]][x] - positions[central_atom][x]);
        }

        rad_specific = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);

        std::cout << "=====================================================================" << std::endl;
        std::cout << "Atom " << central_atom << "(000)<-->Atom " << specific[0] << "(" << specific[1] << specific[2]
                  << specific[3] << ") with radius " << rad_specific << std::endl;

        err = false;
        calc_exchange(central_atom, index_temp, num_orb, num_kpoints, n_max, ntot, spin, cell_vec, k_vec, E, dE,
                      Ham_R, Ham_K, mag_orbs, exchange, err);

        if (err)
        {
            std::cout << "ERROR! Problem with inversion! Please, check the input parameters and Hamiltonian!"
                      << std::endl;
            return 0;
        }

        for (x = 0; x < mag_orbs[central_atom]; x++)
        {
            matrix_trace += exchange[x][x];
        }

        std::cout << std::fixed;

        for (x = 0; x < mag_orbs[central_atom]; x++)
        {
            for (y = 0; y < mag_orbs[central_atom]; y++)
            {
                std::cout << std::setprecision(6) << exchange[x][y] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Trace equals to: " << matrix_trace << " eV" << std::endl;
        std::cout << std::endl;
    }
    else
    {

        num_points = num_mag_atoms * (n_size[0] * n_size[1] * n_size[2]);

        // index = [i, j, k, atom2]
        std::vector<std::vector<int>> index(
            num_points, std::vector<int>(
                            4));

        std::vector<double> radius(num_points);

        //'central_atom' is the index of the atom1

        std::vector<std::vector<std::vector<double>>> occ(
            2, std::vector<std::vector<double>>(
                   mag_orbs[central_atom], std::vector<double>(
                                               mag_orbs[central_atom])));

        std::vector<std::vector<double>> exchange(
            mag_orbs[central_atom], std::vector<double>(
                                        mag_orbs[central_atom]));

        coordination_sort(central_atom, num_mag_atoms, n_min, n_max, cell_vec, positions, radius, index);

        std::cout << "=====================================================================" << std::endl;

        neighbor_num = 1;
        sphere_num = 0;

        for (p = 0; p < num_points; p++)
        {

            if (p == 0)
            {
                // in case of radius = 0 we calculate occupation matrices
                err = false;
                calc_occupation(central_atom, num_orb, num_kpoints, ntot, Ham_K, E, dE, mag_orbs, occ, err);

                if (err)
                {
                    std::cout << "ERROR! Problem with inversion! Please, check the input parameters and Hamiltonian!"
                              << std::endl;
                    return 0;
                }

                matrix_trace = 0;

                for (x = 0; x < mag_orbs[central_atom]; x++)
                {
                    matrix_trace += (occ[0][x][x] - occ[1][x][x]);
                }

                std::cout << "Occupation matrix (N_up - N_dn) for atom " << central_atom << std::endl;
                std::cout << std::fixed;
                for (x = 0; x < mag_orbs[central_atom]; x++)
                {
                    for (y = 0; y < mag_orbs[central_atom]; y++)
                    {
                        std::cout << std::setprecision(3) << occ[0][x][y] - occ[1][x][y] << " ";
                    }
                    std::cout << std::endl;
                }

                std::cout << "Trace equals to: " << matrix_trace << std::endl;
                std::cout << std::endl;
            }
            else
            {

                if (fabs(radius[p - 1] - radius[p]) < 1E-4)
                    neighbor_num++;
                else
                {
                    neighbor_num = 1;
                    sphere_num++;
                }

                if (sphere_num == max_sphere_num)
                    break;

                for (z = 0; z < 4; z++)
                {
                    index_temp[z] = index[p][z];
                }

                std::cout << "Atom " << central_atom << "(000)<-->Atom " << index[p][3] << "(" << index[p][0]
                          << index[p][1]
                          << index[p][2] << ") in sphere #" << sphere_num << " with radius " << radius[p] << " is #"
                          << neighbor_num
                          << ":  " << std::endl;

                matrix_trace = 0;

                for (x = 0; x < mag_orbs[central_atom]; x++)
                {
                    for (y = 0; y < mag_orbs[central_atom]; y++)
                    {
                        exchange[x][y] = 0;
                    }
                }

                err = false;
                calc_exchange(central_atom, index_temp, num_orb, num_kpoints, n_max, ntot, spin, cell_vec, k_vec, E, dE,
                              Ham_R, Ham_K, mag_orbs, exchange, err);

                if (err)
                {
                    std::cout << "ERROR! Problem with inversion! Please, check the input parameters and Hamiltonian!"
                              << std::endl;
                    return 0;
                }

                for (x = 0; x < mag_orbs[central_atom]; x++)
                {
                    matrix_trace += exchange[x][x];
                }

                std::cout << std::fixed;

                for (x = 0; x < mag_orbs[central_atom]; x++)
                {
                    for (y = 0; y < mag_orbs[central_atom]; y++)
                    {
                        std::cout << std::setprecision(6) << exchange[x][y] << " ";
                    }
                    std::cout << std::endl;
                }

                std::cout << "Trace equals to: " << matrix_trace << " eV" << std::endl;
                std::cout << std::endl;
            }
        }
    }

    delete[] Ham_R;
    delete[] Ham_K;
    delete[] k_vec;
    delete[] E;
    delete[] dE;

    td = time(NULL);
    std::cout << std::endl
              << "This run was terminated on: " << std::endl;
    std::cout << ctime(&td) << std::endl;

    std::cout << "JOB DONE" << std::endl;
    std::cout << "=====================================================================" << std::endl;
}
