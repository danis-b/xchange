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
    int i, j, k, x, y, z, e;
    int num_mag_atoms, num_orb, num_kpoints, num_qpoints;
    int n_size[3], n_min[3], n_max[3], vecs[3], orbs[2], kmesh[3], qmesh[3];

    double spin, vol, matrix_trace;
    double cell_vec[3][3], rec_vec[3][3], ham_values[2], r[3], q_vec_temp[3];

    bool err;

    // integration parameters
    int ncol, nrow, ntot, num;
    double smearing, e_low, e_fermi;
    std::complex<double> de, e_const;

    td = time(NULL);
    std::cout << "Program xchange.x v.4.0 (q-version) starts on ";
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
    json_file.at("spin").get_to(spin);
    json_file.at("ncol").get_to(ncol);
    json_file.at("nrow").get_to(nrow);
    json_file.at("smearing").get_to(smearing);
    json_file.at("e_low").get_to(e_low);
    json_file.at("e_fermi").get_to(e_fermi);

    std::vector<int> mag_orbs(num_mag_atoms);

    // read matrices from json file

    i = 0;
    for (auto &element : json_file["kmesh"])
    {
        kmesh[i] = element;
        i++;
    }

    i = 0;
    for (auto &element : json_file["qmesh"])
    {
        qmesh[i] = element;
        i++;
    }

    i = 0;
    for (auto &element : json_file["orbitals_of_magnetic_atoms"])
    {
        mag_orbs[i] = element;
        i++;
    }

    num_kpoints = kmesh[0] * kmesh[1] * kmesh[2];
    num_qpoints = qmesh[0] * qmesh[1] * qmesh[2];

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

    // q-vectors for output
    double **q_vec = new double *[num_qpoints];
    for (z = 0; z < num_qpoints; z++)
    {
        q_vec[z] = new double[3];
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

    e = 0;
    for (i = 0; i < qmesh[0]; i++)
    {
        for (j = 0; j < qmesh[1]; j++)
        {
            for (k = 0; k < qmesh[2]; k++)
            {
                for (z = 0; z < 3; z++)
                {

                    q_vec[e][z] = (i * rec_vec[0][z]) / kmesh[0] + (j * rec_vec[1][z]) / kmesh[1] +
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
    std::cout << "Number of q-points: " << num_qpoints << std::endl;
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

    std::cout << "---------------------------------------------------------------------" << std::endl;

    //===============================================================================
    // Fourier transformation  of Hamiltonian

    calc_hamK(num_orb, num_kpoints, n_min, n_max, cell_vec, k_vec, Ham_R, Ham_K);

    std::cout << "Fourier transformation  of Hamiltonian is completed" << std::endl;

    //===============================================================================
    // Exchange couplings and occupation matrices

    for (e = 0; e < num_qpoints; e++)
    {
        for (z = 0; z < 3; z++)
        {
            q_vec_temp[x] = q_vec[3];
        }

        for (x = 0; x < num_mag_atoms; ++x)
        {
            for (y = x; y < num_mag_atoms; ++y)
            {

                err = false;
                calc_exchange(q_vec_temp, num_orb, num_kpoints, n_max, ntot, spin, cell_vec, k_vec, E, dE,
                              Ham_R, Ham_K, mag_orbs, exchange, err);

                if (err)
                {
                    std::cout << "ERROR! Problem with inversion! Please, check the input parameters and Hamiltonian!"
                              << std::endl;
                    return 0;
                }
            }
        }
    }

    delete[] Ham_R;
    delete[] Ham_K;
    delete[] k_vec;
    delete[] q_vec;
    delete[] E;
    delete[] dE;

    td = time(NULL);
    std::cout << std::endl
              << "This run was terminated on: " << std::endl;
    std::cout << ctime(&td) << std::endl;

    std::cout << "JOB DONE" << std::endl;
    std::cout << "=====================================================================" << std::endl;
}
