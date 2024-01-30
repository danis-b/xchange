#include <complex>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include "functions.h"
#include <omp.h> 
#include <nlohmann/json.hpp> //we use json format for input file

using json = nlohmann::json;

int main()
{

    omp_set_num_threads(8);
    
    time_t td;
    int idx, idx_x, idx_y;

    int central_atom, max_sphere_num;
    int num_mag_atoms, num_orb, num_kpoints, num_points;
    int n_size[3], n_min[3], n_max[3], vecs[3], orbs[2], kmesh[3], specific[4];

    double spin, vol;
    double cell_vec[3][3], rec_vec[3][3], ham_values[2];

    bool specific_true;

    // integration parameters
    int ncol, nrow, ntot;
    double smearing, e_low, e_fermi;
    std::complex<double> de, e_const;

    td = time(NULL);
    std::cout << "Program xchange.x v.3.0 (c++) starts on ";
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
    for (int i = 0; i < 3; ++i)
    {
        n_max[i] = n_min[i] = 0;
    }

    num_orb = 1;
    while (!up.eof())
    {
        up >> vecs[0] >> vecs[1] >> vecs[2] >> orbs[0] >> orbs[1] >> ham_values[0] >> ham_values[1];

        if (orbs[0] > num_orb)
            num_orb = orbs[0];

        for (int i = 0; i < 3; ++i)
        {
            if (n_max[i] < vecs[i])
                n_max[i] = vecs[i];
            else if (n_min[i] > vecs[i])
                n_min[i] = vecs[i];
        }
    }

    for (int i = 0; i < 3; ++i)
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

    idx = 0;
    for (auto &element : json_file["kmesh"])
    {
        kmesh[idx] = element;
        idx++;
    }

    idx = 0;
    for (auto &element : json_file["orbitals_of_magnetic_atoms"])
    {
        mag_orbs[idx] = element;
        idx++;
    }

    idx = 0;
    for (auto &element : json_file["exchange_for_specific_atoms"])
    {
        specific[idx] = element;
        idx++;
    }

    specific_true = false;
    for (int i = 0; i < 4; ++i)
    {
        if (specific[i] != 0)
        {
            specific_true = true;
            break;
        }
    }

    if (specific[3] < 0 || specific[3] >= num_mag_atoms)
    {
        std::cout
            << "ERROR! <exchange_for_specific_atoms: [0]> must be in the range 0 <= input < "
            << num_mag_atoms << std::endl;
        return 0;
    }

    num_kpoints = kmesh[0] * kmesh[1] * kmesh[2];

    idx = 0;
    for (auto &element : json_file["cell_vectors"])
    {
        int j = 0;
        for (auto &element1 : element)
        {
            cell_vec[idx][j] = element1;
            j++;
        }
        idx++;
    }

    idx = 0;
    for (auto &element : json_file["positions_of_magnetic_atoms"])
    {
        int j = 0;
        for (auto &element1 : element)
        {
            positions[idx][j] = element1;
            j++;
        }
        idx++;
    }

    // Hamiltonian in real space
    double ******Ham_R = new double *****[2];
    for (int i = 0; i < 2; ++i)
    {
        Ham_R[i] = new double ****[n_size[0]];
        for (int j = 0; j < n_size[0]; ++j)
        {
            Ham_R[i][j] = new double ***[n_size[1]];
            for (int k = 0; k < n_size[1]; ++k)
            {
                Ham_R[i][j][k] = new double **[n_size[2]];
                for (int x = 0; x < n_size[2]; ++x)
                {
                    Ham_R[i][j][k][x] = new double *[num_orb];
                    for (int y = 0; y < num_orb; ++y)
                    {
                        Ham_R[i][j][k][x][y] = new double[num_orb];
                    }
                }
            }
        }
    }

    // Hamiltonian in reciprocal space
    std::complex<double> ****Ham_K = new std::complex<double> ***[2];
    for (int z = 0; z < 2; ++z)
    {
        Ham_K[z] = new std::complex<double> **[num_kpoints];
        for (int i = 0; i < num_kpoints; ++i)
        {
            Ham_K[z][i] = new std::complex<double> *[num_orb];
            for (int j = 0; j < num_orb; ++j)
            {
                Ham_K[z][i][j] = new std::complex<double>[num_orb];
            }
        }
    }

    // k-vectors for BZ integration
    std::vector<std::array<double,3> > k_vec(num_kpoints);

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

    // reciprocal vectors
    rec_vec[0][0] = (2 * M_PI / vol) * (cell_vec[1][1] * cell_vec[2][2] - cell_vec[1][2] * cell_vec[2][1]);
    rec_vec[0][1] = (2 * M_PI / vol) * (cell_vec[1][2] * cell_vec[2][0] - cell_vec[1][0] * cell_vec[2][2]);
    rec_vec[0][2] = (2 * M_PI / vol) * (cell_vec[1][0] * cell_vec[2][1] - cell_vec[1][1] * cell_vec[2][0]);

    rec_vec[1][0] = (2 * M_PI / vol) * (cell_vec[2][1] * cell_vec[0][2] - cell_vec[2][2] * cell_vec[0][1]);
    rec_vec[1][1] = (2 * M_PI / vol) * (cell_vec[2][2] * cell_vec[0][0] - cell_vec[2][0] * cell_vec[0][2]);
    rec_vec[1][2] = (2 * M_PI / vol) * (cell_vec[2][0] * cell_vec[0][1] - cell_vec[2][1] * cell_vec[0][0]);

    rec_vec[2][0] = (2 * M_PI / vol) * (cell_vec[0][1] * cell_vec[1][2] - cell_vec[0][2] * cell_vec[1][1]);
    rec_vec[2][1] = (2 * M_PI / vol) * (cell_vec[0][2] * cell_vec[1][0] - cell_vec[0][0] * cell_vec[1][2]);
    rec_vec[2][2] = (2 * M_PI / vol) * (cell_vec[0][0] * cell_vec[1][1] - cell_vec[0][1] * cell_vec[1][0]);

    idx = 0;
    for (int i = 0; i < kmesh[0]; ++i)
    {
        for (int j = 0; j < kmesh[1]; ++j)
        {
            for (int k = 0; k < kmesh[2]; ++k)
            {
                for (int z = 0; z < 3; ++z)
                {
                    k_vec[idx][z] = (i * rec_vec[0][z]) / kmesh[0] + (j * rec_vec[1][z]) / kmesh[1] +
                                    (k * rec_vec[2][z]) / kmesh[2];
                }
                idx++;
            }
        }
    }

    //===============================================================================
    // prepare the energy contour for integration

    ntot = ncol + 2 * nrow;
    de = std::complex<double>((e_fermi - e_low) / ncol, smearing / nrow);

    std::vector<std::complex<double>> E;
    E.reserve(ntot);

    std::vector<std::complex<double>> dE;
    dE.reserve(ntot);

    idx_x = 0;
    idx_y = 1;

    idx = 0;
    e_const = std::complex<double>(e_low, 0);

    for (int i = 0; i < ntot; ++i)
    {
        if (i == nrow)
        {
            idx_x = 1;
            idx_y = 0;
            idx = 0;
            e_const = std::complex<double>(e_low, smearing);
        }
        if (i == nrow + ncol)
        {
            idx_x = 0;
            idx_y = -1;
            idx = 0;
            e_const = std::complex<double>(e_fermi, smearing);
        }

        E[i] = e_const + std::complex<double>(idx * idx_x * (de).real(), idx * idx_y * (de).imag());
        dE[i] = std::complex<double>(idx_x * (de).real(), idx_y * (de).imag());
        idx++;
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
    for (int i = 0; i < 3; ++i)
    {
        std::cout << std::showpoint << cell_vec[i][0] << "   " << cell_vec[i][1] << "   " << cell_vec[i][2]
                  << std::endl;
    }

    std::cout << std::endl;
    std::cout << "reciprocal axes: (cart. coord. in units 1/alat)" << std::endl;
    for (int i = 0; i < 3; ++i)
    {
        std::cout << std::showpoint << rec_vec[i][0] << "   " << rec_vec[i][1] << "   " << rec_vec[i][2] << std::endl;
    }

    std::cout << std::endl
              << "atomic positions"
              << "\t"
              << "orbitals" << std::endl;
    for (int i = 0; i < num_mag_atoms; ++i)
    {
        std::cout << std::showpoint << positions[i][0] << "   " << positions[i][1] << "   " << positions[i][2] << "\t" << mag_orbs[i]
                  << std::endl;
    }

    if (specific_true)
    {
        std::cout << std::endl
                  << "Exchange coupling will be calculated only between Atom " << central_atom
                  << "(000)<-->Atom " << specific[3] << "(" << specific[0] << specific[1] << specific[2]
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

    num_points = num_mag_atoms * (n_size[0] * n_size[1] * n_size[2]);

    // index = [i, j, k, atom2]
    std::vector<std::vector<int>> index(
        num_points, std::vector<int>(
                        4));

    std::vector<double> radius(num_points);

    //'central_atom' is the index of the atom1

    std::vector<double> occ;
    occ.reserve(2 * mag_orbs[central_atom] * mag_orbs[central_atom]);

    std::vector<std::complex<double>> Ham_K_numpy;
    Ham_K_numpy.reserve(2 * num_kpoints * num_orb * num_orb);

    for (int z = 0; z < 2; ++z)
    {
        for (int e = 0; e < num_kpoints; ++e)
        {
            for (int x = 0; x < num_orb; ++x)
            {
                for (int y = 0; y < num_orb; ++y)
                {

                    Ham_K_numpy.push_back(Ham_K[z][e][x][y]);
                }
            }
        }
    }

    td = time(NULL);
    std::cout << ctime(&td) << std::endl;
    occ = calc_occupation(central_atom, num_orb, num_kpoints, ntot, Ham_K_numpy, E, dE, mag_orbs);
    td = time(NULL);
    std::cout << ctime(&td) << std::endl;

    

    std::cout <<std::fixed;
    for (int z = 0; z < 2; ++z)
    {
        for (int x = 0; x < mag_orbs[central_atom]; ++x)
        {
            for (int y = 0; y < mag_orbs[central_atom]; ++y)
            {
                std::cout << std::setprecision(3) << occ[z * (mag_orbs[central_atom] * mag_orbs[central_atom]) + mag_orbs[central_atom] * x + y] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // std::vector<int> specific_numpy(4);
    // for(int i = 0; i < 4; ++i)
    // {
    //     specific_numpy[i] = specific[i]; 
    // }


    // std::vector<double> cell_vec_numpy(9);
    // for(int i = 0; i < 3; ++i)
    // {
    //     for(int j = 0; j < 3; j++)
    //     {
    //         cell_vec_numpy[3 * i + j] = cell_vec[i][j];

    //     }
    // }


    // std::vector<double> k_vec_numpy;
    // k_vec_numpy.reserve(3 * num_kpoints);


    // for(int i = 0; i < num_kpoints; ++i)
    // {
    //     for(int j = 0; j < 3; ++j)
    //     {
    //        k_vec_numpy[3 * i + j] =  k_vec[i][j];
    //     }
    // }





    // std::vector<double> exchange;
    // exchange.reserve( mag_orbs[central_atom] * mag_orbs[central_atom]);

    // td = time(NULL);
    // std::cout << ctime(&td) << std::endl;
    // exchange = calc_exchange(central_atom, specific_numpy,  num_orb, num_kpoints, ntot, spin, cell_vec_numpy, k_vec_numpy, Ham_K_numpy, E, dE, mag_orbs); 
    // td = time(NULL);
    // std::cout << ctime(&td) << std::endl;


    // std::cout <<std::fixed;
    //     for (int x = 0; x < mag_orbs[central_atom]; ++x)
    //     {
    //         for (int y = 0; y < mag_orbs[central_atom]; ++y)
    //         {
    //             std::cout << std::setprecision(6) << exchange[mag_orbs[central_atom] * x + y] << " ";
    //         }
    //         std::cout << std::endl;

    //     }



    delete[] Ham_R;
    delete[] Ham_K;
}
