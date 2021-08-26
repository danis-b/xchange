#include <complex>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include "functions.h"
#include <nlohmann/json.hpp> //we use json format for input file

using json = nlohmann::json;



int main() {
    time_t td;
    int i, j, k, x, y, z, e, p, atom, sphere_num, neighbor_num, max_sphere_num;
    int num_mag_atoms, num_orb, num_orb2, num_kpoints, num_points;
    int n_size[3], n_min[3], n_max[3], vecs[3], orbs[2], kmesh[3], specific[5], index_temp[4];

    double spin, vol, ca, sa, matrix_trace, rad_specific;
    double cell_vec[3][3], rec_vec[3][3], ham_values[2], r[3];
    double exchange[5][5], occ[2][5][5], mag_mom[5][5];

    std::complex<double> phase;

    bool specific_true;

    //integration parameters
    int ncol, nrow, ntot, num;
    double  smearing, e_low, e_fermi;
    std::complex<double> de, e_const;

    td = time(NULL);
    std::cout << "Program Xchange.x v.3.0 starts on ";
    std::cout << ctime(&td);
    std::cout << "=====================================================================" << std::endl;


    std::ifstream up;
    up.open("spin_up.dat");
    if (!up) {
        std::cout << "ERROR!Cannot open file <spin_up.dat>!" << std::endl;
        return 0;
    }

    std::ifstream dn;
    dn.open("spin_dn.dat");
    if (!dn) {
        std::cout << "ERROR!Cannot open file <spin_dn.dat>!" << std::endl;
        return 0;
    }


    //define the possible ranges for n_min and n_max from .dat files
    for (i = 0; i < 3; i++) {
        n_max[i] = n_min[i] = 0;
    }

    while (!up.eof()) {
        up >> vecs[0] >> vecs[1] >> vecs[2] >> orbs[0] >> orbs[1] >> ham_values[0] >> ham_values[1];

        for (i = 0; i < 3; i++) {
            if (n_max[i] < vecs[i]) n_max[i] = vecs[i];
            else if (n_min[i] > vecs[i]) n_min[i] = vecs[i];
        }

    }


    for (i = 0; i < 3; i++) {
        n_size[i] = n_max[i] - n_min[i] + 1; // Plus 1 for 0th
    }


    //===============================================================================
    //read information from input files and matrix initialization

    std::ifstream in;
    in.open("in.json");
    if (!in) {
        std::cout << "ERROR! Cannot open input file  in.json" << std::endl;
        return 0;
    }


    json json_file;
    in >> json_file;

    in.close();


    json_file.at("number_of_magnetic_atoms").get_to(num_mag_atoms);
    json_file.at("number_of_wannier_functions").get_to(num_orb);
    json_file.at("max_sphere_num").get_to(max_sphere_num);
    json_file.at("spin").get_to(spin);
    json_file.at("ncol").get_to(ncol);
    json_file.at("nrow").get_to(nrow);
    json_file.at("smearing").get_to(smearing);
    json_file.at("e_low").get_to(e_low);
    json_file.at("e_fermi").get_to(e_fermi);



    std::vector <std::vector<double>> positions(
            num_mag_atoms, std::vector<double>(
                    3)
    );




    // Hamiltonian in real space
    std::vector < std::vector < std::vector < std::vector < std::vector < std::vector < double > > >> >> Ham_R (
            2, std::vector < std::vector < std::vector < std::vector < std::vector < double >> >> > (
                    n_size[0], std::vector < std::vector < std::vector < std::vector < double >> >> (
                            n_size[1], std::vector < std::vector < std::vector < double >> > (
                                    n_size[2], std::vector < std::vector < double >> (
                                            num_orb, std::vector< double >(
                                                    num_orb)
                                    )
                            )
                    )
            )
    );


    // read matrices from json file

    i = 0;
    for (auto &element : json_file["kmesh"]) {
        kmesh[i] = element;
        i++;
    }


    i = 0;
    for (auto &element : json_file["exchange_for_specific_atoms"]) {
        specific[i] = element;
        i++;
    }

    specific_true = false;
    for (i = 0; i < 5; i++) {
        if (specific[i] != 0) {
            specific_true = true;
            break;
        }
    }


    for (i = 0; i < 2; i++) {
        if (specific[i] < 0 || specific[i] >= num_mag_atoms) {
            std::cout
                    << "ERROR! <exchange_for_specific_atoms: [1, 2]> must be in the range 0 < input < "
                    << num_mag_atoms << std::endl;
            return 0;
        }
    }






    num_kpoints = kmesh[0]*kmesh[1]*kmesh[2];

    i = 0;
    for (auto &element : json_file["cell_vectors"]) {
        j = 0;
        for (auto &element1 : element) {
            cell_vec[i][j] = element1;
            j++;
        }
        i++;
    }


    i = 0;
    for (auto &element : json_file["positions_of_magnetic_atoms"]) {
        j = 0;
        for (auto &element1 : element) {
            positions[i][j] = element1;
            j++;
        }
        i++;
    }



    // Hamiltonian in reciprocal space
    std::vector < std::vector < std::vector < std::complex<double> >>>  Ham_K(
            2, std::vector < std::vector < std::complex<double> >> (
                    num_orb, std::vector< std::complex<double> >(
                            num_orb)
            )
    );


    // Eigenvectors in reciprocal space
    std::vector < std::vector < std::vector < std::vector < std::complex<double> >> >> egvec(
            2, std::vector < std::vector < std::vector < std::complex<double> >> > (
                    num_kpoints, std::vector < std::vector < std::complex<double> >> (
                            num_orb, std::vector< std::complex<double> >(
                                    num_orb)
                    )
            )
    );


    // Eigenvalues in reciprocal space
    std::vector < std::vector < std::vector < double >>>  egval(
            2, std::vector < std::vector < double >> (
                    num_kpoints, std::vector< double >(
                            num_orb)
            )
    );


    // k-vectors for BZ integration
    std::vector <std::vector<double>> k_vec(
            num_kpoints, std::vector<double>(
                    3)
    );


    up.clear();
    up.seekg(0, std::ios::beg);



    while (!up.eof()) {
        up >> vecs[0] >> vecs[1] >> vecs[2] >> orbs[0] >> orbs[1] >> ham_values[0] >> ham_values[1];

        Ham_R[0][vecs[0] + n_max[0]][vecs[1] + n_max[1]][vecs[2] + n_max[2]][orbs[0] - 1][orbs[1] - 1] = ham_values[0];
    }
    std::cout << "File spin_up.dat was  scanned  successfully" << std::endl;

    up.close();


    while (!dn.eof()) {
        dn >> vecs[0] >> vecs[1] >> vecs[2] >> orbs[0] >> orbs[1] >> ham_values[0] >> ham_values[1];

        Ham_R[1][vecs[0] + n_max[0]][vecs[1] + n_max[1]][vecs[2] + n_max[2]][orbs[0] - 1][orbs[1] - 1] = ham_values[0];
    }
    std::cout << "File spin_dn.dat was  scanned  successfully" << std::endl;

    dn.close();

    //===============================================================================
    //lapack settings

    char JOBZ = 'V';
    char UPLO = 'U';
    int LWORK = 2 * num_orb;

    num_orb2 = num_orb*num_orb;


    double *W_up = new double [num_orb];
    double *W_dn = new double [num_orb];

    std::complex<double> *h_up = new std::complex<double> [num_orb2];
    std::complex<double> *h_dn = new std::complex<double> [num_orb2];

    std::complex<double> *WORK = new std::complex<double> [LWORK];
    double *RWORK = new double [3 * num_orb];
    int INFO;

    //===============================================================================
    // reciprocal vectors and and kmesh for integration

    vol = cell_vec[0][0] * (cell_vec[1][1] * cell_vec[2][2] - cell_vec[1][2] * cell_vec[2][1]) - cell_vec[0][1] * (cell_vec[1][0] * cell_vec[2][2] - cell_vec[1][2] * cell_vec[2][0]) +
            cell_vec[0][2] * (cell_vec[1][0] * cell_vec[2][1] - cell_vec[1][1] * cell_vec[2][0]);//volume


    rec_vec[0][0] = (2 * M_PI / vol) * (cell_vec[1][1] * cell_vec[2][2] - cell_vec[1][2] * cell_vec[2][1]);
    rec_vec[0][1] = (2 * M_PI / vol) * (cell_vec[1][2] * cell_vec[2][0] - cell_vec[1][0] * cell_vec[2][2]);
    rec_vec[0][2] = (2 * M_PI / vol) * (cell_vec[1][0] * cell_vec[2][1] - cell_vec[1][1] * cell_vec[2][0]);

    rec_vec[1][0] = (2 * M_PI / vol) * (cell_vec[2][1] * cell_vec[0][2] - cell_vec[2][2] * cell_vec[0][1]);
    rec_vec[1][1] = (2 * M_PI / vol) * (cell_vec[2][2] * cell_vec[0][0] - cell_vec[2][0] * cell_vec[0][2]);
    rec_vec[1][2] = (2 * M_PI / vol) * (cell_vec[2][0] * cell_vec[0][1] - cell_vec[2][1] * cell_vec[0][0]);

    rec_vec[2][0] = (2 * M_PI / vol) * (cell_vec[0][1] * cell_vec[1][2] - cell_vec[0][2] * cell_vec[1][1]);
    rec_vec[2][1] = (2 * M_PI / vol) * (cell_vec[0][2] * cell_vec[1][0] - cell_vec[0][0] * cell_vec[1][2]);
    rec_vec[2][2] = (2 * M_PI / vol) * (cell_vec[0][0] * cell_vec[1][1] - cell_vec[0][1] * cell_vec[1][0]);//reciprocal vectors

    e = 0;
    for (i = 0; i < kmesh[0]; i++) {
        for (j = 0; j < kmesh[1]; j++) {
            for (k = 0; k < kmesh[2]; k++) {
                for (z = 0; z < 3; z++) {

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

    std::vector<std::complex<double>> E(ntot);
    std::vector<std::complex<double>> dE(ntot);


    x = 0;
    y = 1;

    num = 0;
    e_const = std::complex<double>(e_low, 0);

    for (i = 0; i <= ntot; i++) {
        if (i == nrow) {
            x = 1;
            y = 0;
            num = 0;
            e_const = std::complex<double>(e_low, smearing);
        }
        if (i == nrow + ncol) {
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
    //print scanned data
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "Parameters of integration:" << std::endl;
    std::cout << "Lowest energy of integration: " << e_low << "; Fermi energy: " << e_fermi << std::endl;
    std::cout << "Smearing: " << smearing << std::endl;
    std::cout << "Vertical steps:  " << nrow << "; Horizontal steps: " << ncol << std::endl;
    std::cout << "number of k-points: " << num_kpoints << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "Crystal structure of system:" << std::endl;
    std::cout << "crystal axes: (cart. coord. in units of alat)" << std::endl;
    for (i = 0; i < 3; i++) { std::cout << std::showpoint << cell_vec[i][0] << "   " << cell_vec[i][1] << "   " << cell_vec[i][2] << std::endl; }

    std::cout << std::endl;
    std::cout << "reciprocal axes: (cart. coord. in units 1/alat)" << std::endl;
    for (i = 0; i < 3; i++) { std::cout << std::showpoint << rec_vec[i][0] << "   " << rec_vec[i][1] << "   " << rec_vec[i][2] << std::endl; }


    std::cout << std::endl << "atomic positions" << std::endl;
    for (i = 0; i < num_mag_atoms; i++) {
        std::cout << std::showpoint << positions[i][0] << "   " << positions[i][1] << "   " << positions[i][2] << std::endl;
    }


    if (specific_true) {
        std::cout << std::endl << "WARNING! Exchange coupling will be calculated only between Atom " << specific[0]
                  << "(000)<-->Atom " << specific[1] << "(" << specific[2] << specific[3] << specific[4]
                  << "). To calculate exchange couplings between all atoms set <exchange_for_specific_atoms>: [0, 0, 0, 0, 0] in <in.json> file"
                  << std::endl;
    }

    std::cout << "---------------------------------------------------------------------" << std::endl;



    //===============================================================================
    //Fourier transformation and diagonalization of Hamiltonian


    for (e = 0; e < num_kpoints; e++) {

        for (x = 0; x < num_orb; x++) {
            for (y = 0; y < num_orb; y++) {
                for (z = 0; z < 2; z++) {
                    Ham_K[z][x][y] = std::complex<double>(0, 0);
                }
            }
        }


        for (x = 0; x < num_orb; x++) {
            for (y = 0; y < num_orb; y++) {
                for (i = n_min[0]; i <= n_max[0]; i++) {
                    for (j = n_min[1]; j <= n_max[1]; j++) {
                        for (k = n_min[2]; k <= n_max[2]; k++) {
                            for (z = 0; z < 3; z++) {
                                r[z] = i * cell_vec[0][z] + j * cell_vec[1][z] + k * cell_vec[2][z];
                            }

                            ca = cos(k_vec[e][0] * r[0] + k_vec[e][1] * r[1] + k_vec[e][2] * r[2]);
                            sa = sin(k_vec[e][0] * r[0] + k_vec[e][1] * r[1] + k_vec[e][2] * r[2]);

                            phase = std::complex<double>(ca, -sa);

                            for (z = 0; z < 2; z++) {
                                Ham_K[z][x][y] +=
                                        phase * std::complex<double>(Ham_R[z][i + n_max[0]][j + n_max[1]][k +
                                                                                                          n_max[2]][x][y],
                                                                     0);//hamiltonian k-space

                            }
                        }
                    }
                }
            }
        }


        for (x = 0; x < num_orb; x++) {
            for (y = 0; y < num_orb; y++) {
                h_up[x * num_orb + y] = Ham_K[0][x][y];
                h_dn[x * num_orb + y] = Ham_K[1][x][y];
            }
        }

        zheev_(&JOBZ, &UPLO, &num_orb, h_up, &num_orb, W_up, WORK, &LWORK, RWORK, &INFO);
        if (INFO) {
            std::cout << "ERROR! Problem with diagonalization of h_up!" << std::endl;
            return 0;
        }

        for (x = 0; x < num_orb; x++) {
            for (y = 0; y < num_orb; y++) {
                egvec[0][e][y][x] = h_up[x * num_orb + y];
            }
        }

        for (x = 0; x < num_orb; x++) {
            egval[0][e][x] = W_up[x];
        }


        zheev_(&JOBZ, &UPLO, &num_orb, h_dn, &num_orb, W_dn, WORK, &LWORK, RWORK, &INFO);
        if (INFO) {
            std::cout << "ERROR! Problem with diagonalization of h_dn!" << std::endl;
            return 0;
        }

        for (x = 0; x < num_orb; x++) {
            for (y = 0; y < num_orb; y++) {
                egvec[1][e][y][x] = h_dn[x * num_orb + y];
            }
        }

        for (x = 0; x < num_orb; x++) {
            egval[1][e][x] = W_dn[x];
        }

    }


    delete[] h_up;
    delete[] h_dn;
    delete[] W_up;
    delete[] W_dn;
    delete[] WORK;
    delete[] RWORK;


    std::cout << "Fourier transformation and diagonalization of Hamiltonian is completed" << std::endl;

    //===============================================================================
    //Exchange couplings and occupation matrices


    if (specific_true) {

        matrix_trace = 0;

        for (x = 0; x < 5; x++) {
            for (y = 0; y < 5; y++) {
                exchange[x][y] = 0;
            }
        }

        atom = specific[0]; //atom 1
        index_temp[0] = specific[2]; // i
        index_temp[1] = specific[3]; // j
        index_temp[2] = specific[4]; // k
        index_temp[3] = specific[1]; //atom 2



        for (x = 0; x < 3; x++) {
            r[x] = specific[2] * cell_vec[0][x] + specific[3] * cell_vec[1][x] + specific[4] * cell_vec[2][x] +
                   (positions[specific[1]][x] - positions[specific[0]][x]);
        }

        rad_specific = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);

        std::cout << "=====================================================================" << std::endl;
        std::cout << "Atom " << specific[0] << "(000)<-->Atom " << specific[1] << "(" << specific[2] << specific[3]
                  << specific[4] << ") with radius " << rad_specific << std::endl;

        calc_exchange(atom, index_temp, num_orb, num_kpoints, n_max, ntot, spin, cell_vec, k_vec, egval, egvec, E, dE,
                      Ham_R, exchange);


        for (x = 0; x < 5; x++) {
            matrix_trace += exchange[x][x];
        }

        std::cout << std::fixed;

        for (x = 0; x < 5; x++) {
            for (y = 0; y < 5; y++) {
                std::cout << std::setprecision(6) << exchange[x][y] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Trace equals to: " << matrix_trace << " eV" << std::endl;
        std::cout << std::endl;

    } else {

        num_points = num_mag_atoms * (n_size[0] * n_size[1] * n_size[2]);

        // index = [i, j, k, atom]
        std::vector <std::vector<int>> index(
                num_points, std::vector<int>(
                        4)
        );

        std::vector<double> radius(num_points);


        //'atom' is the index of the central atom
        for (atom = 0; atom < num_mag_atoms; atom++) {

            coordination_sort(atom, num_mag_atoms, n_min, n_max, cell_vec, positions, radius, index);

            std::cout << "=====================================================================" << std::endl;

            neighbor_num = 1;
            sphere_num = 0;

            for (p = 0; p < num_points; p++) {

                if (p == 0) {
                    //in case of radius = 0 we calculate occupation matrices
                    calc_occupation(atom, num_orb, num_kpoints, ntot, egval, egvec, E, dE, occ);

                    matrix_trace = 0;

                    for (x = 0; x < 5; x++) {
                        for (y = 0; y < 5; y++) {
                            mag_mom[x][y] = occ[0][x][y] - occ[1][x][y];
                        }
                    }

                    for (x = 0; x < 5; x++) {
                        matrix_trace += mag_mom[x][x];
                    }

                    std::cout << "Occupation matrix (N_up - N_dn) for atom " << atom + 1 << std::endl;
                    std::cout << std::fixed;
                    for (x = 0; x < 5; x++) {
                        for (y = 0; y < 5; y++) {
                            std::cout << std::setprecision(3) << mag_mom[x][y] << " ";
                        }
                        std::cout << std::endl;
                    }

                    std::cout << "Trace equals to: " << matrix_trace << std::endl;
                    std::cout << std::endl;
                } else {

                    if (fabs(radius[p - 1] - radius[p]) < 1E-4)neighbor_num++;
                    else {
                        neighbor_num = 1;
                        sphere_num++;
                    }

                    if (sphere_num == max_sphere_num) break;

                    for (z = 0; z < 4; z++) {
                        index_temp[z] = index[p][z];
                    }

                    std::cout << "Atom " << atom << "(000)<-->Atom " << index[p][3]  << "(" << index[p][0]
                              << index[p][1]
                              << index[p][2] << ") in sphere #" << sphere_num << " with radius " << radius[p] << " is #"
                              << neighbor_num
                              << ":  " << std::endl;

                    matrix_trace = 0;

                    for (x = 0; x < 5; x++) {
                        for (y = 0; y < 5; y++) {
                            exchange[x][y] = 0;
                        }
                    }

                    calc_exchange(atom, index_temp, num_orb, num_kpoints, n_max, ntot, spin, cell_vec, k_vec, egval,
                                  egvec, E, dE, Ham_R, exchange);

                    for (x = 0; x < 5; x++) {
                        matrix_trace += exchange[x][x];
                    }

                    std::cout << std::fixed;

                    for (x = 0; x < 5; x++) {
                        for (y = 0; y < 5; y++) {
                            std::cout << std::setprecision(6) << exchange[x][y] << " ";
                        }
                        std::cout << std::endl;
                    }

                    std::cout << "Trace equals to: " << matrix_trace << " eV" << std::endl;
                    std::cout << std::endl;


                }

            }

        }

    }


    td = time(NULL);
    std::cout << std::endl << "This run was terminated on: " << std::endl;
    std::cout << ctime(&td) << std::endl;

    std::cout << "JOB DONE" << std::endl;
    std::cout << "=====================================================================" << std::endl;

}
