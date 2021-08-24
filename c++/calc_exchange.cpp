#include <complex>
#include <iostream>
#include <vector>



//This function calculates exchange coupling parameter between atoms with index 'atom' and 'index_temp'
void
calc_exchange(int atom, int index_temp[4], int num_orb, int num_kpoints, int n_max[3],
              int ntot, double spin, double cell_vec[3][3], std::vector <std::vector<double>> & k_vec,
              std::vector < std::vector < std::vector < double >>>  &egval,
std::vector < std::vector < std::vector < std::vector < std::complex<double> >> >> &egvec,
std::vector<std::complex<double>> &E, std::vector<std::complex<double>> &dE,
std::vector < std::vector < std::vector < std::vector < std::vector < std::vector < double > > >> >> &Ham_R, double exchange[5][5]) {


    int i, j, x, y, z, num, e, zone;
    double r[3], weight, ca, sa;

    std::complex<double> delta_i[5][5], delta_j[5][5]; //on-site energy;
    std::complex<double> greenR_ij[5][5], greenR_ji[5][5]; // Green's function in real space
    std::complex<double> greenK_ij[5][5], greenK_ji[5][5]; // Green's function in k-space;
    std::complex<double> dot_product[5][5];
    std::complex<double> phase;

    weight = pow(num_kpoints, -1);

    for (z = 0; z < 3; z++) {
        r[z] = index_temp[0] * cell_vec[0][z] + index_temp[1] * cell_vec[1][z] + index_temp[2] * cell_vec[2][z];

    }


    //on-site splitting
    for (x = 0; x < 5; x++) {
        for (y = 0; y < 5; y++) {
            delta_i[x][y] = std::complex<double>(Ham_R[0][n_max[0]][n_max[1]][n_max[2]][x + 5 * atom][y + 5 * atom] -
                                                 Ham_R[1][n_max[0]][n_max[1]][n_max[2]][x + 5 * atom][y + 5 * atom], 0);

            delta_j[x][y] = std::complex<double>(
                    Ham_R[0][n_max[0]][n_max[1]][n_max[2]][x + 5 * index_temp[3]][y + 5 * index_temp[3]] -
                    Ham_R[1][n_max[0]][n_max[1]][n_max[2]][x + 5 * index_temp[3]][y + 5 * index_temp[3]],
                    0);

        }
    }


    for (num = 0; num < ntot; num++) {

        for (x = 0; x < 5; x++) {
            for (y = 0; y < 5; y++) {
                greenR_ij[x][y] = std::complex<double>(0, 0);
                greenR_ji[x][y] = std::complex<double>(0, 0);
            }
        }


        for (e = 0; e < num_kpoints; e++) {

            ca = cos(k_vec[e][0] * r[0] + k_vec[e][1] * r[1] + k_vec[e][2] * r[2]);
            sa = sin(k_vec[e][0] * r[0] + k_vec[e][1] * r[1] + k_vec[e][2] * r[2]);

            phase = std::complex<double>(ca, sa);


            for (x = 0; x < 5; x++) {
                for (y = 0; y < 5; y++) {
                    greenK_ij[x][y] = std::complex<double>(0, 0);
                    greenK_ji[x][y] = std::complex<double>(0, 0);
                }
            }


            for (zone = 0; zone < num_orb; zone++) {

                for (x = 0; x < 5; x++) {
                    for (y = 0; y < 5; y++) {
                        greenK_ij[x][y] +=
                                (egvec[1][e][x + 5 * atom][zone] * conj(egvec[1][e][y + 5 * index_temp[3]][zone])) *
                                pow((E[num] - egval[1][e][zone]),
                                    -1);
                        greenK_ji[x][y] +=
                                (egvec[0][e][x + 5 * index_temp[3]][zone] * conj(egvec[0][e][y + 5 * atom][zone])) *
                                pow((E[num] - egval[0][e][zone]), -1); //Green's functions in k-space
                    }
                }
            }

            for (x = 0; x < 5; x++) {
                for (y = 0; y < 5; y++) {
                    greenR_ij[x][y] += weight * conj(phase) * greenK_ij[x][y];
                    greenR_ji[x][y] += weight * phase * greenK_ji[x][y];  //Green's functions in real space
                }
            }
        }


        for (x = 0; x < 5; x++) {
            for (y = 0; y < 5; y++) {
                dot_product[x][y] = std::complex<double>(0, 0);
            }
        }

        for (x = 0; x < 5; x++) {
            for (y = 0; y < 5; y++) {
                for (z = 0; z < 5; z++) {
                    for (i = 0; i < 5; i++) {
                        for (j = 0; j < 5; j++) {

                            dot_product[x][j] +=
                                    delta_i[x][y] * greenR_ij[y][z] * delta_j[z][i] * greenR_ji[i][j];
                        }
                    }
                }
            }
        }

        for (x = 0; x < 5; x++) {
            for (y = 0; y < 5; y++) {
                exchange[x][y] -= (1 / (2 * M_PI * pow(spin, 2))) * (dot_product[x][y] * dE[num]).imag();

            }
        }
    }
}
