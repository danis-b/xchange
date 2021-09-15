#include <complex>
#include <iostream>


//This function calculates occupation matrices for atom with index 'atom'
void
calc_occupation(int atom, int num_orb, int num_kpoints, int ntot,
                std::complex<double> ***egval,
                std::complex<double> ****egvec,
                std::complex<double> *E, std::complex<double> *dE,
                double occ[2][5][5]) {

    int x, y, z, num, e, zone;
    double weight;
    std::complex<double> greenR_ii[2][5][5], greenK_ii[2][5][5];

    weight = pow(num_kpoints, -1);

    for (z = 0; z < 2; z++) {
        for (x = 0; x < 5; x++) {
            for (y = 0; y < 5; y++) {
                occ[z][x][y] = 0;
            }
        }
    }

    for (num = 0; num < ntot; num++) {

        for (z = 0; z < 2; z++) {
            for (x = 0; x < 5; x++) {
                for (y = 0; y < 5; y++) {
                    greenR_ii[z][x][y] = std::complex<double>(0, 0);
                }
            }
        }

        for (e = 0; e < num_kpoints; e++) {

            for (z = 0; z < 2; z++) {
                for (x = 0; x < 5; x++) {
                    for (y = 0; y < 5; y++) {
                        greenK_ii[z][x][y] = std::complex<double>(0, 0);
                    }
                }
            }

            // zone runs along eigenvector
            for (zone = 0; zone < num_orb; zone++) {
                for (z = 0; z < 2; z++) {
                    for (x = 0; x < 5; x++) {
                        for (y = 0; y < 5; y++) {
                            greenK_ii[z][x][y] +=
                                    (egvec[z][e][x + 5 * atom][zone] *
                                     std::conj(egvec[z][e][y + 5 * atom][zone])) *
                                    pow((E[num] -
                                         egval[z][e][zone]), -1);
                        }
                    }
                }
            }

            for (z = 0; z < 2; z++) {
                for (x = 0; x < 5; x++) {
                    for (y = 0; y < 5; y++) {
                        // since R = 0, and phase = 1;
                        greenR_ii[z][x][y] += weight * greenK_ii[z][x][y];
                    }
                }
            }
        }

        for (z = 0; z < 2; z++) {
            for (x = 0; x < 5; x++) {
                for (y = 0; y < 5; y++) {
                    occ[z][x][y] += (-1 / M_PI) * (greenR_ii[z][x][y] * dE[num]).imag();
                }
            }
        }
    }

}