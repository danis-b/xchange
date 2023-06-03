#include <complex>
#include <iostream>
#include <vector>
#include "functions.h"

// This function calculates exchange coupling parameter between atoms with index 'central_atom' and 'index_temp'
void calc_exchange(int central_atom, int index_temp[4], int num_orb, int num_kpoints,
                   int ntot, double spin, double cell_vec[3][3], double **k_vec,
                   std::complex<double> *E, std::complex<double> *dE,
                   std::complex<double> ****Ham_K,
                   std::vector<int> &mag_orbs, std::vector<std::vector<double>> &exchange, bool err)
{
    // index_temp[0] = i, index_temp[1] = j, index_temp[2] = k, index_temp[3] = atom2

    int shift_i, shift_j;
    double r[3], weight, ca, sa;
    std::complex<double> phase;

    // on-site energy;
    std::vector<std::vector<std::complex<double>>> delta_i(
        mag_orbs[central_atom], std::vector<std::complex<double>>(
                                    mag_orbs[central_atom]));

    std::vector<std::vector<std::complex<double>>> delta_j(
        mag_orbs[index_temp[3]], std::vector<std::complex<double>>(
                                     mag_orbs[index_temp[3]]));

    // Green's function in real space
    std::vector<std::vector<std::complex<double>>> greenR_ij(
        mag_orbs[central_atom], std::vector<std::complex<double>>(
                                    mag_orbs[index_temp[3]]));

    std::vector<std::vector<std::complex<double>>> greenR_ji(
        mag_orbs[index_temp[3]], std::vector<std::complex<double>>(
                                     mag_orbs[central_atom]));

    // Green's function in k-space;
    std::vector<std::vector<std::complex<double>>> greenK_ij(
        mag_orbs[central_atom], std::vector<std::complex<double>>(
                                    mag_orbs[index_temp[3]]));

    std::vector<std::vector<std::complex<double>>> greenK_ji(
        mag_orbs[index_temp[3]], std::vector<std::complex<double>>(
                                     mag_orbs[central_atom]));

    std::vector<std::vector<std::complex<double>>> dot_product(
        mag_orbs[central_atom], std::vector<std::complex<double>>(
                                    mag_orbs[central_atom]));

    // local Green's function in k-space;
    std::complex<double> ***loc_greenK = new std::complex<double> **[2];
    for (int z = 0; z < 2; ++z)
    {
        loc_greenK[z] = new std::complex<double> *[num_orb];
        for (int i = 0; i < num_orb; ++i)
        {
            loc_greenK[z][i] = new std::complex<double>[num_orb];
        }
    }

    weight = pow(num_kpoints, -1);

    shift_i = 0;
    for (int x = 0; x < central_atom; ++x)
    {
        shift_i += mag_orbs[x];
    }

    shift_j = 0;
    for (int x = 0; x < index_temp[3]; ++x)
    {
        shift_j += mag_orbs[x];
    }

    for (int z = 0; z < 3; ++z)
    {
        r[z] = index_temp[0] * cell_vec[0][z] + index_temp[1] * cell_vec[1][z] + index_temp[2] * cell_vec[2][z];
    }

    for (int num = 0; num < ntot; ++num)
    {
        for (int x = 0; x < mag_orbs[central_atom]; ++x)
        {
            for (int y = 0; y < mag_orbs[index_temp[3]]; ++y)
            {
                greenR_ij[x][y] = std::complex<double>(0, 0);
                greenR_ji[y][x] = std::complex<double>(0, 0);
            }
        }

        for (int x = 0; x < mag_orbs[central_atom]; ++x)
        {
            for (int y = 0; y < mag_orbs[central_atom]; ++y)
            {
                delta_i[x][y] = std::complex<double>(0, 0);
            }
        }

        for (int x = 0; x < mag_orbs[index_temp[3]]; ++x)
        {
            for (int y = 0; y < mag_orbs[index_temp[3]]; ++y)
            {
                delta_j[x][y] = std::complex<double>(0, 0);
            }
        }

        for (int e = 0; e < num_kpoints; ++e)
        {

            ca = cos(k_vec[e][0] * r[0] + k_vec[e][1] * r[1] + k_vec[e][2] * r[2]);
            sa = sin(k_vec[e][0] * r[0] + k_vec[e][1] * r[1] + k_vec[e][2] * r[2]);

            phase = std::complex<double>(ca, sa);

            for (int z = 0; z < 2; ++z)
            {
                for (int x = 0; x < num_orb; ++x)
                {
                    for (int y = 0; y < num_orb; ++y)
                    {
                        if (x == y)
                        {
                            loc_greenK[z][x][y] = E[num] - Ham_K[z][e][x][y];
                        }
                        else
                        {
                            loc_greenK[z][x][y] = -Ham_K[z][e][x][y];
                        }
                    }
                }
            }

            inverse_matrix(num_orb, loc_greenK, err);

            // read the necessary block
            for (int x = 0; x < mag_orbs[central_atom]; ++x)
            {
                for (int y = 0; y < mag_orbs[index_temp[3]]; ++y)
                {
                    greenK_ij[x][y] = loc_greenK[1][x + shift_i][y + shift_j];
                    greenK_ji[y][x] = loc_greenK[0][y + shift_j][x + shift_i];
                }
            }

            // on-site splitting
            for (int x = 0; x < mag_orbs[central_atom]; ++x)
            {
                // in  case if  we add self-energy
                // delta_i[x][x] += weight * (selfen[0][num][e][x + shift_i] - selfen[1][num][e][x + shift_i]);

                for (int y = 0; y < mag_orbs[central_atom]; ++y)
                {
                    delta_i[x][y] += weight * (Ham_K[0][e][x + shift_i][y + shift_i] - Ham_K[1][e][x + shift_i][y + shift_i]);
                }
            }

            for (int x = 0; x < mag_orbs[index_temp[3]]; ++x)
            {
                // in  case if  we add self-energy
                // delta_j[x][x] += weight * (selfen[0][num][e][x + shift_j] - selfen[1][num][e][x + shift_j]);

                for (int y = 0; y < mag_orbs[index_temp[3]]; ++y)
                {
                    delta_j[x][y] += weight * (Ham_K[0][e][x + shift_j][y + shift_j] - Ham_K[1][e][x + shift_j][y + shift_j]);
                }
            }

            // Green's functions in real space
            for (int x = 0; x < mag_orbs[central_atom]; ++x)
            {
                for (int y = 0; y < mag_orbs[index_temp[3]]; ++y)
                {
                    greenR_ij[x][y] += weight * phase * greenK_ij[x][y];
                    greenR_ji[y][x] += weight * conj(phase) * greenK_ji[y][x];
                }
            }
        }

        for (int x = 0; x < mag_orbs[central_atom]; ++x)
        {
            for (int y = 0; y < mag_orbs[central_atom]; ++y)
            {
                dot_product[x][y] = std::complex<double>(0, 0);
            }
        }

        for (int x = 0; x < mag_orbs[central_atom]; ++x)
        {
            for (int y = 0; y < mag_orbs[central_atom]; ++y)
            {
                for (int z = 0; z < mag_orbs[index_temp[3]]; ++z)
                {
                    for (int i = 0; i < mag_orbs[index_temp[3]]; ++i)
                    {
                        for (int j = 0; j < mag_orbs[central_atom]; ++j)
                        {
                            dot_product[x][j] +=
                                delta_i[x][y] * greenR_ij[y][z] * delta_j[z][i] * greenR_ji[i][j];
                        }
                    }
                }
            }
        }

        for (int x = 0; x < mag_orbs[central_atom]; ++x)
        {
            for (int y = 0; y < mag_orbs[central_atom]; ++y)
            {
                exchange[x][y] -= (1 / (2 * M_PI * pow(spin, 2))) * (dot_product[x][y] * dE[num]).imag();
            }
        }
    }

    delete[] loc_greenK;
}
