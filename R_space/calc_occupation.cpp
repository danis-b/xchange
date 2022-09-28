#include <complex>
#include <iostream>
#include <vector>
#include "functions.h"

// This function calculates occupation matrices for atom with index 'central_atom'
void calc_occupation(int central_atom, int num_orb, int num_kpoints, int ntot,
                     std::complex<double> ****Ham_K,
                     std::complex<double> *E, std::complex<double> *dE, std::vector<int> &mag_orbs,
                     std::vector<std::vector<std::vector<double>>> &occ, bool err)
{

    int x, y, z, num, e;
    int shift_i;
    double weight;

    std::vector<std::vector<std::vector<std::complex<double>>>> greenR_ii(
        2, std::vector<std::vector<std::complex<double>>>(
               mag_orbs[central_atom], std::vector<std::complex<double>>(
                                           mag_orbs[central_atom])));

    std::vector<std::vector<std::vector<std::complex<double>>>> greenK_ii(
        2, std::vector<std::vector<std::complex<double>>>(
               mag_orbs[central_atom], std::vector<std::complex<double>>(
                                           mag_orbs[central_atom])));

    // local Green's function in k-space;
    std::complex<double> ***loc_greenK = new std::complex<double> **[2];
    for (z = 0; z < 2; z++)
    {
        loc_greenK[z] = new std::complex<double> *[num_orb];
        for (x = 0; x < num_orb; x++)
        {
            loc_greenK[z][x] = new std::complex<double>[num_orb];
        }
    }

    weight = pow(num_kpoints, -1);

    shift_i = 0;
    for (x = 0; x < central_atom; x++)
    {
        shift_i += mag_orbs[x];
    }

    for (z = 0; z < 2; z++)
    {
        for (x = 0; x < mag_orbs[central_atom]; x++)
        {
            for (y = 0; y < mag_orbs[central_atom]; y++)
            {
                occ[z][x][y] = 0;
            }
        }
    }

    for (num = 0; num < ntot; num++)
    {

        for (z = 0; z < 2; z++)
        {
            for (x = 0; x < mag_orbs[central_atom]; x++)
            {
                for (y = 0; y < mag_orbs[central_atom]; y++)
                {
                    greenR_ii[z][x][y] = std::complex<double>(0, 0);
                }
            }
        }

        for (e = 0; e < num_kpoints; e++)
        {

            for (z = 0; z < 2; z++)
            {
                for (x = 0; x < num_orb; x++)
                {
                    for (y = 0; y < num_orb; y++)
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
            for (z = 0; z < 2; z++)
            {
                for (x = 0; x < mag_orbs[central_atom]; x++)
                {
                    for (y = 0; y < mag_orbs[central_atom]; y++)
                    {
                        greenK_ii[z][x][y] = loc_greenK[z][x + shift_i][y + shift_i];
                    }
                }
            }

            for (z = 0; z < 2; z++)
            {
                for (x = 0; x < mag_orbs[central_atom]; x++)
                {
                    for (y = 0; y < mag_orbs[central_atom]; y++)
                    {
                        // since R = 0, and phase = 1;
                        greenR_ii[z][x][y] += weight * greenK_ii[z][x][y];
                    }
                }
            }
        }

        for (z = 0; z < 2; z++)
        {
            for (x = 0; x < mag_orbs[central_atom]; x++)
            {
                for (y = 0; y < mag_orbs[central_atom]; y++)
                {
                    occ[z][x][y] += (-1 / M_PI) * (greenR_ii[z][x][y] * dE[num]).imag();
                }
            }
        }
    }

    delete[] loc_greenK;
}