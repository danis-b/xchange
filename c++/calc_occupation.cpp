#include <complex>
#include <iostream>
#include <vector>

//This function calculates occupation matrices for atom with index 'atom'
void calc_occupation(int atom, int num_orb, int num_kpoints, int ntot,
                     std::complex<double> ***egval,
                     std::complex<double> ****egvec,
                     std::complex<double> *E, std::complex<double> *dE, std::vector<int> &mag_orbs,
                     std::vector<std::vector<std::vector<double> > > &occ)
{

    int x, y, z, num, e, zone;
    int shift_i;
    double weight;

    std::vector<std::vector<std::vector<std::complex<double> > > > greenR_ii(
        2, std::vector<std::vector<std::complex<double> > >(
               mag_orbs[atom], std::vector<std::complex<double> >(
                                   mag_orbs[atom])));

    std::vector<std::vector<std::vector<std::complex<double> > > > greenK_ii(
        2, std::vector<std::vector<std::complex<double> > >(
               mag_orbs[atom], std::vector<std::complex<double> >(
                                   mag_orbs[atom])));

    weight = pow(num_kpoints, -1);

    shift_i = 0;
    for (x = 0; x < atom; x++)
    {
        shift_i += mag_orbs[x];
    }

    for (z = 0; z < 2; z++)
    {
        for (x = 0; x < mag_orbs[atom]; x++)
        {
            for (y = 0; y < mag_orbs[atom]; y++)
            {
                occ[z][x][y] = 0;
            }
        }
    }

    for (num = 0; num < ntot; num++)
    {

        for (z = 0; z < 2; z++)
        {
            for (x = 0; x < mag_orbs[atom]; x++)
            {
                for (y = 0; y < mag_orbs[atom]; y++)
                {
                    greenR_ii[z][x][y] = std::complex<double>(0, 0);
                }
            }
        }

        for (e = 0; e < num_kpoints; e++)
        {
            for (z = 0; z < 2; z++)
            {
                for (x = 0; x < mag_orbs[atom]; x++)
                {
                    for (y = 0; y < mag_orbs[atom]; y++)
                    {
                        greenK_ii[z][x][y] = std::complex<double>(0, 0);
                    }
                }
            }

            // zone runs along eigenvector
            for (zone = 0; zone < num_orb; zone++)
            {
                for (z = 0; z < 2; z++)
                {
                    for (x = 0; x < mag_orbs[atom]; x++)
                    {
                        for (y = 0; y < mag_orbs[atom]; y++)
                        {
                            greenK_ii[z][x][y] +=
                                (egvec[z][e][x + shift_i][zone] * std::conj(egvec[z][e][y + shift_i][zone])) *
                                pow((E[num] - egval[z][e][zone]), -1);
                        }
                    }
                }
            }

            for (z = 0; z < 2; z++)
            {
                for (x = 0; x < mag_orbs[atom]; x++)
                {
                    for (y = 0; y < mag_orbs[atom]; y++)
                    {
                        // since R = 0, and phase = 1;
                        greenR_ii[z][x][y] += weight * greenK_ii[z][x][y];
                    }
                }
            }
        }

        for (z = 0; z < 2; z++)
        {
            for (x = 0; x < mag_orbs[atom]; x++)
            {
                for (y = 0; y < mag_orbs[atom]; y++)
                {
                    occ[z][x][y] += (-1 / M_PI) * (greenR_ii[z][x][y] * dE[num]).imag();
                }
            }
        }
    }
}