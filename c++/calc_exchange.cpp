#include <complex>
#include <iostream>
#include <vector>

//This function calculates exchange coupling parameter between atoms with index 'atom' and 'index_temp'
void calc_exchange(int atom, int index_temp[4], int num_orb, int num_kpoints, int n_max[3],
                   int ntot, double spin, double cell_vec[3][3], double **k_vec,
                   std::complex<double> ***egval,
                   std::complex<double> ****egvec,
                   std::complex<double> *E, std::complex<double> *dE,
                   double ******Ham_R, std::vector<int> &mag_orbs, std::vector<std::vector<double> > &exchange)
{
    // index_temp[0] = i, index_temp[1] = j, index_temp[2] = k, index_temp[3] = atom2

    int i, j, x, y, z, num, e, zone;
    int shift_i, shift_j;
    double r[3], weight, ca, sa;
    std::complex<double> phase;

    //on-site energy;
    std::vector<std::vector<std::complex<double> > > delta_i(
        mag_orbs[atom], std::vector<std::complex<double> >(
                            mag_orbs[atom]));

    std::vector<std::vector<std::complex<double> > > delta_j(
        mag_orbs[index_temp[3]], std::vector<std::complex<double> >(
                                     mag_orbs[index_temp[3]]));

    // Green's function in real space
    std::vector<std::vector<std::complex<double> > > greenR_ij(
        mag_orbs[atom], std::vector<std::complex<double> >(
                            mag_orbs[index_temp[3]]));

    std::vector<std::vector<std::complex<double> > > greenR_ji(
        mag_orbs[index_temp[3]], std::vector<std::complex<double> >(
                                               mag_orbs[atom]));

    // Green's function in k-space;
    std::vector<std::vector<std::complex<double> > > greenK_ij(
        mag_orbs[atom], std::vector<std::complex<double> >(
                            mag_orbs[index_temp[3]]));

    std::vector<std::vector<std::complex<double> > > greenK_ji(
        mag_orbs[index_temp[3]], std::vector<std::complex<double> >(
                                               mag_orbs[atom]));

    std::vector<std::vector<std::complex<double> > > dot_product(
        mag_orbs[atom], std::vector<std::complex<double> >(
                                      mag_orbs[atom]));

    weight = pow(num_kpoints, -1);

    shift_i = 0;
    for (x = 0; x < atom; x++)
    {
        shift_i += mag_orbs[x];
    }

    shift_j = 0;
    for (x = 0; x < index_temp[3]; x++)
    {
        shift_j += mag_orbs[x];
    }

    for (z = 0; z < 3; z++)
    {
        r[z] = index_temp[0] * cell_vec[0][z] + index_temp[1] * cell_vec[1][z] + index_temp[2] * cell_vec[2][z];
    }

    //on-site splitting
    for (x = 0; x < mag_orbs[atom]; x++)
    {
        for (y = 0; y < mag_orbs[atom]; y++)
        {
            delta_i[x][y] = std::complex<double>(Ham_R[0][n_max[0]][n_max[1]][n_max[2]][x + shift_i][y + shift_i] -
                                                     Ham_R[1][n_max[0]][n_max[1]][n_max[2]][x + shift_i][y + shift_i],
                                                 0);
        }
    }

    for (x = 0; x < mag_orbs[index_temp[3]]; x++)
    {
        for (y = 0; y < mag_orbs[index_temp[3]]; y++)
        {

            delta_j[x][y] = std::complex<double>(
                Ham_R[0][n_max[0]][n_max[1]][n_max[2]][x + shift_j][y + shift_j] -
                    Ham_R[1][n_max[0]][n_max[1]][n_max[2]][x + shift_j][y + shift_j],
                0);
        }
    }


    for (num = 0; num < ntot; num++)
    {

        for (x = 0; x < mag_orbs[atom]; x++)
        {
            for (y = 0; y < mag_orbs[index_temp[3]]; y++)
            {
                greenR_ij[x][y] = std::complex<double>(0, 0);
                greenR_ji[y][x] = std::complex<double>(0, 0);
            }
        }


        for (e = 0; e < num_kpoints; e++)
        {

            ca = cos(k_vec[e][0] * r[0] + k_vec[e][1] * r[1] + k_vec[e][2] * r[2]);
            sa = sin(k_vec[e][0] * r[0] + k_vec[e][1] * r[1] + k_vec[e][2] * r[2]);

            phase = std::complex<double>(ca, sa);

            for (x = 0; x < mag_orbs[atom]; x++)
            {
                for (y = 0; y < mag_orbs[index_temp[3]]; y++)
                {
                    greenK_ij[x][y] = std::complex<double>(0, 0);
                    greenK_ji[y][x] = std::complex<double>(0, 0);
                }
            }

            for (zone = 0; zone < num_orb; zone++)
            {
                for (x = 0; x < mag_orbs[atom]; x++)
                {
                    for (y = 0; y < mag_orbs[index_temp[3]]; y++)
                    {
                        greenK_ij[x][y] +=
                            (egvec[1][e][x + shift_i][zone] * conj(egvec[1][e][y + shift_j][zone])) *
                            pow((E[num] - egval[1][e][zone]),
                                -1);

                        greenK_ji[y][x] +=
                            (egvec[0][e][y + shift_j][zone] * conj(egvec[0][e][x + shift_i][zone])) *
                            pow((E[num] - egval[0][e][zone]), -1); //Green's functions in k-space
                    }
                }
            }


            for (x = 0; x < mag_orbs[atom]; x++)
            {
                for (y = 0; y < mag_orbs[index_temp[3]]; y++)
                {
                    greenR_ij[x][y] += weight * conj(phase) * greenK_ij[x][y];
                    greenR_ji[y][x] += weight * phase * greenK_ji[y][x]; //Green's functions in real space
                }
            }
        }

        for (x = 0; x < mag_orbs[atom]; x++)
        {
            for (y = 0; y < mag_orbs[atom]; y++)
            {
                dot_product[x][y] = std::complex<double>(0, 0);
            }
        }


        for (x = 0; x < mag_orbs[atom]; x++)
        {
            for (y = 0; y < mag_orbs[atom]; y++)
            {
                for (z = 0; z < mag_orbs[index_temp[3]]; z++)
                {
                    for (i = 0; i < mag_orbs[index_temp[3]]; i++)
                    {
                        for (j = 0; j < mag_orbs[atom]; j++)
                        {

                            dot_product[x][j] +=
                                delta_i[x][y] * greenR_ij[y][z] * delta_j[z][i] * greenR_ji[i][j];
                        }
                    }
                }
            }
        }


        for (x = 0; x < mag_orbs[atom]; x++)
        {
            for (y = 0; y < mag_orbs[atom]; y++)
            {
                exchange[x][y] -= (1 / (2 * M_PI * pow(spin, 2))) * (dot_product[x][y] * dE[num]).imag();
            }
        }
    }
}
