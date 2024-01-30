#include <complex>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

// This function calculates Ham_K at each k-point
void calc_hamK(int num_orb, 
               int num_kpoints, 
               int n_min[3], 
               int n_max[3], 
               std::vector<double> &cell_vec, 
               std::vector<double> &k_vec,
               double ******Ham_R, 
               std::complex<double> ****Ham_K)
{
    double r[3], ca, sa;
    std::complex<double> phase;

    for (int e = 0; e < num_kpoints; ++e)
    {
        for (int z = 0; z < 2; ++z)
        {
            for (int x = 0; x < num_orb; ++x)
            {
                for (int y = 0; y < num_orb; ++y)
                {
                    Ham_K[z][e][x][y] = std::complex<double>(0, 0);
                }
            }
        }

        for (int x = 0; x < num_orb; ++x)
        {
            for (int y = 0; y < num_orb; ++y)
            {
                for (int z = 0; z < 2; ++z)
                {
                    for (int i = n_min[0]; i <= n_max[0]; ++i)
                    {
                        for (int j = n_min[1]; j <= n_max[1]; ++j)
                        {
                            for (int k = n_min[2]; k <= n_max[2]; ++k)
                            {
                                for (int p = 0; p < 3; ++p)
                                {
                                    r[p] = i * cell_vec[0][p] + j * cell_vec[1][p] + k * cell_vec[2][p];
                                }

                                ca = cos(k_vec[e][0] * r[0] + k_vec[e][1] * r[1] + k_vec[e][2] * r[2]);
                                sa = sin(k_vec[e][0] * r[0] + k_vec[e][1] * r[1] + k_vec[e][2] * r[2]);

                                phase = std::complex<double>(ca, -sa);

                                Ham_K[z][e][x][y] +=
                                    phase * std::complex<double>(Ham_R[z][i - n_min[0]][j - n_min[1]][k -
                                                                                                      n_min[2]][x][y],
                                                                 0); // hamiltonian k-space
                            }
                        }
                    }
                }
            }
        }
    }
}
