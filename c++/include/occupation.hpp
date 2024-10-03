#ifndef OCCUPATION_HPP // Include guard to prevent multiple inclusions
#define OCCUPATION_HPP

#include <complex>
#include <iostream>
#include <vector>
#include <Eigen/Dense>

std::vector<double> calc_occupation(int central_atom,
                                    int num_orb,
                                    int num_kpoints,
                                    int num_freq,
                                    std::vector<std::complex<double>> &Ham_K,
                                    std::vector<std::complex<double>> &E,
                                    std::vector<std::complex<double>> &dE,
                                    std::vector<int> &mag_orbs);

/**
 * @brief  Calculates occupation matrix 'occ' using Green's functions G(E) = (E - Ham_K)^-1 defined on real frequencies
 * occ = (-1/PI) * Im[ \int_{-\inf}^E_F  dE * G(E)]
 *
 * @param[in]  central_atom - atomic number of magnetic atom, for which we want to calculate occupation matrix 
 * @param[in]  num_orb - number of Wannier orbitals
 * @param[in]  num_kpoints -  number of kpoints for Brillouin zone integration
 * @param[in]  num_freq -  number if frequency points for energy contour integration
 * @param[in]  Ham_K -  Hamiltonian in k-space with shape Ham_K[num_kpoints * num_orb * num_orb]
 * @param[in]  E -  energy contour for integration with shape E[num_freq]
 * @param[in]  dE -  infinitesimal element of integration with shape dE[num_freq] 
 * @param[in]  mag_orbs -  array of magnetic orbitals
 *  
 * @return     Occupation matrix "occ" for both spin components
 */

#endif