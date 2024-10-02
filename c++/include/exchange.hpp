#ifndef EXCHANGE_HPP  // Include guard to prevent multiple inclusions
#define EXCHANGE_HPP

#include <complex>
#include <iostream>
#include <vector>
#include <Eigen/Dense>

std::vector<double> calc_exchange(int central_atom,
                                  std::vector<int> &index_temp,
                                  int num_orb,
                                  int num_kpoints,
                                  int num_freq,
                                  double spin,
                                  std::vector<double> &cell_vec,
                                  std::vector<double> &k_vec,
                                  std::vector<std::complex<double>> &Ham_K,
                                  std::vector<std::complex<double>> &E,
                                  std::vector<std::complex<double>> &dE,
                                  std::vector<int> &mag_orbs);


/**
 * @brief  Calculates exchange matrix "exchange" based on Green's functions G(E) = (E - Ham_K)^-1
 * exchange = -1/(2 * S^2 PI) * Im[ \int_{-\inf}^E_F  dE * \Delta_i G^dn_ij * \Delta_j * G^up_ji
 *
 * @param[in]  central_atom   atomic number of magnetic atom, for which we want to calculate exchange matrix 
 * @param[in]  index_temp array to define radius-vector  r[z] = index_temp[0] * cell_vec[z] + index_temp[1] * cell_vec[3 + z] + index_temp[2] * cell_vec[6 + z]
 * @param[in]  num_orb   number of Wannier orbitals
 * @param[in]  num_kpoints number of kpoints for Brillouin zone integration
 * @param[in]  num_freq  number if frequency points for energy contour integration
 * @param[in]  spin spin S value
 * @param[in]  cell_vec  matrix of unit cell vectors
 * @param[in]  k_vec  matrix of k-vectors with shape [3 * num_kpoints]
 * @param[in]  Ham_K  Hamiltonian in k-space with shape Ham_K[num_kpoints * num_orb * num_orb]
 * @param[in]  E  energy contour for integration with shape E[num_freq]
 * @param[in]  dE  infinitesimal element of integration with shape dE[num_freq] 
 * @param[in]  mag_orbs array of magnetic orbitals
 *  
 * @return     exchange matrix "exchange" 
 */

#endif 