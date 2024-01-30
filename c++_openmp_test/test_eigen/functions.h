#include <complex>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <omp.h> 

// This function calculates Ham_K at each k-point
void calc_hamK(int num_orb, int num_kpoints, int n_min[3], int n_max[3], double cell_vec[3][3], std::vector<std::array<double,3> > &k_vec,
               double ******Ham_R, std::complex<double> ****Ham_K);

// This function calculates occupation matrices for atom with index 'atom'
std::vector<double> calc_occupation(int central_atom,
                                    int num_orb,
                                    int num_kpoints,
                                    int num_freq,
                                    std::vector<std::complex<double>> &Ham_K,
                                    std::vector<std::complex<double>> &E,
                                    std::vector<std::complex<double>> &dE,
                                    std::vector<int> &mag_orbs);



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