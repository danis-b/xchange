#include <complex>
#include <iostream>
#include <vector>

//This function sorts atoms depending on the radius from the central atom with index 'atom'
void coordination_sort(int atom, int num_atoms, int n_min[3], int n_max[3], double cell_vectors[3][3],
                       std::vector<std::vector<double> > &positions,
                       std::vector<double> &radius, std::vector<std::vector<int> > &index);

//This function calculates Ham_K at each k-point
void calc_hamK(int num_orb, int num_kpoints, int n_min[3], int n_max[3], double cell_vec[3][3], double **k_vec,
               double ******Ham_R, std::complex<double> ****Ham_K);

//This function calculates occupation matrices for atom with index 'atom'
void calc_occupation(int atom, int num_orb, int num_kpoints, int ntot,
                     std::complex<double> ****Ham_K,
                     std::complex<double> *E, std::complex<double> *dE, std::vector<int> &mag_orbs,
                     std::vector<std::vector<std::vector<double> > > &occ, bool err);

//This function calculates exchange coupling parameter between atoms with index 'atom' and 'index_temp'
void calc_exchange(int atom, int index_temp[4], int num_orb, int num_kpoints, int n_max[3],
                   int ntot, double spin, double cell_vec[3][3], double **k_vec,
                   std::complex<double> *E, std::complex<double> *dE,
                   double ******Ham_R, std::complex<double> ****Ham_K,
                   std::vector<int> &mag_orbs, std::vector<std::vector<double> > &exchange, bool err);

//This function inverts  the square complex matrix 'loc_greenK' with dimention 'num_orb x num_orb'
void inverse_matrix(int num_orb, std::complex<double> ***loc_greenK, bool err);
