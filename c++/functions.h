#include <complex>
#include <iostream>
#include <vector>


// external lapack library for diagonalization
extern "C"
void zheev_(char *JOBZ, char *UPLO, int *N, std::complex<double> *A, int *LDA, double *W, std::complex<double> *WORK, int *LWORK,
            double *RWORK, int *INFO);


//This function sorts atoms depending on the radius from the central atom with index 'atom'
void
coordination_sort(int atom, int num_atoms, int n_min[3], int n_max[3], double cell_vectors[3][3],
                  std::vector <std::vector<double>> &positions,
                  std::vector<double> &radius, std::vector <std::vector<int>> &index) ;


//This function calculates occupation matrices for atom with index 'atom'
void
calc_occupation(int atom, int num_orb, int num_kpoints, int ntot,
                std::complex<double> ***egval,
                std::complex<double> ****egvec,
                std::complex<double> *E, std::complex<double> *dE,
                double occ[2][5][5]);


//This function calculates exchange coupling parameter between atoms with index 'atom' and 'index_temp'
void
calc_exchange(int atom, int index_temp[4], int num_orb, int num_kpoints, int n_max[3],
              int ntot, double spin, double cell_vec[3][3], double **k_vec,
              std::complex<double> ***egval,
              std::complex<double> ****egvec,
              std::complex<double> *E, std::complex<double> *dE,
              double ******Ham_R, double exchange[5][5]);




