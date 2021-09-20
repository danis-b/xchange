#include <complex>
#include <iostream>


// external lapack library for diagonalization
extern "C"
void zheev(char *JOBZ, char *UPLO, int *N, std::complex<double> *A, int *LDA, double *W, std::complex<double> *WORK, int *LWORK,
            double *RWORK, int *INFO);


//This function calculates eigenvalues and eigenvectors of Ham_R at each k-point
void 
calc_eigenstates(int num_orb, int num_kpoints, int n_min[3], int n_max[3], double cell_vec[3][3], double **k_vec, 
double ******Ham_R, std::complex<double> ***egval, std::complex<double> ****egvec, bool diag_error)
{
    double r[3], ca, sa;
    std::complex<double> phase;


    // lapack parameters   
    char JOBZ = 'V';
    char UPLO = 'U';
    int LWORK = 2 * num_orb;

    double *W_up = new double[num_orb];
    double *W_dn = new double[num_orb];

    std::complex<double> *h_up = new std::complex<double>[num_orb*num_orb];
    std::complex<double> *h_dn = new std::complex<double>[num_orb*num_orb];

    std::complex<double> *WORK = new std::complex<double>[LWORK];
    double *RWORK = new double[3 * num_orb];
    int INFO_up, INFO_dn;


    // Hamiltonian in reciprocal space
    std::complex<double> ***Ham_K = new std::complex<double> **[2];
    for (int z = 0; z < 2; z++)
    {
        Ham_K[z] = new std::complex<double> *[num_orb];
        for (int i = 0; i < num_orb; i++)
        {
            Ham_K[z][i] = new std::complex<double>[num_orb];
        }
    }

    for (int e = 0; e < num_kpoints; e++)
    {

        for (int x = 0; x < num_orb; x++)
        {
            for (int y = 0; y < num_orb; y++)
            {
                for (int z = 0; z < 2; z++)
                {
                    Ham_K[z][x][y] = std::complex<double>(0, 0);
                }
            }
        }

        for (int x = 0; x < num_orb; x++)
        {
            for (int y = 0; y < num_orb; y++)
            {
                for (int z = 0; z < 2; z++)
                {

                    for (int i = n_min[0]; i <= n_max[0]; i++)
                    {
                        for (int j = n_min[1]; j <= n_max[1]; j++)
                        {
                            for (int k = n_min[2]; k <= n_max[2]; k++)
                            {
                                for (int p = 0; p < 3; p++)
                                {
                                    r[p] = i * cell_vec[0][p] + j * cell_vec[1][p] + k * cell_vec[2][p];
                                }

                                ca = cos(k_vec[e][0] * r[0] + k_vec[e][1] * r[1] + k_vec[e][2] * r[2]);
                                sa = sin(k_vec[e][0] * r[0] + k_vec[e][1] * r[1] + k_vec[e][2] * r[2]);

                                phase = std::complex<double>(ca, -sa);

                                Ham_K[z][x][y] +=
                                    phase * std::complex<double>(Ham_R[z][i - n_min[0]][j - n_min[1]][k -
                                                                                                      n_min[2]][x][y],
                                                                 0); //hamiltonian k-space
                            }
                        }
                    }
                }
            }
        }

        for (int x = 0; x < num_orb; x++)
        {
            for (int y = 0; y < num_orb; y++)
            {
                h_up[x * num_orb + y] = Ham_K[0][x][y];
                h_dn[x * num_orb + y] = Ham_K[1][x][y];
            }
        }

        zheev(&JOBZ, &UPLO, &num_orb, h_up, &num_orb, W_up, WORK, &LWORK, RWORK, &INFO_up);
        zheev(&JOBZ, &UPLO, &num_orb, h_dn, &num_orb, W_dn, WORK, &LWORK, RWORK, &INFO_dn);

        if (INFO_up != 0 || INFO_dn != 0){
            diag_error = true;
        }

        for (int x = 0; x < num_orb; x++)
        {
            egval[0][e][x] = W_up[x];
            egval[1][e][x] = W_dn[x];

            for (int y = 0; y < num_orb; y++)
            {
                egvec[0][e][y][x] = h_up[x * num_orb + y];
                egvec[1][e][y][x] = h_dn[x * num_orb + y];
            }
        }

    }


    delete[] h_up;
    delete[] h_dn;
    delete[] W_up;
    delete[] W_dn;
    delete[] WORK;
    delete[] RWORK;
    delete[] Ham_K;

}
