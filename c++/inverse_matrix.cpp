#include <complex>
#include <iostream>

// external lapack library for matrix inversion
extern "C" void zgetrf(int *N, int *M, std::complex<double> *A, int *LDA, int *IPIV, int *INFO);
extern "C" void zgetri(int *N, std::complex<double> *A, int *LDA, int *IPIV, std::complex<double> *WORK, int *LWORK, int *INFO);

//This function inverts  the square complex matrix 'loc_greenK' with dimention 'num_orb x num_orb'
void inverse_matrix(int num_orb, std::complex<double> ***loc_greenK, bool err)
{

    int LWORK = 2 * num_orb;
    std::complex<double> *WORK = new std::complex<double>[LWORK];
    int *IPIV = new int[num_orb];
    int INFO_up, INFO_dn;

    std::complex<double> *h_up = new std::complex<double>[num_orb * num_orb];
    std::complex<double> *h_dn = new std::complex<double>[num_orb * num_orb];

    for (int x = 0; x < num_orb; ++x)
    {
        for (int y = 0; y < num_orb; ++y)
        {
            h_up[x * num_orb + y] = loc_greenK[0][x][y];
            h_dn[x * num_orb + y] = loc_greenK[1][x][y];
        }
    }

    zgetrf(&num_orb, &num_orb, h_up, &num_orb, IPIV, &INFO_up);
    zgetri(&num_orb, h_up, &num_orb, IPIV, WORK, &LWORK, &INFO_up);

    zgetrf(&num_orb, &num_orb, h_dn, &num_orb, IPIV, &INFO_dn);
    zgetri(&num_orb, h_dn, &num_orb, IPIV, WORK, &LWORK, &INFO_dn);

    if (INFO_up != 0 || INFO_dn != 0)
    {
        err = true;
    }

    for (int x = 0; x < num_orb; ++x)
    {
        for (int y = 0; y < num_orb; ++y)
        {
            loc_greenK[0][x][y] = h_up[x * num_orb + y];
            loc_greenK[1][x][y] = h_dn[x * num_orb + y];
        }
    }

    delete[] h_up;
    delete[] h_dn;
    delete[] WORK;
    delete[] IPIV;
}
