#include <complex>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <omp.h>

std::vector<double> calc_occupation(int central_atom,
                                    int num_orb,
                                    int num_kpoints,
                                    int num_freq,
                                    std::vector<std::complex<double>> &ham_K,
                                    std::vector<std::complex<double>> &freq,
                                    std::vector<std::complex<double>> &d_freq,
                                    std::vector<int> &mag_orbs)
{
    Eigen::MatrixXd occupation_up = Eigen::MatrixXd::Zero(mag_orbs[central_atom], mag_orbs[central_atom]);
    Eigen::MatrixXd occupation_dn = Eigen::MatrixXd::Zero(mag_orbs[central_atom], mag_orbs[central_atom]);

    int shift_i = 0;
    for (int x = 0; x < central_atom; ++x)
    {
        shift_i += mag_orbs[x];
    }

    #pragma omp parallel for default(none) shared(ham_K, shift_i, mag_orbs, freq, d_freq, central_atom, num_orb, num_kpoints, num_freq) reduction(+:occupation_up, occupation_dn)
    for (int num = 0; num < num_freq; ++num)
    {
        double weight = 1.0 / num_kpoints;
        int idx = num_kpoints * num_orb * num_orb;

        Eigen::MatrixXcd greenR_ii_up = Eigen::MatrixXcd::Zero(mag_orbs[central_atom], mag_orbs[central_atom]);
        Eigen::MatrixXcd greenR_ii_dn = Eigen::MatrixXcd::Zero(mag_orbs[central_atom], mag_orbs[central_atom]);

        Eigen::MatrixXcd identity = freq[num] * Eigen::MatrixXcd::Identity(num_orb, num_orb);

        for (int e = 0; e < num_kpoints; ++e)
        {
            const int startIdx = (num_orb * num_orb) * e;
            Eigen::MatrixXcd ham_K_up = Eigen::Map<const Eigen::MatrixXcd>(&ham_K[startIdx], num_orb, num_orb);
            Eigen::MatrixXcd ham_K_dn = Eigen::Map<const Eigen::MatrixXcd>(&ham_K[idx + startIdx], num_orb, num_orb);

            Eigen::MatrixXcd loc_greenK_up = (identity - ham_K_up).inverse();
            Eigen::MatrixXcd loc_greenK_dn = (identity - ham_K_dn).inverse();

            greenR_ii_up.noalias() += weight * loc_greenK_up.block(shift_i, shift_i, mag_orbs[central_atom], mag_orbs[central_atom]);
            greenR_ii_dn.noalias() += weight * loc_greenK_dn.block(shift_i, shift_i, mag_orbs[central_atom], mag_orbs[central_atom]);
        }

        occupation_up.noalias() -= (greenR_ii_up * d_freq[num]).imag();
        occupation_dn.noalias() -= (greenR_ii_dn * d_freq[num]).imag();
    }

    occupation_up *= (1.0 / M_PI);
    occupation_dn *= (1.0 / M_PI);


    std::vector<double> occ;
    occ.reserve(2 * mag_orbs[central_atom] * mag_orbs[central_atom]);

    for (int i = 0; i < mag_orbs[central_atom]; i++)
    {
        for (int j = 0; j < mag_orbs[central_atom]; j++)
        {
            occ.push_back(occupation_up(i, j));
        }
    }


    for (int i = 0; i < mag_orbs[central_atom]; i++)
    {
        for (int j = 0; j < mag_orbs[central_atom]; j++)
        {
            occ.push_back(occupation_dn(i, j));
        }
    }

    return occ;
}
