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
                                    std::vector<int> &mag_orbs)
{
    int shift_i;
    double weight = 1.0 / num_kpoints;

    Eigen::MatrixXd occupation_up = Eigen::MatrixXd::Zero(mag_orbs[central_atom], mag_orbs[central_atom]);
    Eigen::MatrixXd occupation_dn = Eigen::MatrixXd::Zero(mag_orbs[central_atom], mag_orbs[central_atom]);

    shift_i = 0;
    for (int x = 0; x < central_atom; ++x)
    {
        shift_i += mag_orbs[x];
    }

    int idx = num_kpoints * num_orb * num_orb;
    for (int num = 0; num < num_freq; ++num)
    {
        Eigen::MatrixXcd greenR_ii_up = Eigen::MatrixXcd::Zero(mag_orbs[central_atom], mag_orbs[central_atom]);
        Eigen::MatrixXcd greenR_ii_dn = Eigen::MatrixXcd::Zero(mag_orbs[central_atom], mag_orbs[central_atom]);

        Eigen::MatrixXcd identity = E[num] * Eigen::MatrixXcd::Identity(num_orb, num_orb);

        for (int e = 0; e < num_kpoints; ++e)
        {
            const int startIdx = (num_orb * num_orb) * e;
            Eigen::MatrixXcd Ham_K_up = Eigen::Map<const Eigen::MatrixXcd>(&Ham_K[startIdx], num_orb, num_orb);
            Eigen::MatrixXcd Ham_K_dn = Eigen::Map<const Eigen::MatrixXcd>(&Ham_K[idx + startIdx], num_orb, num_orb);

            Eigen::MatrixXcd loc_greenK_up = (identity - Ham_K_up).inverse();
            Eigen::MatrixXcd loc_greenK_dn = (identity - Ham_K_dn).inverse();

            greenR_ii_up += weight * loc_greenK_up.block(shift_i, shift_i, mag_orbs[central_atom], mag_orbs[central_atom]);
            greenR_ii_dn += weight * loc_greenK_dn.block(shift_i, shift_i, mag_orbs[central_atom], mag_orbs[central_atom]);
        }

        occupation_up -= (greenR_ii_up * dE[num]).imag();
        occupation_dn -= (greenR_ii_dn * dE[num]).imag();
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
