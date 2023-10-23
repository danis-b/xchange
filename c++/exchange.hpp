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
                                  std::vector<int> &mag_orbs)
{

    std::complex<double> cplx_i(0, 1);
    double weight = 1.0 / num_kpoints;

    double shift_i = 0;
    for (int x = 0; x < central_atom; ++x)
    {
        shift_i += mag_orbs[x];
    }

    double shift_j = 0;
    for (int x = 0; x < index_temp[3]; ++x)
    {
        shift_j += mag_orbs[x];
    }

    std::vector<double> exchange;
    exchange.reserve(mag_orbs[central_atom] * mag_orbs[central_atom]);

    // on-site energy
    Eigen::MatrixXcd delta_i = Eigen::MatrixXcd::Zero(mag_orbs[central_atom], mag_orbs[central_atom]);
    Eigen::MatrixXcd delta_j = Eigen::MatrixXcd::Zero(mag_orbs[index_temp[3]], mag_orbs[index_temp[3]]);

    // Green's function in R-space
    Eigen::MatrixXcd greenR_ij = Eigen::MatrixXcd::Zero(mag_orbs[central_atom], mag_orbs[index_temp[3]]);
    Eigen::MatrixXcd greenR_ji = Eigen::MatrixXcd::Zero(mag_orbs[index_temp[3]], mag_orbs[central_atom]);
    Eigen::MatrixXcd dot_product = Eigen::MatrixXcd::Zero(mag_orbs[central_atom], mag_orbs[central_atom]);

    Eigen::MatrixXd exchange_temp = Eigen::MatrixXd::Zero(mag_orbs[central_atom], mag_orbs[central_atom]);

    Eigen::MatrixXcd loc_greenK_up(num_orb, num_orb);
    Eigen::MatrixXcd loc_greenK_dn(num_orb, num_orb);

    Eigen::MatrixXcd Ham_K_up(num_orb, num_orb);
    Eigen::MatrixXcd Ham_K_dn(num_orb, num_orb);

    // radius-vector
    double r[3];
    for (int z = 0; z < 3; ++z)
    {
        r[z] = index_temp[0] * cell_vec[z] + index_temp[1] * cell_vec[3 + z] + index_temp[2] * cell_vec[6 + z];
    }

    // phases for all k-points
    std::vector<std::complex<double> > phase(num_kpoints);
    for (int e = 0; e < num_kpoints; ++e)
    {
        double r_dot_k = 0;

        for (int x = 0; x < 3; ++x)
        {
            r_dot_k += k_vec[3 * e + x] * r[x];
        }

        phase[e] = std::exp(cplx_i * r_dot_k);
    }



    int idx = num_kpoints * num_orb * num_orb;
    for (int num = 0; num < num_freq; ++num)
    {
        greenR_ij.setZero();
        greenR_ji.setZero();

        delta_i.setZero();
        delta_j.setZero();

        Eigen::MatrixXcd identity = E[num] * Eigen::MatrixXcd::Identity(num_orb, num_orb);

        for (int e = 0; e < num_kpoints; ++e)
        {
            for (int x = 0; x < num_orb; ++x)
            {
                for (int y = 0; y < num_orb; ++y)
                {
                    Ham_K_up(x, y) = Ham_K[(num_orb * num_orb) * e + num_orb * x + y];
                    Ham_K_dn(x, y) = Ham_K[idx + (num_orb * num_orb) * e + num_orb * x + y];
                }
            }

            loc_greenK_up = (identity - Ham_K_up).inverse();
            loc_greenK_dn = (identity - Ham_K_dn).inverse();

            greenR_ij += weight * phase[e] * loc_greenK_dn.block(shift_i, shift_j, mag_orbs[central_atom], mag_orbs[index_temp[3]]);
            greenR_ji += weight * conj(phase[e]) * loc_greenK_up.block(shift_i, shift_j, mag_orbs[central_atom], mag_orbs[index_temp[3]]).transpose();

            delta_i += weight * (Ham_K_up.block(shift_i, shift_i, mag_orbs[central_atom], mag_orbs[central_atom]) - Ham_K_dn.block(shift_i, shift_i, mag_orbs[central_atom], mag_orbs[central_atom]));
            delta_j += weight * (Ham_K_up.block(shift_j, shift_j, mag_orbs[index_temp[3]], mag_orbs[index_temp[3]]) - Ham_K_dn.block(shift_j, shift_j, mag_orbs[index_temp[3]], mag_orbs[index_temp[3]]));
        }

        dot_product = delta_i * greenR_ij * delta_j * greenR_ji;

        exchange_temp -= (dot_product * dE[num]).imag();
    }

    // Apply the scaling factor at the end
    exchange_temp *= 1 / (2 * M_PI * spin * spin);

    for (int i = 0; i < mag_orbs[central_atom]; i++)
    {
        for (int j = 0; j < mag_orbs[central_atom]; j++)
        {
            exchange.push_back(exchange_temp(i, j));
        }
    }

    return exchange;
}
