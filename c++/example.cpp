  diag_error = false;

#pragma omp parallel for private(x, y, z, p, i, j, k, r, ca, sa, phase, INFO_up, INFO_dn) 
    for (e = 0; e < num_kpoints; e++)
    {

        for (x = 0; x < num_orb; x++)
        {
            for (y = 0; y < num_orb; y++)
            {
                for (z = 0; z < 2; z++)
                {
                    Ham_K[z][x][y] = std::complex<double>(0, 0);
                }
            }
        }

        for (x = 0; x < num_orb; x++)
        {
            for (y = 0; y < num_orb; y++)
            {
                for (z = 0; z < 2; z++)
                {

                    for (i = n_min[0]; i <= n_max[0]; i++)
                    {
                        for (j = n_min[1]; j <= n_max[1]; j++)
                        {
                            for (k = n_min[2]; k <= n_max[2]; k++)
                            {
                                for (p = 0; p < 3; p++)
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

        for (x = 0; x < num_orb; x++)
        {
            for (y = 0; y < num_orb; y++)
            {
                h_up[x * num_orb + y] = Ham_K[0][x][y];
                h_dn[x * num_orb + y] = Ham_K[1][x][y];
            }
        }

        zheev_(&JOBZ, &UPLO, &num_orb, h_up, &num_orb, W_up, WORK, &LWORK, RWORK, &INFO_up);
        zheev_(&JOBZ, &UPLO, &num_orb, h_dn, &num_orb, W_dn, WORK, &LWORK, RWORK, &INFO_dn);

        if (INFO_up != 0 || INFO_dn != 0){
            diag_error = true;
            break;
        }

        for (x = 0; x < num_orb; x++)
        {
            egval[0][e][x] = W_up[x];
            egval[1][e][x] = W_dn[x];

            for (y = 0; y < num_orb; y++)
            {
                egvec[0][e][y][x] = h_up[x * num_orb + y];
                egvec[1][e][y][x] = h_dn[x * num_orb + y];
            }
        }

    }

    if (diag_error)
    {
        std::cout << "ERROR! Problem with diagonalization! Please, check the input parameters and Hamiltonian!"
                  << std::endl;
        return 0;
    }