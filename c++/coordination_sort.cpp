#include <complex>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v)
{

    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
                     [&v](size_t i1, size_t i2)
                     { return v[i1] < v[i2]; });

    return idx;
}

// This function sorts atoms depending on the radius from the central atom with index 'central_atom'
void coordination_sort(int central_atom, int num_atoms, int n_min[3], int n_max[3], double cell_vectors[3][3],
                       std::vector<std::vector<double>> &positions,
                       std::vector<double> &radius, std::vector<std::vector<int>> &index)
{

    int i, j, k, atom2;
    int num_points;

    int n_size[3];
    double r[3];

    for (int z = 0; z < 3; ++z)
    {
        n_size[z] = n_max[z] - n_min[z] + 1; // Plus 1 for 0th
    }

    num_points = num_atoms * (n_size[0] * n_size[1] * n_size[2]);

    // index = [i, j, k, atom]
    std::vector<std::vector<int>> idx(
        num_points, std::vector<int>(
                        4));

    std::vector<double> rad(num_points);

    // Index filling
    i = n_min[0];
    j = n_min[1];
    k = n_min[2];
    atom2 = 0;
    for (int z = 0; z < num_points; ++z)
    {
        idx[z][0] = i;
        idx[z][1] = j;
        idx[z][2] = k;
        idx[z][3] = atom2;

        for (int y = 0; y < 3; ++y)
        {
            r[y] = i * cell_vectors[0][y] + j * cell_vectors[1][y] + k * cell_vectors[2][y] +
                   (positions[atom2][y] - positions[central_atom][y]);
        }

        rad[z] = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
        atom2++;
        if (atom2 == num_atoms)
        {
            atom2 = 0;
            k++;
            if (k == n_max[2] + 1)
            {
                k = n_min[2];
                j++;
            }
            if (j == n_max[1] + 1)
            {
                j = n_min[1];
                i++;
            }
        }
    }

    k = 0;
    for (auto i : sort_indexes(rad))
    {
        for (j = 0; j < 4; j++)
        {
            index[k][j] = idx[i][j];
        }
        radius[k] = rad[i];
        k++;
    }
}
