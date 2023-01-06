# xchange.x
This program calculates the isotropic exchange coupling parameters $J_{ij}$ for Heisenberg model $H = \sum_{i < j} J_{ij} \mathbf{S}_i \mathbf{S}_j$ using Green's function technique (see [1](https://www.sciencedirect.com/science/article/pii/0304885387907219?via%3Dihub), [2](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.71.184434)]):

$$  J_{ij} = \frac{1}{2 \pi S^2}  \int \limits_{-\infty}^{E_F} d \epsilon \, {\mathrm Im} \left( \sum \limits_{m, m^{\prime},  n, n^{\prime}} \Delta^{m m^{\prime}}_i G^{m^{\prime} n}_{ij \downarrow} (\epsilon) \Delta^{n n^{\prime}}_j G^{n^{\prime} m}_{ji \uparrow} (\epsilon) \right), $$

where $m, m^{\prime},  n, n^{\prime}$ are orbital quantum numbers, $S$ is the spin quantum number, $E_F$ is the Fermi energy. $\Delta^{m m^{\prime}}_i$ and $G(\epsilon)$ are the on-site potential and  Green's function, respectively. Python version requires [numba](https://numba.pydata.org), while c++ version needs to be compiled with the additional [json](https://github.com/nlohmann/json) library.

# Usage 
As an example let's calculate isotropic exchange interactions in BaMoP2O8 system [[Phys. Rev. B 98, 094406 (2018)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.094406)]. DFT+U electronic structure near Fermi level was parametrized by Wannier functions of one Mo(d) and eight O(p) orbitals (5 + 8 * 3 = 29 orbitals):
![alt text](https://github.com/danis-b/xchange/blob/main/example/BANDS.png)

in.json file contains the following information:

```json
   "cell_vectors": [[4.880000,   0.000000,   0.000000], [2.028352,   4.438489,   0.000000], [0.547938,   0.352040,  7.788818]],
   "number_of_magnetic_atoms": 1,
   "positions_of_magnetic_atoms": [[0.00000,   0.00000,  0.00000]],
   "central_atom": 0,
   "orbitals_of_magnetic_atoms":[5],
   "max_sphere_num": 3,
   "exchange_for_specific_atoms": [0, 0, 0, 0],
   "spin": 1,
   "ncol": 1000,
   "nrow": 500,
   "smearing": 0.01,
   "e_low": -9,
   "e_fermi": 3.2188,
   "kmesh": [5, 5, 5]
```
* cell_vectors - unit cell vectors;
* number_of_magnetic_atoms - number of  magnetic atoms in unit cell (1 Mo in our case);
 

