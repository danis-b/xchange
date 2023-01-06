# xchange.x
This program calculates the isotropic exchange coupling parameters $J_{ij}$ for Heisenberg model $H = \sum_{i < j} J_{ij} \mathbf{S}_i \mathbf{S}_j$ using Green's function technique (see [1](https://www.sciencedirect.com/science/article/pii/0304885387907219?via%3Dihub), [2](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.71.184434)]):

$$  J_{ij} = \frac{1}{2 \pi S^2}  \int \limits_{-\infty}^{E_F} d \epsilon \, {\mathrm Im} \left( \sum \limits_{m, m^{\prime},  n, n^{\prime}} \Delta^{m m^{\prime}}_i G^{m^{\prime} n}_{ij \downarrow} (\epsilon) \Delta^{n n^{\prime}}_j G^{n^{\prime} m}_{ji \uparrow} (\epsilon) \right), $$

where $m, m^{\prime},  n, n^{\prime}$ are orbital quantum numbers, $S$ is the spin quantum number, $E_F$ is the Fermi energy. $\Delta^{m m^{\prime}}_i$ and $G(\epsilon)$ are the on-site potential and  Green's function, respectively. Python version requires [numba](https://numba.pydata.org), while c++ version needs to be compiled with the additional [json](https://github.com/nlohmann/json) library.

# Usage 
As an example let's calculate isotropic exchange interactions in BaMoP2O8 system [[Phys. Rev. B 98, 094406 (2018)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.094406)]. DFT+U electronic structure near Fermi level was parametrized by Wannier functions of one Mo(d) and eight O(p) orbitals (5 + 8 * 3 = 29 orbitals):
![alt text](https://github.com/danis-b/xchange/blob/main/example/BANDS.png)

in.json file contains the following information:

```json
   "cell_vectors": [[4.880000,   0.000000,   0.000000], 
                    [2.028352,   4.438489,   0.000000], 
                    [0.547938,   0.352040,  7.788818]],
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
* cell_vectors - (3x3)(dfloat) array of unit cell vectors (in Ang);
* number_of_magnetic_atoms - (int) number of magnetic atoms in unit cell (1 Mo in our case);
* positions_of_magnetic_atoms - (3 x number_of_magnetic_atoms)(dfloat) array of magnetic atoms positions in unit cell (in Ang);
* central_atom - (int) atomic number of magnetic atom, for which we want to calculate exchange interactions (0 in our case);
* orbitals_of_magnetic_atoms - (number_of_magnetic_atoms) (int) array of magnetic orbitals (in our case it is [5]) 
* max_sphere_num - (int) maximum number of coordination sphere to calculate exchange couplings around central_atom. This number breaks the loop;
* exchange_for_specific_atoms - (4)(int) array used for calculation of exchange couplings **only** between central_atom and atom given by element[3], connected  with radius-vector **R** = element[0] * **cell_vector1** + element[1] * **cell_vector2** + element[2] * **cell_vector3**. All zero elements [0,0,0,0] makes the program calculate all possible exchange interactions restricted by max_sphere_num; 
* spin - (int) spin number of the system;
* ncol - (int) number of energy point for integration along real axis;
* nrow - (int) number of energy point for integration along imaginary axis;  
* smearing - (dfloat) numerical smearing parameter;
* e_low -  (dfloat) lower boundary of orbital energies;
* e_fermi - (dfloat) Femi energy;
* kmesh [3x1] (int) array - k-mesh for Brillouin zone integration; 

 
Both version of the program needs to be started at the same folder with in.json and hopping parameters from wannier90 (seedname_hr.dat) spin_up.dat and spin_dn.dat files. **Please, make sure that the additional lines before hopping parameters in seedname_up/dn.dat files are removed. Orbitals of magnetic atoms should be at the beginning among all Wannier  functions**.

As a result, program will print the occupation difference between spin up and  down (i.e. magnetization):

Occupation matrix (N_up - N_dn) for atom  0  \
0.871 -0.018 0.002 0.103 -0.018
-0.018 0.826 0.077 0.141 0.051
0.002 0.077 0.014 0.013 0.004
0.103 0.141 0.013 0.087 0.004
-0.018 0.051 0.004 0.004 0.040
 
Trace equals to:  1.836

