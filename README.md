# xchange.x
This program calculates the isotropic exchange coupling parameters $J_{ij}$ for Heisenberg model $H = \sum_{i < j} J_{ij} \mathbf{S}_i \mathbf{S}_j$ using Green's function technique (see [1](https://www.sciencedirect.com/science/article/abs/pii/0304885387907219), [2](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.71.184434)):

$$  J_{ij} = \frac{1}{2 \pi S^2}  {\mathrm Im} \left( \int \limits_{-\infty}^{E_F} d \epsilon \Delta_i G_{ij \downarrow} (\epsilon) \Delta_j G_{ji \uparrow} (\epsilon) \right), $$

where $m, m^{\prime},  n, n^{\prime}$ are orbital quantum numbers, $S$ is the spin quantum number, $E_F$ is the Fermi energy. $\Delta^{m m^{\prime}}_i$ and $G(\epsilon)$ are the on-site potential and  Green's function, respectively. 

# Dependencies
**python** version requires [numba](https://numba.pydata.org) \
**c++** version is implemented as cpp_modules, which requires [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) and [pybind11](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/index.html#)  libraries during compilation via CMake:\
mkdir build \
cd build/ \
cmake -DCMAKE_BUILD_TYPE=Releas .. (add Eigen and pybind11 path)\
make 

# Usage
Run python xchange.py within directory with in.json together with hopping files spin_up.dat and spin_dn.dat from [wannier90](https://wannier.org).

# Example 1 (square lattice)
As an example we calculate exchange interactions in square lattice with nearest neighbor hopping $t = -0.1$ eV and Coulomb potential $U \simeq \Delta = 1$ eV. Here we approximate Coulomb potential with on-site splitting: 
![alt text](https://github.com/danis-b/xchange/blob/main/examples/square_lattice/DOS.png)

in.json file contains the following information:
```json
   "cell_vectors": [[1.000000,   0.000000,   0.000000], 
                    [0.000000,   1.000000,   0.000000], 
                    [0.000000,   0.000000,  10.000000]],
   "number_of_magnetic_atoms": 1,
   "positions_of_magnetic_atoms": [[0.00000,   0.00000,  0.00000]],
   "central_atom": 0,
   "orbitals_of_magnetic_atoms":[1],
   "max_sphere_num": 2,
   "exchange_for_specific_atoms": [[0, 0, 0, 0]],
   "spin": 0.5,
   "ncol": 1000,
   "nrow": 500,
   "smearing": 0.01,
   "e_low": -2,
   "e_fermi": 0,
   "kmesh": [20, 20, 1]
```
* cell_vectors - (3x3)(dfloat) matrix of unit cell vectors (in Ang);
* number_of_magnetic_atoms - (int) number of magnetic atoms in unit cell;
* positions_of_magnetic_atoms - (3 x number_of_magnetic_atoms)(dfloat) matrix of magnetic atoms positions in unit cell (in Ang);
* central_atom - (int) atomic number of magnetic atom, for which we want to calculate exchange interactions (0 in our case);
* orbitals_of_magnetic_atoms - (number_of_magnetic_atoms) (int) array of magnetic orbitals 
* max_sphere_num - (int) maximum number of coordination sphere to calculate exchange couplings around central_atom. This number breaks the loop;
* exchange_for_specific_atoms - (4 x required setups)(int) matrix used for calculation of exchange couplings **only** between central_atom and atom given by element[3], connected  with radius-vector **R** = element[0] * **cell_vector1** + element[1] * **cell_vector2** + element[2] * **cell_vector3**. All zero elements [0,0,0,0] makes the program calculate all possible exchange interactions restricted by max_sphere_num; 
* spin - (int) spin number of the system;
* ncol - (int) number of energy point for integration along real axis;
* nrow - (int) number of energy point for integration along imaginary axis;  
* smearing - (dfloat) numerical smearing parameter;
* e_low -  (dfloat) lower boundary of orbital energies;
* e_fermi - (dfloat) Femi energy;
* kmesh [3x1] (int) array - k-mesh for Brillouin zone integration; 

Resulting value of nearest neighbor exchange coupling $J \sim$ 64.6 meV.  In the limit of $\Delta \gg t$ exchange coupling can be estimated via analytical formula $J = \frac{4t^2}{\Delta}$. You can make sure this increasing on-site splitting $\Delta$ in spin_up.dat and spin_dn.dat files (element 0 0 0 1 1).  


# Example 2 (BaMoP2O8)

Now let's calcualte exchange interactions in BaMoP2O8 system [[Phys. Rev. B 98, 094406 (2018)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.094406)]. DFT+U electronic structure near Fermi level was parametrized by Wannier functions of one Mo(d) and eight O(p) orbitals (5 + 8 * 3 = 29 orbitals):
![alt text](https://github.com/danis-b/xchange/blob/main/examples/BaMoPO/BANDS.png)

in.json file in this case is the following:

```json
   "cell_vectors": [[4.880000,   0.000000,   0.000000], 
                    [2.028352,   4.438489,   0.000000], 
                    [0.547938,   0.352040,  7.788818]],
   "number_of_magnetic_atoms": 1,
   "positions_of_magnetic_atoms": [[0.00000,   0.00000,  0.00000]],
   "central_atom": 0,
   "orbitals_of_magnetic_atoms":[5],
   "max_sphere_num": 3,
   "exchange_for_specific_atoms": [[0, 0, 0, 0]],
   "spin": 1,
   "ncol": 1000,
   "nrow": 500,
   "smearing": 0.01,
   "e_low": -9,
   "e_fermi": 3.2188,
   "kmesh": [5, 5, 5]
```
**Orbitals of magnetic atoms in spin_up.dat and spin_dn.dat go first. Keep this in mind during wannierization**.

As a result, program will print the occupation difference between spin up and  down (i.e. magnetization):\
Occupation matrix (N_up - N_dn) for atom  0  
```math
\begin{pmatrix}
0.871 & -0.018 &  0.002 &  0.103 &  -0.018 \\
-0.018 &  0.826 &  0.077 & 0.141 & 0.051 \\
0.002 & 0.077 & 0.014 & 0.013 & 0.004 \\
0.103 & 0.141 & 0.013 & 0.087 & 0.004 \\
-0.018 & 0.051 & 0.004 & 0.004 & 0.040 
\end{pmatrix}
```

and exchange interactions between atoms:\
Atom 0 (000)<-->Atom 0 ( 0 1 0 ) with radius 4.8800  is # 1 
```math
\begin{pmatrix}
-0.000052 &  0.000110 &  0.000033 &  -0.000105 &  -0.000180 \\
0.000191 &  0.000510 &  0.000163 &  -0.000879 &  -0.000286 \\
0.000016 &  0.000045 &  0.000017 &  -0.000072 &  -0.000029 \\
-0.000007 & 0.000281 &  0.000060 & -0.000186 &  -0.000098 \\
0.000013 &  0.000144 &  0.000055 &  -0.000169 &  -0.000133 
\end{pmatrix}
```
\#  0 0 0 1 0 0.000157 eV

...

Atom 0 (000)<-->Atom 0 ( 1 -1 0 ) with radius 5.2756  is # 1
```math
\begin{pmatrix}
0.003642 & 0.001935 &  0.000210 &  0.000819 &  0.001373 \\
0.001087 & 0.000993 &  0.000105 &  0.000336 &  0.000724 \\
0.000119 & 0.000101 &  0.000017 &  0.000036  & 0.000074 \\
0.000693 &  0.000426 &  0.000040 &  0.000192 &  0.000308 \\
0.000054 & 0.000048 &  0.000005 & 0.000016 & 0.000019 
\end{pmatrix}
```
\#  0 0 1 -1 0 0.004862 eV

...

Atom 0 (000)<-->Atom 0 ( 0 0 1 ) with radius 7.8160  is # 1 
```math
\begin{pmatrix}
0.000469 & 0.000410 & 0.000045 & 0.000138 & -0.000075 \\
-0.000461 & -0.000687 & -0.000075 & -0.000177 & 0.000237 \\
-0.000041 & -0.000063 & -0.000010 & -0.000015 & 0.000021 \\
-0.000001 & -0.000042 & -0.000002 & -0.000018 & 0.000025 \\
0.000036 & 0.000017 & 0.000003 & 0.000008 & 0.000038 
\end{pmatrix}
```
\#  0 0 0 0 1 0.000282 eV

It results $J^\prime \sim$ 0.1 meV, $J_c \sim$ 0.3 meV and $J \sim$ 4.8 meV in figure above. 
