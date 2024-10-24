# NLA Challenge 2 (Group 6)

Hands-on Challenge 2 of the course _Numerical Linear Algebra_ by Professor Antonietti, Polimi, a.y. 2024/25.

The description of the challenge is [here](Challenge2_description.pdf). We really thank the teachers for the opportunity.

## 1. Execution Results

- To run the main code `Challenge2.cpp` on terminal using the Eigen library:

  ```bash
  g++ -I ${mkEigenInc} Challenge2.cpp -o exec
  ./exec einstein.jpg > output.txt
  ```

- To solve the eigenproblem $A^TA \mathbf{x}=\lambda \mathbf{x}$ on terminal using the power method available on Lis (Library of Iterative Solvers for linear systems):

  ```bash
  mpicc -DUSE_MPI -I${mkLisInc} -L${mkLisLib} -llis etest1.c -o eigen1
  mpirun -n 4 ./eigen1 Ata.mtx eigvec1.mtm hist1.txt -e pi -emaxiter 100 -etol 1.e-8 > output_Lis_1.txt
  ```

  To accelerate the convergence you could apply a shift $\mu$ to the system ...  

  The command is

  ```bash
  mpirun -n 4 ./eigen1 Ata.mtx eigvec2.mtm hist2.txt -e pi -emaxiter 100 -etol 1.e-8 -shift 695 > output_Lis_2.txt
  ```

  In addiction, you can speed up the convergence thanks to the inverse power method with shift ...  

  The command is

  ```bash
  mpirun -n 4 ./eigen1 Ata.mtx eigvec3.mtm hist3.txt -e ii -emaxiter 100 -etol 1.e-8 -shift 16083 > output_Lis_3.txt
  ```

---

### 2. Output Results

The detailed output results are shown in files below:

- For the `Challenge2.cpp` output: **[output.txt](output.txt)**

- For the Lis commands outputs: **[output_Lis_1.txt](output_Lis_1.txt)**, **[output_Lis_2.txt](output_Lis_2.txt)**  and **[output_Lis_3.txt](output_Lis_3.txt)**

---

### 3. Image Results

| Einstein                  | C1D1                     | C2D2                     |
| ------------------------- | ------------------------ | ------------------------ |
| ![Einstein](einstein.jpg) | ![C1D1](output_C1D1.png) | ![C2D2](output_C2D2.png) |

| Checkerboard                             | Noised Checkerboard                                    | C3D3                     | C4D4                     |
| ---------------------------------------- | ------------------------------------------------------ | ------------------------ | ------------------------ |
| ![Checkerboard](output_checkerboard.png) | ![Noised Checkerboard](output_checkerboard_noised.png) | ![C3D3](output_C3D3.png) | ![C4D4](output_C4D4.png) |
