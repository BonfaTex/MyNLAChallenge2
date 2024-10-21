#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <unsupported/Eigen/SparseExtra>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
/*****************************************************************************************************************/
// More details and explanation are at this link: 
/*****************************************************************************************************************/
using namespace Eigen;
typedef Eigen::Triplet<double> T;

/******************This is the defined functions field like convolution, filter, export matrix, etc***********************/
// 1: Definite the convolution function as matrix vector production by simulating the whole procedure step by step, return the sparsematrix mn*mn
SparseMatrix<double, RowMajor> convolutionMatrix(const Matrix<double, Dynamic, Dynamic, RowMajor> &kernel, int height, int width)
{
    const int kernel_size = kernel.rows();
    const int m = height;
    const int n = width;
    const int mn = m * n;
    SparseMatrix<double, RowMajor> A(mn, mn);

    std::vector<T> hav2TripletList;
    hav2TripletList.reserve(mn * kernel_size * kernel_size);
    // do convolution from left to right and up to down with zero padding, index_Ai and index_Aj are row and column index
    for (int i = 0; i < m; i++) // We have use zero padding, so m means horizontal steps
    {
        for (int j = 0; j < n; j++) // We have use zero padding, so n means vertical steps
        {
            int index_Ai = i * n + j; // row index of A
            // ki, kj are respectively relative row index and column index of kernel, central point is (0,0)
            for (int ki = -kernel_size / 2; ki <= kernel_size / 2; ki++)
            {
                for (int kj = -kernel_size / 2; kj <= kernel_size / 2; kj++) // Do interation within one kernel
                {
                    // ci: contribute to n(width) shift each time when there is a vertical moving for convolution or inside the kernel
                    int ci = i + ki;
                    // cj: contribute just 1 shift each time, when there is a horizontal moving for convilution or inside the kernel
                    int cj = j + kj;
                    if (ci >= 0 && ci < m && cj >= 0 && cj < n && kernel(ki + kernel_size / 2, kj + kernel_size / 2) != 0) // check if the kernel element itselfe is 0
                    {
                        int index_Aj = ci * n + cj;
                        // push nonzero elements to list
                        hav2TripletList.push_back(Triplet<double>(index_Ai, index_Aj, kernel(ki + kernel_size / 2, kj + kernel_size / 2)));
                    }
                }
            }
        }
    }
    // get the sparsematrix from tripletlist
    A.setFromTriplets(hav2TripletList.begin(), hav2TripletList.end());
    return A;
}

// 2: An alternative definition for kernels H 3 by 3 to realize the convolution, easy to generalize
SparseMatrix<double, RowMajor> convolutionMatrix2(const Matrix<double, Dynamic, Dynamic, RowMajor> &kernel, int m, int n)
{
    const int mn = m * n;
    SparseMatrix<double, RowMajor> A(mn, mn);
    std::vector<T> tripletList;
    tripletList.reserve(mn * 9);
    for (int i = 0; i < mn; ++i)
    {
        // top center (not first n rows)
        if (i - n + 1 > 0 && kernel(0, 1) != 0)
            tripletList.push_back(T(i, i - n, kernel(0, 1)));
        // middle center (always)
        if (kernel(1, 1) != 0)
            tripletList.push_back(T(i, i, kernel(1, 1)));
        // bottom center (not last n rows)
        if (i + n - 1 < mn - 1 && kernel(2, 1) != 0)
            tripletList.push_back(T(i, i + n, kernel(2, 1)));

        if (i % n != 0) // we can go left
        {
            // top left
            if (i - n > 0 && kernel(0, 0) != 0)
                tripletList.push_back(T(i, i - n - 1, kernel(0, 0)));
            // middle left
            if (i > 0 && kernel(1, 0) != 0)
                tripletList.push_back(T(i, i - 1, kernel(1, 0)));
            // bottom left
            if (i + n - 2 < mn - 1 && kernel(2, 0) != 0)
                tripletList.push_back(T(i, i + n - 1, kernel(2, 0)));
        }

        if ((i + 1) % n != 0) // we can go right
        {
            // top right
            if (i - n + 2 > 0 && kernel(0, 2) != 0)
                tripletList.push_back(T(i, i - n + 1, kernel(0, 2)));
            // middle right
            if (i < mn - 1 && kernel(1, 2) != 0)
                tripletList.push_back(T(i, i + 1, kernel(1, 2)));
            // bottom right
            if (i + n < mn - 1 && kernel(2, 2) != 0)
                tripletList.push_back(T(i, i + n + 1, kernel(2, 2)));
        }
    }
    A.setFromTriplets(tripletList.begin(), tripletList.end());
    return A;
}

// Define the function that convert a vector to a Matrix<unsigned char> type and output it to image.png
void outputVectorImage(const VectorXd &vectorData, int height, int width, const std::string &path)
{
    Matrix<double, Dynamic, Dynamic, RowMajor> output_image_matrix(height, width);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            output_image_matrix(i, j) = vectorData(i * width + j);
        }
    }

    // Convert the modified image to grayscale and export it using stbi_write_png
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> new_image_output = output_image_matrix.unaryExpr(
        [](double pixel)
        {
            return static_cast<unsigned char>(std::max(0.0, std::min(255.0, pixel * 255))); // ensure range [0,255]
        });
    if (stbi_write_png(path.c_str(), width, height, 1, new_image_output.data(), width) == 0)
    {
        std::cerr << "Error: Could not save modified image" << std::endl;
    }
    std::cout << "New image saved to " << path << "\n"
              << std::endl;
}

// Export the vector, save it to mtx file. And the index from 1 instead of 0 for meeting the lis input file demand.
void exportVector(VectorXd data, const std::string &path)
{
    FILE *out = fopen(path.c_str(), "w");
    fprintf(out, "%%%%MatrixMarket vector coordinate real general\n");
    fprintf(out, "%d\n", data.size());
    for (int i = 0; i < data.size(); i++)
    {
        fprintf(out, "%d %f\n", i + 1, data(i)); // Attention! here index is from 1, same as lis demand.
    }
    std::cout << "New vector file saved to " << path << std::endl;
    fclose(out);
}

// Export a sparse matrix by saveMarket()
void exportSparsematrix(SparseMatrix<double, RowMajor> data, const std::string &path)
{
    if (saveMarket(data, path))
    {
        std::cout << "New sparse matrix saved to " << path << std::endl;
    }
    else
    {
        std::cerr << "Error: Could not save sparse matrix to " << path << std::endl;
    }
}

// Check if a matrix is symmetric by tolerance 1e-10
bool isSymmetric(SparseMatrix<double, RowMajor> &matrix, const std::string &matrixName)
{
    double tolerance = 1e-10;
    double norm_diff = (matrix - SparseMatrix<double, RowMajor>(matrix.transpose())).norm();
    std::cout << matrixName << "\trow: " << matrix.rows() << "\tcolumns: " << matrix.cols() << std::endl;
    std::cout << "\nCheck if " << matrixName << " is symmetric by norm value of its difference with transpose: "
              << norm_diff << " ..." << std::endl;
    return norm_diff < tolerance;
}

// Function to check if a matrix is positive definite by cholesky
bool isPositiveDefinite(const SparseMatrix<double, RowMajor> &matrix, const std::string &matrixName)
{
    std::cout << "Check if " << matrixName << " is positive definite ... " << std::endl;
    Eigen::SimplicialLLT<SparseMatrix<double, RowMajor>> cholesky(matrix);
    return cholesky.info() == Success;
}

// 1. Lis generated mtx file is marketvector format, but loadMarkerVector() method doesn't match (it needs MatrixMarket matrix array fromat);
// 2. So we use our own menthod to read data from mtx file here, we read each line of the file and put it value into our Eigen::VectorXd data.
// Function to read a vector from a Matrix Market file from lis with 1-based indexing
VectorXd readMarketVector(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return VectorXd();
    }

    std::string line;
    // Skip the header lines
    std::getline(file, line); // %%MatrixMarket vector coordinate real general
    std::getline(file, line); // Dimensions or size of the vector

    int size;
    std::istringstream iss(line);
    if (!(iss >> size))
    {
        std::cerr << "Error reading size from file: " << filename << std::endl;
        return VectorXd();
    }

    VectorXd vectorX(size);

    // Read the vector data line by line, data index staring from 1
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        int index;
        double value;
        if (!(iss >> index >> value))
        {
            std::cerr << "Error reading value from file: " << filename << std::endl;
            return VectorXd();
        }
        // Convert 1-based to 0-based indexing here, because our VectorXd type stores data from 0
        vectorX(index - 1) = value;
    }

    file.close();
    return vectorX;
}
/*--------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------Main()-------------------------------------------------------*/
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    const char *input_image_path = argv[1];

    /*****************************Load the image by using stb_image****************************/
    std::cout << "\nLoading image " << argv[1] << " ..." << std::endl;
    int width, height, channels;
    // for greyscale images force to load only one channel
    unsigned char *image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
    if (!image_data)
    {
        std::cerr << "Error: Could not load image " << input_image_path << std::endl;
        return 1;
    }

    std::cout << "Image " << argv[1] << " loaded: " << height << "x" << width << " pixels with " << channels << " channels" << std::endl;

    /*************Convert the image_data to MatrixXd form, each element value is normalized to [0,1]*************/
    // We use RowMajor notation!
    Matrix<double, Dynamic, Dynamic, RowMajor> A(height, width);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int index = (i * width + j) * channels;
            A(i, j) = static_cast<double>(image_data[index]) / 255; // we use the value range from 0 to 1
        }
    }

    // Report the size of the matrix
    std::cout << "Image " << argv[1] << " has dimension: " << A.rows() << " rows x " << A.cols()
              << " cols = " << A.size() << " entries" << "\n"
              << std::endl;

    /*************************************Compute AtA and its euclidian norm*******************************************/
    std::cout << "Computing AtA ... " << std::endl;
    Matrix<double, Dynamic, Dynamic, RowMajor> AtA = A.transpose() * A;
    std::cout << "AtA has dimension: " << AtA.rows() << " rows x " << AtA.cols() << " cols = " << AtA.size() << " entries" << std::endl;
    std::cout << "AtA has euclidian norm: " << AtA.norm() << "\n" << std::endl;

    /*****************************Solve the eigenvalue problem AtA*x=lambda*x using Eigen***********************************/
    std::cout << "Solving the eigenvalue problem AtA*x=lambda*x using Eigen ..." << std::endl;
    SelfAdjointEigenSolver<MatrixXd> eigensolver(AtA);
    if (eigensolver.info() != Eigen::Success) abort();
    std::cout << "The smallest eigenvalues of AtA is: " << eigensolver.eigenvalues()[0] << std::endl;
    std::cout << "The second largest eigenvalues of AtA is: " << eigensolver.eigenvalues()[width-2] << std::endl;
    std::cout << "The largest eigenvalues of AtA is: " << eigensolver.eigenvalues()[width-1] << std::endl;
    // Define the vector of singular values of A as the square root of the eigenvalues of AtA
    VectorXd sigmaA = eigensolver.eigenvalues().cwiseSqrt();
    std::cout << "The two largest computed singular values of A are: sigma1 = " 
              << sigmaA(width-1) << " and sigma2 = "
              << sigmaA(width-2) << "\n" << std::endl;

    /**********************************************Export matrix AtA****************************************************/
    std::cout << "Exporting AtA ... " << std::endl;
    std::string matrixFileOut("./AtA.mtx");
    saveMarket(AtA, matrixFileOut);
    std::cout << "New dense matrix saved to " << matrixFileOut << "\n" << std::endl;

    /***********************************Solve the eigenvalue problem AtA*x=lambda*x using Eigen***********************************/

    /* The commands are:
        mpicc -DUSE_MPI -I${mkLisInc} -L${mkLisLib} -llis etest1.c -o eigen1
        mpirun -n 4 ./eigen1 Ata.mtx eigvec.mtm hist.txt -e pi -emaxiter 100 -etol 1.e-8 -shift 660 > output_Lis_1.txt
    */


    



    // Free memory
    stbi_image_free(image_data);

    return 0;
}
