#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <cstdlib>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/SVD>
#include <Eigen/Core>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;
typedef Eigen::Triplet<double> T;

void outputImage(const MatrixXd &output_image_matrix, int height, int width, const std::string &path)
{
    // Convert the modified image to grayscale and export it using stbi_write_png, just need this stb-related to be RowMajor
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> new_image_output = output_image_matrix.unaryExpr(
        [](double pixel) { return static_cast<unsigned char>(std::max(0.0, std::min(255.0, pixel * 255))); // ensure range [0,255]
    });
    if (stbi_write_png(path.c_str(), width, height, 1, new_image_output.data(), width) == 0)
    {
        std::cerr << "Error: Could not save modified image" << std::endl;
    }
    std::cout << "New image saved to " << path << std::endl;
}

int main(int argc, char *argv[])
{
    /*******************************************************************************************************************************
                                                          Einstein Part
    *******************************************************************************************************************************/

    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    const char *input_image_path = argv[1];

    /**************************************** Load the image by using stb_image ***************************************/
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

    /************* Convert the image_data to MatrixXd form, each element value is normalized to [0,1] *************/
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
              << " cols = " << A.size() << " entries" << "\n" << std::endl;

    /************************************* Compute AtA and its euclidian norm *******************************************/
    std::cout << "Computing AtA ... " << std::endl;
    Matrix<double, Dynamic, Dynamic, RowMajor> AtA = A.transpose() * A;
    std::cout << "AtA has dimension: " << AtA.rows() << " rows x " << AtA.cols() << " cols = " << AtA.size() << " entries" << std::endl;
    std::cout << "AtA has euclidian norm: " << AtA.norm() << "\n" << std::endl;

    /***************************** Solve the eigenvalue problem AtA*x=lambda*x using Eigen ***********************************/
    std::cout << "Solving the eigenvalue problem AtA*x=lambda*x using Eigen ..." << std::endl;
    SelfAdjointEigenSolver<MatrixXd> eigensolver(AtA);
    if (eigensolver.info() != Eigen::Success) abort();
    double lambda_256 = eigensolver.eigenvalues()[0];
    double lambda_2 = eigensolver.eigenvalues()[width-2];
    double lambda_1 = eigensolver.eigenvalues()[width-1];
    double r = lambda_2 / lambda_1;

    std::cout << "The smallest eigenvalues of AtA is lambda_256 = " << lambda_256 << std::endl;
    std::cout << "The second largest eigenvalues of AtA is lambda_2 = " << lambda_2 << std::endl;
    std::cout << "The largest eigenvalues of AtA is lambda_1 = " << lambda_1 << std::endl;
    std::cout << "The ratio of convergence |lambda_2| / |lambda_1| is r = " << r << std::endl;
    // Define the vector of singular values of A as the square root of the eigenvalues of AtA
    VectorXd sigmaA = eigensolver.eigenvalues().cwiseSqrt();
    std::cout << "The two largest computed singular values of A are: sigma_1 = " 
              << sigmaA(width-1) << " and sigma_2 = "
              << sigmaA(width-2) << "\n" << std::endl;

    /********************************************** Export matrix AtA ****************************************************/
    std::cout << "Exporting matrix AtA ..." << std::endl;
    std::string matrixFileOut("./AtA.mtx");
    saveMarket(AtA, matrixFileOut);
    std::cout << "New dense matrix saved to " << matrixFileOut << "\n" << std::endl;

    /*********************************** Solve the eigenvalue problem AtA*x=lambda*x using Lis ***********************************/
    std::cout << "Computing the largest eigenvalue of AtA using Lis ..." << std::endl;
    std::cout << "The results of the power method are written in output_Lis_1.txt" << std::endl;
    std::cout << "The results of the power method with shift are written in output_Lis_2.txt" << std::endl;
    std::cout << "The results of the inverse power method with shift are written in output_Lis_3.txt" << std::endl;
    /* 
    The Lis terminal commands are:
        mpicc -DUSE_MPI -I${mkLisInc} -L${mkLisLib} -llis etest1.c -o eigen1
        mpirun -n 4 ./eigen1 Ata.mtx eigvec1.mtm hist1.txt -e pi -emaxiter 100 -etol 1.e-8 > output_Lis_1.txt
        mpirun -n 4 ./eigen1 Ata.mtx eigvec2.mtm hist2.txt -e pi -emaxiter 100 -etol 1.e-8 -shift 695 > output_Lis_2.txt
        mpirun -n 4 ./eigen1 Ata.mtx eigvec3.mtm hist3.txt -e ii -emaxiter 100 -etol 1.e-8 -shift 16083 > output_Lis_3.txt
    */

   // The following part is very subtle, therefore I advise you to read the Readme.md file
    std::cout << "\nPerforming a quick check about the power method with shift ..." << std::endl;
    double mu = 695;
    MatrixXd Id = MatrixXd::Identity(width,width);
    MatrixXd AtA_mu = AtA - mu * Id;
    SelfAdjointEigenSolver<MatrixXd> eigensolver2(AtA_mu);
    if (eigensolver2.info() != Eigen::Success) abort();
    std::cout << "The ratio of convergence |lambda_2-mu| / |lambda_1-mu| is r_mu = " 
              << eigensolver2.eigenvalues()[width-2] / eigensolver2.eigenvalues()[width-1] << std::endl;

    std::cout << "\nPerforming a quick check about the inverse power method with shift ..." << std::endl;
    double mu2 = 16083;
    MatrixXd AtA_mu2 = AtA - mu2 * Id;
    SelfAdjointEigenSolver<MatrixXd> eigensolver3(AtA_mu2);
    if (eigensolver3.info() != Eigen::Success) abort();
    std::cout << "The ratio of convergence |lambda_2-mu2| / |lambda_1-mu2| is r_mu2 = " 
              << eigensolver3.eigenvalues()[width-1] / abs(eigensolver3.eigenvalues()[width-2]) << std::endl;

    /*********************************** Compute the SVD of A ***********************************/
    std::cout << "\nComputing the Singular Value Decomposition of A ..." << std::endl;
    Eigen::BDCSVD<Eigen::MatrixXd> svd (A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXd U = svd.matrixU(), V = svd.matrixV(), S = svd.singularValues().asDiagonal();
    VectorXd W = svd.singularValues();
    std::cout << "Size of A     : " << A.rows() << " x " << A.cols() << std::endl;
    std::cout << "Size of Thin U: " << U.rows() << " x " << U.cols() << std::endl;
    std::cout << "Size of Thin S: " << S.rows() << " x " << S.cols() << std::endl;
    std::cout << "Size of Thin V: " << V.rows() << " x " << V.cols() << std::endl;
    std::cout << "Thin S has euclidian norm: " << S.norm() << std::endl;

    std::cout << "\nPerforming quick checks about the SVD ..." << std::endl;
    std::cout << "1) The two largest computed singular values of A are: sigma_1 = " 
              << W(0) << " and sigma_2 = " << W(1) << std::endl;
    MatrixXd Asvd = U * S * V.transpose();
    std::cout << "2) Difference between A_SVD and A: " << (Asvd-A).norm() << std::endl;
    /*
    Eigen::BDCSVD<Eigen::MatrixXd> svd2 (A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    MatrixXd Uf = svd.matrixU(), Vf = svd.matrixV();
    std::cout << "3) Size of Full U: " << Uf.rows() << " x " << Uf.cols() << std::endl;
    std::cout << "   Size of Full V: " << Vf.rows() << " x " << Vf.cols() << std::endl;
    */

    /*********************************** Compute the truncated SVD of A ***********************************/
    std::cout << "\nComputing C1D1 the Truncated SVD of A for k1 = 40 ..." << std::endl;
    int k1 = 40, k2 = 80;
    MatrixXd C1 = U.leftCols(k1), D1 = V.leftCols(k1) * S.topLeftCorner(k1,k1);
    std::cout << "Size of C1: " << C1.rows() << " x " << C1.cols() << std::endl;
    std::cout << "Size of D1: " << D1.rows() << " x " << D1.cols() << std::endl;
    std::cout << "Number of nonzero elements in matrix C1 is " << C1.nonZeros() 
              << " and in matrix D1 is " << D1.nonZeros() << std::endl;
    MatrixXd C1D1 = C1 * D1.transpose();
    std::cout << "Size of C1D1: " << C1D1.rows() << " x " << C1D1.cols() << std::endl;
    std::cout << "Difference between C1D1 and A: " << (C1D1-A).norm() << std::endl;

    std::cout << "\nComputing C2D2 the Truncated SVD of A for k2 = 80 ..." << std::endl; 
    MatrixXd C2 = U.leftCols(k2), D2 = V.leftCols(k2) * S.topLeftCorner(k2,k2);
    std::cout << "Size of C2: " << C2.rows() << " x " << C2.cols() << std::endl;
    std::cout << "Size of D2: " << D2.rows() << " x " << D2.cols() << std::endl;
    std::cout << "Number of nonzero elements in matrix C2 is " << C2.nonZeros() 
              << " and in matrix D2 is " << D2.nonZeros() << std::endl;
    MatrixXd C2D2 = C2 * D2.transpose();
    std::cout << "Size of C2D2: " << C2D2.rows() << " x " << C2D2.cols() << std::endl;
    std::cout << "Difference between C2D2 and A: " << (C2D2-A).norm() << std::endl;

    /********************************************** Export matrices C1, D1, C2, D2 ****************************************************/
    std::cout << "\nExporting matrices C1, D1, C2, D2 ..." << std::endl;
    std::string matrixFileOut2("./C1.mtx");
    saveMarket(C1, matrixFileOut2);
    std::cout << "New dense matrix saved to " << matrixFileOut2 << std::endl;
    std::string matrixFileOut3("./D1.mtx");
    saveMarket(D1, matrixFileOut3);
    std::cout << "New dense matrix saved to " << matrixFileOut3 << std::endl;
    std::string matrixFileOut4("./C2.mtx");
    saveMarket(C2, matrixFileOut4);
    std::cout << "New dense matrix saved to " << matrixFileOut4 << std::endl;
    std::string matrixFileOut5("./D2.mtx");
    saveMarket(D2, matrixFileOut5);
    std::cout << "New dense matrix saved to " << matrixFileOut5 << std::endl;

    /***************************************** Export compressed images C1D1 and C2D2 ***********************************************/
    std::cout << "\nExporting compressed images C1D1 and C2D2 ..." << std::endl;
    outputImage(C1D1, C1D1.rows(), C1D1.cols(), "./output_C1D1.png");
    outputImage(C2D2, C2D2.rows(), C2D2.cols(), "./output_C2D2.png");

    /*******************************************************************************************************************************
                                                          Checkerboard Part
    *******************************************************************************************************************************/
    
    // Free memory
    stbi_image_free(image_data);

    return 0;
}
