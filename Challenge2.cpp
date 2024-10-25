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
        std::cerr << "Error: Could not save new image to " << path << std::endl;
    }
    std::cout << "New image saved to " << path << std::endl;
}

MatrixXd create_checkerboard(int board_size, int blocks_size)
{
    int num_blocks = board_size / blocks_size;
    MatrixXd board(board_size, board_size);
    for (int i = 0; i < num_blocks; i++)
    {
        for (int j = 0; j < num_blocks; j++)
        {
            double color = ((i + j) % 2 == 0) ? 0.0 : 1.0;
            for (int bi = 0; bi < blocks_size; bi++)
            {
                for (int bj = 0; bj < blocks_size; bj++)
                {
                    board(i * blocks_size + bi, j * blocks_size + bj) = color;
                }
            }
        }
    }
    return board;
}

MatrixXd noise_the_image(MatrixXd original_matrix, int a, int b)
{
    Matrix<double, Dynamic, Dynamic, RowMajor> noised_matrix(original_matrix.rows(), original_matrix.cols());
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(a, b);
    for (int i = 0; i < original_matrix.rows(); i++)
    {
        for (int j = 0; j < original_matrix.cols(); j++)
        {
            int noise = distribution(generator);
            double noisedData = original_matrix(i, j) + static_cast<double>(noise) / 255;
            noised_matrix(i, j) = std::max(0.0, std::min(1.0, noisedData));
        }
    }
    return noised_matrix;
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
    double lambda_255 = eigensolver.eigenvalues()[1];
    double lambda_2 = eigensolver.eigenvalues()[width-2];
    double lambda_1 = eigensolver.eigenvalues()[width-1];
    double r = lambda_2 / lambda_1;

    std::cout << "The smallest eigenvalues of AtA is                 : lambda_256 = " << lambda_256 << std::endl;
    std::cout << "The second smallest eigenvalues of AtA is          : lambda_255 = " << lambda_255 << std::endl;
    std::cout << "The second largest eigenvalues of AtA is           : lambda_2 = " << lambda_2 << std::endl;
    std::cout << "The largest eigenvalues of AtA is                  : lambda_1 = " << lambda_1 << std::endl;
    std::cout << "The ratio of convergence |lambda_2| / |lambda_1| is: r = " << r << std::endl;
    // Define the vector of singular values of A as the square root of the eigenvalues of AtA
    VectorXd sigmaA = eigensolver.eigenvalues().cwiseSqrt();
    std::cout << "The two largest computed singular values of A are  : sigma_1 = " 
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
    Eigen::BDCSVD<Eigen::MatrixXd> svdFull (A, Eigen::ComputeFullU | Eigen::ComputeFullV);
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
    
    std::cout << "\nCreating a checkerboard ..." << std::endl;
    int board_size = 200;
    int blocks_size = 25;
    std::cout << "Size of the checkerboard: " << board_size << " x " << board_size << std::endl;
    std::cout << "Size of each block: " << blocks_size << " x " << blocks_size << std::endl;
    std::cout << "The starting top left block is black." << std::endl;
    MatrixXd checkerboard = create_checkerboard(board_size, blocks_size);
    std::cout << "The checkerboard has euclidian norm: " << checkerboard.norm() << std::endl;
    outputImage(checkerboard, board_size, board_size, "./output_checkerboard.png");

    /********************************************** Export checkerboard matrix ****************************************************/
    std::cout << "\nExporting checkerboard matrix ..." << std::endl;
    std::string matrixFileOut6("./checkerboard.mtx");
    saveMarket(checkerboard, matrixFileOut6);
    std::cout << "New dense matrix saved to " << matrixFileOut6 << "\n" << std::endl;

    /***************************************** Add noise to the checkerboard matrix ***********************************************/
    std::cout << "Adding noise to the checkerboard matrix ..." << std::endl;
    MatrixXd noised_checkerboard = noise_the_image(checkerboard, -50, 50);
    outputImage(noised_checkerboard, board_size, board_size, "./output_checkerboard_noised.png");
    std::cout << "Difference between noised_checkerboard and checkerboard: " << (noised_checkerboard-checkerboard).norm() << std::endl;

    /******************************************** Export noised checkerboard matrix **************************************************/
    std::cout << "\nExporting noised checkerboard matrix ..." << std::endl;
    std::string matrixFileOut7("./checkerboard_noised.mtx");
    saveMarket(noised_checkerboard, matrixFileOut7);
    std::cout << "New dense matrix saved to " << matrixFileOut7 << "\n" << std::endl;

    /******************************** Compute the SVD of the noised checkerboard matrix **************************************/
    std::cout << "Computing the SVD (addressed as SVD2) of noised checkerboard matrix ..." << std::endl;
    Eigen::BDCSVD<Eigen::MatrixXd> svd2 (noised_checkerboard, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXd U2 = svd2.matrixU(), V2 = svd2.matrixV(), S2 = svd2.singularValues().asDiagonal();
    VectorXd W2 = svd2.singularValues();
    std::cout << "Size of noised checkerboard matrix: " << noised_checkerboard.rows() << " x " << noised_checkerboard.cols() << std::endl;
    std::cout << "Size of Thin U2                   : " << U2.rows() << " x " << U2.cols() << std::endl;
    std::cout << "Size of Thin S2                   : " << S2.rows() << " x " << S2.cols() << std::endl;
    std::cout << "Size of Thin V2                   : " << V2.rows() << " x " << V2.cols() << std::endl;
    std::cout << "Thin S2 has euclidian norm        : " << S2.norm() << std::endl;
    std::cout << "The two largest computed singular values of noised checkerboard matrix are: sigma_1 = " 
              << W2(0) << " and sigma_2 = " << W2(1) << std::endl;

    std::cout << "\nPerforming a quick check about the SVD2 ..." << std::endl;
    MatrixXd Asvd2 = U2 * S2 * V2.transpose();
    std::cout << "Difference between A_SVD2 and noised checkerboard matrix: " << (Asvd2-noised_checkerboard).norm() << std::endl;
    std::cout << "Difference between A_SVD2 and checkerboard matrix       : " << (Asvd2-checkerboard).norm() << std::endl;

    /*********************************** Compute the truncated SVD of noised checkerboard matrix ***********************************/
    std::cout << "\nComputing C3D3 the Truncated SVD of noised checkerboard matrix for k3 = 5 ..." << std::endl;
    int k3 = 5, k4 = 10;
    MatrixXd C3 = U2.leftCols(k3), D3 = V2.leftCols(k3) * S2.topLeftCorner(k3,k3);
    std::cout << "Size of C3  : " << C3.rows() << " x " << C3.cols() << std::endl;
    std::cout << "Size of D3  : " << D3.rows() << " x " << D3.cols() << std::endl;
    MatrixXd C3D3 = C3 * D3.transpose();
    std::cout << "Size of C1D1: " << C1D1.rows() << " x " << C1D1.cols() << std::endl;
    std::cout << "Difference between C3D3 and noised checkerboard matrix: " << (C3D3-noised_checkerboard).norm() << std::endl;
    std::cout << "Difference between C3D3 and checkerboard matrix       : " << (C3D3-checkerboard).norm() << std::endl;

    std::cout << "\nComputing C4D4 the Truncated SVD of noised checkerboard matrix for k4 = 10 ..." << std::endl; 
    MatrixXd C4 = U2.leftCols(k4), D4 = V2.leftCols(k4) * S2.topLeftCorner(k4,k4);
    std::cout << "Size of C4  : " << C4.rows() << " x " << C4.cols() << std::endl;
    std::cout << "Size of D4  : " << D4.rows() << " x " << D4.cols() << std::endl;
    MatrixXd C4D4 = C4 * D4.transpose();
    std::cout << "Size of C4D4: " << C4D4.rows() << " x " << C4D4.cols() << std::endl;
    std::cout << "Difference between C4D4 and noised checkerboard matrix: " << (C4D4-noised_checkerboard).norm() << std::endl;
    std::cout << "Difference between C4D4 and checkerboard matrix       : " << (C4D4-checkerboard).norm() << std::endl;

    /********************************************** Export matrices C3, D3, C4, D4 ****************************************************/
    std::cout << "\nExporting matrices C3, D3, C4, D4 ..." << std::endl;
    std::string matrixFileOut8("./C3.mtx");
    saveMarket(C3, matrixFileOut8);
    std::cout << "New dense matrix saved to " << matrixFileOut8 << std::endl;
    std::string matrixFileOut9("./D3.mtx");
    saveMarket(D3, matrixFileOut9);
    std::cout << "New dense matrix saved to " << matrixFileOut9 << std::endl;
    std::string matrixFileOut10("./C4.mtx");
    saveMarket(C4, matrixFileOut10);
    std::cout << "New dense matrix saved to " << matrixFileOut10 << std::endl;
    std::string matrixFileOut11("./D4.mtx");
    saveMarket(D4, matrixFileOut11);
    std::cout << "New dense matrix saved to " << matrixFileOut11 << std::endl;

    /***************************************** Export compressed images C3D3 and C4D4 ***********************************************/
    std::cout << "\nExporting compressed images C3D3 and C4D4 ..." << std::endl;
    outputImage(C3D3, C3D3.rows(), C3D3.cols(), "./output_C3D3.png");
    outputImage(C4D4, C4D4.rows(), C4D4.cols(), "./output_C4D4.png");

    /********************************************************* end ***************************************************************/

    // Free memory
    stbi_image_free(image_data);

    return 0;
}
