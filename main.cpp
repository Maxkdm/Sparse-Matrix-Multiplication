#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <algorithm>
#include "main.hpp"
#include <ctime>  
#include <chrono>
#include <omp.h>
#include <mpi.h>
#include <climits>
#include <cstdint> 

using namespace std;

tuple<vector<uint64_t>, vector<uint64_t>, vector<uint64_t>> read_matrices(string inputPathFolder, int rank, int size, uint64_t NumMatrices, uint64_t BlockSize) {
    vector<uint64_t> mat_pos;
    vector<uint64_t> mat_values;
    vector<uint64_t> mat_diff;
    // cout << "reading files by process\n"; 
    #pragma omp parallel
    {
        vector<uint64_t> local_pos;
        vector<uint64_t> local_vals;
        vector<uint64_t> local_diff;
        uint64_t pos = 0;

        #pragma omp for nowait schedule(dynamic)
        for (int i = rank; i < NumMatrices; i += size) {
            string filePath = inputPathFolder + "/matrix" + to_string(i + 1);
            ifstream file(filePath);
            if (!file.is_open()) {
                cerr << "Error: Could not open file: " << filePath << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            uint64_t height, width, numNonZero;
            file >> height >> width >> numNonZero;

            local_diff.push_back(pos);  // Start position for this matrix in val vector

            // Matrix metadata
            local_pos.push_back(i + 1);      // matrix ID
            local_pos.push_back(height);
            local_pos.push_back(width);
            local_pos.push_back(numNonZero);
            local_pos.push_back(BlockSize);

            for (uint64_t j = 0; j < numNonZero; j++) {
                uint64_t row, col;
                file >> row >> col;
                local_pos.push_back(row);
                local_pos.push_back(col);

                // Read block values directly after each position
                for (int b = 0; b < BlockSize * BlockSize; b++) {
                    uint64_t val;
                    file >> val;
                    local_vals.push_back(val);
                    pos++;
                }
            }

            file.close();
        }

        #pragma omp critical
        {
            mat_pos.insert(mat_pos.end(), local_pos.begin(), local_pos.end());
            mat_values.insert(mat_values.end(), local_vals.begin(), local_vals.end());
            mat_diff.insert(mat_diff.end(), local_diff.begin(), local_diff.end());
        }
    }
    // cout << "finished reading files\n";
    return make_tuple(mat_pos, mat_values, mat_diff);
}


void write_matrices(string outPathFile, MatrixNode datax) {
    const vector<uint64_t>& mat_pos = datax.pos;
    const vector<uint64_t>& mat_values = datax.data;
    const uint64_t BlockSize = datax.BlockSize;

    ofstream outFile(outPathFile);
    if (!outFile.is_open()) {
        cerr << "Error: Could not open output file: " << outPathFile << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    uint64_t height = datax.height;
    uint64_t width = datax.width;
    uint64_t numBlocks = datax.numNonZero;
    
    outFile << height << " " << width << "\n";
    outFile << numBlocks << "\n";

    for (uint64_t i = 0; i < numBlocks; ++i) {
        uint64_t row = mat_pos[2 * i];
        uint64_t col = mat_pos[2 * i + 1];
        outFile << row << " " << col << "\n";

        uint64_t startIdx = i * BlockSize * BlockSize;
        for (uint64_t bi = 0; bi < BlockSize; ++bi) {
            for (uint64_t bj = 0; bj < BlockSize; ++bj) {
                outFile << mat_values[startIdx + bi * BlockSize + bj] << " ";
            }
            outFile << "\n";
        }
    }
    outFile.close();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string foldername = argv[1];
    uint64_t NumMatrices, BlockSize;
    chrono::high_resolution_clock::time_point start_time;

    if (rank == 0) {
        auto start_time = chrono::high_resolution_clock::now();
        string size_file_path = foldername + "/size";
        ifstream size_file(size_file_path);
        if (!size_file.is_open()) {
            cerr << "Error: Could not open file: " << size_file_path << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        size_file >> NumMatrices >> BlockSize;
        // cout << "Read from file: N = " << NumMatrices << ", k = " << BlockSize << endl;
        size_file.close();
    }

    MPI_Bcast(&NumMatrices, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&BlockSize, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    // cout << "done broadcasting N and K values around\n";
    tuple<vector<uint64_t>, vector<uint64_t>, vector<uint64_t>> data;
    data = read_matrices(foldername, rank, size, NumMatrices, BlockSize);
    MatrixNode data2;
    data2 = matrixMult(data);
    if(rank == 0){
        write_matrices("matrix", data2);
        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end_time - start_time;

    }
    MPI_Finalize();
    cout << "done with process " << "\n";
}
