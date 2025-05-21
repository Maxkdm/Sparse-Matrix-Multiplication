#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <mpi.h>
#include <unordered_set>
#include <cfloat>
#include <cstdint>   
#include <chrono>
#include <mpi.h>    
#include "main.hpp"

using namespace std;


void printResultArray(const uint64_t* result, uint64_t totalElements) {
    std::cout << "Result Array: [";
    for (uint64_t i = 0; i < totalElements; ++i) {
        std::cout << result[i];
        if (i != totalElements - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

void printVector(const vector<uint64_t>& vec, int rank) {
    cout << "Rank " << rank << ": " << "\n";
    for (const auto& val : vec) {
        cout << val << " ";
    }
    cout << "\n";
    cout << endl;
}

bool hasIntersection(pair<uint64_t, uint64_t> p1, pair<uint64_t, uint64_t> p2) {
    return p1.first == p2.first || p1.first == p2.second ||
           p1.second == p2.first || p1.second == p2.second;
}

MatrixNode launchHostMult(MatrixNode matA, MatrixNode matB){
    // std::cout << "[DEBUG] launchHostMult started\n";
    // std::cout << "[DEBUG] Matrix A: blocks = " << matA.pos.size() / 2 << ", block size = " << matA.BlockSize << "\n";
    // std::cout << "[DEBUG] Matrix B: blocks = " << matB.pos.size() / 2 << ", block size = " << matB.BlockSize << "\n";
    auto start_time = chrono::high_resolution_clock::now();
    vector<uint64_t> pair_indices;
    uint64_t resultBlockSize = matA.BlockSize * matA.BlockSize;
    #pragma omp parallel for schedule(dynamic)
    for (uint64_t i = 0; i < matA.pos.size()-1; i+=2) {
        uint64_t a_row = matA.pos[i];
        uint64_t a_col = matA.pos[i+1];

        for (uint64_t j = 0; j < matB.pos.size()-1; j+=2) {
            uint64_t b_row = matB.pos[j];
            uint64_t b_col = matB.pos[j+1];

            if (a_col == b_row) {
                #pragma omp critical
                {
                    pair_indices.push_back(i/2);
                    pair_indices.push_back(j/2);
                }
            }
        }
    }

    map<pair<uint64_t, uint64_t>, vector<uint64_t>> posToBlockData;
    #pragma omp parallel for
    for (uint64_t p = 0; p < pair_indices.size(); p += 2) {
        uint64_t A_idx = pair_indices[p];
        uint64_t B_idx = pair_indices[p + 1];

        uint64_t a_row = matA.pos[2 * A_idx];
        uint64_t b_col = matB.pos[2 * B_idx + 1];
        vector<uint64_t> blockA(matA.data.begin() + A_idx * resultBlockSize, matA.data.begin() + (A_idx) * resultBlockSize + resultBlockSize);
        vector<uint64_t> blockB(matB.data.begin() + B_idx * resultBlockSize, matB.data.begin() + (B_idx) * resultBlockSize + resultBlockSize);

        vector<uint64_t> result(resultBlockSize, 0);
        for (uint64_t i = 0; i < matA.BlockSize; ++i) {
            for (uint64_t j = 0; j < matA.BlockSize; ++j) {
                for (uint64_t k = 0; k < matA.BlockSize; ++k) {
                    result[i * matA.BlockSize + j] =
                        (result[i * matA.BlockSize + j] + 
                        (blockA[i * matA.BlockSize + k] * blockB[k * matA.BlockSize + j]) % MOD) % MOD;
                }
            }
        }
        pair<uint64_t, uint64_t> new_pos = {a_row, b_col};
        #pragma omp critical
        {
            if (posToBlockData.find(new_pos) == posToBlockData.end()) {
                posToBlockData[new_pos] = result;
            } else {
                for (uint64_t i = 0; i < resultBlockSize; ++i) {
                    posToBlockData[new_pos][i] = 
                        (posToBlockData[new_pos][i] + result[i]) % MOD;
                }
            }
        }
    }
    // cout << "printing final mat values block pos resutls\n";
    // for (const auto& entry : posToBlockData) {
    //     cout << "Key: (" << entry.first.first << ", " << entry.first.second << ") => Values: ";
    //     printVector(entry.second, 0);
    //     cout << endl;
    // }
    auto end_time = chrono::high_resolution_clock::now();
    MatrixNode answer;
    answer.id = matA.id;
    answer.height = matA.height;
    answer.width = matB.width;
    answer.numNonZero = posToBlockData.size();
    answer.BlockSize = matA.BlockSize;
    vector<uint64_t> posAns;
    vector<uint64_t> dataAns;
    for (const auto& entry : posToBlockData) {
        posAns.push_back(entry.first.first);
        posAns.push_back(entry.first.second);
        dataAns.insert(dataAns.end(), entry.second.begin(), entry.second.end());
    }
    answer.pos = posAns;
    answer.data = dataAns;
    
    chrono::duration<double> duration = end_time - start_time;

    // Output elapsed time for matrix multiplication
    cout << "OpenMP Matrix Multiplication Time: " << duration.count() << " seconds\n";
    return answer;
}

__global__ void blockMultKernel(uint64_t* Adata, uint64_t* Bdata, uint64_t* pairIndices, uint64_t BlockSize, uint64_t* Cdata, uint64_t MOD) {
    uint64_t pairIdx = blockIdx.x;
    uint64_t tx = threadIdx.x;
    uint64_t ty = threadIdx.y;
    uint64_t indexA = pairIndices[2 * pairIdx];
    uint64_t indexB = pairIndices[2 * pairIdx + 1]; 
    uint64_t offsetA = indexA * BlockSize * BlockSize;
    uint64_t offsetB = indexB * BlockSize * BlockSize;
    uint64_t offsetC = pairIdx * BlockSize * BlockSize;

    uint64_t val = 0;
    for (uint64_t k = 0; k < BlockSize; ++k) {
        uint64_t a = Adata[offsetA + ty * BlockSize + k];
        uint64_t b = Bdata[offsetB + k * BlockSize + tx];
        val = (val + ((a) * (b)) % MOD) % MOD;
        // printf("Block %lld Thread (%lld, %lld) k=%lld: A[%lld]=%lld * B[%lld]=%lld => partial sum=%lld\n",
        //        pairIdx, ty, tx, k,
        //        offsetA + ty * BlockSize + k, a,
        //        offsetB + k * BlockSize + tx, b,
        //        val);
    }
    Cdata[offsetC + ty * BlockSize + tx] = val;
    // printf("Block %lld Thread (%lld, %lld): C[%lld] = %lld\n",
    //        pairIdx, ty, tx, offsetC + ty * BlockSize + tx, val);

}

MatrixNode launchCudaKernel(MatrixNode matA, MatrixNode matB){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank != 0) {
        MPI_Finalize(); 
    }
    // cout << "Rank " << rank << ": Launching CUDA kernel for matrix multiplication\n";
    vector<uint64_t> pair_indices;
    // printVector(matA.pos, 0);
    // printVector(matB.pos, 0);
    #pragma omp parallel for schedule(dynamic)
    for (uint64_t i = 0; i < matA.pos.size()-1; i+=2) {
        uint64_t a_row = matA.pos[i];
        uint64_t a_col = matA.pos[i+1];

        for (uint64_t j = 0; j < matB.pos.size()-1; j+=2) {
            uint64_t b_row = matB.pos[j];
            uint64_t b_col = matB.pos[j+1];

            if (a_col == b_row) {
                #pragma omp critical
                {
                    pair_indices.push_back(i/2);
                    pair_indices.push_back(j/2);
                }
            }
        }
    }

    uint64_t numPairs = pair_indices.size() / 2;
    uint64_t resultBlockSize = matA.BlockSize * matB.BlockSize;
    uint64_t blocksize = matA.BlockSize;
    uint64_t totalElements = numPairs * resultBlockSize;
    // printVector(pair_indices, 0);
    // printVector(matA.data, 0);
    // printVector(matB.data, 0);

    uint64_t *d_matrix1, *d_matrix2, *d_result, *d_pair_indices;
    cudaMalloc((void**)&d_matrix1, matA.data.size() * sizeof(uint64_t));
    cudaMalloc((void**)&d_matrix2, matB.data.size() * sizeof(uint64_t));
    cudaMalloc((void**)&d_pair_indices, pair_indices.size() * sizeof(uint64_t));
    cudaMalloc((void**)&d_result, totalElements * sizeof(uint64_t));
    cudaMemcpy(d_matrix1, matA.data.data(), matA.data.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, matB.data.data(), matB.data.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pair_indices, pair_indices.data(), pair_indices.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    dim3 threadsPerBlock(blocksize, blocksize);
    dim3 numBlocks(numPairs);
    blockMultKernel<<<numBlocks, threadsPerBlock>>>(d_matrix1, d_matrix2, d_pair_indices, matA.BlockSize, d_result, MOD);
    cudaDeviceSynchronize();

    // Timing end
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "CUDA Matrix Multiplication Time: " << elapsedTime / 1000.0f << " seconds\n"; // Convert ms to seconds


    // cout << "done7\n";
    uint64_t *result = new uint64_t[totalElements];
    cudaMemcpy(result, d_result, totalElements * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    // cout << "results from cuda\n";
    // printResultArray(result, totalElements);
    
    
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_pair_indices);
    cudaFree(result);

    map<pair<uint64_t,uint64_t>, vector<uint64_t>> posToBlockData;
    for (uint64_t p = 0; p < pair_indices.size()-1; p += 2) {
        uint64_t A_idx = pair_indices[p];     
        uint64_t B_idx = pair_indices[p + 1]; 

        uint64_t a_row = matA.pos[2 * A_idx];         
        uint64_t b_col = matB.pos[2 * B_idx + 1];

        // cout << "printing a_row and b_col\n";
        if(posToBlockData.find({a_row,b_col}) == posToBlockData.end()){
            // cout << a_row << " " << b_col << "\n";
            // printVector(vector<uint64_t>(result + p/2 * resultBlockSize, result + p/2 * resultBlockSize + resultBlockSize), 0);
            posToBlockData[{a_row, b_col}] = std::vector<uint64_t>(
                result + p/2 * resultBlockSize,
                result + p/2  * resultBlockSize + resultBlockSize
            );
        }else{
            vector<uint64_t> &existingData = posToBlockData[{a_row,b_col}];
            // cout << a_row << " " << b_col << "\n";
            // cout << "printing existing data\n";
            // printVector(existingData, 0);
            for (uint64_t i = 0; i < resultBlockSize; ++i) {
                // cout << "data to be added: " << result[p/2 * resultBlockSize + i] << "\n";
                existingData[i] += result[p/2 * resultBlockSize + i];
                // cout << "new existing data: " << existingData[i] << "\n";
            }        
        }
    }
    // cout << "printing final mat values block pos resutls\n";
    // for (const auto& entry : posToBlockData) {
    //     cout << "Key: (" << entry.first.first << ", " << entry.first.second << ") => Values: ";
    //     printVector(entry.second, 0);
    //     cout << endl;
    // }

    MatrixNode answer;
    answer.id = matA.id;
    answer.height = matA.height;
    answer.width = matB.width;
    answer.numNonZero = posToBlockData.size();
    answer.BlockSize = matA.BlockSize;
    vector<uint64_t> posAns;
    vector<uint64_t> dataAns;
    for (const auto& entry : posToBlockData) {
        posAns.push_back(entry.first.first);
        posAns.push_back(entry.first.second);
        dataAns.insert(dataAns.end(), entry.second.begin(), entry.second.end());
    }
    answer.pos = posAns;
    answer.data = dataAns;
    return answer;
}

void performHuesristicCalc(uint64_t &globalNumMatrices, unordered_map<uint64_t, MatrixNode> &idToNode, vector<uint64_t> &resultPath){
    double minCost = DBL_MAX;
    double maxCost = DBL_MIN;
    pair<uint64_t, uint64_t> minPair, maxPair;
    size_t minIdx = 0, maxIdx = 0;

    #pragma omp parallel
    {
        double localMinCost = DBL_MAX;
        double localMaxCost = DBL_MIN;
        pair<uint64_t, uint64_t> localMinPair, localMaxPair;
        size_t localMinIdx = 0, localMaxIdx = 0;

        #pragma omp for nowait
        for (size_t i = 0; i < resultPath.size() - 1; ++i) {
            uint64_t idA = resultPath[i];
            uint64_t idB = resultPath[i + 1];
            MatrixNode A = idToNode[idA];
            MatrixNode B = idToNode[idB];
            double cost = (A.width != 0) ? static_cast<double>(A.numNonZero) * B.numNonZero / A.width : DBL_MAX;
            if (cost < localMinCost) {
                localMinCost = cost;
                localMinPair = {idA, idB};
                localMinIdx = i;
            }
            if (cost > localMaxCost) {
                localMaxCost = cost;
                localMaxPair = {idA, idB};
                localMaxIdx = i;
            }
        }
        #pragma omp critical
        {
            if (localMinCost < minCost) {
                minCost = localMinCost;
                minPair = localMinPair;
                minIdx = localMinIdx;
            }
            if (localMaxCost > maxCost) {
                maxCost = localMaxCost;
                maxPair = localMaxPair;
                maxIdx = localMaxIdx;
            }
        }
    }

    // cout << "min pair: " << minPair.first << " " << minPair.second << "\n";
    // cout << "max pair: " << maxPair.first << " " << maxPair.second << "\n";

    if(hasIntersection(minPair,maxPair)){
        // cout << "intersection found\n";
        uint64_t resultId = maxPair.first; 
        MatrixNode result ;
        // cout << "Performing multiplication on matrices with IDs with cuda: " << minPair.first << " and " << minPair.second << endl;
        result = launchCudaKernel(idToNode[maxPair.first],idToNode[maxPair.second]); 
        result.id = resultId;
        idToNode[resultId] = result;
        resultPath.erase(resultPath.begin() + maxIdx + 1);
        globalNumMatrices-=1;   

    }else{
        // cout << "no intersection found\n";
        // cout << "Performing multiplication on matrices with IDs with cuda: " << maxPair.first << " and " << maxPair.second << endl;
        // cout << "Performing multiplication on matrices with IDs with host: " << minPair.first << " and " << minPair.second << endl;
        MatrixNode ans1, ans2 ;
        uint64_t resultId1 = minPair.first;
        uint64_t resultId2 = maxPair.first;
        #pragma omp parallel
        {
            #pragma omp single
            {
                #pragma omp task
                {
                    ans2 = launchCudaKernel(idToNode[maxPair.first], idToNode[maxPair.second]);
                }
                #pragma omp task
                {
                    ans1 = launchHostMult(idToNode[minPair.first], idToNode[minPair.second]);
                }
                #pragma omp taskwait
            }
        }
        idToNode[resultId1] = ans1;
        idToNode[resultId2] = ans2;
        // cout << resultPath.size() << " before\n";
        // cout << "minIdx: " << minIdx << ", maxIdx: " << maxIdx << "\n";
        if (minIdx < maxIdx) {
            resultPath.erase(resultPath.begin() + maxIdx + 1);
            resultPath.erase(resultPath.begin() + minIdx + 1);
        } else {
            resultPath.erase(resultPath.begin() + minIdx + 1);
            resultPath.erase(resultPath.begin() + maxIdx + 1);
        }

        globalNumMatrices-=2; 
    }
}

bool dfs(uint64_t current,
         unordered_set<uint64_t>& visited,
         vector<uint64_t>& path,
         const unordered_map<uint64_t, vector<uint64_t>>& adjList,
         size_t totalNodes,
         vector<uint64_t>& resultPath) {
    visited.insert(current);
    path.push_back(current);

    if (path.size() == totalNodes) {
        resultPath = path;
        return true;
    }

    for (uint64_t neighbor : adjList.at(current)) {
        if (!visited.count(neighbor)) {
            if (dfs(neighbor, visited, path, adjList, totalNodes, resultPath)) {
                return true;
            }
        }
    }

    visited.erase(current);
    path.pop_back();
    return false;
}



MatrixNode matrixMult(tuple<vector<uint64_t>, vector<uint64_t>, vector<uint64_t>> matrixInfo){
    int rank, size;
    MatrixNode emptyNode;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    vector<uint64_t> metaData = get<0>(matrixInfo);
    vector<uint64_t> matrixA = get<1>(matrixInfo);
    vector<uint64_t> diff = get<2>(matrixInfo);

    int metaSize = metaData.size();
    int dataSize = matrixA.size();
    int diffSize = diff.size();

    vector<int> metaSizes(size), dataSizes(size), diffSizes(size);
    MPI_Gather(&metaSize, 1, MPI_INT, metaSizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&dataSize, 1, MPI_INT, dataSizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&diffSize, 1, MPI_INT, diffSizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> metaDispls(size), dataDispls(size), diffDispls(size);
    if (rank == 0) {
        metaDispls[0] = dataDispls[0] = diffDispls[0] = 0;
        for (int i = 1; i < size; ++i) {
            metaDispls[i] = metaDispls[i - 1] + metaSizes[i - 1];
            dataDispls[i] = dataDispls[i - 1] + dataSizes[i - 1];
            diffDispls[i] = diffDispls[i - 1] + diffSizes[i - 1];
        }
    }

    vector<uint64_t> allMeta, allData, allDiff;
    if (rank == 0) {
        allMeta.resize(metaDispls[size - 1] + metaSizes[size - 1]);
        allData.resize(dataDispls[size - 1] + dataSizes[size - 1]);
        allDiff.resize(diffDispls[size - 1] + diffSizes[size - 1]);
    }

    MPI_Gatherv(metaData.data(), metaSize, MPI_UINT64_T,
                allMeta.data(), metaSizes.data(), metaDispls.data(), MPI_UINT64_T,
                0, MPI_COMM_WORLD);

    MPI_Gatherv(matrixA.data(), dataSize, MPI_UINT64_T,
                allData.data(), dataSizes.data(), dataDispls.data(), MPI_UINT64_T,
                0, MPI_COMM_WORLD);

    MPI_Gatherv(diff.data(), diffSize, MPI_UINT64_T,
                allDiff.data(), diffSizes.data(), diffDispls.data(), MPI_UINT64_T,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        vector<MatrixNode> allNodes;
        int mat = 0;
        for (int proc = 0; proc < size; ++proc) {
            uint64_t i = metaDispls[proc];
            uint64_t metaEnd = i + metaSizes[proc];
            uint64_t dataStart = dataDispls[proc];
            uint64_t diffStart = diffDispls[proc];
            uint64_t diffEnd = diffStart + diffSizes[proc];

            int mat_local = 0;  

            // cout << "\n[Processing from rank " << proc << "]\n";
            // cout << "metaStart: " << i << ", metaEnd: " << metaEnd << "\n";
            // cout << "dataStart: " << dataStart << ", dataSize: " << dataSizes[proc] << "\n";
            // cout << "diffStart: " << diffStart << ", diffEnd: " << diffEnd << "\n";

            while (i < metaEnd) {
                // cout << "Parsing matrix " << mat << " (local mat: " << mat_local << ")...\n";

                MatrixNode node;
                node.id = allMeta[i++];
                node.height = allMeta[i++];
                node.width = allMeta[i++];
                node.numNonZero = allMeta[i++];
                node.BlockSize = allMeta[i++];

                // cout << "  ID: " << node.id << ", Size: " << node.height << "x" << node.width
                //     << ", NNZ: " << node.numNonZero << ", BlockSize: " << node.BlockSize << "\n";

                uint64_t posSize = 2 * node.numNonZero;
                node.pos.insert(node.pos.end(), allMeta.begin() + i, allMeta.begin() + i + posSize);
                i += posSize;

                // cout << "  pos range: " << (i - posSize) << " to " << i - 1 << "\n";
                
                uint64_t start = dataStart + allDiff[diffStart + mat_local];
                uint64_t end = (diffStart + mat_local + 1 < diffEnd)
                            ? dataStart + allDiff[diffStart + mat_local + 1]
                            : dataStart + dataSizes[proc];


                // cout << "  data range: " << start << " to " << end - 1 << " (size = " << (end - start) << ")\n";
                node.data.insert(node.data.end(), allData.begin() + start, allData.begin() + end);

                allNodes.push_back(node);
                ++mat_local; 
                ++mat;        
            }
        }


        // std::cout << "Total matrices gathered at rank 0: " << allNodes.size() << "\n";
        // for (const auto& node : allNodes) {
        //     std::cout << "Matrix ID " << node.id
        //               << " — Size: " << node.height << "x" << node.width
        //               << ", Non-zero blocks: " << node.numNonZero
        //               << ", Data size: " << node.data.size() <<  "\n";
        // }
    
        unordered_map<uint64_t, vector<uint64_t>> adjList;
        unordered_map<uint64_t, MatrixNode> idToNode;  
        for (const auto& node : allNodes) {
            idToNode[node.id] = node;
        }

        #pragma omp parallel for schedule(dynamic)
        for (uint64_t i = 0; i < allNodes.size(); ++i) {
            for (uint64_t j = 0; j < allNodes.size(); ++j) {
                if (i != j && allNodes[i].width == allNodes[j].height) {
                    #pragma omp critical
                    adjList[allNodes[i].id].push_back(allNodes[j].id);
                }
            }
            // cout << "printing adjacent vectors\n";
            // cout << "Matrix ID " << allNodes[i].id << " — Adjacent Matrices: ";
            printVector(adjList[allNodes[i].id], rank);
        }

        vector<uint64_t> resultPath;

        for (const auto& node : allNodes) {
            unordered_set<uint64_t> visited;
            vector<uint64_t> path;

            if (dfs(node.id, visited, path, adjList, allNodes.size(), resultPath)) {
                // cout << "Valid multiplication order found:\n";
                for (uint64_t id : resultPath) {
                    cout << "Matrix " << id << " ";
                }
                cout << endl;
                break;  
            }
        }
        if (resultPath.empty()) {
            // cout << "No valid multiplication order found.\n";
            return emptyNode;
        }
        uint64_t globalNumMatrices = allNodes.size();
        
        while(globalNumMatrices > 1) {
            // cout << "Performing heuristic calculation...here we go again\n";
            performHuesristicCalc(globalNumMatrices, idToNode, resultPath);
        }
        // cout << "final answer\n";
        // for (const auto& node : resultPath) {
        //     cout << "Matrix ID " << node << " — Size: " << idToNode[node].height << "x" << idToNode[node].width
        //          << ", Data size: " << idToNode[node].data.size() << endl;
        // }
        // printVector(idToNode[resultPath[0]].data, rank);
        // printVector(idToNode[resultPath[0]].pos, rank);
        return idToNode[resultPath[0]];

    }else{
        return emptyNode;
    } 
}