#ifndef MAIN_HPP
#define MAIN_HPP

#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <algorithm>
#include <climits>
#include <cuda_runtime.h>

using namespace std;
struct MatrixNode {
    uint64_t id;
    uint64_t height;
    uint64_t width;
    uint64_t numNonZero;
    uint64_t BlockSize;
    vector<uint64_t> pos;
    vector<uint64_t> data;
};

const uint64_t MOD = LLONG_MAX;

extern map<pair<uint64_t, uint64_t>, vector<uint64_t>> posToBlockData;
extern unordered_map<uint64_t, MatrixNode> idToNode;
extern unordered_map<uint64_t, vector<uint64_t>> adjList;

void write_matrices(string outPathFile, MatrixNode datax);
tuple<vector<uint64_t>, vector<uint64_t>, vector<uint64_t>> read_matrices(string inputPathFolder, int rank, int size, uint64_t NumMatrices, uint64_t BlockSize);
MatrixNode matrixMult(tuple<vector<uint64_t>, vector<uint64_t>, vector<uint64_t>> matrixInfo);
void performHuesristicCalc(uint64_t &globalNumMatrices, unordered_map<uint64_t, MatrixNode>& idToNode, vector<uint64_t> &resultPath);
MatrixNode launchCudaKernel(MatrixNode matA, MatrixNode matB);
MatrixNode launchHostMult(MatrixNode matA, MatrixNode matB);
bool hasIntersection(pair<uint64_t, uint64_t> p1, pair<uint64_t, uint64_t> p2);

__global__ void blockMultKernel(uint64_t* Adata, uint64_t* Bdata, uint64_t* pairIndices, uint64_t BlockSize, uint64_t* Cdata, uint64_t MOD);

#endif
