#ifndef MATRIX_H
#define MATRIX_H

#include "VectorElement.h"
#include <iostream>
#include <stdint.h>
#include <string>
#include <cstring>
#include <sstream>
#include <fstream>
#include <vector>
#include <time.h>
#include <queue>
#include <cmath>
#include <map>
#include <stdexcept>
#include <algorithm>
#include <cfloat>
#include <cstdlib>

using namespace std;

class Matrix {
protected:
    double *rawData;

public:

    int rowNum;
    int colNum;

    inline Matrix() {
        this->rawData = NULL;
        this->rowNum = 0;
        this->colNum = 0;
    }

    inline ~Matrix() {
        if(rawData){
            delete[] rawData;
        }
    }

    inline double *getRowPtr(const int rowIndex) const {
        return &rawData[rowIndex * colNum];
    }

    inline void init(double *rawData, const int rowNum, const int colNum) {
        this->rowNum = rowNum;
        this->colNum = colNum;
        this->rawData = rawData;
    }

    inline void init(const int rowNum, const int colNum) {
        this->rowNum = rowNum;
        this->colNum = colNum;
        this->rawData = new double[rowNum * colNum];
    }

    inline void setValue(const int rowID, const int colID, const double value){
        this->rawData[rowID * colNum + colID] = value;
    }

    void makePPositive(Matrix &origin, const double maxNorm, vector<VectorElement> &unsortedNorm) {
        this->rowNum = origin.rowNum;
        this->colNum = origin.colNum + 2;
        this->rawData = new double[rowNum * colNum];

        double powMaxNorm = maxNorm * maxNorm;

        // map to d+2
        for (int rowIndex = 0; rowIndex < rowNum; rowIndex++) {

            double *rowPtr = &rawData[rowIndex * colNum];
            double *oriPtr = origin.getRowPtr(rowIndex);
            double originNorm = unsortedNorm[rowIndex].data;

            rowPtr[1] = sqrt((powMaxNorm - originNorm * originNorm)/powMaxNorm) + 1;

            rowPtr[0] = rowPtr[1] * rowPtr[1];
            for (int colIndex = 2; colIndex < colNum; colIndex++) {
                rowPtr[colIndex] = oriPtr[colIndex-2] / maxNorm + 1;
                rowPtr[0] += rowPtr[colIndex] * rowPtr[colIndex];
            }

        }
    }

    void makeQPositive(Matrix &origin, vector<VectorElement> &unsortedNorm) {
        this->rowNum = origin.rowNum;
        this->colNum = origin.colNum + 2;
        this->rawData = new double[rowNum * colNum];


        // map to d+2
        for (int rowIndex = 0; rowIndex < rowNum; rowIndex++) {

            double *rowPtr = &rawData[rowIndex * colNum];
            double *oriPtr = origin.getRowPtr(rowIndex);

            double originalNorm = unsortedNorm[rowIndex].data;

            rowPtr[0] = -1;
            rowPtr[1] = 2;

            for (int colIndex = 2; colIndex < colNum; colIndex++) {
                rowPtr[colIndex] = 2*(oriPtr[colIndex-2] / originalNorm + 1);
            }
        }
    }

    void readData(string dataFilePath) {
        vector <string> lines;
        string line;

        ifstream fin(dataFilePath.c_str());

        int rowNum = 0;
        int colNum = 0;

        while (getline(fin, line)) {
            if (line.length() == 0) {
                continue;
            }
            lines.push_back(line);
        }

        if (lines.size() == 0) {
            return;
        }

        fin.close();

        stringstream test(lines[0]);
        double tempValue;
        while (test >> tempValue) {
            if (test.peek() == ',') {
                test.ignore();
                colNum++;
            }
        }
        colNum++;
        rowNum = lines.size();

        this->rowNum = rowNum;
        this->colNum = colNum;
        this->rawData = new double[rowNum * colNum];
        int colIndex = 0;

        for (int rowIndex = 0; rowIndex < lines.size(); rowIndex++) {
            stringstream ss(lines[rowIndex]);
            colIndex = 0;
            while (ss >> this->rawData[rowIndex * colNum + colIndex]) {
                if (ss.peek() == ',') {
                    ss.ignore();
                    colIndex++;
                }
            }
        }
    }
};

#endif //MATRIX_H