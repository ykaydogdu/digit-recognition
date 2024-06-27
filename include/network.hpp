#ifndef NETWORK_H
#define NETWORK_H

#include "ygzlinalg.hpp"
#include <math.h>

class Network {
public:
    Network(int* sizeArr, int n);
    void SGD(ygzVector<double>** trainingData, int numTraining, int miniBatchSize, double eta, int epochs, ygzVector<double>* testData = nullptr, int numTest = 0);

    void saveWeights(std::string filename);
    void loadWeights(std::string filename);

    void saveBiases(std::string filename);
    void loadBiases(std::string filename);
private:
    int numLayers;
    int* sizes;
    ygzVector<double>* biases;
    ygzMatrix<double>* weights;

    ygzVector<double> feedforward(ygzVector<double> a);
    void updateMiniBatch(ygzVector<double>* miniBatch, int miniBatchSize, double eta);
    void backprop(ygzVector<double> x, ygzVector<double> y, ygzVector<double>* nabla_b, ygzMatrix<double>* nabla_w);
    int evaluate(ygzVector<double>* testData, int testDataSize);
    ygzVector<double> costDerivative(ygzVector<double> outputActivations, ygzVector<double> y);
};

#endif