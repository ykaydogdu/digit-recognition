#ifndef LINALG_H
#define LINALG_H

#include <stdexcept>
#include <cstddef>
#include <cmath>
#include <time.h>
#include <vector>
using std::vector;

#define TOLERANCE 1e-5

template <class T> class ygzVector;
template <class T> class ygzMatrix;

template <class T>
class ygzVector
{
public:
    // Constructors
    ygzVector();
    ygzVector(size_t nDims);
    ygzVector(size_t nDims, const T* inputData);
    ygzVector(vector<T> &inputData);
    ygzVector(const ygzVector<T> &copyVector); // Copy constructor
    ygzVector(ygzVector<T> &&moveVector) noexcept; // Move constructor
    ~ygzVector();

    // Assignment
    ygzVector<T>& operator=(const ygzVector<T> &rhs); // copy assignment
    ygzVector<T>& operator=(ygzVector<T> &&rhs); // move assignment

    // getter
    size_t getNumDims() const;

    // element access
    T getElement(size_t index) const;
    void setElement(size_t index, T value);
    unsigned int argmax() const;

    // Computations
    T norm(int l = 2) const;
    ygzVector<T> normalized() const; // return a normalized vector
    void normalize(); // normalize the vector in place

    // Equality
    bool operator==(const ygzVector<T> &rhs) const;
    bool operator!=(const ygzVector<T> &rhs) const;
    bool compare(const ygzVector<T> &v2, double tolerance = TOLERANCE) const;
    bool closeEnough(T a, T b, double tolerance = TOLERANCE) const;

    // Operators
    ygzVector<T> operator+(const ygzVector<T> &rhs) const;
    ygzVector<T> operator-(const ygzVector<T> &rhs) const;
    ygzVector<T> operator*(const T &rhs) const;
    template <class U> friend ygzVector<U> operator*(const U &lhs, const ygzVector<U> &rhs);

    // Function mapping
    ygzVector<T> map(T (*f)(T)) const;
    void mapInPlace(T (*f)(T));

    // Static Functions
    static T dotProduct(const ygzVector<T> &lhs, const ygzVector<T> &rhs);
    static ygzVector<T> crossProduct(const ygzVector<T> &lhs, const ygzVector<T> &rhs);
    static ygzVector<T> hadamardProduct(const ygzVector<T> &lhs, const ygzVector<T> &rhs);
    static ygzVector<T> random(size_t nDims);
    static ygzVector<T> randn(size_t nDims);

    // Convert to Matrix
    ygzMatrix<T> toMatrix(bool columnVector = true) const;

    // toString()
    std::string toString() const;
    std::string toCSV() const;

private:
    size_t nDims;
    T *data;
};

template <class T>
class ygzMatrix
{
public:
    // Constructors
    ygzMatrix();
    ygzMatrix(size_t nRows, size_t nCols);
    ygzMatrix(size_t nRows, size_t nCols, const T* inputData);
    ygzMatrix(size_t nRows, size_t nCols, const vector<T> *inputData);
    ygzMatrix(const ygzMatrix<T> &copyMatrix); // Copy constructor
    ygzMatrix(ygzMatrix<T> &&moveMatrix) noexcept; // Move constructor

    // Assignment
    ygzMatrix<T>& operator=(const ygzMatrix<T> &rhs); // copy assignment
    ygzMatrix<T>& operator=(ygzMatrix<T> &&rhs); // move assignment
    
    // Destructor
    ~ygzMatrix();

    // Configuration
    bool resize(size_t nRows, size_t nCols);
    void setToIdentity();

    // Element access
    T getElement(size_t nRow, size_t nCol) const;
    bool setElement(size_t nRow, size_t nCol, T value);
    size_t getNumRows() const;
    size_t getNumCols() const;

    // Operations
    bool inverseInPlace();
    ygzMatrix<T> inverse() const;
    T determinant() const;
    bool transposeInPlace();
    ygzMatrix<T> transpose() const;

    // Equality
    bool operator==(const ygzMatrix<T> &rhs) const;
    bool operator!=(const ygzMatrix<T> &rhs) const;
    bool compare(const ygzMatrix<T> & m2, double tolerance = TOLERANCE) const;

    // Arithmetic operations
    template <class U> friend ygzMatrix<U> operator+(const ygzMatrix<U> &lhs, const ygzMatrix<U> &rhs);
    template <class U> friend ygzMatrix<U> operator+(const ygzMatrix<U> &matrix, const U &scalar);
    template <class U> friend ygzMatrix<U> operator+(const U &scalar, const ygzMatrix<U> &matrix);

    template <class U> friend ygzMatrix<U> operator-(const ygzMatrix<U> &lhs, const ygzMatrix<U> &rhs);
    template <class U> friend ygzMatrix<U> operator-(const ygzMatrix<U> &matrix, const U &scalar);
    template <class U> friend ygzMatrix<U> operator-(const U &scalar, const ygzMatrix<U> &matrix);

    template <class U> friend ygzMatrix<U> operator*(const ygzMatrix<U> &lhs, const ygzMatrix<U> &rhs);
    template <class U> friend ygzMatrix<U> operator*(const ygzMatrix<U> &matrix, const U &scalar);
    template <class U> friend ygzMatrix<U> operator*(const U &scalar, const ygzMatrix<U> &matrix);

    // matrix * vector
    template <class U> friend ygzVector<U> operator*(const ygzMatrix<U> &matrix, const ygzVector<U> &vector);
    template <class U> friend ygzMatrix<U> operator*(const ygzVector<U> &vector, const ygzMatrix<U> &r_vector);

    // Special matrix generation
    static ygzMatrix<T> random(size_t nRows, size_t nCols);
    static ygzMatrix<T> randn(size_t nRows, size_t nCols);
    static ygzMatrix<T> identity(size_t n);

    // Hadamard product
    static ygzMatrix<T> hadamardProduct(const ygzMatrix<T> &lhs, ygzMatrix<T> &rhs);

    // toString()
    std::string toString() const;
    std::string toCSV() const;

private:
    size_t sub2Ind(size_t r, size_t c) const;
    bool isSquare() const;
    bool closeEnough(T a, T b, double tolerance = TOLERANCE) const;

    // Row and column operations
    void swapRow(int i, int j);
    void multRow(int i, T factor);
    void multAddRow(int i, int j, T factor);
    // void swapCol(int i, int j);
    // void multCol(int i, T factor);
    // void multAddCol(int i, int j, T factor);

    // Augmented matrix operations
    ygzMatrix<T> join(const ygzMatrix<T> &m);
    void separate(ygzMatrix<T> *m1, ygzMatrix<T> *m2, int colNum) const;

    int findRowWithMax(int col, int startingRow) const; // pivot finding

    ygzMatrix<T> findMinor(int row, int col) const; // find minor matrix

private:
    size_t nRows, nCols, nElements;
    T *data;
};

/* Implementation of ygzVector */
/* ************************************************************************* */
/*                        Constructors and Destructor                        */
/* ************************************************************************* */
// Default constructor
template <class T>
ygzVector<T>::ygzVector()
{
    nDims = 0;
    data = nullptr;
}

// Constructor with size
template <class T>
ygzVector<T>::ygzVector(size_t nDims) : nDims(nDims)
{
    data = new T[nDims];
    for (size_t i = 0; i < nDims; i++)
    {
        data[i] = 0.0;
    }
}

// Constructor with input data
template <class T>
ygzVector<T>::ygzVector(size_t nDims, const T *inputData) : nDims(nDims)
{
    data = new T[nDims];
    for (size_t i = 0; i < nDims; i++)
    {
        data[i] = inputData[i];
    }
}

// Constructor with std::vector input data
template <class T>
ygzVector<T>::ygzVector(vector<T> &inputData) : nDims(inputData.size())
{
    data = new T[nDims];
    for (size_t i = 0; i < nDims; i++)
    {
        data[i] = inputData.at(i);
    }
}

// Copy Constructor
template <class T>
ygzVector<T>::ygzVector(const ygzVector<T> &copyVector) : nDims(copyVector.nDims)
{
    data = new T[nDims];
    for (size_t i = 0; i < nDims; i++)
    {
        data[i] = copyVector.data[i];
    }
}

// Move Constructor
template <class T>
ygzVector<T>::ygzVector(ygzVector<T> &&moveVector) noexcept : nDims(moveVector.nDims), data(moveVector.data)
{
    moveVector.nDims = 0;
    moveVector.data = nullptr;
}

// Destructor
template <class T>
ygzVector<T>::~ygzVector()
{
    if (data != nullptr)
    {
        delete[] data;
    }
}

/* ************************************************************************* */
/*                             Assignment Operators                          */
/* ************************************************************************* */
// Copy assignment
template <class T>
ygzVector<T>& ygzVector<T>::operator=(const ygzVector<T> &rhs)
{
    if (this == &rhs)
        return *this;

    if (nDims != rhs.nDims)
    {
        delete[] data;
        nDims = rhs.nDims;
        data = new T[nDims];
    }

    for (size_t i = 0; i < nDims; i++)
    {
        data[i] = rhs.data[i];
    }

    return *this;
}

// Move assignment
template <class T>
ygzVector<T>& ygzVector<T>::operator=(ygzVector<T> &&rhs)
{
    if (this == &rhs)
        return *this;

    delete[] data;
    nDims = rhs.nDims;
    data = rhs.data;
    rhs.nDims = 0;
    rhs.data = nullptr;

    return *this;
}

/* ************************************************************************* */
/*                             Property Access                               */
/* ************************************************************************* */
// Get the number of dimensions
template <class T>
size_t ygzVector<T>::getNumDims() const
{
    return nDims;
}

// Get the element at the specified index
template <class T>
T ygzVector<T>::getElement(size_t index) const
{
    if (index < 0)
    {
        throw std::invalid_argument("Index must be positive");
    }
    size_t i = index;
    if (i >= nDims)
    {
        throw std::invalid_argument("Index out of bounds");
    }
    return data[i];
}

// Set the element at the specified index to the given value
template <class T>
void ygzVector<T>::setElement(size_t index, T value)
{
    if (index < 0 || index >= nDims)
    {
        throw std::invalid_argument("Index out of bounds");
    }
    data[index] = value;
}

// Return the index of the maximum element
template <class T>
unsigned int ygzVector<T>::argmax() const
{
    if (nDims == 0)
    {
        throw std::invalid_argument("Cannot find argmax of an empty vector");
    }

    unsigned int maxIndex = 0;
    T maxValue = data[0];
    for (size_t i = 1; i < nDims; i++)
    {
        if (data[i] > maxValue)
        {
            maxValue = data[i];
            maxIndex = i;
        }
    }

    return maxIndex;
}

/* ************************************************************************* */
/*                              Computations                                 */
/* ************************************************************************* */
// Compute the norm of the vector
template <class T>
T ygzVector<T>::norm(int l) const
{
    if (l != 1 && l != 2)
        throw std::invalid_argument("Only L1 and L2 norms are supported");
    T result = 0;
    if (l == 1)
    {
        for (size_t i = 0; i < nDims; i++)
        {
            result += fabs(data[i]);
        }
        return result;
    } else 
    {
        for (size_t i = 0; i < nDims; i++)
        {
            result += data[i] * data[i];
        }
        return sqrt(result);
    }
}

// Return a normalized vector
template <class T>
ygzVector<T> ygzVector<T>::normalized() const
{
    T n = norm();
    if (n == 0)
    {
        throw std::invalid_argument("Cannot normalize a zero vector");
    }

    ygzVector<T> result(nDims, data);
    return result * (static_cast<T>(1.0) / n);
}

// Normalize the vector in place
template <class T>
void ygzVector<T>::normalize()
{
    T n = norm();
    if (n == 0)
    {
        throw std::invalid_argument("Cannot normalize a zero vector");
    }

    for (size_t i = 0; i < nDims; i++)
    {
        data[i] /= n;
    }
}

/* ************************************************************************* */
/*                                Equality                                   */
/* ************************************************************************* */
// Check if two vectors are equal
template <class T>
bool ygzVector<T>::operator==(const ygzVector<T> &rhs) const
{
    if (nDims != rhs.nDims)
        return false;

    for (size_t i = 0; i < nDims; i++)
    {
        // if (data[i] != rhs.data[i])
        if (!closeEnough(data[i], rhs.data[i]))
            return false;
    }

    return true;
}

// Check if two vectors are not equal
template <class T>
bool ygzVector<T>::operator!=(const ygzVector<T> &rhs) const
{
    return !(*this == rhs);
}

// Compare two matrices with tolerance using MSE
template <class T>
bool ygzVector<T>::compare(const ygzVector<T> & m2, double tolerance) const
{
    if (nDims != m2.nDims)
        return false;

    double sum = 0.0;
    for (size_t i = 0; i < nDims; i++)
    {
        sum += (data[i] - m2.data[i]) * (data[i] - m2.data[i]);
    }

    return sqrt(sum / (nDims - 1)) < tolerance;
}

template <class T>
bool ygzVector<T>::closeEnough(T a, T b, double tolerance) const
{
    return fabs(a - b) < tolerance;
}

/* ************************************************************************* */
/*                                Operations                                 */
/* ************************************************************************* */
// Vector addition
template <class T>
ygzVector<T> ygzVector<T>::operator+(const ygzVector<T> &rhs) const
{
    if (nDims != rhs.nDims)
    {
        throw std::invalid_argument("Vector dimensions do not match");
    }

    ygzVector<T> result;
    result.nDims = nDims;
    result.data = new T[nDims];
    for (size_t i = 0; i < nDims; i++)
    {
        result.data[i] = data[i] + rhs.data[i];
    }

    return result;
}

// Vector subtraction
template <class T>
ygzVector<T> ygzVector<T>::operator-(const ygzVector<T> &rhs) const
{
    if (nDims != rhs.nDims)
    {
        throw std::invalid_argument("Vector dimensions do not match");
    }

    T* resData = new T[nDims];
    for (size_t i = 0; i < nDims; i++)
    {
        resData[i] = data[i] - rhs.data[i];
    }
    ygzVector<T> result(nDims, resData);
    delete[] resData;
    return result;
}

// Scalar multiplication
template <class T>
ygzVector<T> ygzVector<T>::operator*(const T &rhs) const
{
    T* resultData = new T[nDims];
    for (size_t i = 0; i < nDims; i++)
    {
        resultData[i] = data[i] * rhs;
    }
    ygzVector<T> result(nDims, resultData);
    delete[] resultData;
    return result;
}

// Scalar multiplication (friend function)
template <class T>
ygzVector<T> operator*(const T &lhs, const ygzVector<T> &rhs)
{
    return rhs * lhs;
}

// Dot product
template <class T>
T ygzVector<T>::dotProduct(const ygzVector<T> &lhs, const ygzVector<T> &rhs)
{
    if (lhs.nDims != rhs.nDims)
    {
        throw std::invalid_argument("Vector dimensions do not match");
    }

    T result = 0;
    for (size_t i = 0; i < lhs.nDims; i++)
    {
        result += lhs.data[i] * rhs.data[i];
    }

    return result;
}

// Cross product
template <class T>
ygzVector<T> ygzVector<T>::crossProduct(const ygzVector<T> &lhs, const ygzVector<T> &rhs)
{
    if (lhs.nDims != 3 || rhs.nDims != 3)
    {
        throw std::invalid_argument("Cross product is only defined for 3D vectors");
    }

    ygzVector<T> result;
    result.nDims = 3;
    result.data = new T[3];
    result.data[0] = lhs.data[1] * rhs.data[2] - lhs.data[2] * rhs.data[1];
    result.data[1] = lhs.data[2] * rhs.data[0] - lhs.data[0] * rhs.data[2];
    result.data[2] = lhs.data[0] * rhs.data[1] - lhs.data[1] * rhs.data[0];

    return result;
}

// Hadamard product
template <class T>
ygzVector<T> ygzVector<T>::hadamardProduct(const ygzVector<T> &lhs, const ygzVector<T> &rhs)
{
    if (lhs.nDims != rhs.nDims)
    {
        throw std::invalid_argument("Vector dimensions do not match");
    }

    ygzVector<T> result;
    result.nDims = lhs.nDims;
    result.data = new T[lhs.nDims];
    for (size_t i = 0; i < lhs.nDims; i++)
    {
        result.data[i] = lhs.data[i] * rhs.data[i];
    }

    return result;
}

/* ************************************************************************* */
/*                             Function Mapping                              */
/* ************************************************************************* */
// Apply a function to each element of the vector
template <class T>
ygzVector<T> ygzVector<T>::map(T (*f)(T)) const
{
    T* resultData = new T[nDims];
    for (size_t i = 0; i < nDims; i++)
    {
        resultData[i] = f(data[i]);
    }
    ygzVector<T> result(nDims, resultData);
    delete[] resultData;
    return result;
}

// Apply a function to each element of the vector in place
template <class T>
void ygzVector<T>::mapInPlace(T (*f)(T))
{
    for (size_t i = 0; i < nDims; i++)
    {
        data[i] = f(data[i]);
    }
}

/* ************************************************************************* */
/*                             Convert to Matrix                             */
/* ************************************************************************* */
// Convert the vector to a matrix
template <class T>
ygzMatrix<T> ygzVector<T>::toMatrix(bool columnVector) const
{
    ygzMatrix<T> result;
    if (columnVector)
    {
        result = ygzMatrix<T>(nDims, 1);
        for (size_t i = 0; i < nDims; i++)
        {
            result.setElement(i, 0, data[i]);
        }
    } else 
    {
        result = ygzMatrix<T>(1, nDims);
        for (size_t i = 0; i < nDims; i++)
        {
            result.setElement(0, i, data[i]);
        }
    }

    return result;
}

/* ************************************************************************* */
/*                        Special Vector Generation                          */
/* ************************************************************************* */
// Generate a random vector
template <class T>
ygzVector<T> ygzVector<T>::random(size_t nDims)
{
    T* data = new T[nDims];
    for (size_t i = 0; i < nDims; i++)
    {
        data[i] = static_cast<T>(rand()) / RAND_MAX;
    }
    ygzVector<T> result = ygzVector<T>(nDims, data);
    delete[] data;
    return result;
}

// Generate a random vector with normal distribution (mean = 0, std = 1)
template <class T>
ygzVector<T> ygzVector<T>::randn(size_t nDims)
{
    T* data = new T[nDims];
    for (size_t i = 0; i < nDims; i++)
    {
        double u1 = static_cast<double>(rand()) / RAND_MAX;
        double u2 = static_cast<double>(rand()) / RAND_MAX;
        data[i] = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
    }
    ygzVector<T> result = ygzVector<T>(nDims, data);
    delete[] data;
    return result;
}

/* ************************************************************************* */
/*                             toString()                                    */
/* ************************************************************************* */
// Convert the vector to a string
template <class T>
std::string ygzVector<T>::toString() const
{
    std::string result = "[";
    for (size_t i = 0; i < nDims; i++)
    {
        result += std::to_string(data[i]);
        if (i < nDims - 1)
        {
            result += ", ";
        }
    }
    result += "]";
    return result;
}

// Convert the vector to a CSV string
template <class T>
std::string ygzVector<T>::toCSV() const
{
    std::string result = "";
    for (size_t i = 0; i < nDims; i++)
    {
        result += std::to_string(data[i]);
        if (i < nDims - 1)
        {
            result += ",";
        }
    }
    result += "\n";
    return result;
}


/* Implementation of ygzMatrix */
/* ********************************************************************************************* */
/* 					    		Constructors & Destructor 										 */
/* ********************************************************************************************* */
// Default constructor
template <class T>
ygzMatrix<T>::ygzMatrix()
{
    nRows = 0;
    nCols = 0;
    nElements = 0;
    data = nullptr;
}

// Construct empty matrix (all elements are 0)
template <class T>
ygzMatrix<T>::ygzMatrix(size_t nRows, size_t nCols)
{
    this->nRows = nRows;
    this->nCols = nCols;
    nElements = nRows * nCols;
    data = new T[nElements];
    for (size_t i = 0; i < nElements; i++)
        data[i] = 0.0; // Initialize elements with default value
}

// Construct matrix from array
template <class T>
ygzMatrix<T>::ygzMatrix(size_t nRows, size_t nCols, const T* inputData)
{
    this->nRows = nRows;
    this->nCols = nCols;
    nElements = nRows * nCols;
    data = new T[nElements];
    for (size_t i = 0; i < nElements; i++)
        data[i] = inputData[i];
}

// Construct matrix from vector
template <class T>
ygzMatrix<T>::ygzMatrix(size_t nRows, size_t nCols, const vector<T> *inputData)
{
    this->nRows = nRows;
    this->nCols = nCols;
    nElements = nRows * nCols;
    data = new T[nElements];
    for (size_t i = 0; i < nElements; i++)
        data[i] = inputData->at(i);
}

// Copy constructor
template <class T>
ygzMatrix<T>::ygzMatrix(const ygzMatrix<T> &copyMatrix)
{
    nRows = copyMatrix.nRows;
    nCols = copyMatrix.nCols;
    nElements = nRows * nCols;
    data = new T[nElements];
    for (size_t i = 0; i < nElements; i++)
        data[i] = copyMatrix.data[i];
}

// Move constructor
template <class T>
ygzMatrix<T>::ygzMatrix(ygzMatrix<T> &&moveMatrix) noexcept : nRows(moveMatrix.nRows), nCols(moveMatrix.nCols), nElements(moveMatrix.nElements), data(moveMatrix.data)
{
    moveMatrix.nRows = 0;
    moveMatrix.nCols = 0;
    moveMatrix.nElements = 0;
    moveMatrix.data = nullptr;
}

// Destructor
template <class T>
ygzMatrix<T>::~ygzMatrix()
{
    delete[] data;
}

/* ********************************************************************************************* */
/* 					    		Assignment Operators 											 */
/* ********************************************************************************************* */
// Copy assignment
template <class T>
ygzMatrix<T>& ygzMatrix<T>::operator=(const ygzMatrix<T> &rhs)
{
    if (this == &rhs)
        return *this;

    if (nRows != rhs.nRows || nCols != rhs.nCols)
    {
        delete[] data;
        nRows = rhs.nRows;
        nCols = rhs.nCols;
        nElements = nRows * nCols;
        data = new T[nElements];
    }

    for (size_t i = 0; i < nElements; i++)
    {
        data[i] = rhs.data[i];
    }

    return *this;
}

// Move assignment
template <class T>
ygzMatrix<T>& ygzMatrix<T>::operator=(ygzMatrix<T> &&rhs)
{
    if (this == &rhs)
        return *this;

    delete[] data;
    nRows = rhs.nRows;
    nCols = rhs.nCols;
    nElements = rhs.nElements;
    data = rhs.data;
    rhs.nRows = 0;
    rhs.nCols = 0;
    rhs.nElements = 0;
    rhs.data = nullptr;

    return *this;
}

/* ********************************************************************************************* */
/* 					    			  Special Matrix Generation									 */
/* ********************************************************************************************* */
// Generate a random matrix with values between 0 and 1
template <class T>
ygzMatrix<T> ygzMatrix<T>::random(size_t nRows, size_t nCols)
{
    T* data = new T[nRows * nCols];
    for (size_t i = 0; i < nRows * nCols; i++)
        data[i] = static_cast<T>(rand()) / RAND_MAX; // Random value between 0 and 1

    ygzMatrix<T> randomMatrix(nRows, nCols, data);
    delete[] data;
    return randomMatrix;
}

// Generate a random matrix using normal distribution (mean = 0, std = 1)
template <class T>
ygzMatrix<T> ygzMatrix<T>::randn(size_t nRows, size_t nCols)
{
    T* data = new T[nRows * nCols];
    for (size_t i = 0; i < nRows * nCols; i++)
    {
        double u1 = static_cast<double>(rand()) / RAND_MAX;
        double u2 = static_cast<double>(rand()) / RAND_MAX;
        data[i] = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
    }

    ygzMatrix<T> randomMatrix(nRows, nCols, data);
    delete[] data;
    return randomMatrix;
}

// Generate an identity matrix
template <class T>
ygzMatrix<T> ygzMatrix<T>::identity(size_t n)
{
    T* data = new T[n * n];
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
            data[i * n + j] = i == j ? 1.0 : 0.0;
    }

    ygzMatrix<T> identityMatrix(n, n, data);
    delete[] data;
    return identityMatrix;
}

/* ********************************************************************************************* */
/* 					    		        Configuration 											 */
/* ********************************************************************************************* */
// Resize matrix
template <class T>
bool ygzMatrix<T>::resize(size_t nRows, size_t nCols)
{
    if (nRows == this->nRows && nCols == this->nCols)
        return true;

    delete[] data;
    this->nRows = nRows;
    this->nCols = nCols;
    nElements = nRows * nCols;
    data = new T[nElements];
    if (data == nullptr)
        return false;
    for (size_t i = 0; i < nElements; i++)
        data[i] = 0.0; // Initialize elements with default value

    return true;
}

// Set matrix to identity matrix
template <class T>
void ygzMatrix<T>::setToIdentity()
{
    if (!isSquare())
        throw std::invalid_argument("Matrix must be square to set to identity");

    for (size_t row = 0; row < nRows; row++)
    {
        for (size_t col = 0; col < nCols; col++)
        {
            setElement(row, col, row == col ? 1.0 : 0.0);
        }
    }
}

/* ********************************************************************************************* */
/* 					    			  Subscript Converter      									 */
/* ********************************************************************************************* */
// Convert subscripts to linear index
template <class T>
size_t ygzMatrix<T>::sub2Ind(size_t r, size_t c) const
{
    if (r < 0 || r >= nRows || c < 0 || c >= nCols)
        throw std::out_of_range("Index out of range");
    return r * nCols + c;
}

/* ********************************************************************************************* */
/* 					    			  Element access 											 */
/* ********************************************************************************************* */
// Get element at position (nRow, nCol)
template <class T>
T ygzMatrix<T>::getElement(size_t nRow, size_t nCol) const
{
    if (nRow >= nRows || nCol >= nCols)
        throw std::out_of_range("Index out of range");

    return data[sub2Ind(nRow, nCol)];
}

// Set element at position (nRow, nCol)
template <class T>
bool ygzMatrix<T>::setElement(size_t nRow, size_t nCol, T value)
{
    if (nRow >= nRows || nCol >= nCols)
        return false;

    data[sub2Ind(nRow, nCol)] = value;
    return true;
}

// Get number of rows
template <class T>
size_t ygzMatrix<T>::getNumRows() const
{
    return nRows;
}

// Get number of columns
template <class T>
size_t ygzMatrix<T>::getNumCols() const
{
    return nCols;
}

/* ********************************************************************************************* */
/* 					    			  Equality Operations										 */
/* ********************************************************************************************* */
// Check if two matrices are equal
template <class T>
bool ygzMatrix<T>::operator==(const ygzMatrix<T> &rhs) const
{
    if (nRows != rhs.nRows || nCols != rhs.nCols)
        return false;

    for (size_t i = 0; i < nElements; i++)
    {
        // if (data[i] != rhs.data[i])
        if (!closeEnough(data[i], rhs.data[i]))
            return false;
    }

    return true;
}

// Check if two matrices are not equal
template <class T>
bool ygzMatrix<T>::operator!=(const ygzMatrix<T> &rhs) const
{
    return !(*this == rhs);
}

// Compare two matrices with tolerance using MSE
template <class T>
bool ygzMatrix<T>::compare(const ygzMatrix<T> & m2, double tolerance) const
{
    if (nRows != m2.nRows || nCols != m2.nCols)
        return false;

    double sum = 0.0;
    for (size_t i = 0; i < nElements; i++)
    {
        sum += (data[i] - m2.data[i]) * (data[i] - m2.data[i]);
    }

    return sqrt(sum / (nElements - 1)) < tolerance;
}

template <class T>
bool ygzMatrix<T>::closeEnough(T a, T b, double tolerance) const
{
    return fabs(a - b) < tolerance;
}

/* ********************************************************************************************* */
/* 					    			  Arithmetic Operations										 */
/* ********************************************************************************************* */
// Matrix addition
template <class T>
ygzMatrix<T> operator+(const ygzMatrix<T> &lhs, const ygzMatrix<T> &rhs)
{
    if (lhs.nRows != rhs.nRows || lhs.nCols != rhs.nCols)
        throw std::invalid_argument("Matrices must have the same dimensions");

    T* result = new T[lhs.nElements];
    for (size_t i = 0; i < lhs.nElements; i++)
        result[i] = lhs.data[i] + rhs.data[i];

    ygzMatrix<T> resultMatrix(lhs.nRows, lhs.nCols, result);
    delete[] result;
    return resultMatrix;
}

// Matrix-scalar addition
template <class T>
ygzMatrix<T> operator+(const ygzMatrix<T> &matrix, const T &scalar)
{
    T* result = new T[matrix.nElements];
    for (size_t i = 0; i < matrix.nElements; i++)
        result[i] = matrix.data[i] + scalar;

    ygzMatrix<T> resultMatrix(matrix.nRows, matrix.nCols, result);
    delete[] result;
    return resultMatrix;
}

// Scalar-matrix addition
template <class T>
ygzMatrix<T> operator+(const T &scalar, const ygzMatrix<T> &matrix)
{
    return matrix + scalar; // Addition is commutative
}

// Matrix subtraction
template <class T>
ygzMatrix<T> operator-(const ygzMatrix<T> &lhs, const ygzMatrix<T> &rhs)
{
    if (lhs.nRows != rhs.nRows || lhs.nCols != rhs.nCols)
        throw std::invalid_argument("Matrices must have the same dimensions");

    T* result = new T[lhs.nElements];
    for (size_t i = 0; i < lhs.nElements; i++)
        result[i] = lhs.data[i] - rhs.data[i];

    ygzMatrix<T> resultMatrix(lhs.nRows, lhs.nCols, result);
    delete[] result;
    return resultMatrix;
}

// Matrix-scalar subtraction
template <class T>
ygzMatrix<T> operator-(const ygzMatrix<T> &matrix, const T &scalar)
{
    T* result = new T[matrix.nElements];
    for (size_t i = 0; i < matrix.nElements; i++)
        result[i] = matrix.data[i] - scalar;

    ygzMatrix<T> resultMatrix(matrix.nRows, matrix.nCols, result);
    delete[] result;
    return resultMatrix;
}

// Scalar-matrix subtraction
template <class T>
ygzMatrix<T> operator-(const T &scalar, const ygzMatrix<T> &matrix)
{
    T* result = new T[matrix.nElements];
    for (size_t i = 0; i < matrix.nElements; i++)
        result[i] = scalar - matrix.data[i];

    ygzMatrix<T> resultMatrix(matrix.nRows, matrix.nCols, result);
    delete[] result;
    return resultMatrix;
}

// Matrix multiplication
template <class T>
ygzMatrix<T> operator*(const ygzMatrix<T> &lhs, const ygzMatrix<T> &rhs)
{
    if (lhs.nCols != rhs.nRows)
        throw std::invalid_argument("Number of columns in the first matrix must be equal to the number of rows in the second matrix");

    int nRows = lhs.nRows;
    int nCols = rhs.nCols;
    T* result = new T[nRows * nCols]; // Resulting matrix has dimensions (lhs.nRows, rhs.nCols)
    for (size_t i = 0; i < nRows; i++)
    {
        for (size_t j = 0; j < nCols; j++)
        {
            int index = i * nCols + j;
            result[index] = 0; // Initialize element to 0
            for (size_t c = 0; c < lhs.nCols; c++) // for the ith row of the lhs
                for (size_t r = 0; r < rhs.nCols; r++) // for the jth column of the rhs
                    result[index] += lhs.getElement(i, c) * rhs.getElement(r, j); // Multiply and accumulate (dot product)
        }
    }

    ygzMatrix<T> resultMatrix(nRows, nCols, result);
    delete[] result;
    return resultMatrix;
}

// Matrix-scalar multiplication
template <class T>
ygzMatrix<T> operator*(const ygzMatrix<T> &matrix, const T &scalar)
{
    T* result = new T[matrix.nElements];
    for (size_t i = 0; i < matrix.nElements; i++)
        result[i] = matrix.data[i] * scalar;

    ygzMatrix<T> resultMatrix(matrix.nRows, matrix.nCols, result);
    delete[] result;
    return resultMatrix;
}

// Scalar-matrix multiplication
template <class T>
ygzMatrix<T> operator*(const T &scalar, const ygzMatrix<T> &matrix)
{
    return matrix * scalar; // Multiplication is commutative
}

/* ********************************************************************************************* */
/* 					    			  Checks                  									 */
/* ********************************************************************************************* */
template <class T>
bool ygzMatrix<T>::isSquare() const
{
    return nRows == nCols;
}

/* ********************************************************************************************* */
/* 					    			  Augmented Matrices    									 */
/* ********************************************************************************************* */
// Join two matrices into one (augmented matrix)
template <class T>
ygzMatrix<T> ygzMatrix<T>::join(const ygzMatrix<T> &m)
{
    if (nRows != m.nRows)
        throw std::invalid_argument("Matrices must have the same number of rows");

    T* newData = new T[nRows * (nCols + m.nCols)];
    for (size_t i = 0; i < nRows; i++)
    {
        for (size_t j = 0; j < (nCols + m.nCols); j++)
        {
            int index = i * (nCols + m.nCols) + j;
            if (j < nCols)
                newData[index] = getElement(i, j);
            else
                newData[index] = m.getElement(i, j - nCols);
        }
    }
    ygzMatrix<T> result(nRows, (nCols + m.nCols), newData);
    delete[] newData;
    return result;
}

// Separate augmented matrix into two matrices
// Output is generated to m1 and m2 given as the input arguments
template <class T>
void ygzMatrix<T>::separate(ygzMatrix<T> *m1, ygzMatrix<T> *m2, int colNum) const
{
    if (colNum < 0 || colNum >= nCols)
        throw std::out_of_range("Column number out of range");

    m1->resize(nRows, colNum);
    m2->resize(nRows, nCols - colNum);

    for (size_t i = 0; i < nRows; i++)
    {
        for (size_t j = 0; j < colNum; j++)
            m1->setElement(i, j, getElement(i, j));
        for (size_t j = 0; j < nCols - colNum; j++)
            m2->setElement(i, j, getElement(i, colNum + j));
    }
}

/* ********************************************************************************************* */
/* 					    			  Gauss Jordan Elimination									 */
/* ********************************************************************************************* */
// Swaps row i with row j (in place)
template <class T>
void ygzMatrix<T>::swapRow(int i, int j)
{
    if (i == j)
        return;
    for (size_t c = 0; c < nCols; c++)
    {
        T temp = getElement(i, c);
        setElement(i, c, getElement(j, c));
        setElement(j, c, temp);
    }
}

// Multiplies row i by a factor (in place)
template <class T>
void ygzMatrix<T>::multRow(int i, T factor)
{
    for (size_t c = 0; c < nCols; c++)
        setElement(i, c, getElement(i, c) * factor);
}

// Multiplies row i by a factor and adds it to row j (in place)
template <class T>
void ygzMatrix<T>::multAddRow(int i, int j, T factor)
{
    for (size_t c = 0; c < nCols; c++)
        setElement(j, c, getElement(j, c) + factor * getElement(i, c));
}

// Function to find the row with the maximum value in a given column
// Returns the row index
template <class T>
int ygzMatrix<T>::findRowWithMax(int col, int startingRow) const
{
    T max = getElement(startingRow, col);
    int rowIndex = startingRow;
    for (size_t i = startingRow + 1; i < nRows; i++)
    {
        T val = getElement(i, col);
        if (val > max)
        {
            max = val;
            rowIndex = i;
        }
    }
    return rowIndex;
}

/* ********************************************************************************************* */
/* 					    			  Inverse Operations										 */
/* ********************************************************************************************* */
// Inverts the matrix in place
template <class T>
bool ygzMatrix<T>::inverseInPlace()
{
    if (!isSquare())
        throw std::invalid_argument("Matrix must be square to invert");

    // Create an identity matrix
    ygzMatrix<T> identityMatrix = ygzMatrix<T>::identity(nRows);
    ygzMatrix<T> augmentedMatrix = join(identityMatrix); // Augment the matrix with the identity matrix

    // Perform Gauss-Jordan elimination
    ygzMatrix<T>* lhs = new ygzMatrix<T>(*this);
    ygzMatrix<T>* res = new ygzMatrix<T>(nRows, nCols);

    int cRow, cCol;
    int maxCount = 100;
    int count = 0;
    bool complete = false;
    while ((!complete) && (count < maxCount))
    {
        for (size_t diagIndex = 0; diagIndex < nRows; diagIndex++)
        {
            cRow = diagIndex;
            cCol = diagIndex;
            int maxRow = augmentedMatrix.findRowWithMax(cCol, cRow); // Find row with max element in the cCol th column
            if (maxRow == -1)
                return false; // something went wrong
            augmentedMatrix.swapRow(cRow, maxRow); // Swap the row with the maximum value to the current row
            if (augmentedMatrix.getElement(cRow, cCol) != 1)
                augmentedMatrix.multRow(cRow, 1 / augmentedMatrix.getElement(cRow, cCol)); // Divide the row by the diagonal element to make it 1

            // Now process the rows and columns
            for (size_t r = cRow + 1; r < nRows; r++)
            {
                T currentElement = augmentedMatrix.getElement(r, cCol);
                T diagElement = augmentedMatrix.getElement(cRow, cCol);
                if (closeEnough(currentElement, 0.0) || closeEnough(diagElement, 0.0))
                    continue;

                T factor = -(currentElement / diagElement);
                augmentedMatrix.multAddRow(cRow, r, factor); // Make the elements below the diagonal element 0
            }
            for (size_t c = cCol + 1; c < nCols; c++)
            {
                T currentElement = augmentedMatrix.getElement(cRow, c);
                T diagElement = augmentedMatrix.getElement(cRow, cCol);
                if (closeEnough(currentElement, 0.0) || closeEnough(diagElement, 0.0))
                    continue;

                T factor = -(currentElement / diagElement);
                augmentedMatrix.multAddRow(c, cCol, factor); // Make the elements to the right of the diagonal element 0
            }
        }
        // Now seperate
        augmentedMatrix.separate(lhs, res, nCols);
        if (*lhs == identityMatrix) // we are done
        {
            complete = true;
            // res is now the inverse of the original matrix
            for (size_t i = 0; i < nRows; i++)
            {
                for (size_t j = 0; j < nCols; j++)
                    setElement(i, j, res->getElement(i, j));
            }
        }
        count++;
    }
    return complete;
}

// Returns the inverse of the matrix
template <class T>
ygzMatrix<T> ygzMatrix<T>::inverse() const
{
    ygzMatrix<T> copy(*this);
    copy.inverseInPlace();
    return copy;
}

/* ********************************************************************************************* */
/* 					    			  Determinant Operations									 */
/* ********************************************************************************************* */
// Find the minor with respect to given row and column
template <class T>
ygzMatrix<T> ygzMatrix<T>::findMinor(int row, int col) const
{
    T* minorData = new T[(nRows - 1) * (nCols - 1)];
    int minorIndex = 0;
    for (size_t i = 0; i < nRows; i++)
    {
        if (i == row)
            continue;
        for (size_t j = 0; j < nCols; j++)
        {
            if (j == col)
                continue;
            minorData[minorIndex++] = getElement(i, j);
        }
    }
    ygzMatrix<T> minorMatrix(nRows - 1, nCols - 1, minorData);
    delete[] minorData;
    return minorMatrix;
}

// Calculate the determinant of the matrix
template <class T>
T ygzMatrix<T>::determinant() const
{
    if (!isSquare())
        throw std::invalid_argument("Matrix must be square to calculate the determinant");

    if (nRows == 1)
        return getElement(0, 0);
    if (nRows == 2)
        return getElement(0, 0) * getElement(1, 1) - getElement(0, 1) * getElement(1, 0);

    T det = 0;
    for (size_t i = 0; i < nCols; i++)
    {
        ygzMatrix<T> minor = findMinor(0, i);
        det += (i % 2 == 0 ? 1 : -1) * getElement(0, i) * minor.determinant();
    }
    return det;
}

/* ********************************************************************************************* */
/* 					    			  Transpose Operations										 */
/* ********************************************************************************************* */
// Transpose the matrix in place
template <class T>
bool ygzMatrix<T>::transposeInPlace()
{
    T* newData = new T[nElements];
    for (size_t i = 0; i < nRows; i++)
    {
        for (size_t j = 0; j < nCols; j++)
            newData[j * nRows + i] = getElement(i, j);
    }
    delete[] data;
    data = newData;
    int temp = nRows;
    nRows = nCols;
    nCols = temp;
    return true;
}

// Return the transpose of the matrix
template <class T>
ygzMatrix<T> ygzMatrix<T>::transpose() const
{
    T* newData = new T[nElements];
    for (size_t i = 0; i < nRows; i++)
    {
        for (size_t j = 0; j < nCols; j++)
            newData[j * nRows + i] = getElement(i, j);
    }
    ygzMatrix<T> transposedMatrix(nCols, nRows, newData);
    delete[] newData;
    return transposedMatrix;
}

/* ********************************************************************************************* */
/* 					    			Matrix - Vector Operations									 */
/* ********************************************************************************************* */
// Matrix-vector multiplication
template <class T>
ygzVector<T> operator*(const ygzMatrix<T> &matrix, const ygzVector<T> &vector)
{
    if (matrix.getNumCols() != vector.getNumDims())
        throw std::invalid_argument("Number of columns in the matrix must be equal to the number of dimensions in the vector");

    T* result = new T[matrix.nRows];
    for (size_t i = 0; i < matrix.nRows; i++)
    {
        result[i] = 0;
        for (size_t j = 0; j < matrix.nCols; j++)
            result[i] += matrix.getElement(i, j) * vector.getElement(j);
    }

    ygzVector<T> resultVector(matrix.nRows, result);
    delete[] result;
    return resultVector;
}

// Vector - Row Vector Multiplication
template <class T>
ygzMatrix<T> operator*(const ygzVector<T> &vector, const ygzMatrix<T> &r_vector)
{
    if (r_vector.getNumRows() != 1)
        throw std::invalid_argument("The right matrix must be a row vector");
    
    T* result = new T[vector.getNumDims() * r_vector.getNumCols()];
    for (size_t i = 0; i < vector.getNumDims(); i++)
    {
        for (size_t j = 0; j < r_vector.getNumCols(); j++)
            result[i * r_vector.getNumCols() + j] = vector.getElement(i) * r_vector.getElement(0, j);
    }

    ygzMatrix<T> resultMatrix(vector.getNumDims(), r_vector.getNumCols(), result);
    delete[] result;
    return resultMatrix;
}

/* ********************************************************************************************* */
/* 					    			  Hadamard Product Operations								 */
/* ********************************************************************************************* */
// Hadamard product of two matrices
template <class T>
ygzMatrix<T> ygzMatrix<T>::hadamardProduct(const ygzMatrix<T> &lhs, ygzMatrix<T> &rhs)
{
    if (lhs.nRows != rhs.nRows || lhs.nCols != rhs.nCols)
        throw std::invalid_argument("Matrices must have the same dimensions");

    T* result = new T[lhs.nElements];
    for (size_t i = 0; i < lhs.nElements; i++)
        result[i] = lhs.data[i] * rhs.data[i];

    ygzMatrix<T> resultMatrix(lhs.nRows, lhs.nCols, result);
    delete[] result;
    return resultMatrix;
}

/* ********************************************************************************************* */
/* 					    			  toString()                								 */
/* ********************************************************************************************* */
// Convert the matrix to a string
template <class T>
std::string ygzMatrix<T>::toString() const
{
    std::string result = "[";
    for (size_t i = 0; i < nRows; i++)
    {
        result += "[";
        for (size_t j = 0; j < nCols; j++)
        {
            result += std::to_string(getElement(i, j));
            if (j < nCols - 1)
                result += ", ";
        }
        result += "]";
        if (i < nRows - 1)
            result += ", ";
    }
    result += "]";
    return result;
}

// Convert the matrix into CSV format
template <class T>
std::string ygzMatrix<T>::toCSV() const
{
    std::string result = "";
    for (size_t i = 0; i < nRows; i++)
    {
        for (size_t j = 0; j < nCols; j++)
        {
            result += std::to_string(getElement(i, j));
            if (j < nCols - 1)
                result += ",";
        }
        result += "\n";
    }
    return result;
}

#endif