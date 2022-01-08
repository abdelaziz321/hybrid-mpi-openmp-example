#include <iostream>
#include <fstream>
#include <sstream>  
#include <omp.h>
#include <string>
#include <vector>
#include <mpi.h>
#include <math.h>

using namespace std;

int NUMBER_OF_THREADS = 4;

int SUM_VALUE_TAG = 1;
int AVERAGE_VALUE_TAG = 2;
int MAX_VALUE_TAG = 3;
int MIN_VALUE_TAG = 4;

vector<vector<double>> convertDatasetFileToMatrix(string fileName);

double findSum(vector<double> vec);
double findAverage(vector<double> vec);
double findMax(vector<double> vec);
double findMin(vector<double> vec);

int main(int argc, char *argv[])
{
    /**
     * vector[0] => vector of the `age` column in the `heartdata.csv` file
     * vector[1] => vector of the `trestbps` column in the `heartdata.csv` file
     * vector[2] => vector of the `chol` column in the `heartdata.csv` file
     * vector[3] => vector of the `thalact` column in the `heartdata.csv` file
    */
    vector<vector<double>> datasetMatrix = convertDatasetFileToMatrix("heartdata.csv");

    // configure openMP
    omp_set_num_threads(NUMBER_OF_THREADS);

    // initialize mpi => multiple processes
    int processRank;
    int numberOfProcesses;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);

    // each process will work on a specific column
    for (int i = 1; i < datasetMatrix.size() + 1; i++) {
        if (processRank != i) continue;

        // send the max of the column to the processRank '0'
        double calculatedColumnMax = findMax(datasetMatrix[i - 1]);
        MPI_Send(&calculatedColumnMax, 1, MPI_DOUBLE, 0, MAX_VALUE_TAG, MPI_COMM_WORLD);

        // send the min of the column to the processRank '0'
        double calculatedColumnMin = findMin(datasetMatrix[i - 1]);
        MPI_Send(&calculatedColumnMin, 1, MPI_DOUBLE, 0, MIN_VALUE_TAG, MPI_COMM_WORLD);

        // send the min of the column to the processRank '0'
        double calculatedColumnSum = findSum(datasetMatrix[i - 1]);
        MPI_Send(&calculatedColumnSum, 1, MPI_DOUBLE, 0, SUM_VALUE_TAG, MPI_COMM_WORLD);

        // send the min of the column to the processRank '0'
        double calculatedColumnAverage = findAverage(datasetMatrix[i - 1]);
        MPI_Send(&calculatedColumnAverage, 1, MPI_DOUBLE, 0, AVERAGE_VALUE_TAG, MPI_COMM_WORLD);
    }

    // should we wait for all processes or not?
    // MPI_Barrier(MPI_COMM_WORLD);

    // processRank 0 will receive from the other processRanks '1, 2, 3, 4'
    if (processRank == 0)
    {
        vector<double> columnsMaxValues;
        vector<double> columnsMinValues;
        vector<double> columnsSumValues;
        vector<double> columnsAverageValues;

        for (int i = 1; i < datasetMatrix.size() + 1; i++) {
            // receiveing the max value from the other processess
            double receivedColumnMax;
            MPI_Recv(&receivedColumnMax, 1, MPI_DOUBLE, i, MAX_VALUE_TAG, MPI_COMM_WORLD, &status);
            columnsMaxValues.push_back(receivedColumnMax);

            // receiveing the min value from the other processess
            double receivedColumnMin;
            MPI_Recv(&receivedColumnMin, 1, MPI_DOUBLE, i, MIN_VALUE_TAG, MPI_COMM_WORLD, &status);
            columnsMinValues.push_back(receivedColumnMin);

            // receiveing the sum value from the other processess
            double receivedColumnSum;
            MPI_Recv(&receivedColumnSum, 1, MPI_DOUBLE, i, SUM_VALUE_TAG, MPI_COMM_WORLD, &status);
            columnsSumValues.push_back(receivedColumnSum);

            // receiveing the sum value from the other processess
            double receivedColumnAverage;
            MPI_Recv(&receivedColumnAverage, 1, MPI_DOUBLE, i, AVERAGE_VALUE_TAG, MPI_COMM_WORLD, &status);
            columnsAverageValues.push_back(receivedColumnAverage);
        }

        cout << "==== max values ====" << endl;
        for (int i = 0; i < columnsMaxValues.size(); i++) {
            cout << "max of column <" << i << ">: " << columnsMaxValues[i] << endl;
        }

        cout << "==== min values ====" << endl;
        for (int i = 0; i < columnsMinValues.size(); i++) {
            cout << "min of column <" << i << ">: " << columnsMinValues[i] << endl;
        }

        cout << "==== sum values ====" << endl;
        for (int i = 0; i < columnsSumValues.size(); i++) {
            cout << "sum of column <" << i << ">: " << columnsSumValues[i] << endl;
        }

        cout << "==== average values ====" << endl;
        for (int i = 0; i < columnsAverageValues.size(); i++) {
            cout << "sum of column <" << i << ">: " << columnsAverageValues[i] << endl;
        }
    }

    MPI_Finalize();
    return 0;
}

vector<vector<double>> convertDatasetFileToMatrix(string fileName)
{
    vector<vector<double>> datasetMatrix;

    string line;
    ifstream datasetFile(fileName);

    // pop first line
    getline(datasetFile, line);

    int columnIndex;
    bool isFirstRaw = true;
    while (getline(datasetFile, line)) {
        // read the next line from the first column
        columnIndex = 0;

        string element;
        stringstream lineStream(line);

        while (getline(lineStream, element, ';')) {
            // add the possible columns in the given file
            if (isFirstRaw) datasetMatrix.push_back(vector<double>());

            datasetMatrix[columnIndex].push_back(
                stod(element)
            );
            columnIndex++;
        }

        if (isFirstRaw) isFirstRaw = false;
    }

    datasetFile.close();

    return datasetMatrix;
}

double findMax(vector<double> vec)
{
    // calculate each thread chunck depending on the NUMBER_OF_THREADS
    int vectorSize = vec.size();
    int threadChunk = ceil(((double) vectorSize) / NUMBER_OF_THREADS);

    double max = vec[0];

    #pragma omp parallel shared(vec, threadChunk, max)
    {
        #pragma omp for schedule(dynamic, threadChunk)
        for (int i = 1; i < vec.size(); i++)
        {
            // for testing: output is a number between 0 and NUMBER_OF_THREADS - 1
            // cout << omp_get_thread_num() << endl;
            // #pragma omp critical
            // {
            if (vec[i] > max) max = vec[i];
            // }
        }
    }

    return max;
}

double findMin(vector<double> vec)
{
    // calculate each thread chunck depending on the NUMBER_OF_THREADS
    int vectorSize = vec.size();
    int threadChunk = ceil(((double) vectorSize) / NUMBER_OF_THREADS);

    double min = vec[0];

    #pragma omp parallel shared(vec, threadChunk, min)
    {
        #pragma omp for schedule(dynamic, threadChunk)
        for (int i = 1; i < vec.size(); i++)
        {
            if (vec[i] < min) min = vec[i];
        }
    }

    return min;
}

double findSum(vector<double> vec)
{
    // calculate each thread chunck depending on the NUMBER_OF_THREADS
    int vectorSize = vec.size();
    int threadChunk = ceil(((double)vectorSize) / NUMBER_OF_THREADS);

    double sum = 0;

    #pragma omp parallel shared(vec, threadChunk, sum)
    {
        #pragma omp for schedule(dynamic, threadChunk)
        for (int i = 1; i < vec.size(); i++)
        {
            sum += vec[i];
        }
    }

    return sum;
}

double findAverage(vector<double> vec)
{
    return findSum(vec) / vec.size();
}