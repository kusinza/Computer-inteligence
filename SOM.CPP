// SOM Implementation
#include<iostream>
#include<fstream>
#include<cstdlib>
#include<cstdio>
#include<cmath>
#include<ctime>

using namespace std;

float **Aloc2DAry(int m, int n);
void Free2DAry(float **Ary2D, int n);

int main() {

    int i, j, NumSamples = 5000, NumFeatures = 784;
    ifstream fin;
    ofstream fout1, fout2, fout3;
    fout1.open("som_results_1.csv");
    fout2.open("som_results_2.csv");
    fout3.open("som_results_3.csv");
    char Line[500], Tmp[20], FName[2000];
    cout << "Enter data filename: ";
    cin >> FName; cin.ignore();

    fin.open(FName);
    if (!fin.good()) { cout << "File not found!\n"; exit(1); }
    float **allData = Aloc2DAry(NumSamples, NumFeatures);

    for (i = 0; i < NumFeatures; i++)
        for (j = 0; j < NumSamples; j++)
            fin >> allData[i][j];
    fin.close();
    // Set the number of rows and columns of the 2D lattice in SOM;
    int rowNum = 10;
    int colNum = 10;

    // Set the number of neurons(equal rowNum * colNum);
    int neuronNum = rowNum * colNum;
    // Define the weight vector of each neuron and randomly initialise them;
    int dim = NumFeatures;
    float **neuronWeight = Aloc2DAry(dim, neuronNum);
    float **oldneuronWeight = Aloc2DAry(dim, neuronNum);
    int *rowIndex = new int[neuronNum];
    int *colIndex = new int[neuronNum];
    // matlab code : neuronWeight = rand(dim, neuronNum);
    // oldneuronWeight = neuronWeight; % for checking the convergence;
    srand((unsigned int) time(0));
    for (i = 0; i < dim; i++)
        for (j = 0; j < neuronNum; j++) {
            neuronWeight[i][j] = float(rand()) / RAND_MAX;
            oldneuronWeight[i][j] = neuronWeight[i][j];
            fout1 << neuronWeight[i][j];
        }

    int epochNumOrd = 100;// 100
    int epochNumCov = 1400;// 100

    float eta_0 = 0.2;
    float eta_t = eta_0;
    // Define the attenuation speed of the learning rate with epoches;
    float tau_2 = 1000;

    float sigma_0 = sqrt(pow(0.5*rowNum, 2.0) + pow(0.5*colNum, 2.0));
    // Define the sigma value in each epoch t;
    float sigma_t = sigma_0;
    // Define the attenuation speed of sigma with epoches;
    float tau_1 = 1000 / log(sigma_0);

    // matlab code [rowIndex, colIndex] = ind2sub([rowNum, colNum], 1:neuronNum);
    // Reshape them to a column vector;
    // matlab code rowIndex = reshape(rowIndex, length(rowIndex), 1);
    // matlab code colIndex = reshape(colIndex, length(colIndex), 1);
    for (i = 0; i < neuronNum; i++)
    {
        rowIndex[i] = i / rowNum;
        colIndex[i] = i % rowNum;
    }

    // matlab code weightDiff = zeros(epochNumOrd + epochNumCov, 1);
    float *weightDiff = new float[epochNumOrd + epochNumCov];

    // disp(['Training SOM starts. Total number of epochs is ', num2str(epochNumOrd + epochNumCov), '...']);
    cout << "Training SOM starts. Total number of epochs is " << epochNumOrd + epochNumCov << "..." << endl;
    /*
    for t = 1:epochNumOrd + epochNumCov
        disp(['Now in epoch ', num2str(t), '...']);
    */

    int *index = new int[NumSamples];
    for (int i = 0; i < NumSamples; i++)
        index[i] = i;  // fill the array in order


    for (i = 0; i < epochNumOrd + epochNumCov; i++) {
        cout << "Now in epoch " << i << "..." << endl;

        // At each epoch, randomly shuffle data first;
        for (int j = 0; j < NumSamples; j++) {
            int r = rand() % NumSamples;  // generate a random position
            int temp = index[j]; index[j] = index[r]; index[r] = temp;
        }

        for (j = 0; j < NumSamples; j++) {
            // Calculate the(squared) Euclidean distance of x from each neuron;
            // Identify the winning neuron by finding the minimum one in sqEucDist;
            // -- [minDist, minInd] = min(sqEucDist);
            int minInd = 0; float minVal = 1e6;
            for (int k = 0; k < neuronNum; k++)
            {
                float diff = 0;

                for (int t = 0; t < dim; t++)
                    diff += pow(neuronWeight[t][k] - allData[t][index[j]], 2);

                if (diff < minVal) {
                    minInd = k; minVal = diff;
                }
            }

            // Obtain the coordinates of the winning neuron in the 2D lattice;
            // -- winCoor = [rowIndex(minInd) colIndex(minInd)];

            // Calculate the(squared) Euclidean distance of this winnung neuron from all neurons;
            // This is to obtain the neighborhood of the winning neuron;
            // -- sqEucDist = sum(([rowIndex colIndex] - repmat(winCoor, neuronNum, 1)). ^ 2, 2);
            // Compute the value of the neighborhood function h();
            // -- h_winNeuron = exp(-sqEucDist / (2 * sigma_t * sigma_t));

            // Updating the weight vectors of all neurons;
            // -- for j = 1:neuronNum
            // --    neuronWeight(:, j) = neuronWeight(:, j) + eta_t * h_winNeuron(j) * (x - neuronWeight(:, j));
            // -- endm.par
            for (int k = 0; k < neuronNum; k++)
                for (int t = 0; t < dim; t++) {
                    float h = pow(rowIndex[k] - rowIndex[minInd], 2) + pow(colIndex[k] - colIndex[minInd], 2);
                    neuronWeight[t][k] = neuronWeight[t][k] + eta_t * exp(-h / (2 * sigma_t * sigma_t))* (allData[t][index[j]] - neuronWeight[t][k]);
                }
        }
        // Compute weight difference;
        // matlab code weightDiff(t) = sum(sum((neuronWeight - oldneuronWeight). ^ 2));
        weightDiff[i] = 0;
        for (int k = 0; k < neuronNum; k++)
            for (int t = 0; t < dim; t++)
                weightDiff[i] += pow(neuronWeight[t][k] - oldneuronWeight[t][k], 2);

        // weightDiff(t) = sqrt(weightDiff(t) / (dim * neuronNum));
        weightDiff[i] = sqrt(weightDiff[i] / (dim * neuronNum));
        cout<<weightDiff[i]<<endl;

        // oldneuronWeight = neuronWeight;
        for (int k = 0; k < neuronNum; k++)
            for (int t = 0; t < dim; t++) {
                oldneuronWeight[t][k] = neuronWeight[t][k];
                if (i == epochNumOrd + epochNumCov - 1)
                    fout3 << neuronWeight[t][k];
                else if (i == round((epochNumOrd + epochNumCov) / 2))
                    fout2 << neuronWeight[t][k];
            }
        // Attenuate the learning rate for the next epoch;
        // Note that the learning rate shall remain above 0.01;
        // matlab code eta_t = max(eta_0 * exp(-t / tau_2), 0.01);
        eta_t = eta_0 * exp(-i / tau_2);
        if (eta_t < 0.01)
            eta_t = 0.01;
	if(i>=epochNumOrd)
		eta_t = 0.01;

        // Attenuate the sigma value of h() for the next epoch;
        //sigma_t = sigma_0 * exp(-t / tau_1);

        sigma_t = sigma_0 * exp(-i / tau_1);
    }
    Free2DAry(allData, NumFeatures);
    return 0;
}

float **Aloc2DAry(int m, int n) {
    //Allocates memory for 2D array
    float **Ary2D = new float*[m];
    if (Ary2D == NULL) { cout << "No memory!\n"; exit(1); }
    for (int i = 0; i<m; i++) {
        Ary2D[i] = new float[n];
        if (Ary2D[i] == NULL) { cout << "No memory!\n"; exit(1); }
    }
    return Ary2D;
}

void Free2DAry(float **Ary2D, int n) {
    //Frees memory in 2D array
    for (int i = 0; i<n; i++)
        delete[] Ary2D[i];
    delete[] Ary2D;
}
