/*!
 *  \file main.c
 *  \author LEFLOCH Thomas <leflochtho@eisti.eu>
 *  \version 0.1
 *  \date Mar 24 Janvier 2023 - 14:06:38
 *
 *  \brief Wirow-Woff method for neural networks
 *
 *
 */

// Including library headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
/*!
 \def EXIT_SUCCESS
 \brief Exit code confirming program proper execution
*/
#define EXIT_SUCCESS 0
/*!
 \def RW_ERROR
 \brief Exit code stating that an error occured
*/
#define RW_ERROR -1

/*!
 *  \def FEATURES_AMOUNT
 *  \brief Number of features
 */
#define FEATURES_AMOUNT 48

/*!
 *  \def NOISE_LOOPS
 *  \brief Number of loops to compute noise value
 */
#define NOISE_LOOPS 50

/*!
 *  \def MAX_NOISE_AMOUNT
 *  \brief Maximum noise amount
 */
#define MAX_NOISE_AMOUNT 100

/*!
 *  \def NUMBER_OF_CLASSES
 *  \brief Number of classes
 */
#define NUMBER_OF_CLASSES 10

const char *arrayOfClasses[] = {"./assets/zero.txt", "./assets/un.txt", "./assets/deux.txt", "./assets/trois.txt",
                                "./assets/quatre.txt", "./assets/cinq.txt", "./assets/six.txt", "./assets/sept.txt",
                                "./assets/huit.txt", "./assets/neuf.txt"};
int featureArray[FEATURES_AMOUNT];
int yKd[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
int flatIndex = 0;
float teta = 0.0;
float epsilon = 0.01;
float precision = 0.001;
float weightValuesZero[FEATURES_AMOUNT];
float weightValuesOne[FEATURES_AMOUNT];
float weightValuesTwo[FEATURES_AMOUNT];
float weightValuesThree[FEATURES_AMOUNT];
float weightValuesFour[FEATURES_AMOUNT];
float weightValuesFive[FEATURES_AMOUNT];
float weightValuesSix[FEATURES_AMOUNT];
float weightValuesSeven[FEATURES_AMOUNT];
float weightValuesEight[FEATURES_AMOUNT];
float weightValuesNine[FEATURES_AMOUNT];

float *weightValues[] = {
    weightValuesZero,
    weightValuesOne,
    weightValuesTwo,
    weightValuesThree,
    weightValuesFour,
    weightValuesFive,
    weightValuesSix,
    weightValuesSeven,
    weightValuesEight,
    weightValuesNine};

/*!
 *  \fn void fillWeight()
 *  \author LEFLOCH Thomas <leflochtho@eisti.eu>
 *  \version 0.1
 *  \date Mer 11 Janvier 2023 - 14:06:10
 *  \brief fills weight array with random values
 */
void fillWeight()
{
    for (int i = 0; i < NUMBER_OF_CLASSES; i++)
    {
        for (int j = 0; j < FEATURES_AMOUNT; j++)
        {
            weightValues[i][j] = ((float)rand() / (float)RAND_MAX) / (float)FEATURES_AMOUNT;
        }
    }
}

/*!
 *  \fn void softmax()
 *  \author LEFLOCH Thomas <leflochtho@eisti.eu>
 *  \version 0.1
 *  \date Mer 11 Janvier 2023 - 14:06:10
 *  \brief computes the softmax fonction
 *  \param float* input : input array
 *  \param size_t input_len : size of the input array
 */
void softmax(float *input, size_t input_len)
{

    float m = -INFINITY;
    for (size_t i = 0; i < input_len; i++)
    {
        if (input[i] > m)
        {
            m = input[i];
        }
    }

    float sum = 0.0;
    for (size_t i = 0; i < input_len; i++)
    {
        sum += expf(input[i] - m);
    }

    float offset = m + logf(sum);
    for (size_t i = 0; i < input_len; i++)
    {
        input[i] = expf(input[i] - offset);
    }
}

/*!
 *  \fn void fillFeaturesArray(FILE* myFile)
 *  \author LEFLOCH Thomas <leflochtho@eisti.eu>
 *  \version 0.1
 *  \date Mer 11 Janvier 2023 - 14:28:03
 *  \brief fills feature array with file values
 *  \param FILE* myFile
 */
void fillFeaturesArray(FILE *myFile)
{
    char c;
    int i = 0;
    while ((c = getc(myFile)) != EOF)
    {
        if (c == '*')
        {
            featureArray[i] = 1;
            i++;
        }
        else if (c == '.')
        {
            featureArray[i] = 0;
            i++;
        }
        else if (c >= 48 && c <= 57)
        {
            yKd[(c - '0')] = 1;
        }
    }
}

/*!
 *  \fn void showArray(int* myArray, int size)
 *  \author LEFLOCH Thomas <leflochtho@eisti.eu>
 *  \version 0.1
 *  \date Mer 11 Janvier 2023 - 14:36:59
 *  \brief prints
 *  \param int* myArray : array
 *  \param int size : size of the array
 *  \remarks
 */
void showArray(int *myArray, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%d ", myArray[i]);
    }
}

/*!
 *  \fn void showArrayF(float* myArray, int size)
 *  \author LEFLOCH Thomas <leflochtho@eisti.eu>
 *  \version 0.1
 *  \date Mer 11 Janvier 2023 - 14:36:59
 *  \brief prints
 *  \param float* myArray : array
 *  \param int size : size of the array
 *  \remarks
 */
void showArrayF(float *myArray, int size)
{
    printf("[");
    for (int i = 1; i < size; i++)
    {
        printf("%f,", myArray[i]);
    }
    printf("]");
}

/*!
 *  \fn int enabledClassNeuron(float* localYkd)
 *  \author LEFLOCH Thomas <leflochtho@eisti.eu>
 *  \version 0.1
 *  \date Ven 03 Mars 2023 - 16:47:43
 *  \brief returns the index of the enabled neuron
 *  \param int* localYkd : array of classes
 *  \return int : index of the enabled neuron
 */
int enabledClassNeuron(float *localYkd)
{
    float max = -INFINITY;
    int maxIndex = -1;
    for (int i = 0; i < NUMBER_OF_CLASSES; i++)
    {
        if (localYkd[i] > max)
        {
            max = localYkd[i];
            maxIndex=i;
        }
    }
    return (maxIndex);
}

/*!
 *  \fn float propagate()
 *  \author LEFLOCH Thomas <leflochtho@eisti.eu>
 *  \version 0.1
 *  \date Mer 11 Janvier 2023 - 14:57:56
 *  \brief propagate correction over neurons
 *  \return potential
 */
float propagate(int inputNeuron)
{
    float potential = 0.0;
    for (int i = 0; i < FEATURES_AMOUNT; i++)
    {
        potential += (weightValues[inputNeuron][i] * (float)featureArray[i]);
    }
    return (potential);
}

/*!
 *  \fn void learn(float error)
 *  \author LEFLOCH Thomas <leflochtho@eisti.eu>
 *  \version 0.1
 *  \date Mer 11 Janvier 2023 - 15:14:37
 *  \brief learn new weight values
 *  \param float error : error rate
 *  \remarks
 */
void learn(float error, int flatIndex)
{
    for (int i = 0; i < FEATURES_AMOUNT; i++)
    {
        weightValues[flatIndex][i] = weightValues[flatIndex][i] + epsilon * (float)error * (float)featureArray[i];
    }
}

/*!
 *  \fn float verify(const char* file)
 *  \author LEFLOCH Thomas <leflochtho@eisti.eu>
 *  \version 0.1
 *  \date Mer 11 Janvier 2023 - 15:26:03
 *  \brief verify if the network is strong enough
 *  \param const char* file : file to verify
 * \param int originalValue : original value of the file
 *  \return err : error rate
 */
float verify(const char *file, int originalValue)
{
    float error = 0.0;
    FILE *myFile = fopen(file, "r");
    fillFeaturesArray(myFile);
    fclose(myFile);
    error = (float)yKd[originalValue] - propagate(originalValue);
    return (error);
}

/*!
 *  \fn void noisy(float percentage)
 *  \author LEFLOCH Thomas <leflochtho@eisti.eu>
 *  \version 0.1
 *  \date Mar 24 Janvier 2023 - 10:13:20
 *  \brief Add some noise to the image
 *  \param float percentage : percentage of noise
 *  \remarks
 */
void noisy(float percentage)
{
    int amountOfPixels = percentage * FEATURES_AMOUNT / 100;
    for (int i = 0; i < amountOfPixels; i++)
    {
        int pixel = (int)rand() % FEATURES_AMOUNT;
        featureArray[pixel] = !featureArray[pixel];
    }
}

// totalPropagate
/*!
 *  \fn void totalPropagate()
 *  \author LEFLOCH Thomas <leflochtho@eisti.eu>
 *  \version 0.1
 *  \date Mar 24 Janvier 2023 - 10:13:20
 *  \brief propagate correction over neurons
 *  \return potential
 */
void totalPropagate(float *outputArray)
{
    for (int i = 0; i < NUMBER_OF_CLASSES; i++)
    {
        outputArray[i] = propagate(i);
        // printf("Valeur de propagation pour %d = %f\n", i, outputArray[i]);
    }
}

/*!
 *  \fn void testMyNet(const char *inputFile, const char *outputFile)
 *  \author LEFLOCH Thomas <leflochtho@eisti.eu>
 *  \version 0.1
 *  \date Mar 24 Janvier 2023 - 11:14:17
 *  \brief test my neural net for a certain input
 *  \param const char* inputFile : inputFile
 *  \param const char* outputFile : outputFile
 *  \param int originalValue : original value of the input
 *  \remarks
 */
void testMyNet(const char *inputFile, const char *outputFile, int originalValue)
{
    printf("Tests for %s\n", inputFile);
    FILE *outputfd = NULL;
    int numberOfErrors;
    float error = 0;
    outputfd = fopen(outputFile, "w");
    for (int i = 1; i <= MAX_NOISE_AMOUNT; i++)
    {
        numberOfErrors = 0;
        for (int j = 0; j < NOISE_LOOPS; j++)
        {
            FILE *myFile = fopen(inputFile, "r");
            fillFeaturesArray(myFile);
            fclose(myFile);
            noisy(i);
            float outputArray[10];
            totalPropagate(outputArray);
            softmax(outputArray, NUMBER_OF_CLASSES);
            // printf("Array for number %d and noise %d\n", originalValue, i);
            // showArrayF(outputArray, NUMBER_OF_CLASSES);
            int outputValue = enabledClassNeuron(outputArray);
            //printf("Output value : %d\tExpected value : %d\tError : %d\n\r", outputValue, originalValue, (originalValue == outputValue ? 0 : 1));
            error = (originalValue == outputValue) ? 0 : 1;
            numberOfErrors += fabs(error);
        }
        fprintf(outputfd, "%d\t%f\r\n", i, ((((float)NOISE_LOOPS - (float)numberOfErrors) * 100) / (float)NOISE_LOOPS));
    }
    fclose(outputfd);
}

/*!
 *  \fn FILE* pickAClass()
 *  \author LEFLOCH Thomas <leflochtho@eisti.eu>
 *  \version 0.1
 *  \date Mer 11 Janvier 2023 - 14:17:06
 *  \brief pick a random class from the assets file and opens the file
 *  \return fd of the file containing the class
 */
const char *pickAClass()
{
    int zeroToNine = rand() % 10;
    flatIndex = zeroToNine;
    return arrayOfClasses[flatIndex];
}

/*!
 *  \fn int main (int argc, char** argv)
 *  \author LEFLOCH Thomas <leflochtho@eisti.eu>
 *  \version 0.1
 *  \date Mar 24 Janvier 2023 - 11:37:50
 *  \brief Main program
 *  \param argc : Number of params at runtime
 *  \param argv : Parameters
 *  \return 0 if everything was fine
 */
int main(int argc, char const *argv[])
{
    float error = 0.0;
    int iterations = 0;
    float errorRate = 0.0;
    srand(time(NULL));
    fillWeight();
    FILE *outputfd = fopen("neuralNetIterations.dat", "w");
    do
    {
        FILE *myFile = fopen(pickAClass(), "r");
        fillFeaturesArray(myFile);
        fclose(myFile);
        error = (float)yKd[flatIndex] - propagate(flatIndex);
        learn(error, flatIndex);
        errorRate = ((float)fabs(verify("./assets/un.txt", 1)) + (float)fabs(verify("./assets/zero.txt", 0)) +
                     (float)fabs(verify("./assets/deux.txt", 2)) + (float)fabs(verify("./assets/trois.txt", 3)) + (float)fabs(verify("./assets/quatre.txt", 4)) +
                     (float)fabs(verify("./assets/cinq.txt", 5)) + (float)fabs(verify("./assets/six.txt", 6)) + (float)fabs(verify("./assets/sept.txt", 7)) +
                     (float)fabs(verify("./assets/huit.txt", 8)) + (float)fabs(verify("./assets/neuf.txt", 9))) /
                    10;
        fprintf(outputfd, "%d\t%f\r\n", iterations, errorRate);
        iterations++;
    } while (errorRate > precision);
    fclose(outputfd);
    testMyNet("./assets/zero.txt", "./errorWithNoiseForZero.dat", 0);
    testMyNet("./assets/un.txt", "errorWithNoiseForOne.dat", 1);
    testMyNet("./assets/deux.txt", "errorWithNoiseForTwo.dat", 2);
    testMyNet("./assets/trois.txt", "errorWithNoiseForThree.dat", 3);
    testMyNet("./assets/quatre.txt", "errorWithNoiseForFour.dat", 4);
    testMyNet("./assets/cinq.txt", "errorWithNoiseForFive.dat", 5);
    testMyNet("./assets/six.txt", "errorWithNoiseForSix.dat", 6);
    testMyNet("./assets/sept.txt", "errorWithNoiseForSeven.dat", 7);
    testMyNet("./assets/huit.txt", "errorWithNoiseForEight.dat", 8);
    testMyNet("./assets/neuf.txt", "errorWithNoiseForNine.dat", 9);
    return 0;
}