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

int featureArray[FEATURES_AMOUNT];
int yKd;
float teta = 0.0;
float epsilon = 0.01;
float precision = 0.001;
float weightValues[FEATURES_AMOUNT];

/*!
 *  \fn void fillWeight()
 *  \author LEFLOCH Thomas <leflochtho@eisti.eu>
 *  \version 0.1
 *  \date Mer 11 Janvier 2023 - 14:06:10
 *  \brief fills weight array with random values
 */
void fillWeight()
{
    for (int i = 0; i < FEATURES_AMOUNT; i++)
    {
        weightValues[i] = ((float)rand() / (float)RAND_MAX) / (float)FEATURES_AMOUNT;
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
            yKd = (c - '0');
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
    for (int i = 0; i < size; i++)
    {
        printf("%f ", myArray[i]);
    }
}

/*!
 *  \fn float propagate()
 *  \author LEFLOCH Thomas <leflochtho@eisti.eu>
 *  \version 0.1
 *  \date Mer 11 Janvier 2023 - 14:57:56
 *  \brief propagate correction over neurons
 *  \return potential
 */
float propagate()
{
    float potential = 0.0;
    for (int i = 0; i < FEATURES_AMOUNT; i++)
    {
        potential += (weightValues[i] * (float)featureArray[i]);
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
void learn(float error)
{
    for (int i = 0; i < FEATURES_AMOUNT; i++)
    {
        weightValues[i] = weightValues[i] + epsilon * (float)error * (float)featureArray[i];
    }
}

/*!
 *  \fn float verify(const char* file)
 *  \author LEFLOCH Thomas <leflochtho@eisti.eu>
 *  \version 0.1
 *  \date Mer 11 Janvier 2023 - 15:26:03
 *  \brief verify if the network is strong enough
 *  \param const char* file : file to verify
 *  \return err : error rate
 */
float verify(const char *file)
{
    float error = 0.0;
    FILE *myFile = fopen(file, "r");
    fillFeaturesArray(myFile);
    fclose(myFile);
    error = (float)yKd - propagate();
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

/*!
 *  \fn void testMyNet(const char *inputFile, const char *outputFile)
 *  \author LEFLOCH Thomas <leflochtho@eisti.eu>
 *  \version 0.1
 *  \date Mar 24 Janvier 2023 - 11:14:17
 *  \brief test my neural net for a certain input
 *  \param const char* inputFile : inputFile
 *  \param const char* outputFile : outputFile
 *  \remarks
 */
void testMyNet(const char *inputFile, const char *outputFile)
{
    printf("Tests for %s\n", inputFile);
    FILE *outputfd;
    int numberOfErrors;
    float error;
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
            float outputValue = fabs(propagate()) >= 0.5 ? 1 : 0;
            error = (float)yKd - outputValue;
            numberOfErrors += fabs(error);
            // printf("ITERATION %d || NOISE %d %% || INPUT %d // OUTPUT %f || ERRORATE %f\r\n", j, i, yKd, outputValue, error);
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
    int oneOrZero = rand() & 1;
    return ((oneOrZero) ? "./assets/un.txt" : "./assets/zero.txt");
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
        error = (float)yKd - propagate();
        learn(error);
        errorRate = ((float)fabs(verify("./assets/un.txt")) + (float)fabs(verify("./assets/zero.txt")))/2;
        printf("ErrorRate : %f\r\n", errorRate);
        fprintf(outputfd, "%d\t%f\r\n", iterations, errorRate);
        iterations++;
    } while (errorRate > precision);
    testMyNet("./assets/zero.txt", "errorWithNoiseForZero.dat");
    testMyNet("./assets/un.txt", "errorWithNoiseForOne.dat");
    fclose(outputfd);
    return 0;
}