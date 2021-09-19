### Use the Bayesian clasifer to detect the ground in a greenhouse
### @Dr. Zhaozheng Yin, Stony Brook University, Fall 2021
### housekeeping
import math

import cv2
import numpy as  np
import matplotlib.pyplot as plt

### File directory
datadir = 'data-greenhouse/Tunnel-' #the photos must be in a folder named data-greenhouse with only the name Tunnel-0a where a means
trainingImages = [3, 1]
testingImages = [1, 2]
### Parameters
nDim = 8 #number of bins for the color likelihood distribution. This is too big. Try to have smaller bins such as 8, 16, 32, etc.

### Training process
Pr_x_given_y_equalsTo_1 = np.zeros((nDim, nDim, nDim))  # likelihood for the ground class
Pr_x_given_y_equalsTo_0 = np.zeros((nDim, nDim, nDim))  # likelihood for the non-ground class
N_GroundPixels = 0  # Pr_y_equalsTo_1 = N_GroundPixels/N_totalPixels
N_totalPixels = 0
Pr_y_equalsTo_1 = 0 #probability of ground pixel
Pr_y_equalsTo_0 = 0 #probability of not ground pixel
print("Loading training images. This will take a while")
for iFile in trainingImages:
    ### Load the training image and labeled image regions
    origIm = cv2.imread(datadir + '0' + str(iFile) + '.jpg')
    labels = cv2.imread(datadir + '0' + str(iFile) + '-label.png',cv2.IMREAD_GRAYSCALE)  # label=1 representing the ground class
    labels = cv2.threshold(labels, 127, 1, cv2.THRESH_BINARY)[1]
    ### Visualization input image and its labels
    nrows, ncols = origIm.shape[0], origIm.shape[1]
    # showIm = origIm.copy()
    # showIm[labels == 1] = 255;
    ### Be sure to convert the color space of the image from BGR (Opencv) to RGB (Matplotlib) before you show a color image read from OpenCV
    # plt.figure(figsize=(18, 6))
    # plt.subplot(1, 3, 1)
    # plt.imshow(cv2.cvtColor(origIm, cv2.COLOR_BGR2RGB))
    # plt.title('Training image')
    # plt.axis("off")
    #
    # plt.subplot(1, 3, 2)
    # plt.imshow(labels, 'gray')
    # plt.title('GT')
    # plt.axis("off")
    #
    # plt.figure(figsize=(18, 6))
    # plt.subplot(1, 3, 3)
    # plt.imshow(cv2.cvtColor(showIm, cv2.COLOR_BGR2RGB))
    # plt.title('GT overlyed on the training image')
    # plt.axis("off")
    #
    # plt.show()

    ### Prior-related codes:
    # Fill in your code here
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            N_totalPixels += 1
            if labels[i][j] == 1:
                N_GroundPixels += 1

    ### Likelihood-related codes:
    #Loops through an image and checks each pixel to the label which determines if it is a ground pixel if so increment Pr_x_given_y_equalsTo_1[r][g][b]
    for i in range(len(origIm)):
        for j in range(len(origIm[i])):
            r = round(origIm[i][j][0] / 32 - 1)
            g = round(origIm[i][j][1] / 32 - 1)
            b = round(origIm[i][j][2] / 32 - 1)
            if labels[i][j] == 1:
                Pr_x_given_y_equalsTo_1[r][g][b] += 1
            else:
                Pr_x_given_y_equalsTo_0[r][g][b] += 1
### Some other codes such as normalizing the likelihood/prior and computing Pr_y_equalsTo_0:
# Fill in your code here
#Normalizing the likelihood
equalsTo_1_count = 0 #counter for equals to 1
equalsTo_0_count = 0 #counter for equals to 0
for i in range(len(Pr_x_given_y_equalsTo_1)):
    for j in range(len(Pr_x_given_y_equalsTo_1[i])):
        for k in range(len(Pr_x_given_y_equalsTo_1[i][j])):
            equalsTo_0_count += Pr_x_given_y_equalsTo_0[i][j][k]
            equalsTo_1_count += Pr_x_given_y_equalsTo_1[i][j][k]
for i in range(len(Pr_x_given_y_equalsTo_1)):
    for j in range(len(Pr_x_given_y_equalsTo_1[i])):
        for k in range(len(Pr_x_given_y_equalsTo_1[i][j])):
            Pr_x_given_y_equalsTo_0[i][j][k] = Pr_x_given_y_equalsTo_0[i][j][k]/equalsTo_0_count
            Pr_x_given_y_equalsTo_1[i][j][k] = Pr_x_given_y_equalsTo_1[i][j][k]/equalsTo_1_count
Pr_y_equalsTo_1 = N_GroundPixels / N_totalPixels
Pr_y_equalsTo_0 = 1 - Pr_y_equalsTo_1

### Testing
print("Training complete, now Loading testing images. This will take a while")
truePositives = 0;
falsePositives = 0;
falseNegatives = 0;
testingImages = [ 2]
for iFile in testingImages:
    ### Load the testing image and ground truth regions
    origIm = cv2.imread(datadir + '0' + str(iFile) + '.jpg')
    gtMask = cv2.imread(datadir + '0' + str(iFile) + '-label.png', cv2.IMREAD_GRAYSCALE)
    gtMask = cv2.threshold(gtMask, 127, 1, cv2.THRESH_BINARY)[1]
    nrows, ncols = origIm.shape[0], origIm.shape[1]
    detectedMask = np.zeros((nrows, ncols))
    ### Define the posteriors
    Pr_y_equalsTo_1_given_x = np.zeros((nrows, ncols))
    Pr_y_equalsTo_0_given_x = np.zeros((nrows, ncols))

    ### Codes to infer the posterior:
    #ignores the Pr(x) as it may be in the formula but is not necessary in the algebra so we exclude it to save on resources
    for i in range(len(origIm)):
        for j in range(len(origIm[i])):
            r = round(origIm[i][j][0] / 32 - 1)
            g = round(origIm[i][j][1] / 32 - 1)
            b = round(origIm[i][j][2] / 32 - 1)
            Pr_y_equalsTo_1_given_x[i][j] = Pr_x_given_y_equalsTo_1[r][g][b] * Pr_y_equalsTo_1
            Pr_y_equalsTo_0_given_x[i][j] = Pr_x_given_y_equalsTo_0[r][g][b] * Pr_y_equalsTo_0
    ### Codes to obtain the final classification result (detectedMask):
    # Pr_y_equalsTo_1_given_x[i][j] >= Pr_y_equalsTo_0_given_x[i][j] then pixel i,j must be a ground pixel
    for i in range(len(detectedMask)):
        for j in range(len(detectedMask[i])):
            if Pr_y_equalsTo_1_given_x[i][j] >= Pr_y_equalsTo_0_given_x[i][j]:
                detectedMask[i][j] = 1
                detectedMask[i][j] = 1
                detectedMask[i][j] = 1
            else:
                detectedMask[i][j] = 0
                detectedMask[i][j] = 0
                detectedMask[i][j] = 0

    ### Codes to calculate the TP, FP, FN:
    # A simple counting loop that detects where ground pixels are detected, truths, or both
    for i in range(len(detectedMask)):
        for j in range(len(detectedMask[i])):
            if detectedMask[i][j] == 1 and gtMask[i][j] == 1:
                truePositives += 1
            elif detectedMask[i][j] == 1 and gtMask[i][j] == 0:
                falsePositives += 1
            elif detectedMask[i][j] == 0 and gtMask[i][j] == 1:
                falseNegatives += 1

    ### Visualize the classification results
    showIm = origIm.copy()
    showIm[detectedMask==1] = 255

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(origIm, cv2.COLOR_BGR2RGB))
    plt.title('testing image')
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(detectedMask, 'gray')
    plt.title('detected mask')
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(showIm, cv2.COLOR_BGR2RGB))
    plt.title('detected mask overlyed on the testing image')
    plt.axis("off")
    plt.show()

### Codes to calculate the precision, recall, and fscore:
precision = truePositives / (truePositives + falsePositives)
recall = truePositives / (truePositives + falseNegatives)
fscore = (2 * precision * recall) / (precision + recall)
print("The precision, recall, and fscore of the algorithm is:")
print("Precision: " + precision.__str__())
print("Recall: " + recall.__str__())
print("Fscore: " + fscore.__str__())

