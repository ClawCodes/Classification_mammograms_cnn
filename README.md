# Classification of Mammogram images: Benign or Malignant
This data was taken from the Curated Breast Imaging Subset of the Digital Database for Screening Mammography (CBIS-DDSM). It was put together in an effort to create a large dataset of mammograms as most studies have a very limited amount of images. This dataset is a pool of images from 6775 studies with 6671 patients, resulting in 10239 total images. The data set contains four different image types: Craniocaudal view (CC), Mediolateral oblique view (MLO), a ROI (region of interest) cropped image of a potentially problematic mass (mass or calcification), and an image that highlights the pixels where the mass is located in the CC view. The data can be downloaded together, but the curators set up a train-test split in the following structure:

| Mass or Calcification|
|---|
|Train Full Mammogram Images|
|Train ROI and Cropped Images|
|Test Full Mammogram Images|
|Test ROI and Cropped Images|

Each of these files are accompanied with a csv specific to each Image file. The csvs important information, such as patient ID, breast density score, image view, abnormality id, mass shape, and most notably, Pathology (Benign or Malignant).


Please note that these images were plotted with matplotlib's default colormap and they are actually grayscale.

![](https://github.com/Clawton92/Classification_mammograms_cnn/blob/master/proposal/full_cc_mlo_roi.png)

## Goal:
In my prior work experience, I worked on a project with a goal of assessing the link between breast density and an increased risk in developing breast cancer. The current literature suggests that dense breasts can be 6 times more likely to develop cancer. So, in this projects I hope to classify mammograms as benign or malignant. To do this, I decided to create a simple Convolution Neural Network (CNN). For this model, I will only be using the CC view as that is the view my prior project was using and it would capture over all breast density. Additionally, to keep computation time down, I will only be using the Mass images not the Calcification images.


Basic Convolutional Neural Net architecture with a binary output. [Source](https://www.researchgate.net/figure/Illustration-of-Convolutional-Neural-Network-CNN-Architecture_fig3_322477802)

![](https://www.researchgate.net/profile/Qianzhou_Du2/publication/322477802/figure/fig3/AS:582461356511232@1515881017676/Illustration-of-Convolutional-Neural-Network-CNN-Architecture.png)

The Neural net is based off of a paper that also classified mammograms as Benign or Malignant [https://arxiv.org/pdf/1612.00542.pdf](https://arxiv.org/pdf/1612.00542.pdf).

# EDA

Exploring the Data across only the Mass data set

There are 1592 total images in the Mass set, this includes CC and MLO.

Count| |
------|---|
Benign| 852
Malignant| 740

![](https://github.com/Clawton92/Classification_mammograms_cnn/blob/master/graphs_images/path_image_view.png)

![](https://github.com/Clawton92/Classification_mammograms_cnn/blob/master/graphs_images/breast_density_pathology.png)

Although the above plot does not show a big difference in pathology between density values, I decided to move forward due to my domain knowledge, and the fact that the density values range from 1-4. Currently there is no consensus on how to measure breast density, so these values are a subjective number given to each image by a radiologist.


## The Data
All images are in the standard medical imaging format, DICOM (Digital Imaging and Communications in Medicine). DICOM images are widely adopted across health systems. These images contain an image as well as metadata. This meta data can hold important information such as patient ID, image type, diagnosis, etc. Each DICOM file ranges from 12-25 MB. Additionally, the DICOM pixel arrays are not consistent in size, but do hover around 5900x3200 for the full mammogram images.

Structure of the each file:

<p align="center"> <img src=https://github.com/Clawton92/Classification_mammograms_cnn/blob/master/data_challenges/Screen%20Shot%202018-11-15%20at%206.16.50%20PM.png width = "60%">

Note that each image is 3 directories deep from the parent and primarily named the same. In order to identify what these images actually contain, I used a python library called pydicom [https://pydicom.github.io/](https://pydicom.github.io/). When reading in an image with pydicom you can access the meta data and image as attributes.

![](https://github.com/Clawton92/Classification_mammograms_cnn/blob/master/data_challenges/dicom_meta_data.png)

###### Notable attributes:

Patient ID: Same name as the second directory for each image.

Patient Orientation: image view (CC, MLO)

Pixel array: allows my to grab the array into a numpy array using pydicom. Each array is uint16 data type, the pixel values range from 0-65535.

## Organizing and structuring the data  
As I stated above, to build the simple CNN, I will only be focusing on the Mass CC view. To capture this, I downloaded the train and test files that contain the full mammogram images. The image file was to large for my local machine, so they were downloaded to an external hard drive.

To use flow_from_directory in Keras, I will need the CC views from the test and train folders split into my own train, test, and hold out sets along with the pathology (diagnosis) of each image. Initially I thought the pathology would be accessible via the meta data, however, I found pathology was only available via the provided csvs. My first approach to giving each image its respective diagnosis was to walk through the directories, access the meta data and put it into a dataframe. From there I would map the patient IDs in the provided csv to the patient IDs I abstracted. However, when I did this, I found the dataframes to be of different lengths. It was difficult to find the exact reason as to why this was the case, but I speculate that it is due to the abnormality id column. In most cases there is a single abnormality per image, but in some cases there are more. Each patient seemed to have two images each, one CC and one MLO, and they can have multiple abnormalities for a single image. My next approach was to get the unique patients where abnormality ID == 1. But, that didn't work either, it seemed there are some images that don't start with and abnormality ID == 1, however this could represent duplicate images. After attempting several other methods, I finally found a string of numbers (patient UID) in the image path column in the csv that corresponded to a field in the meta data of each DICOM image. With this information, I walked through each directory with os.walk, matched the ID, view, and UID. I saved each image as a grayscale png and sent the images into a train and test folder and into their respective pathology folders, Benign or Malignant. I then further divided them into train, test, and hold out folders using a 70%(train), 20%(test), and 10%(hold) split.

Original csv structure:
![](https://github.com/Clawton92/Classification_mammograms_cnn/blob/master/data_challenges/csv_sample_mass.png)

![](https://github.com/Clawton92/Classification_mammograms_cnn/blob/master/data_challenges/csv_sample_mass_multiple_abnorm.png)

Final Mass CC images:

| | Benign   |Malignant|
|---|---|---|
 | Train|  277 | 243|
 |Test | 79  | 69|
 | Hold out | 40  | 34|
 |Total | 396 | 346 |

742 total images

# More data struggles
Again, this data was initially to large to put on my local machine, so I conducted all the file transfer and conversion on the hard drive. When it came time to model, my data stream would break. After some sleuthing, I found several corrupted images that were created from the file transfer. I was able to clear some space to get the images onto my local machine and from there I complete the transfer again. This time, it seemed that all images were intact.


# Approach
Choosing the right image size.

##### (50x50)
![](https://github.com/Clawton92/Classification_mammograms_cnn/blob/master/graphs_images/img_50_compare.png)

##### (150x150)
![](https://github.com/Clawton92/Classification_mammograms_cnn/blob/master/graphs_images/img_150_compare.png)

I chose 150x150 for the image input size into my model as it was one of the smaller scales that I felt the density of a breast could potentially be measured.


I based my initial architecture for the CNN on the paper mentioned above. This is similar the my final model, shown below, but lacks a few layers and hyperparameters. This consisted of 3 convolutional blocks in the order conv2d with a 3x3 kernel - activation(relu) - maxpooling2 (2x2) these blocks were then followed by flatten layer, a dense layer with 128 neurons, activation with relu, a final dense layer of one neuron with the last activation function being sigmoid for binary classification, the loss function used is binary crossentropy.

# Training the CNN

Training my first model showed no learning through 7 epochs with a batch size of 60. The model remained around an accuracy of 0.55 and a loss of roughly 0.69. Note the y axis on both plots, these are essentially straight lines.

![](https://github.com/Clawton92/Classification_mammograms_cnn/blob/master/graphs_images/basic_model_accuracy.png)

![](https://github.com/Clawton92/Classification_mammograms_cnn/blob/master/graphs_images/basic_model_loss.png)

From here I tested various hyperparameters: batch_size, activation functions, kernel initializers, number of epochs, optimizers, learning rates, image augmentation (rotation, shifting, flipping), dropout, etc. However, no combination seemed to improve the metrics.

Unfortunately, the processing required for these models is large and my computer was having a hard time keeping up. In more than half of my runs, the model would get half way through one of the first 4 epochs and then break. I would like to explore more combinations using cloud computing.

The final architecture follows:


![](https://github.com/Clawton92/Classification_mammograms_cnn/blob/master/graphs_images/final_model.png)

The kernel and pool size remained the same, (3x3) and (2x2) respectively. A dense and activation layer was added as well as glorot normal kernel initialization in each conv2d and dense layer. This was recommended from the paper stated above. Training this final model showed the following. Again, note the y axis. There is no significant improvement in accuracy or loss.

![](https://github.com/Clawton92/Classification_mammograms_cnn/blob/master/graphs_images/final_model_train_val_accuracy.png)

![](https://github.com/Clawton92/Classification_mammograms_cnn/blob/master/graphs_images/final_model_train_val_loss.png)

# Visualizing convolution feature maps
Since the model did not seem to train well, I wanted to see what the feature maps held. Unfortunately, there were no easily interpretable patters. I would expect a feature map to find the edge of a breast, a general mass shape, or possibly a pattern in breast density, however, these feature maps do not display much interpretability of the model.

2nd Convolution layer
![](https://github.com/Clawton92/Classification_mammograms_cnn/blob/master/graphs_images/stitched_filters_4x4_conv2d_2.png)

3rd Convolution layer
![](https://github.com/Clawton92/Classification_mammograms_cnn/blob/master/graphs_images/stitched_filters_4x4_conv2d_3.png)

# Results

The final results of predicting on the hold out set.

|   |  Loss | Accuracy  |   
|---|---|---|
|Basic model | 0.69  | 0.54|
|Final model| 0.68  | 0.54|

All predictions probabilities in the basic model were within 0.005 of 0.47 and the final model predications were all around 0.46. So, depending on the threshold used, all images would be classified as Malignant if > 0.46 and Benign if < 0.46.


# Discussion
The final predictions on the hold out set suggest that the network is not learning and is taking a guess on each image. Beyond this, it is concerning that every change in the architecture and hyperparameters did not yield different results. My thought as to why this is occurring is due to sacrificing the resolution of the input images. Recall that the shape of the original images are around 5900x3200. Downsizing to 150x150 sacrifices a lot of important information. I believe that training a network with larger images and with greater computational power would allow the network to begin to learn patters.

# Future directions
Train the network using higher resolution images. Potentially crop the images so there is not as much empty space.

Missed EDA.

![](https://github.com/Clawton92/Classification_mammograms_cnn/blob/master/graphs_images/path_common_mass.png)

It would be interesting to train the network on the provided cropped images as it seems that learning the shapes of problematic masses could potentially lead to a better rate of classification.

After tuning the simple CNN, I would like to move on to Transfer learning. In the paper provided the researchers used GoogLeNet (inception v1). I would like to move to transfer learning with inception v3 in Keras.
