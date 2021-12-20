# Classifying MLB Baseball Pitches with Supervised ML Techiques

## Introduction / Motivation
In this project, my group and I use various ML techiniques to classify baseball pitches. Through this process we hope to discover which models predict pitches best, show which features have the most predictive power in classifying pitches, and familiarize ourselves with MLB Statcast data.

## Dataset
To approach this project we used pitch-by-pitch Statcast data from the 2018 season. The final dataset we used after cleaning is located in the final upload, named michaels_cleaned_mlb_data.csv

###### **Data Cleaning (skip if uninterested)**
Before we had michaels_cleaned_mlb_data.csv, we downloaded a group of 5 datatables representing every play, player, and game from 2018 season. The cleaning steps are listed below:

- merge datatables
- take out pitch-outs, intentional walks, and rare pitches (ex ephus)
- correct datatypes
- downsample data from 3 million pitches to 50k pitches to improve runtime of our code (this decision could lower our test accuracy so if this model were ever to deploy, we would simply rerun our code with all the datapoints)
- remove features that were either redundent, would cause overfitting (name of pitcher/batter), or had obviously little significance


## Visualizations 
###### **Principal Component Analysis**
To prove the the separability of the different pitches, we decided to perform a pricipal component analysis. Below are the 20+ features, reduced to two principal components/features.

![Screen Shot 2021-12-19 at 8 07 35 PM](https://user-images.githubusercontent.com/70180470/146710483-d0f02293-045c-46d6-ba55-99e845395ee9.png)

NOTE: FT - two-seam fastball, SI - sinker, FF - four-seam fastball, CH - Changeup, SL - slider, CU - curveball, FC - cutter, KC - knuckle curve, KN - Knuckle ball

This visualization shows that the models should have success in classifying pitches as they appear separable. We can use this visualization to see which pitches are the most unique and which pitches are most similar. For example, the four-seam fastball looks the most distinguishable while the two-seam fastball and the sinker seem most similar. This visualizations will serve as a good reference point when approaching our task.

###### **Looking at Features/Attributes**
Below are some select visualizations from the in-depth visualizations file in the project folder. The rest can be viewed there. I used violin plots to view the distrubution of an attribute for the various pitches and KDE plots for 2D visualizations of the distrubution density.


![Screen Shot 2021-12-19 at 8 23 21 PM](https://user-images.githubusercontent.com/70180470/146711499-b6e8f87c-b5b3-4086-a700-dbd7df387f30.png)

Above is the distrubution of the speed of various pitches as they cross homeplate. We can see that it will serve as an important feature as the different pitches have different ranges for their speed.

![Screen Shot 2021-12-19 at 8 25 26 PM](https://user-images.githubusercontent.com/70180470/146711636-a8f0c515-12e0-42a4-af78-9606a386a096.png)

Above we can see that there are different spin rates for different pitches. A general trend is that off-speed pitches have less spin rate, which is why the 4-seam fastball has the greatest spin rate.

![Screen Shot 2021-12-19 at 8 27 56 PM](https://user-images.githubusercontent.com/70180470/146711816-fa09f51c-f5b8-48f0-8e7f-38c4d718220f.png)

Statcast defines break length as the total movement of the pitch in both the vertical/horizontal direction. This seems like it will be an important features.

![Screen Shot 2021-12-19 at 8 40 15 PM](https://user-images.githubusercontent.com/70180470/146712718-0cc129fe-18b2-4cf5-8f5a-96c870804f67.png)

This shows the acceleration in the x and z directions will be a helpful feature.

When creating visualizations we would see that some features will help with separability while others are less helpful. Some examples of features that we saw were irrelevant are below:

![Screen Shot 2021-12-19 at 8 37 36 PM](https://user-images.githubusercontent.com/70180470/146712543-796bfc7b-7188-46c0-ad59-1f6753afee74.png)

This visualization shows that release point of a pitch depends more on the pitcher and less on the pitch.

![Screen Shot 2021-12-19 at 8 39 02 PM](https://user-images.githubusercontent.com/70180470/146712634-affff070-f1c9-4c13-9187-978178c1ea75.png)

This visualization shows that location of pitch in the strikezone matters less as well.

Also, in the file, I broke down pitch likelihood per ball/strike count.

Again, these are a subset of the visualizations. The rest are in the in-depth visualizations file.

## Models
Our model choices for the classification project were a Neural Netwok, a random forest, and a decision tree bagging classifier.
We expected the Neural Network and the bagging tree bagging classifier to perform with the higher accuracies while the random forest provided the better interpretability. Details/process of each model can be found in each corresponding file in the project folder, while below will offer a high level discussion of the models.

###### **Neural Network**
TEST ACCURACY: 86.2%
Parameters/Hyperparameters
- Two hidden layers with ReLu activation
- Softmax function output layer
- Stochastic Gradient Descent, alpha = .005
- Loss: Categorical cross entropy

![Screen Shot 2021-12-19 at 8 53 13 PM](https://user-images.githubusercontent.com/70180470/146713678-f73bc791-17e4-4397-bc82-093eae435e79.png)

Perhaps with more data (the full 3 million piches) this model could improve a bit.

###### **Random Forest**
TEST ACCURACY: 85.9%
Parameters/Hyperparameters
- Max depth = 100
- n estimators = 20
- min samples split = 4

![Screen Shot 2021-12-19 at 8 59 01 PM](https://user-images.githubusercontent.com/70180470/146714085-00b24f17-55d9-42ee-860d-fd3586033177.png)

Again, this was our most interpretable model as it was able to show feature importance as shown in the graph below. As we can see, break length, acceleration in the vertical direction, and break in the vertical direction were the biggest indicators of pitch type.

![Screen Shot 2021-12-19 at 9 03 31 PM](https://user-images.githubusercontent.com/70180470/146714371-bd36a26e-1db7-408a-929f-966a8254c9b4.png)

###### **Decision Tree Bagging classifier (located in Nueral Net Notebook)**
TEST ACCURACY: 90.6%
Parameters/Hyperparameters
- Base estimator: decision tree classifier
- n estimators = 10

![Screen Shot 2021-12-19 at 9 17 04 PM](https://user-images.githubusercontent.com/70180470/146715330-9d83f89f-d316-45c1-93a2-26113d0f1351.png)

Overall, our models were very prone to overfitting which is perhaps why bagging, an ensembling method created to combat this, performs the best out of all the models. 

## Results 
The change-up was the hardest pitch to classify over all the models. Also, fastballs (four-seam, four-seam, sinker, ect) and breaking balls (curve, slider, ect) tended to misclassify among themselves when the model made a mistake. Overall, the model performed very well.


## Final Thoughts
This project helped our group familiarize ourselves with MLB Statcast data, making us more confident when using this data for other projects. Also, we learned how to boost model performance when faced with overfitting. 

