# wine

In this repo, I show how to use artial least squares regression to make an emsemble model to predict wine quality. 

## Data
The data can be downloaded in https://archive.ics.uci.edu/ml/datasets/Wine+Quality

I was considering if I should keep the labels of wine color for the training set. Then before the final prediction, I can first classify wine color. In this way, I can increase the accuracy by building two models for red wine and white wine separately.

Eventually, I decided not to separate the two datasets because:

(1) When visualizing the data on 2D space after LDA dimensional reduction, with or without combining the two datasets look similar.

(2) The red wine data only has 1599 examples. This may not be enough to build a robust model.

(3) The prediction results seem similar with or without training the model separately for red and white wine. The mean absolute error (MAE) of the combined model is 0.53. For the same method, if I used it on only the red wine data, the MAE is 0.52 and for white wine data, it is 0.57.

As I was building an ensemble data, I decided to keep the data in three parts in order
to minimize the possibility of overfitting caused by the ensembled model.

ensembleData -> used to train and select individual models, accounts for 40% of the
total data

blenderData -> used to train the final ensemble model, accounts for 40% of the total
data

testingData -> unseen data, only used for evaluation, accounts for 20% of the total
data

## EDA

<img src = images/correlation.png>

Correlation of independent variables (white and red wine combined)

<img src = images/pairplot.png>

Pair plot of the independent variables (white and red wine combined)

EDA incidate strong correlation between variables, therefore, PLS is a good method in this application.

<img src = images/biplot.png>

Biplot extracted from simple PLS

From the biplot in Fig3, which shows the projection of all data and the independent variables to the first two components in PLS, we can see that alcohol has the largest vector length, implying that it is the strongest factor in wine quality. We can also see that critic acid and fixed acidity is closed to each other, and are opposite to PH. This also fits into the chemical model as the higher the acidity, the lower the PH value.

## Feature engineering

There are several reasons that feature generation is important here:

1. PLS is the only regression model that I use in this data challenge. PLS is a linear model, which means that if the feature space is non-linear, it won't be able to model it well.

2. Besides, the advantage of PLS is that it can handle high dimensional feature space, especially when there is collinearity among the features. That means that we should generate nonlinear features and let the PLS do it job in filtering out the useful features and handling the collinearity.

3. In the plsRglm package, we have "pls", "pls-glm-Gamma", "pls-glm-gaussian", "pls-glm-inverse.gaussian", "pls-glm-logistic", "pls-glm-poisson", "pls-glm-polr". Usually, an ensemble model has better performance when the model has low correlation, if we only use plsRglm,there is little improvement when stacking different plsRglm models. Therefore, instead of working on different plsRglm family, I focus on creating different datasets for the ensemble.

#### Polynomial features with Linear Discriminant Analysis (LDA) reduction
Assume there are features x1, x2, then the generated 2nd order polynomial features will be x1*x1, x1*x2, x2*x2. And the 3rd order polynomial features will be x1*x1*x1, x1*x1*x2, x1*x2*x2, x2*x2*x2. The generated polynomial features were passed through an LDA classifier to extract LDA factors as the new features. The reason behind this is that I would like to extend the distance between data of different wine quality as far as possible, making the PLS prediction more robust. The feature generation code is in the Appendix (Hand_crafted_model_part1). It also includes a function to use a kernel function to generate new features. However, due to the high dimensionality of the kernel features – more than 6000 features, the training becomes very slow. Therefore, kernel feature was not used in the final model

<img src = images/LDA.png>

This is the scatter plot based on the first two components in the LDA model based on 3rd order polynomial features I generated. In Fig 4, two different trends are being circled out: the low to medium quality wine (quality <7) and the medium to high quality wine (quality >6). This inspires me to build two models, one for the low to medium quality wine and one for medium to high quality wine. The final prediction is the combination of the two models based on the following conditions: 

(1) when the prediction is <6, take the prediction from the low-quality model,

(2) when the prediction is >6, take the prediction from the high-quality model and

(3) when it is in the middle, take the average.

The MAE of the mixed model is around 0.54, while for the low-quality model alone,
the MAE is around 0.63, and for the high-quality model alone, it is 0.72.

The code of this mixed model is in Hand_crafted_model_part2.

## Final model

As the final task is to be able to run the entire ensemble model resulting from the training script through a single call to predict (model, newData=newData), I decide to use caret’s built-in function for the ensemble. As discussed earlier, using different data is the key to getting a good prediction here. In the ensemble model, I chose to do different preprocessing to create different datasets, including ‘expoTrans’, ‘pca’ and ‘YeoJohnson’. ‘expoTrans’ apply a power transform to change the data distribution. ‘pca’ perform principal component analysis on the data, and extract the principal components to replace the original data. ‘YeoJohnson’ is a method that commonly used to convert the data distribution to a normal distribution. From the EDA, I know that data is not normally distributed, therefore, it should be able to generate a new dataset by using the ‘YeoJohnson’ method.

<img src = images/mae.png>

## Other model to try
Guassian process
https://distill.pub/2019/visual-exploration-gaussian-processes/
