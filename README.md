# Income Prediction Using Supervised and Unsupervised Learning

<b>Abstract—As the title of the paper, the research is conducted to predict whether the persons’ income exceeds $50K/year or not. The input data about the different factors are collected from a given UCI dataset which is based on census data. The dataset is then processed through data cleaning, handling of missing values, Outlier removal, feature selection, data normalization, data visualization, and splitting of data into a training set and testing set. The training set is then fitted into Supervise Learning model, Gaussian Naïve Bayes model, and into the Unsupervised learning model, KMeans from scikit learn library. The Gaussian Naïve Bayes model is tested with the testing dataset, whereas the optimal number of clusters is calculated in KMeans<br/>
Keywords—Income, Missing Values, Outliers, Feature Selection, Normalization, Gaussian Naïve Bayes, KMeans</b>
  
  <h2>I. Introduction</h2><br/>
  People of different ages with different educational backgrounds and different years of experience have different amounts of income. The amount of income is dependent on different factors. We can predict whether the persons’ income is greater than 50K or not based on different factors. 
  <br/>

<h2>II. Methodology </h2><br>
<h3>A. Data Acquisition </h3><br/>
There is variety in peoples’ income on different basis. Barry Becker extracted the data from the 1994 Census database and created an dataset named Adult data set which is currently found in UCI Machine Learning Repository[1]. This dataset was used for the prediction of income. <br/><br/>

<h3>B. Data Preprocessing </h3> <br/>
While checking the data, the target feature ‘Salary’ only contained two unique values: <=50K and >50K. So, its values were converted to 0 and 1 for salary <=50K and >50K, respectively.<br/> 
Then the data was split into 70% of the training set and the rest into the testing set using the python array splitting method. We moved on to the next steps with train data.<br/><br/>

<h4>1.	Missing Value Handling</h4> <br/>
The null or missing values are represented as ‘?’ by the dataset collector. Three features, ‘workclass’, ‘occupation’, and  ‘native-country’. As all three features were categorical, all the data with missing values were removed. The duplicate datas were also removed. <br/><br/>

<h4>2.	Categorical to numerical</h4><br/>
The categorical columns were differentiated from the dataset, and their categorical data were converted into numerical data using a label encoder found in preprocessing module from scikit learn library.<br/><br/>

<h4>3.	Outlier Removal</h4><br/>
The data was visualized using histogram and box plot to determine its dispersion and the outliers. <br/><br/>

<h5>i)	Using Median-IQR</h5><br/>
Features ‘age’, ‘education-num’, ‘fnlwgt’, and ‘capital-gain’ were analyzed for outliers by using Median-IQR method and the Outliers were removed.<br/><br/>
Here, the outliers are determined as: <br/>
Q1: First Quartile, Q3: Third Quartile<br/>
IQR = Q3 – Q1<br/>
OutlierConstant = 1.5<br/>
OLB = Q1 - (IQR * outlierConstant)<br/>
OUB = Q3 + (IQR * outlierConstant)<br/>
OUB < Outlier & Outlier < OLB <br/>
Where, OLB: Outlier Lower Boundary, OUB: Outlier Upper Boundary <br/><br/>

<h5>ii)	 Using Mean-SDV</h5><br/>
Features ‘Capital-loss' and ‘hours-per-week’ were analyzed for outliers by using Mean-SDV method and the Outliers were removed.  <br/>
Here, the outliers are determined as: <br/>
OLB = mean – 3*std <br/>
ULB = mean + 3*std <br/>
OUB < Outlier & Outlier < OLB <br/>
Where, std: standard deviation, OLB: Outlier Lower Boundary, OUB: Outlier Upper Boundary <br/><br/>

<h4>4.	Feature Selection</h4><br/>
The correlation between the features was calculated and rounded to the decimal point of two decimals. Heatmaps were created using the correlation data of all 15 columns. Four feature columns were selected by keeping the threshold of 20% correlation percentage. They are: ‘age’, ‘education-num’, ‘sex’, and ‘hours-per-week'. <br/><br/>
  
![Heatmap_of_correlations_between_the_features](https://github.com/samirkhanal35/Income-Prediction-Using-Supervised-and-Unsupervised-Learning/blob/main/Heatmap.png)
  <br/><br/>
<h4>5.	Data Visualization</h4><br/>
The distribution of the target variable ‘Salary’ was visualized. Then, the distribution of each selected feature was visualized with a histogram plot along with their central tendency by using a Normal distribution graph.<br/><br/>
 
![Distribution of target variable](https://github.com/samirkhanal35/Income-Prediction-Using-Supervised-and-Unsupervised-Learning/blob/main/fig2.png)
 

![Data distribution of age](https://github.com/samirkhanal35/Income-Prediction-Using-Supervised-and-Unsupervised-Learning/blob/main/fig3.png)
 

![Data distribution of education-num](https://github.com/samirkhanal35/Income-Prediction-Using-Supervised-and-Unsupervised-Learning/blob/main/fig4.png)


![Data distribution of hours-per-week](https://github.com/samirkhanal35/Income-Prediction-Using-Supervised-and-Unsupervised-Learning/blob/main/fig5.png)

 
![Normal distribution of age](https://github.com/samirkhanal35/Income-Prediction-Using-Supervised-and-Unsupervised-Learning/blob/main/fig6.png)

 

![Normal distribution of education-num](https://github.com/samirkhanal35/Income-Prediction-Using-Supervised-and-Unsupervised-Learning/blob/main/fig7.png)

 

![Normal distribution of sex](https://github.com/samirkhanal35/Income-Prediction-Using-Supervised-and-Unsupervised-Learning/blob/main/fig8.png)


![Normal distribution of hours-per-week](https://github.com/samirkhanal35/Income-Prediction-Using-Supervised-and-Unsupervised-Learning/blob/main/fig9.png)

<br/><br/>
<h4>6.	Data Normalization </h4><br/>
To normalize the data; the scaling data normalization method was applied to the feature columns of the dataset using the formula in equation (1).<br/>

![equation-1](https://github.com/samirkhanal35/Income-Prediction-Using-Supervised-and-Unsupervised-Learning/blob/main/equation.png)
        
<br/><br/>
<h3>	Data Modelling </h3> <br/>

<h4>1.	Supervised Learning</h4><br/>

The target variable or dependent variable only had two categories of data 0 and 1. So, the binary classification technique, Gaussian Naïve Bayes model was used from Scikit learn library.
<br/><br/>

<h4>2.	Unsupervised Learning</h4><br/>
The target variable was removed from the data, and other selected features were fed to the KMeans model from Scikit learn library with different numbers of clusters.<br/><br/>

<h2>III.	PERFORMANCE AND RESULTS</h2><br>

A Gaussian Naïve Bayes model from scikit-learn was used to fit the data. The model gave 82.65% accuracy to the training dataset and 80.84% accuracy for the test dataset. The confusion matrix of the model for the test dataset is shown below. It has 6964 True Positives and 933 True Negatives, 1374 False Negatives, and 498 False Positives.<br/><br/>
 
![Model evaluation with confusion matrix](https://github.com/samirkhanal35/Income-Prediction-Using-Supervised-and-Unsupervised-Learning/blob/main/confusion-matrix.png)

<br/><br/>
A KMeans model from scikit-learn was used to fit the data. We used different numbers of clusters to calculate the distortion and inertial of the model. In the elbow graph of both of them, the value of k at elbow is 2. So, the optimal number of clusters for this data is 2.<br/><br/>

 


![Elbow Method using Distortion Values](https://github.com/samirkhanal35/Income-Prediction-Using-Supervised-and-Unsupervised-Learning/blob/main/Elbow_with_distortion.png)
 

![Elbow Method using Inertia Values](https://github.com/samirkhanal35/Income-Prediction-Using-Supervised-and-Unsupervised-Learning/blob/main/Elbow_with_Inertia.png)

<br/><br/>
<h2>IV.	CONCLUSION</h2><br/>
The two methods presented in this paper can process the data in Adult Data Dataset, clean data, deal with missing values and outliers, feature selection, normalization, visualization, modeling, training, and predicting the Salary whether less than 50K OR greater than 50K. <br/>

There is still some further research to do. For example, Other classification models can be used and compared with our models.<br/><br/>

<h2>ACKNOWLEDGMENT</h2> <br/>
This research article would not have been possible without the exceptional support of my mentor Prof. Mohammad Islam and the UCI dataset [1]. I am also grateful for the insightful comments offered by the anonymous peer reviewers from my batchmates. The generosity and insights of  Prof. Mohammad Islam have improved this study in innumerable ways. Thank you all for your support. <br/><br/>

<h2>REFERENCES</h2><br/>
[1]	Kohavi, R. and Becker, B. (1994). UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/adult <br/>
[2]	Gupta, A. (22 August, 2022). Elbow Method for optimal value of k in KMeans. https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/ <br/>
[3]	Kohavi, R. (1996). Scaling Up the Accuracy of Naive-Bayes Classifiers. Proceedings of the Second International Conference on Knowledge Discovery and Data Mining. http://robotics.stanford.edu/~ronnyk/nbtree.pdf <br/>
