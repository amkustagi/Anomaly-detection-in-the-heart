# CARDIOTOCOGRAPHY Data analysis to find out the Fetal status
#### Fetal status defined as (Normal/Suspect/ Pathologic) - It's a Categorical prediction with Nominal in nature

#### Methodology Used:
###### 1. Stepwise Regression using StepAIC to reduce the no. of independent variable 
###### 2. Multinomial Logistic Regression - Which helps to predict nominal output variables 
###### 3. Decision Tree Using rpart & Validation using test dataset & used Party package to improvise the visualization of tree
###### 4. Bagging & Random Forest Approach to improve the model


### 1. Stepwise Regression using StepAIC function to reduce the no. of independent variable



```{r include=FALSE}
library(caret)
library(MASS)
library(nnet)
getwd()
setwd("F:/MS BIA/4T. Applied Data Mining - SPRING B 2016/Project")
NSP <- read.csv("F:/MS BIA/4T. Applied Data Mining - SPRING B 2016/Project/Cardiotocographic_NSP.csv")
NSP$NSP <- as.factor(NSP$NSP)
```

##### Metrics considered for NSP classification
1. LB	Baseline value 		
2. AC	accelerations 		
3. FM	foetal movement 		
4. UC	uterine contractions 
5. DL	light decelerations	
6. DS	severe decelerations	
7. DP	Prolonged decelerations	
8. ASTV	percentage of time with abnormal short term variability  	
9. MSTV	mean value of short term variability  	
10. ALTV	percentage of time with abnormal long term variability  	
11. MLTV	mean value of long term variability  	
12. Width	histogram width
13. Min	low freq. of the histogram
14. Max	high freq. of the histogram
15. Nmax	number of histogram peaks
16. Nzeros	number of histogram zeros
17. Mode	histogram mode
18. Mean	histogram mean
19. Median	histogram median
20. Variance	histogram variance
21. Tendency	histogram tendency: -1=left asymmetric; 0=symmetric; 1=right asymmetric
22. CLASS 	FHR pattern class code (1 to 10) 
23. NSP	Normal=1; Suspect=2; Pathologic=3	


##### 1. The step with least AIC Value is the better model ( Least AIC = 345.97)
```{r}
fit <- glm(NSP~.,data=NSP, family = "binomial")
step <- stepAIC(fit, direction="both")
``` 

##### Final Model with list of independent variables to be eliminated 
```{r}
step$anova # display results
```


##### Eliminating Independent variables & Splitting the dataset (70:30)
```{r}
NSP_new <- NSP[,-c(1,6,9,11,14,16,19,21)]
set.seed(123)
NSP_rand <- NSP_new[order(runif(2126)), ] # randomize observations
NSP_rand <- sample(1:2126, 1488) #We want 1488 observations in training and the rest in test. Approximately 70-30 split.
NSP_train <- NSP_new[NSP_rand,]
NSP_test  <- NSP_new[-NSP_rand,]


dim(NSP_train) #checking the split
dim(NSP_test) #checking the split
prop.table(table(NSP_train$NSP)) #checking to see the class proportions between the training and test sets. 
prop.table(table(NSP_test$NSP))



```

##### Final Metrics considered for NSP classification after step regression
1. AC	accelerations 		
2. FM	foetal movement 		
3. UC	uterine contractions 
4. DL	light decelerations	
5. DP	Prolonged decelerations	
6. ASTV	percentage of time with abnormal short term variability  	
7. ALTV	percentage of time with abnormal long term variability  	
8. Width	histogram width
9. Min	low freq. of the histogram
10. Nmax	number of histogram peaks
11. Mode	histogram mode
12. Mean	histogram mean
13. Variance	histogram variance
14. CLASS 	FHR pattern class code (1 to 10) 
15. NSP	Normal=1; Suspect=2; Pathologic=3	

##### Since the Class variable is the Morphological Pattern (1 to 10) classification by Medical expertise, that could be the reason for overfitting!!
##### We cannot use the Classifications by Medical expertise 
##### Creating alternative dataset without class variable to check how the model behaves 


##### Creating New dataset without Class Variable 
```{r}

NSP_train_No.class <- NSP_train[, -c(14)]
NSP_test_No.class <- NSP_test[, -c(14)]

#checking the split of data without Class Variable 
dim(NSP_train_No.class)
dim(NSP_test_No.class)

```





### 2. Multinomial Logistic Regression - Which helps to predict nominal output variables 

```{r include=FALSE}
options(scipen=0)
NSP.train <- multinom(NSP ~., data = NSP_train)
```
#### Summary - Coefficients & Standard error 
##### 1. Coefficients for N-1 outputs in ODDS (2 OuTputs- as there are totally 3 categorical outputs)
##### 2. Standard errors for N-1 Outputs (2 OuTputs- as there are totally 3 categorical outputs)

```{r}
summary(NSP.train)
z <- summary(NSP.train)$coefficients/summary(NSP.train)$standard.errors

```

##### P Value indicating the significance of independent variables
```{r}
p <- (1 - pnorm(abs(z), 0, 1))*2
p
```

##### Coefficients for N-1 outputs in ODDS (2 Outputs- as there are totally 3 categorical outputs)
```{r}
exp(coef(NSP.train))

```

##### Output predictions
```{r}

options(scipen=999)
head(pp <- fitted(NSP.train))
options(scipen=0)
```


### 3. Decision Tree Using rpart 
```{r include=FALSE}
library(rpart)
library(rpart.plot)
library(partykit)
library(party)
```

##### Decision Tree using Gini Index With Class Variable 
```{r}

NSP.rpart<-rpart(NSP_train$NSP ~ ., method="class", parms=list(split="gini"), data=NSP_train)
rpart.plot(NSP.rpart, type=4, extra=101, cex = 0.7)

```


#### Validation using test dataset
##### 1. Accuracy of the model is 98.59% - Overfitting due to the use of Class Variable ( Class - Morphological classifications by medical experties)
```{r}
actual <- NSP_test$NSP 
NSP_predicted <- predict(NSP.rpart, newdata=NSP_test, type="class") 
NSP_results.matrix <- confusionMatrix(NSP_predicted, actual, positive="yes") 
print(NSP_results.matrix) 

```


##### Decision Tree using Gini Index Without Class Variable 
```{r}

NSP.rpart.class<-rpart(NSP_train_No.class$NSP ~ ., method="class", parms=list(split="gini"), data=NSP_train_No.class)
rpart.plot(NSP.rpart.class, type=4, extra=101, cex = 0.7)

```


####  Validation of model without Class Variable
##### 1. Accuracy of the model is to 92.3%,
##### 2. The previous model was overfitting due to Class variable (direct use of morphological classifications of medical experties)
##### 3. Hence, eliminated the class variable from Bagging and Randomforest Models in the further analysis 
```{r}
actual.class <- NSP_test_No.class$NSP 
NSP_predicted.class <- predict(NSP.rpart.class, newdata=NSP_test_No.class, type="class") 
NSP_results.matrix.class <- confusionMatrix(NSP_predicted.class, actual.class, positive="yes") 
print(NSP_results.matrix.class) 

```



### 4. Bagging & Random Forest Approach to improve the model

#### Model using Bagging - Considering all predictors at each split
##### 1.OOB estimate of  error rate: 6.45%
##### 2. ASTV, ALTV & Mean are predicted as most important Independent variable in predicting the output
```{r}
library(caret)
library(randomForest)
set.seed(123) 


NSP.bag <- randomForest(NSP_train_No.class$NSP ~., data=NSP_train_No.class, mtry=13, na.action=na.omit, importance=TRUE)

print(NSP.bag) #note the "out of bag" (OOB) error rate. 

importance(NSP.bag) #shows the importance of each variable. Variable importance is computed using the mean decrease in the Gini index.

varImpPlot(NSP.bag)

```

##### Validating the Bagging model using test dataset
##### 1. Model accuracy has increase to 93.89% Vs 92.3% Comapare to rpart model
```{r}
actual <- NSP_test_No.class$NSP
NSP_predicted.bag <- predict(NSP.bag, newdata=NSP_test_No.class, type="class") 
NSP_results.matrix.bag <- confusionMatrix(NSP_predicted.bag, actual, positive="yes") 
print(NSP_results.matrix.bag)

```


#### Model using Random Forest Approach
##### 1.OOB estimate of  error rate: 6.05%
##### 2. Again ASTV, ALTV & the Mean are predicted as most important Independent variable in predicting the output by Random forest method
```{r}
NSP.RForest <- randomForest(NSP_train_No.class$NSP ~.,data=NSP_train_No.class, mtry=3, ntree=600,na.action = na.omit, importance=TRUE) #default mtry = 3 and ntree= 500.
print(NSP.RForest) #shows OOB of model and confusion matrix
importance(NSP.RForest) #shows the importance of each variable
varImpPlot(NSP.RForest) #plots the importance of each variable
```


#### Validating the Random Forest model using test dataset
##### 1. Model has predicted with 94.04% accuracy (little higher than Bagging Model) 
##### 2. A small increase in the Sensitivity and Specificity are better compared to previous model
##### 2. Random Forest turned out as the best model among rest of the models.
``````{r}

actual <- NSP_test_No.class$NSP 
NSP.RForest_predict<-predict(NSP.RForest, NSP_test_No.class ,type="response") 
NSP.RForest_results.matrix <- confusionMatrix(NSP.RForest_predict, actual,positive="yes") 
print(NSP.RForest_results.matrix)
print(NSP_results.matrix.bag)

```

## FOR - Insights

The Cardiotocogrphy dataset contains the records of 2126 samples of medical recording contain fetal Cardiotocograph (CTGs) records with the details as show below.

1. LB	Baseline value 		
2. AC	accelerations 		
3. FM	fetal movement 		
4. UC	uterine contractions 
5. DL	light decelerations	
6. DS	severe decelerations	
7. DP	Prolonged decelerations	
8. ASTV	percentage of time with abnormal short term variability  	
9. MSTV mean value of short term variability  	
10. ALTV percentage of time with abnormal long term variability  	
11. MLTV mean value of long term variability  	
12. Width histogram width
13. Min	low freq. of the histogram
14. Max	high freq. of the histogram
15. Nmax number of histogram peaks
16. Nzeros number of histogram zeros
17. Mode histogram mode
18. Mean histogram mean
19. Median histogram median
20. Variance histogram variance
21. Tendency histogram tendency: -1=left asymmetric; 0=symmetric; 1=right asymmetric

Using these variables modeling has been done to predict the fetal status as (N=normal; S=suspect; P=pathologic) or morphologic patterns ( 1 to 10) This predictions could help us taking further steps,	
1. Precautionary measures
2. Decisions on medical care etc.



### FACTS
According to the data set:
  1. We divided the 2126 medical records Cardio graphic data into 2 datasets with 70% sample size for building the model and 30% sample size for testing and validation.
2. Our goal is to create a model for NSP Classification using training data set (Subset of database) and Validating the model using Testing dataset (another subset). This could help in predicting the fetal status.

### OPTIONS
We have developed multiple prediction algorithms, which could help us in classifying the fetal status, most of the models could help us classifying up to 92% accurately. 

Example of model: 
  1.	Decision tree with rpart predicts with the accuracy of 92.3%
2.	Bagging model (by considering all the variables at each node) predicts with the accuracy of 93.89%. This model predicts ALTV, ASTV and Mean are the most important to predicting the NSP categories. 
3.	Random Forest model (by considering only 3 variables at each node) predicts with the accuracy of 94.67%. Even this model predicts ALTV, ASTV and Mean as the most important factors in predicting the NSP categories. 

### RECOMMENDATIONS
1.	Random Forest model could be used to classify the NSP categories with the accuracy 94.67%, which is highest among all the models. 
2.	Bagging model can also be used for confirming the result, even this model is performing the equally good with almost 94.04% accuracy.
