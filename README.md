# R-project-ML-Basic(02)

## Multivariate_Analysis / Machine Learning playground with R

### [Contents] 

__Lab-04.__ Classification  
  - package: rpart, partykit, party, nnet, adabag, randomForest 
  - func:

__Lab-05.__ Dimension reduction 
  - package: 
  - func:
  
----------------------------------------------------------------------
#### >Lab-04. Classification

<img src="https://user-images.githubusercontent.com/31917400/32509112-536aa38a-c3e4-11e7-86f5-63fb57c48fa6.jpg" />
<img src="https://user-images.githubusercontent.com/31917400/32509133-604e5b5a-c3e4-11e7-99d2-3943783c7a60.jpg" />
<img src="https://user-images.githubusercontent.com/31917400/32509155-6ba7c9c8-c3e4-11e7-95fe-d6adf0901633.jpg" />
<img src="https://user-images.githubusercontent.com/31917400/32509171-78e32dbc-c3e4-11e7-8a96-eb6ca10bc92e.jpg" />

```
## entropy and gini ==> variance of categorical variables##
##if var=0, monopolization, if var=log(k), evenly distributed (all entries have equal propability)##
##we use these variances to find the threshold that differenciates the predictors##

##Entropy: 0 < var < log(k) and p is prob of each categorical variable entry##
entropy = function(p) {
  p < -p[p>0] 
  sum(-p*log(p))
}

##Gini: 0 < var < 1-(1/k) and p is prob of each categorical variable entry##
gini = function(p) {
  1-sum(p^2)
}
```

#### *|Implementing "Classification Trees" (for simple dataset- 5 variables)|*

__Data:__ On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships. Our data were collected recording passangers' information including "Name","PClass(socio-economic status)","Age","Sex","Survived or not".
Extra details of the votes can be found at (https://www.kaggle.com/c/titanic/data/). 

__Story:__ One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class. In this challenge, we want to predict what sorts of people were likely to survive the tragedy.

<img src="https://user-images.githubusercontent.com/31917400/32564854-545e8918-c4ad-11e7-9f39-a287a1af860f.jpg" width="600" height="200" />

```
#Ensure that "any categorical variables" are coded as factors##
is.factor(titanic$Name)
is.factor(titanic$PClass)
is.factor(titanic$Sex)
is.factor(titanic$Survived)

##fit the model##
library(rpart)
fit.r <- rpart(Survived ~ Sex+Age+PClass, data = titanic); fit.r
summary(fit.r)

##plotting Classification Tree##
plot(fit.r)
text(fit.r, use.n = T, cex=0.8, xpd=T, col='red')

library(partykit)
plot(as.party(fit.r))

library(party)
fit.c <- ctree(Survived ~ Sex+Age+PClass, data = titanic); fit.c
plot(fit.c)
plot(Survived ~ Sex+Age+PClass, data = titanic)
```

<img src="https://user-images.githubusercontent.com/31917400/32566468-ec9574ce-c4af-11e7-8a2e-085dfb2cca30.jpeg" />

When choosing splits in the tree, the distinction between nominal and ordinal categorical variables is important. Passenger class is clearly a categorical variable with ordering 1st, 2nd and 3rd. We can check if it is stored as an ordinal or a nominal variable. 
```
is.ordered(titanic$PClass) 
titanic$PClass <- ordered(titanic$PClass, c('1st', '2nd', '3rd'))

##plotting Classification Tree again##
fit.c <- ctree(Survived ~ Sex+Age+PClass, data = titanic); fit.c
plot(fit.c)
plot(Survived ~ Sex+Age+PClass, data = titanic)

```
The categorical variable is split into groups of those variables less than some value and greater than or equal to the value.

<img src="https://user-images.githubusercontent.com/31917400/32567070-fdc7eebe-c4b1-11e7-84e8-ac57df330a7d.jpeg" width="600" height="300" />

In particular, this tree growing method doesn't split any node with less than 20 observations(controlled by 'minsplit'). It also doesn't fit very complex looking trees(controlled by 'cp'). 


#### *|Implementing "Bagging / randomForest" (for complex dataset - 37 variables)|*

Bagging is a method that uses bootstrapping to increase the stability of a classiﬁcation method.
Random forests extend this further to make the trees more diverse.
We can compare a single classiﬁcation tree and logistic regression to bagging and random forests. 
Using `set.seed(123)` to make the results repeatable.

__Data:__ Lower Back Pain Symptoms Data set, Collection of physical spine data (36 Binary Predictors, 1 three-Class Attribute). Lower back pain can be caused by a variety of problems with any parts of the complex, interconnected network of spinal muscles, nerves, bones, discs, tendons, etc in the lumbar spine. This data set is about to identify a symtom is Nociceptive or Central Neuropathic or Peripheral Neuropathic, using collected physical spine details. Extra details of the votes can be found at (https://www.kaggle.com/sammy123/lower-back-pain-symptoms-dataset). 
   
<img src="https://user-images.githubusercontent.com/31917400/32578427-1b6824fc-c4d5-11e7-8bda-b40dcef0f626.jpg" width="600" height="230" />

__Issue:__ We can test how well our classiﬁcation method works on the data that is used to build the classiﬁer. Because the data is used to build the classiﬁer, we cannot avoid overfitting and over estimating our performance. A better approach would be to ﬁt to one data set, choose method on a second and to test on third. We will split the data into three to complete this task. In addition, before starting, we need to deal with some missing values detected. In this case, random Forest does not work. Accordingly, those values need to be eliminated (N: 464 => 425)

 - __Comparing classifiers__ (assessing performance on the **same** data)
```
is.factor(b.data$assigned.labels)

##missing values detected##
is.na(b.data$X19) 
sum(is.na(b.data)) 

##delete rows of NA##
b.data = na.omit(b.data) 
dim(b.data) 
sum(is.na(b.data))

##model fitting 'rpart' + assessing performance on the same data##
fit.r <- rpart(assigned.labels ~., data=b.data); fit.r
pred <- predict(fit.r, type="class", newdata = b.data)
tab <- table(b.data$assigned.labels, pred); tab
sum(diag(tab))/sum(tab) ##accurracy: 0.905(with missing values) -> 0.913(without missing values)## 
plot(as.party(fit.r))
varImp(fit.r)

##model fitting 'logistic regression' + assessing performance on the same data##
fitlog <- multinom(assigned.labels ~., data=b.data); fitlog 
pred <- predict(fitlog, type='class', newdata = b.data)
tab <- table(b.data$assigned.labels, pred); tab
sum(diag(tab))/sum(tab) ##accurracy: 1.00(with missing values) -> 1.00(without missing values)## 
varImp(fitlog) 

##model fitting 'bagging' + assessing performance on the same data##
fitbag <- bagging(assigned.labels ~., data=b.data)
pred <- predict(fitbag, type="class", newdata = b.data)
tab <- pred$confusion; tab
sum(diag(tab))/sum(tab) ##accurracy:0.920(with missing values) -> 0.927(without missing values)## 

##model fitting 'randomForest' + assessing performance on the same data##
fitrf <- randomForest(assigned.labels ~., data=b.data); fitrf
pred <- predict(fitrf, type="class", newdata = b.data)
tab <- table(b.data$assigned.labels, pred); tab
sum(diag(tab))/sum(tab) ##accurracy: NA -> 0.995##
plot(fitrf)
varImp(fitrf) 
```
<img src="https://user-images.githubusercontent.com/31917400/32579694-fe6ff08c-c4d9-11e7-8d6e-65f2b7856ced.jpg" />

__Interpretation:__ Judging from the outputs above, their performance seems quite great, and notably, logistic regression classifier registers 100% accuracy. This seemingly **overfitting** issue stems from the fact that every data is used to build those classifiers. We know there are several methods to reassess those classifiers and build some confidence in their performance. 
 
 - >_A. General validation: we split the data into three parts – training 50%/validation25%/test25%, and build classifiers based on the training data. Then compare full prediction results yielded by the classifier with validation data and test data from the original so that we can compare their performances.  

 - >_B. Bootstrapping validation: We use a bootstrap sample as training data. When we do bootstraping, classifier is built on the dataset that has the same number of observations of the original data, and it does some of observations repeated, whereas in general splittting, classifier is built on smaller dataset. More importantly, the values bootstrapping is missing becomes the test data. 

 - >_C. K-fold cross validation: We divide the data into K groups and differentiate one of those groups (test data) from the rest of them (training data), then build the classifiers based on those K-1 groups and compare the prediction results with the test data. This process continues K times. 
 
 - __General validation__ (assessing performance on the **validate & test** data)
```
## 1> general ##
N <- nrow(b.data)
ind.train <- sample(1:N,size=0.50*N,replace=FALSE)
ind.train <- sort(ind.train)
ind.valid <- sample(setdiff(1:N,ind.train),size=0.25*N)
ind.valid <- sort(ind.valid)
ind.test <- setdiff(1:N,union(ind.train,ind.valid)) 


##model fitting 'rpart' + assessing performance on the validate data##
fit.r <- rpart(assigned.labels ~., data=b.data, subset = ind.train); fit.r
pred <- predict(fit.r, type = 'class', newdata = b.data); pred
tab <- table(b.data$assigned.labels[ind.valid], pred[ind.valid]); tab
sum(diag(tab))/sum(tab)  # accuracy: 0.896

##model fitting 'logistic regression'+ assessing performance on the validate data##
fitlog <- multinom(assigned.labels ~., data=b.data, subset = ind.train); fitlog 
pred <- predict(fitlog, type='class', newdata = b.data); pred
tab <- table(b.data$assigned.labels[ind.valid], pred[ind.valid]); tab
sum(diag(tab))/sum(tab) # accurracy: 0.858

##model fitting 'bagging'+ assessing performance on the validate data##
fitbag <- bagging(assigned.labels ~., data=b.data[ind.train, ])
pred <- predict(fitbag, type="class", newdata = b.data[ind.valid, ])
tab <- pred$confusion; tab
sum(diag(tab))/sum(tab) # accurracy: 0.896

##model fitting 'randomForest'+ assessing performance on the validate data##
fitrf <- randomForest(assigned.labels ~., data=b.data, subset = ind.train); fitrf
pred <- predict(fit.r, type="class", newdata = b.data); pred
tab <- table(b.data$assigned.labels[ind.valid], pred[ind.valid]); tab
sum(diag(tab))/sum(tab) # accurracy: 0.896

## All in all, bagging shows the best performance, thus we try bagging on the test data to get an accurate assessment of its performance. Finally, model fitting 'bagging'+ assessing performance on the test data##
pred <- predict(fitbag, type="class", newdata = b.data[ind.test, ])
tab <- pred$confusion; tab
sum(diag(tab))/sum(tab) # accurracy: 0.906
```
 - __Bootstrapping validation__ (assessing performance on the **test** data)
```
## 2> bootstrapping ##
N <- nrow(b.data)
ind.train <- sample(1:N,replace=TRUE);ind.train ## bootstrapping new data => training data! ##
ind.train <- sort(ind.train);ind.train
ind.test <- setdiff(1:N,ind.train);ind.test ##values bootstrapping is missing##


##model fitting 'rpart' + assessing performance on the test data##
fit.r <- rpart(assigned.labels ~., data=b.data, subset = ind.train)
pred <- predict(fit.r,type="class",newdata=b.data)
pred[ind.test]
tab <- table(b.data$assigned.labels[ind.test],pred[ind.test]); tab
sum(diag(tab))/sum(tab) # accurracy: 0.8387097

##model fitting 'logistic regression'+ assessing performance on the test data##
fitlog <- multinom(assigned.labels ~., data=b.data, subset = ind.train) 
pred <- predict(fitlog, type='class', newdata = b.data); pred
tab <- table(b.data$assigned.labels[ind.test], pred[ind.test]); tab
sum(diag(tab))/sum(tab) # accurracy: 0.8387097

##model fitting 'bagging'+ assessing performance on the test data##
fitbag <- bagging(assigned.labels ~., data=b.data[ind.train, ])
pred <- predict(fitbag,type="class", newdata = b.data[ind.test, ])
tab <- pred$confusion; tab
sum(diag(tab))/sum(tab) # accurracy: 0.8774194

##model fitting 'randomForest'+ assessing performance on the test data##
fitrf <- randomForest(assigned.labels ~., data=b.data, subset = ind.train)
pred <- predict(fitrf,type="class", newdata = b.data); pred
tab <- table(b.data$assigned.labels[ind.test], pred[ind.test]); tab
sum(diag(tab))/sum(tab) # accurracy: 0.8903226

## All in all, randomForest shows the best performance.
```
 - __K-fold cross validation__ (assessing performance on the **test** data) **K=10**
```




```
__Interpretation:__ All in all, it seems that random Forest is the best classifier to this data set, which is underpinned by those multiple validation procedures. When our data is subject to limited usage, bagging, interestingly, shows the greatest performance; however, once extending the scope of data set, and lifting the usage limit - using bootstrap or K-fold cross validation – we were able to obtain more reliable outcomes.   


























