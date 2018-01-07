#Data Mining HW3

# Load the PainData from the internet##################################
#load(url("http://mathsci.ucd.ie/~brendan/data/PainData.Rdata")) #? WTF
#################################################################################################################################
#work through various methods for assessing the performance of a single classi???er (or multiple classi???ers). 
#The task for this laboratory is to implement these methods on the back pain dataset.

#Choose the best classiﬁer from at least three of those seen in class. 

#A.Classification tree
#B.Logistic regression
#C.bagging
#D.Random forest

#What do you think is an honest measure of classiﬁer accuracy for this classiﬁer on the data that you chose? 

b.data <- read.table('C:/Users/Minkun/Desktop/r_practice/+++++++++++School R/LAB_data mining/data/Physio.txt', header = T)
names(b.data)
tail(b.data)
summary(b.data$assigned.labels)

is.factor(b.data$assigned.labels) #true
is.factor(b.data$X1) #false

# we found missing values..
is.na(b.data$X19) #true
is.na(b.data$X20) #true
sum(is.na(b.data)) #44

b.data = na.omit(b.data) # delete NA
dim(b.data) #425 x 37
sum(is.na(b.data))


#Central Neuropathic:1
#Nociceptive:2
#Peripheral Neuropathic:3

set.seed(123)
#---------------------------why set.seed????
#> sample(LETTERS, 5)
#[1] "K" "N" "R" "Z" "G"
#> sample(LETTERS, 5)
#[1] "L" "P" "J" "E" "D"
#-------------------------------------------
#> set.seed(42); sample(LETTERS, 5)
#[1] "X" "Z" "G" "T" "O"
#> set.seed(42); sample(LETTERS, 5)
#[1] "X" "Z" "G" "T" "O"
#-------------------------------------------




library(rpart) #classificationtree
library(adabag) #bagging
library(randomForest) #randomForest
library(nnet) #logistic regression


# 01. ############################################################################################################################
####model fitting 'rpart' + assessing performance on the same data
fit.r <- rpart(assigned.labels ~., data=b.data); fit.r
pred <- predict(fit.r,type="class", newdata = b.data)
tab <- table(b.data$assigned.labels, pred); tab
sum(diag(tab))/sum(tab) # accurracy: 0.905 -> 0.913 
plot(as.party(fit.r))
varImp(fit.r) # which variable is important? it gives the variable importance plot for random forests

####model fitting 'logistic regression'
fitlog <- multinom(assigned.labels ~., data=b.data); fitlog 
pred <- predict(fitlog, type='class', newdata = b.data)
tab <- table(b.data$assigned.labels, pred); tab
sum(diag(tab))/sum(tab) # accurracy: 1.00 -> 1.00
varImp(fitlog) 

####model fitting 'bagging'
fitbag <- bagging(assigned.labels ~., data=b.data)
pred <- predict(fitbag,type="class", newdata = b.data)
tab <- pred$confusion; tab
sum(diag(tab))/sum(tab) # accurracy:0.920 -> 0.927
#plot(fitbag)
#varImp(fitbag) # does not work..

####model fitting 'randomForest'
fitrf <- randomForest(assigned.labels ~., data=b.data); fitrf
pred <- predict(fitrf,type="class", newdata = b.data)
tab <- table(b.data$assigned.labels, pred); tab
sum(diag(tab))/sum(tab) # accurracy: NA -> 0.913 
plot(fitrf)
varImp(fitrf) 





# comparing classifiers using validation######################################################################################
####Assessing performance: building training and test data
#splitting
#N = nrow(b.data)
#ind.train = sort(sample(1:N, size = floor(N*0.50)))
#ind.not = setdiff(1:N, ind.train)
#ind.valid = sort(sample(ind.not, size = length(ind.not)/2))
#ind.test = sort(setdiff(ind.not, ind.valid))

# 1> general
N <- nrow(b.data)
ind.train <- sample(1:N,size=0.50*N,replace=FALSE)
ind.train <- sort(ind.train)
ind.valid <- sample(setdiff(1:N,ind.train),size=0.25*N)
ind.valid <- sort(ind.valid)
ind.test <- setdiff(1:N,union(ind.train,ind.valid)) # check 'union()'function!

####model fitting 'rpart' + assessing performance on the validate data
fit.r <- rpart(assigned.labels ~., data=b.data, subset = ind.train); fit.r
pred <- predict(fit.r, type = 'class', newdata = b.data); pred
tab <- table(b.data$assigned.labels[ind.valid], pred[ind.valid]); tab
sum(diag(tab))/sum(tab)  # accuracy: 0.896

####model fitting 'bagging'+ assessing performance on the validate data
fitbag <- bagging(assigned.labels ~., data=b.data[ind.train, ])
pred <- predict(fitbag,type="class", newdata = b.data[ind.valid, ])
tab <- pred$confusion; tab
sum(diag(tab))/sum(tab) # accurracy: 0.896

####model fitting 'randomForest'+ assessing performance on the validate data
fitrf <- randomForest(assigned.labels ~., data=b.data, subset = ind.train); fitrf
pred <- predict(fit.r,type="class", newdata = b.data); pred
tab <- table(b.data$assigned.labels[ind.valid], pred[ind.valid]); tab
sum(diag(tab))/sum(tab) # accurracy: 0.896

####model fitting 'logistic regression'+ assessing performance on the validate data
fitlog <- multinom(assigned.labels ~., data=b.data, subset = ind.train); fitlog 
pred <- predict(fitlog, type='class', newdata = b.data); pred
tab <- table(b.data$assigned.labels[ind.valid], pred[ind.valid]); tab
sum(diag(tab))/sum(tab) # accurracy: 0.858


#-----------------------------------------------------------------------------------------------------------------------------
#As can be seen from above, bagging shows the best performance here, thus we try bagging on the test data to get an accurate 
#assessment of its performance.
#-----------------------------------------------------------------------------------------------------------------------------
####model fitting 'bagging'+ assessing performance on the test data
pred <- predict(fitbag,type="class", newdata = b.data[ind.test, ])
tab <- pred$confusion; tab
sum(diag(tab))/sum(tab) # accurracy: 0.906





# 2> bootstrapping
N <- nrow(b.data)
ind.train <- sample(1:N,replace=TRUE);ind.train #bootstrapping...new data = training data!!
ind.train <- sort(ind.train);ind.train
ind.test <- setdiff(1:N,ind.train);ind.test # values bootstrapping is missing..


####model fitting 'rpart' + assessing performance on the test data
fit.r <- rpart(assigned.labels ~., data=b.data, subset = ind.train)
pred <- predict(fit.r,type="class",newdata=b.data)
pred[ind.test]
tab <- table(b.data$assigned.labels[ind.test],pred[ind.test]); tab
sum(diag(tab))/sum(tab)

####model fitting 'bagging'+ assessing performance on the test data
fitbag <- bagging(assigned.labels ~., data=b.data[ind.train, ])
pred <- predict(fitbag,type="class", newdata = b.data[ind.test, ])
tab <- pred$confusion; tab
sum(diag(tab))/sum(tab) 

####model fitting 'randomForest'+ assessing performance on the test data
fitrf <- randomForest(assigned.labels ~., data=b.data, subset = ind.train)
pred <- predict(fitrf,type="class", newdata = b.data); pred
tab <- table(b.data$assigned.labels[ind.test], pred[ind.test]); tab
sum(diag(tab))/sum(tab) 

####model fitting 'logistic regression'+ assessing performance on the test data
fitlog <- multinom(assigned.labels ~., data=b.data, subset = ind.train) 
pred <- predict(fitlog, type='class', newdata = b.data); pred
tab <- table(b.data$assigned.labels[ind.test], pred[ind.test]); tab
sum(diag(tab))/sum(tab) 






# 3> k-fold cross validation
#聽First, let's assign the observations to folds.
K <- 10 #k fold cross validation
#if, K <- N #Leave one Out cross validation..this methods is quite unstable,,

folds <- rep(1:K,ceiling(N/K)); folds # 1:10 x 43  
folds <- sample(folds); folds
folds <- folds[1:N]; folds # drop the last five values.....
table(folds) # the size of each fold is 42 or 43...


#聽Set up res to store results
res<-matrix(NA,K,1) #nrow=k, ncol=1
# We will need to drop each fold in turn.
iterlim <- K




####model fitting 'rpart' + assessing performance on the test data
for (iter in 1:iterlim)
{
  ind.train <- (1:N)[!(folds==iter)] #drop any obv that belongs to 1st fold?????,and call remaining values of training data..??
  ind.test <- setdiff(1:N,ind.train) #abandoned data...(fold)
  
  #聽Fit a classifier to only the training data
  fit.r <- rpart(assigned.labels ~., data=b.data, subset = ind.train)
  
  # Classify for ALL of the observations
  pred.r <- predict(fit.r,type="class",newdata=b.data)
  
  # Look at table for the validation data only (rows=truth, cols=prediction)
  tab.r <- table(b.data$assigned.labels[ind.test],pred.r[ind.test])
  
  #聽Let's see how well we did on the fold that we dropped
  #聽res[iter,1] <- sum(diag(tab.r))/sum(tab.r)
  res[iter,1]<-sum(pred.r[ind.test]==b.data$assigned.labels[ind.test])/length(ind.test)
}; res

colnames(res)<-c("test")
apply(res,2,summary)




####model fitting 'bagging'+ assessing performance on the test data
for (iter in 1:iterlim)
{
  ind.train <- (1:N)[!(folds==iter)] #drop any obv that belongs to 1st fold?????,and call remaining values of training data..??
  ind.test <- setdiff(1:N,ind.train) #abandoned data...(fold)
  
  #聽Fit a classifier to only the training data
  fitbag <- bagging(assigned.labels ~., data=b.data[ind.train, ])
  
  # Classify for ALL of the observations
  predbag <- predict(fitbag,type="class",newdata=b.data[ind.test, ])
  
  #聽Let's see how well we did on the fold that we dropped
  #聽res[iter,1] <- sum(diag(tab.r))/sum(tab.r)
  res[iter,1]<-sum(predbag[ind.test]==b.data$assigned.labels[ind.test])/length(ind.test)
}; res







####model fitting 'randomForest'+ assessing performance on the test data
for (iter in 1:iterlim)
{
  ind.train <- (1:N)[!(folds==iter)] #drop any obv that belongs to 1st fold?????,and call remaining values of training data..??
  ind.test <- setdiff(1:N,ind.train) #abandoned data...(fold)
  
  #聽Fit a classifier to only the training data
  fitrf <- randomForest(assigned.labels ~., data=b.data, subset = ind.train)
  
  # Classify for ALL of the observations
  predrf <- predict(fitrf,type="class",newdata=b.data)
  
  # Look at table for the validation data only (rows=truth, cols=prediction)
  tabrf <- table(b.data$assigned.labels[ind.test],predrf[ind.test])
  
  #聽Let's see how well we did on the fold that we dropped
  #聽res[iter,1] <- sum(diag(tab.r))/sum(tab.r)
  res[iter,1]<-sum(predrf[ind.test]==b.data$assigned.labels[ind.test])/length(ind.test)
}; res

colnames(res)<-c("test")
apply(res,2,summary)




####model fitting 'logistic regression'+ assessing performance on the test data
for (iter in 1:iterlim)
{
  ind.train <- (1:N)[!(folds==iter)] #drop any obv that belongs to 1st fold?????,and call remaining values of training data..??
  ind.test <- setdiff(1:N,ind.train) #abandoned data...(fold)
  
  #聽Fit a classifier to only the training data
  fitlog <- multinom(assigned.labels ~., data=b.data, subset = ind.train)
  
  # Classify for ALL of the observations
  predlog <- predict(fitlog,type="class",newdata=b.data)
  
  # Look at table for the validation data only (rows=truth, cols=prediction)
  tablog <- table(b.data$assigned.labels[ind.test],predlog[ind.test])
  
  #聽Let's see how well we did on the fold that we dropped
  #聽res[iter,1] <- sum(diag(tab.r))/sum(tab.r)
  res[iter,1]<-sum(predlog[ind.test]==b.data$assigned.labels[ind.test])/length(ind.test)
}; res

colnames(res)<-c("test")
apply(res,2,summary)






#-------------
[,1]
[1,] 0.8888889 # when drop the first fold, we get 88% correct classification.. 
[2,] 0.9444444 # when drop the second fold, we get..
[3,] 0.9444444
[4,] 0.9444444
[5,] 0.8823529
[6,] 0.8333333
[7,] 0.9444444
[8,] 0.9444444
[9,] 0.8235294
[10,] 0.7222222
#--------------































