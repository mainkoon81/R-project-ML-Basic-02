# lab 07. bagging + random forest

#Bagging is a method that uses bootstrapping to increase the "stability" of a classiﬁcation method.
#Random forests extend this further to make the trees more "diverse."
#In this laboratory session, we compare a single classiﬁcation tree to bagging and random forests.


#Set the seed of the random number generator in R so that the results are repeatable...
set.seed(123)

library(rpart) #a single classification trees
library(adabag) #a bagging
library(randomForest) #a random Forest

load("C:/Users/Minkun/Desktop/r_practice/+++++++++++School R/LAB_data mining/data/GermanCredit.Rdata")
summary(GermanCredit)
names(GermanCredit)
head(GermanCredit)
is.factor(GermanCredit$Class)
attach(GermanCredit)
Class

#Fit a classification tree and assess performance on the same data.
fit.r <- rpart(Class ~ ., data = GermanCredit); fit.r
plot(as.party(fit.r)) #fuck...
pred = predict(fit.r, type = 'class', newdata = GermanCredit); pred
tab = table(GermanCredit$Class, pred); tab 
sum(diag(tab))/sum(tab) # aweful....0.809
varImp(fit.r) # which variable is important? it gives the variable importance plot for random forests

#Complete bagging and assess performance on the same data.
fitbag <- bagging(Class ~ ., data = GermanCredit)  #fuck...
#plot(fitbag) #does not work...
pred = predict(fitbag, type = 'class', newdata = GermanCredit); pred 
tab = pred$confusion #it already has a table...
sum(diag(tab))/sum(tab) #aweful...0.863
#varImp(fitbag) #does not work...

#Fit a random Forest and assess performance on the same data.
fitrf <- randomForest(Class ~ ., data = GermanCredit); fitrf
plot(fitrf) #not what i want..
pred = predict(fitrf, type = 'class', newdata = GermanCredit); pred
tab = table(GermanCredit$Class, pred); tab
sum(diag(tab))/sum(tab) # wowwww! overfitting?...100%
varImp(fitrf) # which variable is important? it returns scaled results in range 0-100.

#------------------------------------------------------------------------------------------------------------------------------
#Which method looks best? Do you suspect one method might be "over ﬁtting"? 
#There is a problem with our analysis above (as we have seen already in the course).
#We have tested how well our classiﬁcation method worked on the data that was used to build the classiﬁer. 
#Because the data was used to build the classiﬁer, we are over estimating our performance 
#(of course we should predict this data well).
#A better approach would be to ﬁt to one data set, choose method on a second and to test on third. 
#Because we don’t have three German credit data sets, we will split the data into three to complete this task
#------------------------------------------------------------------------------------------------------------------------------

# Split the data into three sets
N = nrow(GermanCredit)
ind.train = sort(sample(1:N, size = floor(N*0.70)))
ind.not = setdiff(1:N, ind.train)
ind.valid = sort(sample(ind.not, size = length(ind.not)/2))
ind.test = sort(setdiff(ind.not, ind.valid))

#compare..this splitting..0.5 / 0.25 /0.25 ####################################################################################
#N <- nrow(wine)
#ind.train <- sample(1:N,size=0.50*N,replace=FALSE)
#ind.train <- sort(ind.train)
#ind.valid <- sample(setdiff(1:N,ind.train),size=0.25*N)
#ind.valid <- sort(ind.valid)
#ind.test <- setdiff(1:N,union(ind.train,ind.valid)) # check 'union()'function!################################################

#Fit to training data and asses performance on validation 
# rpart
fit.r <- rpart(Class ~ ., data = GermanCredit, subset = ind.train); fit.r
pred <- predict(fit.r, type = 'class', newdata = GermanCredit); pred

tab <- table(GermanCredit$Class[ind.valid], pred[ind.valid]); tab # why original data [] ? 
sum(diag(tab))/sum(tab) #aweful...0.75

# bagging
fitbag <- bagging(Class ~ ., data = GermanCredit[ind.train, ]) #fuck # note! you select col!!

pred <- predict(fitbag, type = 'class', newdata = GermanCredit[ind.valid, ]); pred #fuck!
tab <- pred$confusion; tab
sum(diag(tab))/sum(tab) #aweful...0.76

# randomForest
fitrf <- randomForest(Class ~ ., data = GermanCredit, subset = ind.train); fitrf
pred <- predict(fitrf, type = 'class', newdata = GermanCredit); pred

tab <- table(GermanCredit$Class[ind.valid], pred[ind.valid]); tab
sum(diag(tab))/sum(tab) #better...0.78

#-----------------------------------------------------------------------------------------------------------------------------
#In this case, bagging reduces the error on the validation data. Random forests may be a bit better. So, we try random forests 
#on the test data to get an accurate assessment of its performance.
#-----------------------------------------------------------------------------------------------------------------------------

tab.test <- table(GermanCredit$Class[ind.test], pred[ind.test]); tab.test
sum(diag(tab))/sum(tab) #better...0.78



