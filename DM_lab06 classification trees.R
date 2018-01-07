
# lab06 classification trees

# entropy and gini ==> var of categorical variables..
#if var=0, monopolization, if var=log(k), evenly distributed..go together..all entries have equal prob..

#for entropy, 0 < var < log(k) k is the size of the categorical variable.
#for gini, 0 < var < 1-(1/k)

# p is prob of each categorical variable entry...
# Entropy
entropy = function(p) {
  p < -p[p>0] ####why?????
  sum(-p*log(p))
}

# Gini
gini = function(p) {
  1-sum(p^2)
}

#we use these var to find the threshold that differenciates the predictors...

#------------------------------------------------------------------------------------------------------------------------------

titanic = read.table('C:/Users/Minkun/Desktop/r_practice/+++++++++++School R/LAB_data mining/data/titanic.txt', header=T, sep='\t')
summary(titanic)
names(titanic)
head(titanic, 10)
is.factor(titanic$Name)
is.factor(titanic$PClass)
is.factor(titanic$Sex)
is.factor(titanic$Survived) #not a factor...but how can it be used in the formular ?????

library(rpart)

#fit the model
fit.r <- rpart(Survived ~ Sex+Age+PClass, data = titanic); fit.r
summary(fit.r)
plot(fit.r)
text(fit.r, use.n = T, cex=0.8, xpd=T, col='red')# after plot!!! cex means size of letter..


library(partykit)
plot(as.party(fit.r))


library(party)
fit.c <- ctree(Survived ~ Sex+Age+PClass, data = titanic); fit.c
plot(fit.c)
plot(Survived ~ Sex+Age+PClass, data = titanic)



#When choosing splits in the tree, the distinction between nominal and ordinal categorical variables is important. 
#Passenger class is clearly a categorical variable with ordering 1st, 2nd and 3rd. 
#We can check if it is stored as an ordinal or a nominal variable

is.ordered(titanic$PClass) 
#If it isn't ordered, then we can make it so
titanic$PClass <- ordered(titanic$PClass, c('1st', '2nd', '3rd')) # isn't it 'as.ordered' ????? 

#In this case, the categorical variable will be split into groups of those variables less than some value and greater than or 
#equal to the value. Does this change the results for the titanic data? 

#The rpart() function has some default settings that you may wish to change. 
#In particular, the tree growing method doesn't split any node with less than 20 observations (controlled by minsplit). 
#It also doesn't ???t very complex looking trees (controlled by cp) and there many other options set by default. 
#Let's look at when we change some of them...

fit.r <- rpart(Survived ~ Sex+Age+PClass, data = titanic, cp=0.0005); fit.r #cp means depth????
plot(fit.r)
text(fit.r, use.n = T, cex=0.5, xpd=T, col='red')# after plot!!! cex means size of letter..
#--------------------------------------------------------------------------------------------------------------------------------

# SIMULATION (diagonal division)

X <- matrix(runif(1000), 500, 2); X
Y <- (X[,1] > X[,2]); Y
L <- as.factor(Y+1); L

plot(X, col=Y+1, pch=Y+1)
abline(a=0, b=1)

fit.r <- rpart(L ~ X); fit.r

pred.r <- predict(fit.r, type = 'class')
table(Y, pred.r)






























