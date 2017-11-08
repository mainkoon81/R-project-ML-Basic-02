# R-project-ML-Basic(02)

## Multivariate_Analysis / Machine Learning playground with R

### [Contents] 

__Lab-04.__ Classification  
  - package: rpart, partykit, party, adabag, randomForest 
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

__Data:__ On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships. Our data were collected recording passangers' information including "Name","PClass(socio-economic status)","Age","Sex","Survived or not".
Extra details of the votes can be found at (https://www.kaggle.com/c/titanic/data/). 

__Story:__ One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class. In this challenge, we want to predict what sorts of people were likely to survive the tragedy.

#### *|Implementing Classification Trees|*

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
















