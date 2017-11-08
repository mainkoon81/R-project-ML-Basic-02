# R-project-ML-Basic(02)

## Multivariate_Analysis / Machine Learning playground with R

### [Contents] 

__Lab-04.__ Classification  
  - package: rpart, party, adabag, randomForest 
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
## p is prob of each categorical variable entry##

##Entropy (0 < var < log(k))##
entropy = function(p) {
  p < -p[p>0] 
  sum(-p*log(p))
}

##Gini (0 < var < 1-(1/k)##
gini = function(p) {
  1-sum(p^2)
}
```

__Data:__ The Dat















