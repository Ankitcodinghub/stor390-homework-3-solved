# stor390-homework-3-solved
**TO GET THIS SOLUTION VISIT:** [STOR390 Homework 3 Solved](https://www.ankitcodinghub.com/product/stor-390-homework-3-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;122160&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;STOR390 Homework 3 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
In this homework, we will discuss support vector machines and tree-based methods. I will begin by simulating some data for you to use with SVM.

library(e1071) set.seed(1)

x=matrix(rnorm(200*2),ncol=2)

x[1:100,]=x[1:100,]+2 x[101:150,]=x[101:150,]-2 y=c(rep(1,150),rep(2,50)) dat=data.frame(x=x,y=as.factor(y)) plot(x, col=y)

x[,1]

0.1

Quite clearly, the above data is not linearly separable. Create a training-testing partition with 100 random observations in the training partition. Fit an svm on this training data using the radial kernel, and tuning parameters 𝛾=1, cost =1. Plot the svm on the training data.

set.seed(1)

trainIndices = sample(1:nrow(dat), 100) trainData = dat[trainIndices, ] testData = dat[-trainIndices, ]

svmmod = svm(y ~ x.1 + x.2, data = trainData, kernel = “radial”, gamma = 1, cost = 1) plot(svmmod, trainData)

SVM classification plot

x.2

0.2

Notice that the above decision boundary is decidedly non-linear. It seems to perform reasonably well, but there are indeed some misclassifications. Let’s see if increasing the cost helps our classification error rate. Refit the svm with the radial kernel, 𝛾=1, and a cost of 10000. Plot this svm on the training data.

svmmod1 = svm(y ~ x.1 + x.2, data = trainData, kernel = “radial”, gamma = 1, cost = 10000) plot(svmmod1, trainData)

SVM classification plot

x.2

0.3

It would appear that we are better capturing the training data, but comment on the dangers (if any exist), of such a model.

SVM does a great job of capturing the extremely complex decision boundary present in the features, however, the dangers of utilizing an extremely high cost come from overfitting on the training data, as characterized by a model’s excessive complexity. This excessive complexity arises not from the model fitting to the underlying pattern in the training data, but from it also fitting the noise within that data. These models tend to possess a characteristic of poorly performing on unseen test data.

0.4

Create a confusion matrix by using this svm to predict on the current testing partition. Comment on the confusion matrix. Is there any disparity in our classification results?

table(true=dat[-trainIndices,”y”], pred=predict(svmmod1, newdata=dat[-trainIndices,]))

## pred

## true 1 2

## 1 67 12

## 2 2 19

Addressing overfitting requires careful tuning of the model’s parameters, such as the cost and 𝛾, and the introduction of regularization techniques. Additionally, using cross-validation, as mentioned in my previous homework, to select the parameters can help ensure that the model generalizes well to new data. Balancing the dataset or adjusting the decision threshold based on the costs of different types of misclassifications could also help mitigate the disparity in classification accuracy between the classes.

0.5

Is this disparity because of imbalance in the training/testing partition? Find the proportion of class 2 in your training partition and see if it is broadly representative of the underlying 25% of class 2 in the data as a whole.

table(true=dat[-trainIndices,”y”], pred=predict(svmmod1, newdata=dat[-trainIndices,]))

## pred

## true 1 2

## 1 67 12 ## 2 2 19

classProportions = table(dat[trainIndices, “y”]) / length(trainIndices)

proportionOfClass2 = classProportions[“2”]

print(paste0(“The proportion of class 2 is “, proportionOfClass2 * 100, “%.”))

## [1] “The proportion of class 2 is 29%.”

A proportion of 29% for class 2 indicates a slight imbalance but is relatively close to the underlying 25% of class 2 in the data as a whole.

0.6

Let’s try and balance the above to solutions via cross-validation. Using the tune function, pass in the training data, and a list of the following cost and 𝛾 values: {0.1, 1, 10, 100, 1000} and {0.5, 1,2,3,4}. Save the output of this function in a variable called tune.out.

set.seed(1)

tuneGrid &lt;- expand.grid(cost = c(0.1, 1, 10, 100, 1000), gamma = c(0.5, 1, 2, 3, 4)) tune.out &lt;- tune(svm, y ~ x.1 + x.2, data = trainData,

kernel = “radial”, ranges = tuneGrid, scale = FALSE)

print(tune.out)

##

## Parameter tuning of ‘svm’:

##

## – sampling method: 10-fold cross validation

##

## – best parameters:

## cost gamma

## 1 0.5

##

## – best performance: 0.15

plot(tune.out)

Performance of ‘svm’

cost

I will take tune.out and use the best model according to error rate to test on our data. I will report a confusion matrix corresponding to the 100 predictions. table(true=dat[-trainIndices,”y”], pred=predict(tune.out$best.model, newdata=dat[-trainIndices,]))

0.7

Comment on the confusion matrix. How have we improved upon the model in question 2 and what qualifications are still necessary for this improved model.

True Positives (TP): 73 (correctly predicted as class 1) False Negatives (FN): 6 (class 1 incorrectly predicted as class 2) False Positives (FP): 3 (class 2 incorrectly predicted as class 1) True Negatives (TN): 18 (correctly predicted as class 2) The confusion matrix indicates a higher accuracy, with the model correctly classifying the majority of the data. However, there are still a few misclassifications, as seen in the presence of both false positives and false negatives, though they are relatively low in number. This outcome demonstrates that the tuned model, with the best combination of cost and 𝛾 parameters identified through cross-validation, performs well on the test set. This is because the model effectively balances the trade-off between sensitivity (true positive rate) and specificity (true negative rate).

1

Let’s turn now to decision trees.

library(kmed) data(heart) library(tree)

1.1

The response variable is currently a categorical variable with four levels. Convert heart disease into binary categorical variable. Then, ensure that it is properly stored as a factor.

heart$class_binary &lt;- ifelse(heart$class &gt; 0, 1, 0) heart$class_binary &lt;- as.factor(heart$class_binary) levels(heart$class_binary) &lt;- c(“absent”, “present”) str(heart$class_binary)

## Factor w/ 2 levels “absent”,”present”: 1 2 2 1 1 1 2 1 2 2 …

1.2

Train a classification tree on a 240 observation training subset (using the seed I have set for you). Plot the tree.

set.seed(101)

trainIndices &lt;- sample(1:nrow(heart), 240) trainData &lt;- heart[trainIndices,]

treeMod &lt;- tree(class_binary ~ . – class, data = trainData) plot(treeMod) text(treeMod, pretty = 0)

1.3

Use the trained model to classify the remaining testing points. Create a confusion matrix to evaluate performance. Report the classification error rate.

testData &lt;- heart[-trainIndices, ]

predictions &lt;- predict(treeMod, newdata = testData, type = “class”) confMatrix &lt;- table(Predicted = predictions, Actual = testData$class_binary) print(confMatrix)

## Actual

## Predicted absent present

## absent 28 3 ## present 8 18

errorRate &lt;- 1 – sum(diag(confMatrix)) / nrow(testData)

print(paste0(“The classification error rate is “, round(errorRate, 4)* 100, “%”))

## [1] “The classification error rate is 19.3%”

1.4

Above we have a fully grown (bushy) tree. Now, cross validate it using the cv.tree command. Specify cross validation to be done according to the misclassification rate. Choose an ideal number of splits, and plot this tree. Finally, use this pruned tree to test on the testing set. Report a confusion matrix and the misclassification rate.

cvTree &lt;- cv.tree(treeMod, FUN=prune.misclass)

plot(cvTree$size, cvTree$dev, type=’b’, xlab=”Number of Terminal Nodes”, ylab=

“Misclassification Rate”)

Number of Terminal Nodes

optimalSize &lt;- cvTree$size[which.min(cvTree$dev)] prunedTree &lt;- prune.misclass(treeMod, best=optimalSize)

plot(prunedTree) text(prunedTree, pretty=0)

prunedPredictions &lt;- predict(prunedTree, newdata = testData, type = “class”) confMatrixPruned &lt;- table(Predicted = prunedPredictions, Actual = testData$class_binary) print(confMatrixPruned)

## Actual

## Predicted absent present

## absent 28 9 ## present 8 12

errorRatePruned &lt;- 1 – sum(diag(confMatrixPruned)) / nrow(testData)

print(paste0(“The classification error rate of the pruned tree is “, round(errorRatePruned, 4) * 100, “% ## [1] “The classification error rate of the pruned tree is 29.82%”

1.5

Discuss the trade-off in accuracy and interpretability in pruning the above tree.

In contrast, the pruned tree, optimized through cross-validation to minimize the misclassification rate, demonstrated a slightly higher error rate of 29.82%. This increase in the error rate following pruning indicates a reduction in accuracy. The confusion matrix revealed that while the pruned tree continued to correctly identify the absence of disease, its ability to correctly predict the presence of heart disease diminished slightly. This result suggests that pruning, by simplifying the model, might have removed some of the nuance and detail necessary for capturing more complex patterns associated with the disease’s presence in the data.

Despite this reduction in accuracy, the pruned tree offers a significant advantage in terms of interpretability. By reducing the tree to a smaller number of terminal nodes (2 nodes), pruning makes the model’s logic more straightforward and easier for users to understand. This simplification allows non-data scientists to grasp the key factors influencing the model’s predictions, allowing for easier validation of the model’s decision rules in clinical settings.

1.6

Discuss the ways a decision tree could manifest algorithmic bias.
