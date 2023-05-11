# classifying_rhotics
A tutorial on how to automatically extract rhotic measurements in Praat and rank measurements according to their predictive power with random forest. 

Measuring rhotics in diverse languages has long been a challenge for (socio)phoneticians given the relative difficulty in identifying and classifying the acoustic signal(s) associated with their realizations. If you’d like to measure rhotics and you’re visiting this tutorial, you have likely asked yourself the existential question: “is it more appropriate to measure duration, F3-F2 difference, F5-F4 difference, COG, F1, F2, F3, etc”? The list goes on! Or, “if I choose one or more of these dependent variables, how do I know which measure is a better fit for my data without running multiple mixed effects models and finding out retrospectively”? The literature on measuring rhotics, even in normative varieties of languages, is complex and, oftentimes, conflicting. Due to the continuous and gradient nature of rhotics ([Bradley, 2019](https://www.researchgate.net/profile/Travis-Bradley-4/publication/347389020_Spanish_rhotics_and_the_phonetics-phonology_interface/links/63928c38e42faa7e75aa9f8b/Spanish-rhotics-and-the-phonetics-phonology-interface.pdf)), employing the acoustic measures used in previous literature may not be sufficient or appropriate for your unique sample of speakers. 

This issue is further complicated by the potential multicollinearity of independent variables. In a linguistic dataset, we might not be able to assume that the extralinguistic effect of a speaker’s gender or sexual orientation is independent from the linguistic effect of the rhotic’s preceding or following sound, for example. In data of this type, a classification method like linear discriminant analysis would not provide reliable results that give insight into the classification or relative importance of the independent variables at play specifically because they experience a collinear relationship ([Büyüköztürk & Çokluk-Bökeoğlu, 2008](https://ejer.com.tr/wp-content/uploads/2021/01/ejer_2008_issue_33.pdf#page=76)). However, this disparity can be mitigated by a nonparametric classification method—one that does not assume independence among variables. 

Random forests are highly nonparametric and, as explained by [Gardner (2023) in Random Forests: The Basics](https://lingmethodshub.github.io/content/R/lvc_r/090_lvcr.html), are useful in ranking independent variables that are collinear. This tutorial ranks several continuous and categorical independent variables (i.e. F3-F2 difference, duration, COG, social factors, etc.) based on their effect on a single categorical dependent variable (allophone type: [r], [ɾ], or elision). Random forests are particularly useful because both the independent and dependent variables can be either categorical or continuous. For a global overview of random forest, I defer to Gardner (2023). This tutorial intends to serve the dual purpose of aiding in both the acoustic and statistical analysis of rhotics with materials for Praat ([Boersma & Weenink, 2023](https://www.fon.hum.uva.nl/praat/)) scripting and the optimization of your random forest classification model. 

To begin, the [Praat script](AnalyzingRhotics_PraatScript) adapts and expands upon the scripting tutorials from [Stanley and Lipani’s (2019) Praat Scripting Workshop Series](https://joeystanley.com/downloads/190911-intro_to_Praat). The script comprises several commonly used measures to analyze rhotics. It is designed to work with data aligned by the Montreal Forced Aligner ([McAuliffe, 2017](https://www.researchgate.net/profile/Morgan-Sonderegger/publication/319185277_Montreal_Forced_Aligner_Trainable_Text-Speech_Alignment_Using_Kaldi/links/59b84d450f7e9bc4ca393755/Montreal-Forced-Aligner-Trainable-Text-Speech-Alignment-Using-Kaldi.pdf)), but could be easily adapted to support manually aligned data. The script identifies the target phoneme, the word associated with that phoneme, and the preceding and following segments. It then measures the duration of the target phoneme, duration of the preceding phoneme to account for speech rate (for more information, see [Balukas & Koops, 2015](https://journals.sagepub.com/doi/pdf/10.1177/1367006913516035?casa_token=OIM6jhe9_akAAAAA:NAYt0LFQJ3vXvh05MySzlLL5D2UjmBhUwjbkkjkGRmcdmgUHk_5UNbjvSHrVF7bqgJZK2OSbuPnCM08)), the midpoints of F1, F2, F3, and F4, F3-F2 difference, center of gravity, standard deviation, skewness, and kurtosis. This list of rhotic measures, however, is by no means exhaustive; for further review, I recommend referencing [Amengual (2016)](https://brill.com/view/journals/hlj/13/2/article-p88_2.xml), [Bradley (2019)](https://www.researchgate.net/profile/Travis-Bradley-4/publication/347389020_Spanish_rhotics_and_the_phonetics-phonology_interface/links/63928c38e42faa7e75aa9f8b/Spanish-rhotics-and-the-phonetics-phonology-interface.pdf), [Campbell et al. (2018)](https://www.tandfonline.com/doi/abs/10.1080/17549507.2017.1359334?role=button&needAccess=true&journalCode=iasl20), [Colantoni (2006)](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=58178f615551e2e02af16f08d6de2db204f587fe), [Jongman et al. (2000)](https://kuscholarworks.ku.edu/bitstream/handle/1808/13393/Jongman%20et%20al.%20JASA2000.pdf?sequence=1&isAllowed=y), and [Zhou et al. (2008)](https://bpb-us-e1.wpmucdn.com/blog.umd.edu/dist/c/619/files/2019/11/journal_zhou_etal_jasa_08.pdf) to name a few.

Once you have successfully completed your acoustic analysis and exported your measures to a .csv file, we take on the statistical analysis. The script for the random forest in this tutorial was largely adapted from [Rai (2020)](https://www.taylorfrancis.com/chapters/edit/10.1201/9781351123303-2/supervised-machine-learning-application-example-using-random-forest-bharatendra-rai) and this researcher’s [Youtube tutorial](https://www.youtube.com/watch?v=dJclNIN-TPo). Before partitioning your data, it is essential to convert the strings to factors. This ensures that the categorical variables are listed as factors and that the continuous variables are numeric. Check the structure of the data to see if you have any missing values or discrepancies in this categorization. A categorical dependent variable that is not classified as a factor will prevent the random forest from running.

Begin by importing and familiarizing yourself with your data. 

```R
# Import and Get to Know Your Data
setwd("~/Desktop")
data <- read.csv("CodaRhotics.csv", stringsAsFactors = TRUE)
str(data)
```
Before partitioning, set the seed to make the analysis replicable, as this model uses random sampling (see [Gardner, 2023](https://lingmethodshub.github.io/content/R/lvc_r/090_lvcr.html)). Partitioning 100% of the data into two sets of 70% and 30% allows us to train the model. Doing this allows us to see how well we trained the model—that is, how well 70% of the data is able to predict the outcome of the other 30%. It is also common to see model partitions of 80% and 20%.  

```R
# Separate Data into Training and Testing Datasets  
set.seed(123)
ind <- sample(2,nrow(data), replace = TRUE, prob = c(0.7, 0.3))
train <- data[ind==1,]
test <- data[ind==2,]
```
Next, install and load…you guessed it!...random forest. Set the seed again for this model to make it replicable. Name the random forest model and include the categorical dependent variable—in this case, it is the type of rhotic (allophone) which has 3 levels—as a function of the independent variables you’d like to test. You can either select only specific independent variables to include in your model, or use a period before the comma to include all other variables. 

```R
# Use a period if You Want to Test All Variables 
rf <- randomForest(allophone~., data=train). 

# Create the Random Forest Model
install.packages("randomForest")
library(randomForest)
set.seed(222)
rf <- randomForest(allophone~performance+interlocuterGender+interlocuterSexOrientation+stress+manner_precedingSeg+precedingSegDuration+manner_followingSeg+duration+F1+F2+F3+F4+F3.F2_distance+cog+stdev+skewness+kurtosis, data=train)
print(rf)
```
Print the model and examine the results. 
```R
Call:
 randomForest(formula = allophone ~ performance + interlocuterGender + interlocuterSexOrientation + stress + manner_precedingSeg +      precedingSegDuration + manner_followingSeg + duration + F1 + F2 + F3 + F4 + F3.F2_distance + cog + stdev + skewness + kurtosis, data = train) 
               Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 4

        OOB estimate of  error rate: 12.26%
Confusion matrix:
        elision  r   ɾ class.error
elision      34  0   0  0.00000000
r             0 18  28  0.60869565
ɾ             0 10 220  0.04347826
```
We are able to verify that, because the dependent variable is categorical, this is a classification model. The number of trees is set to 500, the default. Next, we see the number of variables tried at each split. This number represents the approximate square root of the number of variables included in the model. The out of bag (OOB) estimate of error rate indicates that this model is about 87% accurate. Lastly, the confusion matrix shows that the predictions are rather good at predicting the elision and [ɾ] classes (0% error and 4.4% error, respectively), but errors are higher for [r] (60.9%). The confusion matrix table can be interpreted as follows: elision was predicted as elision 34 times, trills [r] were predicted as elision zero times, and taps [ɾ] were predicted as elision zero times; elision was predicted as a trill zero times, trills were predicted as trills 18 times, and taps were predicted as trills 28 times; lastly, elision was predicted as a tap zero times, trills were predicted as taps 10 times, and taps were predicted as taps 220 times.

To further examine the prediction and confusion matrix of the trained dataset, install and load the Caret package. Name your first prediction and use the predict function to examine the confusion matrix of the random forest model you previously created within the training set. 
```R
# Prediction & Confusion Matrix on Train Data
library(caret)
p1 <- predict(rf, train)
confusionMatrix(p1, train$allophone)

Confusion Matrix and Statistics

          Reference
Prediction elision   r   ɾ
   elision      34   0   0
   r             0  46   0
   ɾ             0   0 230

Overall Statistics
                                     
               Accuracy : 1          
                 95% CI : (0.9882, 1)
    No Information Rate : 0.7419     
    P-Value [Acc > NIR] : < 2.2e-16  
                                     
                  Kappa : 1          
                                     
 Mcnemar's Test P-Value : NA         

Statistics by Class:

                     Class: elision Class: r Class: ɾ
Sensitivity                  1.0000   1.0000   1.0000
Specificity                  1.0000   1.0000   1.0000
Pos Pred Value               1.0000   1.0000   1.0000
Neg Pred Value               1.0000   1.0000   1.0000
Prevalence                   0.1097   0.1484   0.7419
Detection Rate               0.1097   0.1484   0.7419
Detection Prevalence         0.1097   0.1484   0.7419
Balanced Accuracy            1.0000   1.0000   1.0000
```
In the confusion matrix, we see that there were no misclassifications in any of the rhotic categories. To get the accuracy interval, calculate the total instances along the diagonal of the confusion matrix and divide by the number of tokens in the training interval. Since the confusion accurately predicted all allophones, the accuracy interval displays a score of 1, or 100%. The 95% confidence interval (CI) is also rather high, indicating 95% confidence between 98.82% and 100%. 

When comparing the previous two outputs of the same data, we see that there is a discrepancy between the accuracy of the first model (88%) and the second (100%). This is because the first model utilizes an OOB rate of error which predicts the error based on data that the model has not seen, whereas the second only uses data that the model has seen. 

Now, run the same code used previously on the test data and name the new model accordingly. 
```R
# Prediction & Confusion Matrix on Test Data
p2 <- predict(rf, test)
confusionMatrix(p2, test$allophone)

Confusion Matrix and Statistics

          Reference
Prediction elision  r  ɾ
   elision      14  0  0
   r             0  7  5
   ɾ             0 14 84

Overall Statistics
                                          
               Accuracy : 0.8468          
                 95% CI : (0.7711, 0.9052)
    No Information Rate : 0.7177          
    P-Value [Acc > NIR] : 0.0005576       
                                          
                  Kappa : 0.6204          
                                          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: elision Class: r Class: ɾ
Sensitivity                  1.0000  0.33333   0.9438
Specificity                  1.0000  0.95146   0.6000
Pos Pred Value               1.0000  0.58333   0.8571
Neg Pred Value               1.0000  0.87500   0.8077
Prevalence                   0.1129  0.16935   0.7177
Detection Rate               0.1129  0.05645   0.6774
Detection Prevalence         0.1129  0.09677   0.7903
Balanced Accuracy            1.0000  0.64239   0.7719
```
This model of the test data resulted in a decrease of accuracy (85.68%) and confidence interval (77.11% to 90.52%). However, we can consider the test data a more precise assessment of the random forest model because the model has been exposed to the test data. We can see that the test data’s accuracy mirrors the original model’s OOB estimate of error rate and subsequent accuracy rate for a similar reason. 

Next, we will visualize the OOB error by plotting the random forest. 

```R
# Error Rate of Random Forest
plot(rf)
```
![Random Forest Error Rate](https://user-images.githubusercontent.com/133238472/237765339-f1843fc3-2210-452c-b008-970b1fdfad1b.png)

In the plot, the OOB error decreases sharply and then levels out. This shows us that the error does not improve after about 100 trees. 

The first step of tuning the model is only necessary if you did not include every independent variable in your model from the dataset. Since this is the case for my dataset, I will remove the variables not included in the model from my train data. Next, we use the tuneRF function to determine how to improve the model. Using the train data, we remove the fifth input variable—the dependent variable (allophone/rhotic type)—and then include the fifth variable as the response variable. By assigning a number to stepFactor, mtry either improves or worsens. Begin by assigning it to “0.5”. After following the corresponding steps, if the output doesn’t complete an effective search, the mtry value can be subsequently increased or decreased. According to [RStudio Team (2023)](https://cran.r-project.org/), “plot” decides “whether to plot the OOB error as a function of mtry”. We will decide to do this by writing “TRUE”. Because we saw previously that the error does not improve after about 100 trees, we will try the model with 100 instead of the standard 500. By using the trace function, it “allows you to to insert debugging code”. The improve function tells us that “the (relative) improvement in OOB error must be by this much for the search to continue”. We will assign a small number to this function. 
```R
# Remove variables not included in the model from train data
train <- data[ , ! names(train) %in% c("file", "performance_interlocuterGender_SexOrientation","interlocuterGender.SexOrientation","word","precedingSeg","followingSeg","time")]

# Tune the Model
t <- tuneRF(train[,-5], train[,5], stepFactor = 0.5, plot = TRUE, ntreeTry = 100, trace = TRUE, improve = .05)
```
![mtry](https://user-images.githubusercontent.com/133238472/237772102-14607853-df4f-4067-bc31-2b5f7c44d2a6.png)

The plot shows us that the OOB error is lowest when mtry is at 16. With this information extracted, we will rerun the random forest model and add a few more details. The number of trees will be set to 100, mtry will be set to 16, the importance function will ask the model for information about variable importance measures, and proximity will be used. 

Now, you can compare the earlier model with the ~tuned up~ model and, ideally, your error rate will have improved. 
```R
Call:
 randomForest(formula = allophone ~ performance + interlocuterGender + interlocuterSexOrientation + stress + manner_precedingSeg +      precedingSegDuration + manner_followingSeg + duration + F1 + F2 + F3 + F4 + F3.F2_distance + cog + stdev + skewness + kurtosis, data = train, ntree = 100, mtry = 16, importance = TRUE, proximity = TRUE) 
               Type of random forest: classification
                     Number of trees: 100
No. of variables tried at each split: 16

        OOB estimate of  error rate: 11.29%
Confusion matrix:
        elision  r   ɾ class.error
elision      34  0   0  0.00000000
r             0 23  23  0.50000000
ɾ             0 12 218  0.05217391
```
You will see that the error rate did, indeed, improve by .97%. Next, you will check the prediction and confusion matrix on both train data and test data to see if these models improved as well. 
```R
# Rerun Tuned up Prediction & Confusion Matrix on Train & Test Data
library(caret)
p1 <- predict(rf, train)
confusionMatrix(p1, train$allophone)

p2 <- predict(rf, test)
confusionMatrix(p2, test$allophone)
```
In the 100 tree model, you can examine the size of the trees in terms of the number of nodes from the rf model. 
```R
# Number of Nodes for Trees
hist(treesize(rf), main = "Number of Nodes for the Trees", col = "blue")
```
![Number of Nodes for the Trees](https://user-images.githubusercontent.com/133238472/237775029-8292dc8e-d13c-464c-b100-95bb4b50f5df.png)

The histogram shows the distribution of the number of nodes in each of the 100 trees included in the model. It shows that in our model, there are over 35 trees with above 40 nodes. There is a small number of trees with 25 nodes and a small number of trees with 55 nodes. In other words, the distribution of nodes in the trees is from 25 to 55. 

Essential for the interpretability of the random forest, the varImpPlot gives you the ranked importance of each independent variable included in the model. 
```R
# Variable Importance
varImpPlot(rf, sort = T, main = "Variable Importance")
```
![Variable Importance](https://user-images.githubusercontent.com/133238472/237776998-e07c8b0d-ebd4-4f37-948f-e083f1a7547b.png)

The first chart in the plot of variable importance tells us that duration is the most important factor by a long shot; excluding duration would cause the model’s accuracy to decrease by more than 50%. The least important variable is rhotic stress (stressed vs. unstressed). The MeanDecreaseGini chart measures how “pure” the nodes are when the variable is excluded. Purity refers to how evenly split a node is; when all of the data belongs to a single class, it is maximally pure and when the data is split evenly, it is maximally impure. Duration is the greatest contributing factor to Gini as well. 

You may further investigate the numerical values associated with the varImpPlot above by using the importance() command. Lastly, varUsed allows us to see how often the variables appeared in the random forest. 
```R
> importance(rf)
                            elision            r           ɾ
performance                  0.0000  2.196466169  0.08946103
interlocuterGender           0.0000 -0.129076191  0.58323473
interlocuterSexOrientation   0.0000 -1.005037815 -0.52290874
stress                       0.0000  1.228677066  0.23175070
manner_precedingSeg          0.0000  0.000000000  0.00000000
precedingSegDuration         0.0000  1.385330734  2.64385007
manner_followingSeg          0.0000  0.519018714 -0.09948453
duration                   113.2237 24.393693180 33.96058055
F1                           0.0000 -0.005796602  3.58503043
F2                           0.0000  2.110664702  8.71219781
F3                           0.0000 -1.288560348  3.49142980
F4                           0.0000 -0.598624126  2.14306273
F3.F2_distance               0.0000 -0.248645443  4.47204780
cog                          0.0000 -1.947694886  3.79707309
stdev                        0.0000  1.565677763  2.84824145
skewness                     0.0000  1.311208090  2.81607764
kurtosis                     0.0000 -2.106750177  2.37189334
                           MeanDecreaseAccuracy MeanDecreaseGini
performance                          1.63387783        0.9196072
interlocuterGender                   0.31201815        0.4173132
interlocuterSexOrientation          -0.82902701        0.2256313
stress                               0.57499582        0.7691910
manner_precedingSeg                  0.00000000        0.0000000
precedingSegDuration                 3.34178742        4.0853319
manner_followingSeg                  0.09674065        0.1515332
duration                            51.73754422       86.2037644
F1                                   3.42764755        4.9058118
F2                                   8.30695849        7.5758581
F3                                   3.27834837        2.0111964
F4                                   2.00181179        3.3653960
F3.F2_distance                       4.52442586        3.8073735
cog                                  3.04311852        4.7563355
stdev                                3.76823500        3.2593162
skewness                             3.27585557        3.5153831
kurtosis                             1.95735811        2.5321187

> varUsed(rf)
 [1]  29  14   6  58   0 161   8 442 189 255  93 137 169 187 152 169 119

The fifth variable with the smallest number in the table (o) corresponds with the fifth predictor variable in the model—manner_followingSeg. This variable’s importance value  zero because it did not appear in the model compared to the other predictors. However, if we compare this with the eighth variable, duration, it has occurred 442 times in the model, indicating that it is maximally important. 

You did it! Now that you’ve created your random forest classification, you can have greater confidence in your selected measure(s) that are tuned to your unique dataset. Depending on your goals, your next step might be running a linear mixed effects model with duration as your dependent variable. 






