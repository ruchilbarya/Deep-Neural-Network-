
### Prediction of MBA refinance Index (Mortgage prepayment)
_Deep Neural Network based Model_


The ability to predict mortgage prepayment is of critical use to financial institutions from an interest rate risk perspective. The goal of this project to predict the MBA refinance index.



The main components of this project are 
* Data Cleaning 
* Exploratory data analysis 
* Model Creation 
* Model Evaluation 

# Data
Data for MBA originated across USA between 1990-2021
Weekly performance update as a target variable
Mortgage Bankers Association Refi Index (Bloomberg Terminal)

Data for national economic factors from FRED (Federal Reserve Economic Data)
Economically significant data 

11,451 daily observations, each described by roughly 26 feature variables 
Correlation was used to choose feature selection
![image](![image](https://user-images.githubusercontent.com/70984576/149645151-130f58a8-7904-4659-9c45-d1202217f287.png))
![image](https://user-images.githubusercontent.com/70984576/149645157-648a9d4c-a127-4a84-a875-c08df49d6f7b.png)


# Results 

Best model was with 5- layered sequential NN model (with hyper parameter tuning)
The MAPE value of 8.44%. Although DNN (Deep Neural Network) is being used mostly in image recognition and NLP, it could be used to predict the quantitative data. The hyperparameter tuning was able to increase the performance of our model.
