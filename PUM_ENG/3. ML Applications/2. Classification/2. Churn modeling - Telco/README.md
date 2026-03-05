# Identifying customers at high risk of leaving for competitors (CHURN)

You are an analyst for a telecommunications company. Your task is to develop a model to identify customers at high risk of leaving to competitors (CHURN) and to evaluate the economic effectiveness of its implementation.

The current situation is as follows:

1. We have a 700 USD margin on one customer.
2. to a customer whose contract is about to end, we offer a 100$ incentive (bonus) to stay with us. We do not use any model for this purpose.
3. the cost of making such contact (telemarketing work) is 50$
4. not every customer we call decides to renew the contract: in such a situation we incur a cost of 50 USD, but we do not spend 100$ on a bonus.

From the sales department you received a file with the following data:

**State**: the US state in which the customer resides, indicated by a two-letter abbreviation; for example, OH or NJ

**Area Code**: the three-digit area code of the corresponding customer’s phone number

**Phone**: the remaining seven-digit phone number

**Account Length**: the number of days that this account has been active

**Int’l Plan**: whether the customer has an international calling plan

**VMail Plan**: whether the customer has a voice mail feature

**VMail Message**: presumably the average number of voice mail messages per month

**Day Mins**: the total number of calling minutes used during the day

**Day Calls**: the total number of calls placed during the day

**Day Charge**: the billed cost of daytime calls

**Eve Mins, Eve Calls, Eve Charge**: the billed time, # of calls and cost for calls placed during the evening

**Night Mins, Night Calls, Night Charge**: the billed time, # of calls and cost for calls placed during nighttime

**Intl Mins, Intl Calls, Intl Charge**: the billed time, # of calls and cost for international calls

**CustServ Calls**: the number of calls placed to Customer Service

**Churn?**: whether the customer left. 0: stayed (no churn), 1: left (churn)



## Tasks

1. Descriptive analysis and baseline model 

   - [ ] Review the data ("Analyze").
   - [ ] Create a baseline model: Lab > ML Prediction > Churn
   - [ ] Identify the winner and record the ROC AUC value.

2. Compute the cost/benefits using Cost Matrix

   1. model predicts 1
      1. And reality is 1: 700 - (100 + 50)
      2. And reality is 0: - (100 + 50)

   2. model predicts 0
      1. And reality is 0: 0
      2. And reality is 1: -700

3. model 1.0: first feature engineering 

   - [ ] Create a "Total_mins" column containing the sum of the number of minutes of daytime, evening, nighttime and international calls
   - [ ] Create a "Total_charge" column containing the sum of daytime, evening, nighttime and international costs
   - [ ] Create model 1.0, incorporating new variables
   - [ ] Identify the winner, record the ROC AUC value

   What was the effect of the new variables on the effectiveness of the model?

3. Segmentation 

   - [ ] Create a segmentation model (ML Clustering). Turn off "phone" and "churn"!

4. model 2.0: taking segmentation into account 

   - [ ] Create model 2.0, taking into account new variables (clusters)

   - [ ] Identify the winner, record the ROC AUC value

     What was the effect of the new variables on the effectiveness of the model?

5. Model 3.0: feature engineering, new algorithms  

   - [ ] Create a classification model, excluding various variables from the analysis

   - [ ] Identify the winner, record the value of the selected measure assessing the quality of the model

   - [ ] Create a classification model, trying new algorithms

   - [ ] Identify the winner, record the value of the ROC AUC

     What was the effect of new exclusion of new variables and introduction of new algorithms on the efficiency of the model?

6. Economic efficiency of the best model 
   - [ ] For the best model, enter the Performance > Confusion Matrix parameters based on the assumptions presented in the task
   - [ ] Calculate the profit per record

7. model 4.0: selection and optimization of the best measure to evaluate the quality of the model

   - [ ] Select, in your opinion, the best measure to evaluate the quality of the model

   - [ ] optimize the training of the model for the new measure
   - [ ] calculate the gain per record
     How did the selection of your new measure affect the profit per record?
