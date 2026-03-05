# Identification of customers at high risk of defaulting on loans

You are an analyst at a bank. Your task is to develop a model that identifies customers at high risk of not repaying a loan on time and to evaluate the economic effectiveness of its implementation.

You have received a file from the sales department with the following data:

**ID**: ID of each client 

**LIMIT_BAL**: Amount of given credit in NT dollars (includes individual and family/supplementary credit)  

**SEX**: Gender (1=male, 2=female)  

**EDUCATION**: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)  

**MARRIAGE**: Marital status (1=married, 2=single, 3=others)  

**AGE**: Age in years  

**PAY_0 to PAY_6**: Repayment status by n months ago (PAY_0 = last month ... PAY_6 = 6 months ago) (Labels: -1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)  

**BILL_AMT1 to BILL_AMT6**: Amount of bill statement by n months ago ( BILL_AMT1 = last_month .. BILL_AMT6 = 6 months ago)  

**PAY_AMT1 to PAY_AMT6**: Amount of payment by n months ago ( BILL_AMT1 = last_month .. BILL_AMT6 = 6 months ago)  

**default**: Default payment (1=yes, 0=no) Target Column  

Source:

Lichman, M. (2013). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science,  https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

Good Luck!

## Tasks

1. Exploratory data analysis and the baseline model

   - [ ] Analyze data
   - [ ] Train Your first model
   - [ ] Identify the winner and record the ROC AUC value

2. Improving the baseline model

   - [ ] Do some experiments in identifying a better-than-baseline model:

     - [ ] Feature engineering...
     
       - [ ] Algorithms
          - [ ] ...
     
   - [ ] Identify the winner, record the ROC AUC value

     ***What was the impact of your experiments on the effectiveness of the model?***

3. Economic efficiency of the best model 

   - [ ] Propose the first version of the cost matrix:
     - [ ] For inspiration, see, for example, the *Profit Curve* section here: http://inseaddataanalytics.github.io/INSEADAnalytics/CourseSessions/ClassificationProcessCreditCardDefault.html
   - [ ] For the best model, enter the Performance > Confusion Matrix parameters based on the assumptions presented in the task
   - [ ] Calculate profit per record.
     ***What business benefits can the bank expect when accepting Your model to be used in operations?***
