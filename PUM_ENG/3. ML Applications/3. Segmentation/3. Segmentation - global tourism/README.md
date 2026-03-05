## Identifying the country with the highest potential for foreign tour sales

You're in charge of marketing for a global travel company offering tours for senior citizens.

Your task: to identify the countries with the highest sales potential in order to start marketing activities in them.

In the data directory you will find a file with the most important characteristics (in the context of foreign tourism) of several hundred countries from all over the world: countries_prepared.csv

Based on these, you are to prepare a list of several(a dozen) countries with the most potential customers.

Good luck!



## Strategy

1. use Lab > AutoML Clustering to segment countries
2. select K-means as your algorithm
3. identify the optimal model (highest Silhouette factor):
   1. try different numbers of clusters (4, 5, 6, 7).
   2. include and exclude different features (e.g. `count`,`population_total_avg`).
   3. try to achieve a Silhouette at least at 0.358
4. for each cluster in the optimal (highest) model:
   1. name it
   2. write a brief description of it (`Description` section)
5. run the model and apply it to the input dataset
6. filter the dataset for the most promising cluster and identify the best countries to start your promotion. 



## Solution

List of recommended countries:

1. ...
2. ...
3. ...
4. ...
5. ...
6. ...
7. 

