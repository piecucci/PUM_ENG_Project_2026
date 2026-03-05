## Segmentation of online store customers

Your goal is  to segment the customers of an online store using the RMF (Recency > Monetary Value > Frequency) method.

The source dataset is available at: https://archive.ics.uci.edu/ml/datasets/online+retail

Columns description:
**InvoiceNo**: Invoice number. Nominally, a 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter "c", it means that the transaction has been cancelled.
**StockCode**: Product (item) code. Nominally, a 5-digit integral number uniquely assigned to each distinct product.
**Description**: The name of the product (item). Nominal.
**Quantity**: Quantity of each product (item) per transaction. Numeric.
**InvoiceDate**: Date and time of invoice generation. Numeric, the day and time each transaction was generated.
**UnitPrice**: Unit price. Numeric, the price of the product per unit in pounds sterling.
**CustomerID**: Customer number. A nominal, 5-digit integral number uniquely assigned to each customer.
**Country**: Country name. Nominal, name of the country in which each customer resides.

## Strategy

1. clear the data
   1. invoice number: delete all records starting with "c" (ie: canceled).
   2. delete unnecessary columns: StockCode, Description, Country

2. prepare the data
   1. create Recency column: the number of days from the invoice date to 30.12.2011 (year-end)
   2. create Monetary column: calculate the order value by multiplying Quantity by Unit Price.
   3. group the Dataset to get a single entry for each customer

3. create segmentation
   1. use Lab > ML Clustering to segment your customers
   2. experiment with different algorithms and their hyper-parameters
   3. for the best algorithm: **interpret the meaning of clusters**.
   4. implement the best model and apply to the dataset



