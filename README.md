# CS-460---Final Project: Adapting An Existing Algorithm (Apriori Algorithm)


# Employee Retention Analysis Using Apriori Algorithm

This project aims to analyze employee data to uncover patterns and associations that influence employee retention. By utilizing the Apriori algorithm, the project identifies frequent itemsets and generates association rules, providing insights into the key factors that contribute to employee turnover.

**Project Overview**

The primary objective of this project is to help HR departments understand the combinations of employee attributes that are most strongly associated with employee retention or turnover. The insights gained can guide the development of targeted interventions to improve employee satisfaction and reduce turnover rates.


**Dataset**

The dataset includes various attributes related to employee characteristics and their work environment:

- Job Satisfaction: Low, Medium, High
- Training Opportunities: Few, Moderate, Many
- Years of Service: <1, 1-2, 3-5, 6-10, 10+
- Work-Life Balance: Poor, Average, Good
- Performance Score: Low, Medium, High
- Commute Time: <30min, 30-60min, 60-90min, 90+min
- Promotion History: Never, Once, Twice, Thrice+
- Department: HR, Engineering, Sales, Marketing, Finance
- Age: Integer values representing employee ages
- Left: Yes, No (whether the employee has left the company)
- Getting Started
- Prerequisites
- Ensure you have the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- networkx


**Output**

The program will output:

- The initial and transformed (one-hot encoded) DataFrames.
- Frequent itemsets identified by the Apriori algorithm.
- Generated association rules based on the frequent itemsets.
- Filtered retention rules focusing on factors associated with employees leaving the company.
- Visualizations of employee data distributions and the network graph of association rules.
- Explanation of the Apriori Algorithm
- The Apriori algorithm is used to identify frequent itemsets and generate association rules from these itemsets. It helps in understanding which combinations of employee attributes are most strongly linked to turnover. By setting minimum support and confidence thresholds, the algorithm filters out less significant patterns, focusing on the most relevant ones.

**Insights**

Even with strict filtering criteria, the project is able to identify key factors influencing employee retention, such as job satisfaction and training opportunities. These insights have the potential guide HR departments in developing targeted strategies to improve employee satisfaction and reduce turnover.


**Conclusion**

This project demonstrates the power of the Apriori algorithm in analyzing employee data to uncover meaningful patterns and associations. By understanding these patterns, organizations can implement more effective policies and practices, ultimately enhancing overall employee retention.

