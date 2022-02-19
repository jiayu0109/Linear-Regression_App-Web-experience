# # Linear Regression Project 
# Analyzing customer data-- Should we focus on Mobile app experience or website? 

### Imports Library 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


### Get the Data
# It is a sampled Ecommerce Customers csv file from an Ecommerce company selling clothes. 
# Columns contains:
# 1. Customer info, suchas Email, Address, and their color Avatar.
# 2. Avg. Session Length: Average session of in-store style advice sessions.
# 3. Avg. Session Length: Average session of in-store style advice sessions.
# 4. Time on App: Average time spent on App in minutes
# 5. Time on Website: Average time spent on Website in minutes
# 6. Length of Membership: How many years the customer has been a member. 
 

EC = pd.read_csv('Ecommerce Customers')


### Brief view on the Data
EC.head()
EC.describe()
EC.info()


### Exploratory Data Analysis
# I'll only use the numerical data of the csv file.

# 1. Compare the Time on Website and Yearly Amount Spent columns with seaborn.
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=EC)
# More time on site, more money's spent.

# 2. Compare the Time on App and Yearly Amount Spent columns
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=EC)

# 3. Compare Time on App and Length of Membership with jointplot to create a 2D hex bin plot.
sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=EC)

# 4. Use pairplot to explore relationships across the entire data set. 
sns.pairplot(EC)
# Based on this plot, Length of Membership& Time on APP look to be the most correlated feature with Yearly Amount Spent.

# 5. Create a linear model plot (using seaborn's lmplot) of  Yearly Amount Spent vs. Length of Membership.
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=EC)


### Create Training and Testing Data
# Split the data into training and testing sets.
# Variable X: numerical features of the customers,variable y: "Yearly Amount Spent" column
EC.columns
x = EC[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y = EC['Yearly Amount Spent']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)


### Training the Model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()


### Train/fit lm on the training data
lm.fit(x_train,y_train)


### The coefficients of the model
lm.coef_


### Predicting Test Data
lm.predict(x_test) 


### Create a scatterplot showing real test values versus predicted values.
predictions = lm.predict(x_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


### Evaluating the Model
# Evaluate model performance by Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


### Residuals
# Plot the residuals and check whether it looks normally distributed with a histogram using seaborn distplot
sns.displot((y_test-predictions),bins=50, legend=True);


### Conclusion: So should we put more efforst on mobile app or website development? 
### Check the coefficient of each variable below
df = pd.DataFrame(lm.coef_,x.columns,columns=['Coefficient'])
df

# Length of Membership does impact the most, but comparing "Time on App" and "Time on Website", "Time on App" siginificantly impacts more than "Time on Website".
# The company should focus more on their mobile app, but they would probably want to explore the relationship between 
# Length of Membership and the App or the Website before coming to a conclusion.
