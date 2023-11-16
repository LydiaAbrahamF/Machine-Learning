import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
data = pd.read_csv('data_for_lr.csv')

print(data.head(10))

print(data.info())
data = data.dropna()
print(data.shape)

X=data.x
Y=data.y


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

train_input = np.array(X_train).reshape(len(X_train), 1)
train_output = np.array(y_train).reshape(len(y_train), 1)

test_input = np.array(X_test).reshape(len(X_test), 1)
test_output = np.array(y_test).reshape(len(y_test),1)
linear_regressor = LinearRegression()
linear_regressor.fit(train_input, train_output)

print(linear_regressor.coef_)
print(linear_regressor.intercept_)

predicted_value = linear_regressor.predict(test_input)
cost = mean_squared_error(test_output, predicted_value)
print(test_output)
print(predicted_value)
print(cost)


plt.plot(test_input, test_output, '+', color = "green")
plt.plot(test_input, predicted_value, '*', color = "red")
plt.title("Performance testing")
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()

filename = 'finalized_model_scratch.sav'
pickle.dump(linear_regressor, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))

result = loaded_model.fit(train_input, train_output)
print(result)

a = loaded_model.predict(test_input)
print(a)
