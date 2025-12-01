# 1. Data (x = hours studied, y = exam score) ---
xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ys = [52, 56, 61, 66, 68, 72, 74, 79, 83, 88]

# --- 2. Split into train (first 8) and test (last 2) ---
train_size = 8
x_train = xs[:train_size]
y_train = ys[:train_size]
print(list(zip(x_train, y_train)))
x_test = xs[train_size:]
y_test = ys[train_size:]

# --- 3. Compute mean of x and y (for train set) ---
x_mean = sum(x_train) / len(x_train)
y_mean = sum(y_train) / len(y_train)
print("Mean of X:", x_mean)
print("Mean of Y:", y_mean)

# 4. Compute slope (m) and intercept (b) 
num = 0
den = 0
for x, y in zip(x_train, y_train):
    num += (x - x_mean) * (y - y_mean)
    den += (x - x_mean) ** 2

m = num / den
b = y_mean - m * x_mean

print("Slope m:", m)
print("Intercept b:", b)

# --- 5. Prediction on test set ---
y_pred_test = []
for x in x_test:
    y_pred = m * x + b
    y_pred_test.append(y_pred)

print("\nTest X:", x_test)
print("Actual Y:", y_test)
print("Predicted Y:", y_pred_test)

# --- 6. Simple evaluation: Mean Squared Error (MSE) ---
error_sum = 0
for yt, yp in zip(y_test, y_pred_test):
    error_sum += (yt - yp) ** 2

mse = error_sum / len(y_test)
print("\nTest MSE:", mse)
