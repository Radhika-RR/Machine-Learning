import csv
import math

# ---------- Load dataset ----------

xs = []
ys = []

with open("exam_pass_data.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        xs.append(float(row["hours"]))
        ys.append(int(row["passed"]))

# ---------- Train-test split ----------

train_size = 8
x_train = xs[:train_size]
y_train = ys[:train_size]
x_test = xs[train_size:]
y_test = ys[train_size:]

# ---------- Logistic regression math ----------

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def predict_prob(x, w, b):
    return sigmoid(w * x + b)

def predict_label(x, w, b, threshold=0.5):
    p = predict_prob(x, w, b)
    return 1 if p >= threshold else 0

def compute_loss(x_list, y_list, w, b):
    n = len(x_list)
    total = 0.0
    for x, y in zip(x_list, y_list):
        p = predict_prob(x, w, b)
        p = min(max(p, 1e-10), 1 - 1e-10)
        total += y * math.log(p) + (1 - y) * math.log(1 - p)
    return -total / n

# ---------- Training (gradient descent) ----------

w = 0.0
b = 0.0
learning_rate = 0.1
epochs = 1000
n = len(x_train)

for epoch in range(epochs):
    dw = 0.0
    db = 0.0

    for x, y in zip(x_train, y_train):
        p = predict_prob(x, w, b)
        error = p - y
        dw += error * x
        db += error

    dw /= n
    db /= n

    w = w - learning_rate * dw
    b = b - learning_rate * db

    if (epoch + 1) % 100 == 0:
        loss = compute_loss(x_train, y_train, w, b)
        print(f"Epoch {epoch+1:4d} | loss = {loss:.4f} | w = {w:.4f}, b = {b:.4f}")

print("\nTraining finished")
print("w =", w)
print("b =", b)
print("Learning rate =", learning_rate)
print("Epochs =", epochs)

decision_boundary = -b / w
print("Decision boundary (hours where prob ≈ 0.5):", decision_boundary)

# ---------- Test prediction ----------

correct = 0
for x, y_true in zip(x_test, y_test):
    y_pred = predict_label(x, w, b)
    p_pred = predict_prob(x, w, b)
    print(f"x = {x:.1f} | true = {y_true} | predicted = {y_pred} | prob(pass) = {p_pred:.3f}")
    if y_pred == y_true:
        correct += 1

accuracy = correct / len(x_test) if len(x_test) > 0 else 0
print(f"Test accuracy: {accuracy * 100:.2f}%")
