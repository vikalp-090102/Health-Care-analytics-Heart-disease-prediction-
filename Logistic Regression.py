
class LogisticRegression():
    
    def __init__(self, learning_rate=0.001, no_of_iterations=1000):
        self.lr = learning_rate #alpha in gradient descent
        self.no_of_iterations = no_of_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # w and b=> parameters for model
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.no_of_iterations):
            # Z=X.w+b
            Z = np.dot(X, self.weights) + self.bias
            # sigmoid func=>Y_hat=1/1+e^-Z
            y_predicted = self._sigmoid(Z)

            # gradients using gradient descent and cost function dJ(w,b)
            #dw=>dJ(w,b)/dw
            #db=>dJ(w,b)/db
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        #Z=X.w+b
        Z = np.dot(X, self.weights) + self.bias
        #y_hat=1/1+e^-Z
        y_predicted = self._sigmoid(Z)
        #classify as 1 if y_hat>0.5 else as 0
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_class)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

lr=LogisticRegression(learning_rate=0.001,no_of_iterations=100000)
lr.fit(X_train,Y_train)
train_pred=lr.predict(X_train)
print(accuracy_score(train_pred,Y_train))