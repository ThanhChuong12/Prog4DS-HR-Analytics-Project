import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler not fitted.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
class ThresholdOptimizer:
    """
    Tune threshold to maximize a metric (default: recall).
    """
    def __init__(self, metric="recall"):
        self.metric = metric
        self.best_threshold = 0.5
        self.best_score = -np.inf

    def fit(self, y_true, y_proba):
        y_true = np.asarray(y_true).astype(int)
        y_proba = np.asarray(y_proba)

        thresholds = np.linspace(0.05, 0.95, 50)
        for th in thresholds:
            y_pred = (y_proba >= th).astype(int)
            score = getattr(Metrics, self.metric)(y_true, y_pred)
            if score > self.best_score:
                self.best_score = score
                self.best_threshold = th
        return self

    def predict(self, y_proba):
        return (y_proba >= self.best_threshold).astype(int)

class LogisticRegressionFromScratch:
    """
    Logistic Regression implemented from scratch with:
      - L2 regularization
      - mini-batch gradient descent
      - early stopping (optional)
      - reproducible initialization via random_state
    """
    def __init__(self, learning_rate=0.01, n_iters=1000, reg_lambda=0.0, reg_type="l2", l1_ratio=0.5, class_weight=None, 
                 batch_size=None, tol=1e-6, early_stopping=False, patience=10,
                 verbose=True, random_state=None):
        self.learning_rate = learning_rate
        self.n_iters = int(n_iters)
        self.reg_lambda = float(reg_lambda)
        self.reg_type = reg_type
        self.l1_ratio = l1_ratio
        self.class_weight = class_weight or {"0": 1.0, "1": 1.0}
        self.batch_size = batch_size  # None -> full batch
        self.tol = tol
        self.early_stopping = early_stopping
        self.patience = int(patience)
        self.verbose = verbose
        self.random_state = random_state

        # will be set in initialize_parameters or fit
        self.weights = None  # shape (n_features, 1)
        self.bias = 0.0

        self.loss_history = []
        self.val_loss_history = []
        self.best_weights = None
        self.best_bias = None

    def _rng(self):
        return np.random.RandomState(self.random_state)

    @staticmethod
    def _sigmoid(z):
        # clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _bce_weighted(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        w1 = self.class_weight["1"]
        w0 = self.class_weight["0"]

        loss = -np.mean(
            w1 * (y_true * np.log(y_pred)) +
            w0 * ((1 - y_true) * np.log(1 - y_pred))
        )
        return loss

    def initialize_parameters(self, n_features):
        # Heuristic init small random values; use RNG for reproducibility
        rng = self._rng()
        # shape (n_features, 1)
        self.weights = rng.normal(0, 0.01, size=(n_features, 1)).astype(float)
        self.bias = 0.0
        # reset histories
        self.loss_history = []
        self.val_loss_history = []
        self.best_weights = None
        self.best_bias = None

    def _check_shapes(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return X, y.astype(float)

    def fit(self, X, y, validation_data=None):
        """
        Train model. Returns histories dict.
        If validation_data provided, use for early stopping and tracking.
        """
        X, y = self._check_shapes(X, y)
        n_samples, n_features = X.shape

        if self.weights is None:
            self.initialize_parameters(n_features)

        # Basic batch size
        batch_size = self.batch_size or n_samples

        # For reproducibility shuffling
        rng = self._rng()

        # convert validation data
        if validation_data is not None:
            X_val, y_val = self._check_shapes(*validation_data)
        else:
            X_val = y_val = None

        best_val_loss = np.inf
        wait = 0

        for it in range(self.n_iters):
            # shuffle indices each epoch
            indices = np.arange(n_samples)
            rng.shuffle(indices)

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                # forward
                linear = np.dot(X_batch, self.weights) + self.bias
                y_pred = self._sigmoid(linear)

                # compute loss (batch loss)
                loss = self._bce_weighted(y_batch, y_pred)
                # add L2 regularization term (do not regularize bias)

                # weights for each sample
                w_sample = np.where(y_batch == 1, self.class_weight["1"], self.class_weight["0"])
                w_sample = w_sample.reshape(-1, 1)

                # gradient with sample weights
                error = (y_pred - y_batch) * w_sample
                dw = (1.0 / X_batch.shape[0]) * np.dot(X_batch.T, error)
                db = (1.0 / X_batch.shape[0]) * np.sum(error)


                if self.reg_lambda > 0:
                    if self.reg_type == "l2":
                        loss += 0.5 * self.reg_lambda * np.mean(self.weights ** 2)
                        dw += (self.reg_lambda / X_batch.shape[0]) * self.weights

                    elif self.reg_type == "l1":
                        loss += self.reg_lambda * np.mean(np.abs(self.weights))
                        dw += (self.reg_lambda / X_batch.shape[0]) * np.sign(self.weights)

                    elif self.reg_type == "elasticnet":
                        l1 = self.l1_ratio * self.reg_lambda
                        l2 = (1 - self.l1_ratio) * self.reg_lambda
                        loss += l1 * np.mean(np.abs(self.weights)) + 0.5*l2*np.mean(self.weights**2)
                        dw += (l2 / X_batch.shape[0]) * self.weights + (l1 / X_batch.shape[0]) * np.sign(self.weights)



                # update
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            # end of epoch: compute full-batch train loss
            train_pred_full = self._sigmoid(np.dot(X, self.weights) + self.bias)
            train_loss = self._bce_weighted(y, train_pred_full)
            if self.reg_lambda > 0:
                train_loss += 0.5 * self.reg_lambda * np.mean(self.weights ** 2)
            self.loss_history.append(train_loss)

            # validation
            if X_val is not None:
                val_pred_full = self._sigmoid(np.dot(X_val, self.weights) + self.bias)
                val_loss = self._bce_weighted(y_val, val_pred_full)
                if self.reg_lambda > 0:
                    val_loss += 0.5 * self.reg_lambda * np.mean(self.weights ** 2)
                self.val_loss_history.append(val_loss)

                # early stopping logic: store best
                if val_loss + self.tol < best_val_loss:
                    best_val_loss = val_loss
                    self.best_weights = self.weights.copy()
                    self.best_bias = float(self.bias)
                    wait = 0
                else:
                    wait += 1

                if self.early_stopping and wait >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {it+1}, best val loss: {best_val_loss:.6f}")
                    # restore best
                    if self.best_weights is not None:
                        self.weights = self.best_weights
                        self.bias = self.best_bias
                    break

            # verbose
            if self.verbose and (it % 100 == 0 or it == self.n_iters - 1):
                if X_val is not None:
                    print(f"Iter {it}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
                else:
                    print(f"Iter {it}: train_loss={train_loss:.6f}")

        return {
            "train_loss": np.array(self.loss_history),
            "val_loss": np.array(self.val_loss_history)
        }

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        linear = np.dot(X, self.weights) + self.bias
        probs = self._sigmoid(linear)
        # return 1D array for convenience
        return probs.reshape(-1)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def get_parameters(self):
        return self.weights.copy(), float(self.bias)

    def score(self, X, y, threshold=0.5):
        y_pred = self.predict(X, threshold=threshold)
        return Metrics.accuracy(y, y_pred)

class Metrics:
    @staticmethod
    def _prepare(y_true, y_pred):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        return y_true.astype(int), y_pred.astype(int)

    @staticmethod
    def accuracy(y_true, y_pred):
        y_true, y_pred = Metrics._prepare(y_true, y_pred)
        return np.mean(y_true == y_pred)

    @staticmethod
    def precision(y_true, y_pred):
        y_true, y_pred = Metrics._prepare(y_true, y_pred)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @staticmethod
    def recall(y_true, y_pred):
        y_true, y_pred = Metrics._prepare(y_true, y_pred)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @staticmethod
    def f1_score(y_true, y_pred):
        p = Metrics.precision(y_true, y_pred)
        r = Metrics.recall(y_true, y_pred)
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        y_true, y_pred = Metrics._prepare(y_true, y_pred)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        return np.array([[tn, fp], [fn, tp]])

    @staticmethod
    def classification_report(y_true, y_pred):
        y_true, y_pred = Metrics._prepare(y_true, y_pred)
        acc = Metrics.accuracy(y_true, y_pred)
        prec = Metrics.precision(y_true, y_pred)
        rec = Metrics.recall(y_true, y_pred)
        f1 = Metrics.f1_score(y_true, y_pred)
        cm = Metrics.confusion_matrix(y_true, y_pred)
        out = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm
        }
        # Print concise nicely formatted report
        print("CLASSIFICATION REPORT")
        print(f"    Accuracy : {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall   : {rec:.4f}")
        print(f"    F1-score : {f1:.4f}")
        print("\nConfusion Matrix (rows = pred, cols = true):")
        print(cm)
        print("\n")
        return out

class KFoldCrossValidation:
    def __init__(self, k=5, random_state=None, stratify=False):
        self.k = int(k)
        self.random_state = random_state
        self.stratify = stratify

    def split(self, X, y):
        n = len(X)
        rng = np.random.RandomState(self.random_state)
        indices = np.arange(n)

        if not self.stratify:
            rng.shuffle(indices)
            fold_sizes = (n // self.k) * np.ones(self.k, dtype=int)
            fold_sizes[: n % self.k] += 1
            current = 0
            folds = []
            for fs in fold_sizes:
                start, stop = current, current + fs
                test_idx = indices[start:stop]
                train_idx = np.concatenate([indices[:start], indices[stop:]])
                folds.append((train_idx, test_idx))
                current = stop
            return folds
        else:
            # simple stratified split: maintain class proportions per fold
            y = np.asarray(y).reshape(-1)
            classes, class_counts = np.unique(y, return_counts=True)
            # bucket indices per class
            class_indices = {c: indices[y == c] for c in classes}
            for c in classes:
                rng.shuffle(class_indices[c])
            folds_indices = [[] for _ in range(self.k)]
            
            for c in classes:
                idxs = class_indices[c]
                for i, idx in enumerate(idxs):
                    fold_id = i % self.k
                    folds_indices[fold_id].append(idx)
            splits = []
            for i in range(self.k):
                test_idx = np.array(folds_indices[i], dtype=int)
                train_mask = np.ones(n, dtype=bool)
                train_mask[test_idx] = False
                train_idx = indices[train_mask]
                splits.append((train_idx, test_idx))
            return splits

    def evaluate(self, X, y, model_class, model_params=None, threshold=0.5, metrics=['f1_score', 'recall', 'precision']):
        model_params = model_params or {}
        folds = self.split(X, y)
        results = {m: [] for m in metrics}

        print(f"Running {self.k}- Fold CV with Threshold = {threshold}")

        for fold_i, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Initialization and Training
            model = model_class(**model_params)
            model.fit(X_train, y_train, validation_data=None)
            
            # Predict with specified Threshold
            y_pred = model.predict(X_test, threshold=threshold)
            
            # Calculate Metrics
            for m in metrics:
                if hasattr(Metrics, m):
                    score = getattr(Metrics, m)(y_test, y_pred)
                    results[m].append(score)
            
            # print(f"   Fold {fold_i+1}/{self.k}: F1={results['f1_score'][-1]:.4f}")
        avg = {m: {'mean': float(np.mean(results[m])), 'std': float(np.std(results[m]))} for m in metrics}
        return avg, results