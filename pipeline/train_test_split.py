from sklearn.model_selection import train_test_split as sklearn_train_test_split
from prefect import task

@task
def train_test_split(X, y, test_size=0.2, random_state=1):
    X_train, X_test, y_train, y_test = sklearn_train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = sklearn_train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)
    return X_train, X_test, X_val, y_train, y_test, y_val