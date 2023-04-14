import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from tensorflow.keras import models, layers
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import GridSearchCV
from flask import Flask, request, jsonify

def create_model(neurons=256, dropout=0.1):
    model = models.Sequential()
    model.add(layers.Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(neurons // 2, activation='relu'))
    model.add(layers.Dense(neurons // 4, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def simple_heuristic(Horizontal_Distance_To_Roadways, Wilderness_Area4, Wilderness_Area1, Elevation):
    if Elevation > 3250:
        return 7
    elif Elevation > 3000:
        return 1
    elif Elevation > 2750:
        return 5
    elif Elevation > 2300 and not Wilderness_Area1:
        return 6

    if Horizontal_Distance_To_Roadways > 1000:
        return 2

    if Wilderness_Area4:
        return 4

    return 3

def prepare_environment():
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz", header=None)
    colnames = ['Elevation',
                'Aspect',
                'Slope',
                'Horizontal_Distance_To_Hydrology',
                'Vertical_Distance_To_Hydrology',
                'Horizontal_Distance_To_Roadways',
                'Hillshade_9am',
                'Hillshade_Noon',
                'Hillshade_3pm',
                'Horizontal_Distance_To_Fire_Points',
                'Wilderness_Area1',
                'Wilderness_Area2',
                'Wilderness_Area3',
                'Wilderness_Area4']
    soil_types = ["Soil_Type" + str(i) for i in range(1, 41)]
    cover_types = ['Cover_Type']
    colnames = colnames + soil_types + cover_types
    data.columns = colnames

    X = data.drop(['Cover_Type'], axis=1)
    y = data.Cover_Type
    y = y - 1

    pca = PCA(n_components=5)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    X = pca.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, pca, scaler

def search_hyperparam(X_train, y_train):
    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, verbose=0)
    param_grid = {'neurons': [128, 256], 'dropout': [0.1, 0.2]}
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_result = grid.fit(X_train, y_train)
    return "Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)


def prepare_nn(X_train, y_train, X_test, y_test):
    model = create_model(256, 0.1)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return model, history

def prepare_knn(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=7, p=1)
    knn.fit(X_train, y_train)
    y_pred_test = knn.predict(X_test)
    y_pred_train = knn.predict(X_train)

    print("Train accuracy: ", accuracy_score(y_train, y_pred_train))
    print("Test accuracy: ", accuracy_score(y_test, y_pred_test))
    return knn

def prepare_dct(X_train, y_train, X_test, y_test):
    dtc = DecisionTreeClassifier(random_state=42)
    dtc.fit(X_train, y_train)
    y_pred_dtc = dtc.predict(X_test)
    print("Decision Tree accuracy:", accuracy_score(y_test, y_pred_dtc))
    return dtc

def make_plot_for_nn(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

X_train, X_test, y_train, y_test, pca, scaler = prepare_environment()

app = Flask(__name__)

# Define the API endpoints
@app.route('/', methods=['GET'])
def home():
    return "Covertype classification API"

@app.route('/predict', methods=['POST'])
def predict():
    # Parse the request body
    model = request.args.get('model')
    data = request.json
    X = pd.DataFrame(data, index=[0])

    # Use the appropriate model to make predictions
    if model == 'heuristic':
        y_pred = np.zeros(len(X.index))
        for i in range(len(X.index)):
            y_pred[i] = simple_heuristic(X.Horizontal_Distance_To_Roadways[i], X.Wilderness_Area4[i], X.Wilderness_Area1[i], X.Elevation[i])

    elif model == 'dct':
        dct = prepare_dct(X_train, y_train, X_test, y_test)
        X = pca.transform(X)
        X = scaler.transform(X)
        y_pred = dct.predict(X)

    elif model == 'nn':
        nn, history = prepare_nn(X_train, y_train, X_test, y_test)
        X = pca.transform(X)
        X = scaler.transform(X)
        y_pred = nn.predict(X)
        y_pred = np.argmax(y_pred, axis=1) + 1
        make_plot_for_nn(history)

    else:
        knn = prepare_knn(X_train, y_train, X_test, y_test)
        X = pca.transform(X)
        X = scaler.transform(X)
        y_pred = knn.predict(X)

    # Return the predictions as JSON
    return jsonify({'predictions': y_pred.tolist()})

# Run the server
if __name__ == '__main__':
    app.run()
