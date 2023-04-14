Applied algorithms:
  Neural Network,
  K-Nearest Neighbours,
  Decision Tree,
  simple heuristic

Comparison of models is based on accuracy. For parameters in the project KNN was the best
choice according to accuracy after PCA, but decision tree was the fastest and most accurate before using PCA.

You can find solution in app.py.
REST API is build with Flask.

req.py is a file that can be used to send POST request. 

For windows users:
set FLASK_APP=app.py

For bash users:
export FLASK_APP=app.py

To run app:
run flask

When server is on, run req.py to get predictions. For KNN and NN you need to wait a little longer, than for Decision Tree and simple heuristic. 
