python3 -m pytest -k "add"
python3 -m pytest -k "parse_mnist"
python3 -m pytest -k "softmax_loss"
python3 -m pytest -k "softmax_regression_epoch and not cpp"
python3 -m pytest -k "nn_epoch"
python3 -m pytest -k "softmax_regression_epoch_cpp"
python3 src/simple_ml.py