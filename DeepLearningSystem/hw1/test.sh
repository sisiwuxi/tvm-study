python3 -m pytest -v -k "forward"
python3 -m pytest -v -k "backward"
python3 -m pytest -k "topo_sort"
python3 -m pytest -k "compute_gradient"
python3 -m pytest -k "softmax_loss_ndl"
python3 -m pytest -l -k "nn_epoch_ndl"