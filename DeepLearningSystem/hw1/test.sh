python3 -m pytest -v -k "forward"
python3 -m pytest -v -k "backward"
python3 -m pytest -k "topo_sort"
python3 -m pytest -k "compute_gradient"