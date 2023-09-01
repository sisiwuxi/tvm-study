
# test
- ./tests/python/
  - integration/test_auto_tensorize.py:145
  - contrib/test_hexagon/test_meta_schedule.py:335
  - unittest/test_meta_schedule_mma_m16n8k8_auto_tensorization.py:1228
  - unittest/test_meta_schedule_postproc_rewrite_tensorize.py:497
# src
- ./python/tvm/
  - meta_schedule/postproc/rewrite_tensorize.py:24
- ./src/meta_schedule/postproc
  - postproc.cc:65
  - rewrite_tensorize.cc:111