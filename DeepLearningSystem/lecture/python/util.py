import needle as ndl

def print_node(node, str=""):
  # import pdb;pdb.set_trace()
  print("==========%s==========" % str)
  print("id=%d" % id(node))
  print("op=%s" % type(node.op))
  print("inputs=%s" % [id(x) for x in node.inputs])
  print("num_outputs=%d" % node.num_outputs)
  print("cached_data=%s" % node.cached_data)
  print("requires_grad=%d" % node.requires_grad)
  return