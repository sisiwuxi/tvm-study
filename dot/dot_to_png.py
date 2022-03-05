import os
import sys
import pdb

def dot_to_png():
    if len(sys.argv) < 2:
        print("Please input the full path")
        return
    path = sys.argv[1]
    print(path)
    filelist = os.listdir(path)
    for file in filelist:
        if file.endswith('.dot'):
            input_file = path + "/" + file
            output_file = input_file + ".png"
            cmd = "dot -Tpng " + input_file + " -o " + output_file
            # cmd = "dot -Tpng dfg.dot -o dfg.png"
            os.system(cmd)

if __name__ == "__main__":
    dot_to_png()