import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(parent_dir)
sys.path.append('.')

from utils import input_parser, get_cfg_n
args = input_parser() 
_, _, cfg_n = get_cfg_n(args)

print(f"{cfg_n}")