#######
## Assignment 3 - Query expansion with word embeddings
####### 

# Importing packages
import sys
import os
import argparse
import pandas as pd
import numpy as np
import gensim.downloader as api

sys.path.append(os.path.join('..'))

#######
## Functions
#######

# Argument parser
def parse_arguments():
    parser =  argparse.ArgumentParser(description="Query expansion with word embeddings")
    parser.add_argument("--datapath", type=str, default="in/archive/", help="Write the path to the data")
    
    return parser.parse_args()
    

def load_data(path_to_file):
    # Load the data
    data = pd.read_csv(path_to_file)
    return data

def main():
    # Parse the arguments
    args = parse_arguments()

    # Load the data
    load_data(args.datapath)
    
if __name__=="__main__":
    main()