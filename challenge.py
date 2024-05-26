"""
Streamlined Distillat des Python Notebooks
"""

from analysis import Analysis

if __name__ == "__main__": 
    analysis = Analysis()
    analysis.load_and_clean()
    analysis.transform_features()
    analysis.train_test_split()
    analysis.train_dtr()
    analysis.train_nnls()
    analysis.eval_dtr()
    analysis.eval_nnls()