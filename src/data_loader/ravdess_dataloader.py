import pandas as pd

class RavdessDataloader:
    def __init__(self, annot_path, split, random_seed):
        self.annot_path = annot_path
        self.split = split
        self.random_seed = random_seed
        
        self.annotations = pd.read_csv(annot_path).sample(frac=1, )