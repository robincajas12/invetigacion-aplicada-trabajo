import pandas as pd
import os

data_dir = "data"
filepath = os.path.join(data_dir, "battlesStaging_12282020_WL_tagged.csv")

# Load a small sample
df = pd.read_csv(filepath, nrows=10)
print(df['battleTime'].head(10))
