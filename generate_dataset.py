import numpy as np
import pandas as pd

def get_output(x,y,z):
    return x*x + 4*y +z

data = np.random.randint(1, 5, size=(100))
out_file = "data.csv"
df = pd.DataFrame(columns=['X', 'Y', 'Z'])
df['X'] = data
df['Y'] = data
df['Z'] = data

df['output'] = df.apply(lambda k: get_output(k['X'],k['Y'],k['Z']), axis=1)

print(df)
df.to_csv(out_file)