import sys
import pandas as pd
from sklearn.model_selection import train_test_split

if len(sys.argv) != 4:
    print("usage: {} <datafile (csv)> <problemname> <test_size (fraction)>".format(sys.argv[0]))
    sys.exit(1)

datafile = sys.argv[1]
problemname = sys.argv[2]
test_size = float(sys.argv[3])

df = pd.read_csv(datafile)

print(df)
train, test = train_test_split(df, test_size=test_size)

train.to_csv('train_data_{}.csv'.format(problemname), index=False)
test.to_csv('test_data_{}.csv'.format(problemname), index=False)
#train.to_csv('train_data_{}.csv'.format(problemname), index=False, float_format="%.6f")
#test.to_csv('test_data_{}.csv'.format(problemname), index=False, float_format="%.6f")

