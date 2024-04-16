# the following script reads an arguments named training_data, which specifies the path to the training data.

'''it is just an example it will not show anything because the file is asking here is not in the system
and i am not making any file there just to show that how a code will run remember that, My dear! '''
 # import libraries
"""import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression

def main(args):
    # read data
    df = get_data(args.training_data)

# function that reads the data
def get_data(path):
    df = pd.read_csv(path)
    
    return df

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":

    # parse args
    args = parse_args()

    # run main function
    main(args)"""


#this is an example how to create and train a logistic regression model using scikit-learn.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load your dataset (replace with your own data)
# X: Features (independent variables)
# y: Target variable (dependent variable)
# Example:
# X, y = load_your_data()

# Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
# logreg = LogisticRegression()

# Train the model
# logreg.fit(X_train, y_train)

# Evaluate the model (e.g., accuracy, precision, recall, etc.)
# Example:
# accuracy = logreg.score(X_test, y_test)
# print(f"Accuracy: {accuracy:.2f}")


# in there we does't put the value that why it will give ys ouput but not in number(integer format).
# it is just an format remeber that you have to work upon it later after completing azure machine learning model.
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum the integers (default: find the max)')
args = parser.parse_args()
print(args.accumulate(args.integers))


