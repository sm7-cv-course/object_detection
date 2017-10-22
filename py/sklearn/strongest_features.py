"""
Determine the strongest features
via Random Forest classifier.
http://localhost:8888/notebooks/ch04.ipynb
"""
from sklearn.ensemble import RandomForestClassifier
import pickle
import argparse
import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--table", required=True, help="File with table {[label, features vector]}, list of tuples.")
ap.add_argument("-m", "--model", required=True, help="File with trained classifier params (load or save).")
args = vars(ap.parse_args())

# Read table {labels, HOG features} from saved in advance file. Contains list of tuples.
with open(args["table"], 'rb') as f:
    table = pickle.load(f)

# Initialize "forest" classifier.
random_forest = RandomForestClassifier(n_estimators=10000,
                                       random_state=0,
                                       n_jobs=-1)

# Load or train "forest" classifier.
Labels, Features = zip(*table)
Labels = np.array(Labels)
Features = np.array(Features)

# Load trained classifier.
random_forest = joblib.load(args["model"])

if random_forest is None:
    # Training of classifier.
    random_forest.fit(Features, Labels)
    # Save trained classifier.
    joblib.dump(random_forest, args["model"])

# Get features importances.
importances = random_forest.feature_importances_
indices = np.argsort(importances)[::-1]

# Print out sorted list of features importance.
for f in range(indices.shape[0]):
    print("%2d) %-*s %f" % (f + 1, 30,
                             Labels[indices[f]],
                             importances[indices[f]]))

# Plot histogram  of features importance.
plt.title('Features importance')

plt.bar(range(importances.shape[0]),
        importances[indices],
        color='lightblue',
        align='center')

plt.xticks(range(importances.shape[0]),
           Labels[indices], rotation=90)

plt.xlim([-1, importances.shape[0]])
plt.tight_layout()
# plt.savefig('./random_forest.png', dpi=300)
plt.show()



# Another examples of loading/saving classifier.
#
# import pickle
#
# classifier = RandomForestClassifier(etc)
# output = open('classifier.pkl', 'wb')
# pickle.dump(classifier, output)
# output.close()
#
# The “other people” could then reload the pickled object as follows:
#
# import pickle
#
# f = open('classifier.pkl', 'rb')
# classifier = pickle.load(f)
# f.close()
