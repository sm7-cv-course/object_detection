"""
Determining of the strongest features
via Random Forest classifier.
"""
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

# Read table from saved in advance file
with open('hog_descriptors.src', 'rb') as f:
    readtable = pickle.load(f)

# Initialize "forest" classifier
forest = RandomForestClassifier(n_estimators=10000,
                                random_state=0,
                                n_jobs=-1)

# Train "forest" classifier
Labels, Features = zip(*readtable)
Labels = np.array(Labels)
Features = np.array(Features)

print("X_train = ", Labels)
print("Y_train = ", Features)
print(type(Labels))
print(type(Features))
print(Labels.shape)
print(Features.shape)

# Training of classifier
forest.fit(Features, Labels)

# Estimate features importance
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(Labels.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(Labels.shape[1]),
        importances[indices],
        color='lightblue',
        align='center')

plt.xticks(range(Labels.shape[1]),
           feat_labels[indices], rotation=90)
plt.xlim([-1, Labels.shape[1]])
plt.tight_layout()
#plt.savefig('./random_forest.png', dpi=300)
plt.show()
