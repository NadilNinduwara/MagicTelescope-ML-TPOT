from tpot import TPOT
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load the data
telescope = pd.read_csv('MagicTelescope.csv')

# Clean the data
telescope_shuffle = telescope.iloc[np.random.permutation(len(telescope))]
tele = telescope_shuffle.reset_index(drop=True)

# Store 2 classes
tele['Class'] = tele['Class'].map({'g': 0, 'h': 1})
tele_class = tele['Class'].values

# Split training, testing, and validation data
training_indices, testing_indices = train_test_split(tele.index, stratify=tele_class, train_size=0.6, test_size=0.2)
training_indices, validation_indices = train_test_split(training_indices, stratify=tele_class[training_indices], train_size=0.75, test_size=0.25)

# Let Genetic Programming find the best ML model and Hyperparameters
tpot = TPOT(generations=5, verbosity=2)
tpot.fit(tele.drop('Class', axis=1).loc[training_indices].values,
         tele.loc[training_indices, 'Class'].values)

# Score the accuracy
accuracy = tpot.score(tele.drop('Class', axis=1).loc[validation_indices].values,
                      tele.loc[validation_indices, 'Class'].values)
print("Validation accuracy:", accuracy)

# Export the generated code
tpot.export('pipeline.py')
