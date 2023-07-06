Cherenkov Telescope Gamma Radiation Classifier


This project aims to build a gamma radiation classifier using machine learning techniques and Genetic Programming. The goal is to find the best machine learning model and its hyperparameters automatically using TPOT (Tree-based Pipeline Optimization Tool) library.

Dataset
The dataset used for this project is the "MagicTelescope.csv" file, which contains data from Cherenkov telescopes. The dataset can be downloaded from Cherenkov Telescope website. It includes various features related to gamma radiation detection, along with corresponding class labels ('g' or 'h') indicating the type of radiation.

Prerequisites
To run this project, the following libraries need to be installed using pip:

tpot
pandas
numpy
sklearn

Usage
Download the "MagicTelescope.csv" dataset from the Cherenkov Telescope website.
Clone the repository to your local machine.
Place the "MagicTelescope.csv" dataset in the project directory.
Open a Python IDE or Jupyter Notebook and navigate to the project directory.
Run the provided code to perform the following steps:
Load the dataset and shuffle the data.
Prepare the data by mapping the class labels to binary values.
Split the data into training, testing, and validation sets.
Use Genetic Programming with TPOT to find the best machine learning model and hyperparameters.
Evaluate the accuracy of the model using the validation set.
Export the generated code for the best model and hyperparameters to "pipeline.py".
Review the validation accuracy printed during the execution of the code.

Results
The TPOT algorithm utilizes Genetic Programming to automatically search for the best machine learning model and its hyperparameters for the gamma radiation classification task. The validation accuracy obtained provides an estimate of the model's performance on unseen data.

Contributing
Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

License
This project is licensed under the MIT License. Feel free to use and modify the code as per the terms of the license.

Acknowledgments
This project utilizes the TPOT library and follows the approach provided by the Cherenkov Telescope website for gamma radiation classification.

References
TPOT: Tree-based Pipeline Optimization Tool
Cherenkov Telescope
