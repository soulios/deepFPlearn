# Running deepFPlearn

Here you will find example code for running deepFPlearn in all five modes: train, predict, traingnn, predictgnn, and convert. The input data for each of these modes can be found in the example/data folder.

The pre-computed output of the train mode can be found in the assets of the release, for the predict mode it is stored in the respective example/results_predict folder. Trained models that are used in the prediction mode are stored in the models folder.

NOTE: Before you proceed with running deepFPlearn, activate the conda environment or use the container as described in the main README.md of the repository.
## Train
The train mode is used to train models to predict the association of molecular structures to biological targets. 

The encoding of the molecules is done based on molecular fingerprints(we chose 2048 as the fp length). 

The training data contains three targets and you may train models for each using the following command:

``` 
python -m dfpl train -f example/train.json 
```
Training with the configurations from the example/train.json file will take approximately 4min on a single CPU.

The trained models, training histories and respective plots, as well as the predictions on the test data are stored in the example/results_train folder as defined in the example/train.json file (you may change this).

The train.json options file provides various options for training a model on the input data.

* One option is **trainAC**, which when set to true, trains an autoencoder model. 
* Another option is **trainRBM**, which when set to true, trains a deep belief network (DBN) using Restricted Boltzmann Machines (RBM). 
* The **useRBM** option specifies whether the RBM weights should be used to initialize the model parameters for training the final neural network.

* The **split_type** and **aeSplitType** option specifies the type of data splitting used for training the models. It can be set to either scaffold_balanced or random. In scaffold_balanced splitting, molecules are split based on their scaffold structure to ensure that similar scaffolds are present in both training and validation sets. In random splitting, the data is randomly split into training and validation sets.
**If you choose k-fold cross-validation while using scaffold split, the evaluation will be done with different seeds because the high imbalance of the data, does not allow for actual cross validation.**
* In addition, train.json also includes options for setting the type of fingerprint (fpType), the size of the fingerprint (_fpSize_), the type of neural network (_fnnType_), the optimizer (_optimizer_), the loss function (_lossFunction_), the number of epochs (_epochs_), the batch size (_batchSize_), the learning rate (_learningRate_), the L2 regularization parameter (_l2reg_), and more.

* The **wabTracking** or **aeWabTracking** option specifies whether Weights and Biases (WANDB) tracking should be used to monitor model performance during training. If either is set to true, the wabTarget option can be used to specify the target value for WAB tracking.
## Predict

The predict mode is used to predict the from molecular structures. Use this command to predict the provided data for prediction:

```
python -m dfpl predict -f example/predict.json
```
The compounds are predicted with the (provided) AR model and results are returned as a float number between 0 and 1.
* inputFile: The path to the input file containing the molecules to be predicted.
* outputDir: The directory where the output file will be saved.
* outputFile: The name of the output file containing the predicted values.
* ecModelDir: The directory where the autoencoder model is saved.
* ecWeightsFile: The name of the file containing the weights of the autoencoder. This is not needed for predicting with an AR model.
* fnnModelDir: The directory where the FNN model is saved.
* compressFeatures: Whether to compress the features using the autoencoder or not.
* useRBM: Whether to use a trained deep belief network or not. This is not needed for predicting with an AR model.
* trainAC: Whether to train a new autoencoder or use a pre-trained one. 
* trainFNN: Whether to train a new FNN model or use a pre-trained one.
## Traingnn
The traingnn mode is used to train models using a graph neural network to predict binary fingerprints from molecular structures. The training data contains three targets and you may train models for each using the following command:
```
python -m dfpl traingnn -f example/traingnn.json
```
The trained models, training histories and respective plots, as well as the predictions on the test data are stored in the example/results_traingnn folder as defined in the example/traingnn.json file (you may change this).
Similar and even more options are offered via the GNN model. Go to chemprop/args.py to take a peek and set your options.
## Predictgnn
The predictgnn mode is used to predict binary fingerprints from molecular structures using a graph neural network. Use this command to predict the provided data for prediction:
```
python -m dfpl predictgnn -f example/predictgnn.json
```
The compounds are predicted with the graph neural network model and results are returned as a float number between 0 and 1.

## Interpretgnn
Given a trained model, the interpretation mode can be used to explain the model's prediction by attributing to them responsible substructures and generating toxicity score (for more details, see Chemprop's documentation on "predicting"):
```
python -m dfpl interpretgnn -f example/interpretgnn.json
```
Returns a table of 3 columns: "smiles", target, "rationale", and "rationale_score". These respectively are: molecule, target receptor, substructure (explanation of the toxicity), and the toxicity score based on MTCS. Explained bein one property (target) at a time. It can be changed in "property_id" in interpretgnn.json file.

To tune interpretation complexity, max and min number of atoms included in "rationale" can be set. I can be done by adjusting "max_atoms" and "min_atoms" parameters in interpretgnn.json file respectively.

The parameters of interpretgnn.json include:

* py/object: dfpl.options.GnnOptions object with other important, lower-level parameters.
* data_path: path to the dataset.
* output_dir: path to a directory where the results will be saved into.
* checkpoint_dir: path to the dir where trained model's stored
* max_atoms: max # of atoms rationale (substructure that may be responsible for toxicity) can include
* min_atoms: min # of atoms respectively
* visualise_smiles: whether or not visualise rationale (substructures) if found responsible for toxicity (rationale_score > 0.5)


## Convert

The convert mode is used to convert .csv or .tsv files into .pkl files for easy access in Python and to reduce memory on disk. The .pkl files then already contain the binary