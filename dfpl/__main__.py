import dataclasses
import logging
import os.path
from argparse import Namespace
from os import path

import chemprop as cp
from keras.models import load_model
import pandas as pd
from dfpl import autoencoder as ac
from dfpl import feedforwardNN as fNN
from dfpl import fingerprint as fp
from dfpl import options, predictions
from dfpl import single_label_model as sl
from dfpl import shap_dfpl as sd
from dfpl import interpret as explain
from dfpl import visualise
from dfpl import vae as vae
from dfpl.utils import createArgsFromJson, createDirectory, makePathAbsolute


def traindmpnn(opts: options.GnnOptions) -> None:
    """
    Train a D-MPNN model using the given options.
    Args:
    - opts: options.GnnOptions instance containing the details of the training
    Returns:
    - None
    """
    # Load options from a JSON file and replace the relevant attributes in `opts`
    arguments = createArgsFromJson(jsonFile=opts.configFile)
    opts = cp.args.TrainArgs().parse_args(arguments)
    logging.info("Training DMPNN...")
    mean_score, std_score = cp.train.cross_validate(
        args=opts, train_func=cp.train.run_training
    )
    logging.info(f"Results: {mean_score:.5f} +/- {std_score:.5f}")


def predictdmpnn(opts: options.GnnOptions) -> None:
    """
    Predict the values using a trained D-MPNN model with the given options.
    Args:
    - opts: options.GnnOptions instance containing the details of the prediction
    Returns:
    - None
    """
    # Load options and additional arguments from a JSON file
    arguments = createArgsFromJson(jsonFile=opts.configFile)
    opts = cp.args.PredictArgs().parse_args(arguments)

    cp.train.make_predictions(args=opts)

def interpretdmpnn(opts: options.GnnOptions) -> None:
    """
    Interpret the values using a trained D-MPNN model with the given options.
    Args:
    - opts: options.GnnOptions instance containing the details of the prediction
    - JSON_ARG_PATH: path to a JSON file containing additional arguments for interpretation
    Returns:
    -
    """
    ignore_elements = ["py/object"]
    # Load options and additional arguments from a JSON file
    arguments = createArgsFromJson(
        jsonFile=opts.configFile
    )
    opts = cp.args.InterpretArgs().parse_args(arguments)
    # Define the target properties and their corresponding IDs
    target_properties = {'AR': 1, 'ER': 2, 'ED': 7}

    for target_name, property_id in target_properties.items():
        opts.property_id = property_id
        df = explain.interpret(
            args=opts,visualise_smiles=True
        )
        df.to_csv(f'interpretation_{target_name}.csv', index=False)

def interpretffn(opts: options.Options) -> None:
    """
    Interpret the values using a trained FFN model with the given options.

    Args:
    - opts: options.Options instance containing the details of the prediction

    Returns:
    -
    """
    args = createArgsFromJson(
        jsonFile=opts.configFile
    )

    def list_to_dict(args_list):
        if len(args_list) % 2 != 0:
            raise ValueError("List length must be even, with alternating keys and values.")
        return {args_list[i]: args_list[i + 1] for i in range(0, len(args_list), 2)}

    args = list_to_dict(args)
    #read all except the last column
    x_train = pd.read_csv(path.join(opts.outputDir, f"{args.get('--target')}_train_fold_0.csv"), index_col=-1)
    x_test = pd.read_csv(path.join(opts.outputDir, f"{args.get('--target')}_test_fold_0.csv"), index_col=-1)
    print(x_test.shape,x_train.shape)
    model = tf.keras.models.load_model(path.join(opts.outputDir, "NR-AR_saved_model"),
                                       custom_objects={'balanced_accuracy': sl.balanced_accuracy})
    print(model.summary())
    shap_values = sd.shap_explain(x_train=x_train, x_test=x_test, model=model,
                                  target=args.get("--target"),
                                  outputDir=opts.outputDir,
                                  drop_values=args.get("--drop_values"),
                                  threshold=args.get("--threshold"),
                                  save_values=args.get("--save_values")
                                  )
    plot_types = ["bar", "waterfall", "heatmap", "force"]
    plot_types = ["waterfall"]
    sd.shap_plots(shap_values, opts, args.get("--target"), plot_types)

def train(opts: options.Options):
    """
    Run the main training procedure
    :param opts: Options defining the details of the training
    """
    # import data from file and create DataFrame
    if "tsv" in opts.inputFile:
        df = fp.importDataFile(
            opts.inputFile, import_function=fp.importDstoxTSV, fp_size=opts.fpSize
        )
    else:
        df = fp.importDataFile(
            opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize
        )
    # initialize (auto)encoders to None
    encoder = None
    autoencoder = None
    if opts.trainAC:
        if opts.aeType == "deterministic":
            encoder, train_indices, test_indices = ac.train_full_ac(df, opts)
        elif opts.aeType == "variational":
            encoder, train_indices, test_indices = vae.train_full_vae(df, opts)
        else:
            raise ValueError(f"Unknown autoencoder type: {opts.aeType}")

    # if feature compression is enabled
    if opts.compressFeatures:
        if not opts.trainAC:
            if opts.aeType == "variational":
                (autoencoder, encoder) = vae.define_vae_model(opts=options.Options())
            else:
                (autoencoder, encoder) = ac.define_ac_model(opts=options.Options())

            if opts.ecWeightsFile == "":
                encoder = load_model(opts.ecModelDir)
            else:
                autoencoder.load_weights(
                    os.path.join(opts.ecModelDir, opts.ecWeightsFile)
                )
        # compress the fingerprints using the autoencoder
        df = ac.compress_fingerprints(df, encoder)
        if opts.visualizeLatent:
            ac.visualize_fingerprints(
                df,
                before_col="fp",
                after_col="fpcompressed",
                train_indices=train_indices,
                test_indices=test_indices,
                save_as=f"UMAP_{opts.aeSplitType}.png",
            )
    # train single label models if requested
    if opts.trainFNN and not opts.enableMultiLabel:
        sl.train_single_label_models(df=df, opts=opts)

    # train multi-label models if requested
    if opts.trainFNN and opts.enableMultiLabel:
        fNN.train_nn_models_multi(df=df, opts=opts)


def predict(opts: options.Options) -> None:
    """
    Run prediction given specific options
    :param opts: Options defining the details of the prediction
    """
    # import data from file and create DataFrame
    if "tsv" in opts.inputFile:
        df = fp.importDataFile(
            opts.inputFile, import_function=fp.importDstoxTSV, fp_size=opts.fpSize
        )
    else:
        df = fp.importDataFile(
            opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize
        )

    if opts.compressFeatures:
        # load trained model for autoencoder
        if opts.aeType == "deterministic":
            (autoencoder, encoder) = ac.define_ac_model(opts=options.Options())
        if opts.aeType == "variational":
            (autoencoder, encoder) = vae.define_vae_model(opts=options.Options())
        # Load trained model for autoencoder
        if opts.ecWeightsFile == "":
            encoder = load_model(opts.ecModelDir)
        else:
            encoder.load_weights(os.path.join(opts.ecModelDir, opts.ecWeightsFile))
        df = ac.compress_fingerprints(df, encoder)

    # Run predictions on the compressed fingerprints and store the results in a dataframe
    df2 = predictions.predict_values(df=df, opts=opts)

    # Extract the column names from the dataframe, excluding the 'fp' and 'fpcompressed' columns
    names_columns = [c for c in df2.columns if c not in ["fp", "fpcompressed"]]

    # Save the predicted values to a CSV file in the output directory
    df2[names_columns].to_csv(path_or_buf=path.join(opts.outputDir, opts.outputFile))

    # Log successful completion of prediction and the file path where the results were saved
    logging.info(
        f"Prediction successful. Results written to '{path.join(opts.outputDir, opts.outputFile)}'"
    )


def createLogger(filename: str) -> None:
    """
    Set up a logger for the main function that also saves to a log file
    """
    # get root logger and set its level
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create file handler which logs info messages
    fh = logging.FileHandler(filename, mode="w")
    fh.setLevel(logging.INFO)
    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatterFile = logging.Formatter(
        "{asctime} - {name} - {levelname} - {message}", style="{"
    )
    formatterConsole = logging.Formatter("{levelname} {message}", style="{")
    fh.setFormatter(formatterFile)
    ch.setFormatter(formatterConsole)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


def main():
    """
    Main function that runs training/prediction defined by command line arguments
    """

    parser = options.createCommandlineParser()
    prog_args: Namespace = parser.parse_args()
    try:
        if prog_args.method == "convert":
            directory = makePathAbsolute(prog_args.f)
            if path.isdir(directory):
                createLogger(path.join(directory, "convert.log"))
                logging.info(f"Convert all data files in {directory}")
                fp.convert_all(directory)
            else:
                raise ValueError("Input directory is not a directory")
        elif prog_args.method == "traingnn":
            traingnn_opts = options.GnnOptions.fromCmdArgs(prog_args)
            createLogger("traingnn.log")
            traindmpnn(traingnn_opts)

        elif prog_args.method == "predictgnn":
            predictgnn_opts = options.GnnOptions.fromCmdArgs(prog_args)
            fixed_opts = dataclasses.replace(
                predictgnn_opts,
                test_path=makePathAbsolute(predictgnn_opts.test_path),
                preds_path=makePathAbsolute(predictgnn_opts.preds_path),
            )
            createLogger("predictgnn.log")
            predictdmpnn(fixed_opts)

        elif prog_args.method == "train":
            train_opts = options.Options.fromCmdArgs(prog_args)
            fixed_opts = dataclasses.replace(
                train_opts,
                inputFile=makePathAbsolute(train_opts.inputFile),
                outputDir=makePathAbsolute(train_opts.outputDir),
            )
            createDirectory(fixed_opts.outputDir)
            createLogger(path.join(fixed_opts.outputDir, "train.log"))
            logging.info(
                f"The following arguments are received or filled with default values:\n{fixed_opts}"
            )
            train(fixed_opts)
        elif prog_args.method == "predict":
            predict_opts = options.Options.fromCmdArgs(prog_args)
            fixed_opts = dataclasses.replace(
                predict_opts,
                inputFile=makePathAbsolute(predict_opts.inputFile),
                outputDir=makePathAbsolute(predict_opts.outputDir),
                outputFile=makePathAbsolute(
                    path.join(predict_opts.outputDir, predict_opts.outputFile)
                ),
                ecModelDir=makePathAbsolute(predict_opts.ecModelDir),
                fnnModelDir=makePathAbsolute(predict_opts.fnnModelDir),
            )
            createDirectory(fixed_opts.outputDir)
            createLogger(path.join(fixed_opts.outputDir, "predict.log"))
            logging.info(
                f"The following arguments are received or filled with default values:\n{prog_args}"
            )
            predict(fixed_opts)
        elif prog_args.method == "interpretgnn":
            interpretgnn_opts = options.GnnOptions.fromCmdArgs(prog_args)
            interpretdmpnn(interpretgnn_opts)

        elif prog_args.method == "interpretffn":
            interpretffn_opts = options.Options.fromCmdArgs(prog_args)
            interpretffn(interpretffn_opts)

    except AttributeError as e:
        print(e)
        parser.print_usage()


if __name__ == "__main__":
    main()
