import pickle
from os import path
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import shap
import pandas as pd

# Initialise JavaScript
# shap.initjs()
def shap_explain(
    x_train,
    x_test,
    model,
    target,
    outputDir: str,
    drop_values=False,
    threshold=100,
    save_values=False,
):
    # Convert True/False and other non-numeric data to numeric (1/0) before any computation
    x_train_numeric = x_train.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    x_test_numeric = x_test.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)

    # Convert DataFrame to NumPy arrays for numerical operations
    x_train_np = x_train_numeric.to_numpy()
    x_test_np = x_test_numeric.to_numpy()

    y_pred = model.predict(x_test_np[:10])
    print(f"Predictions: {y_pred}")

    # Get the selected indices based on predictions
    selected_indices = np.where(np.round(y_pred) == 1)[0]
    print(f"Selected indices: {selected_indices}")

    # Select the test data based on these indices
    x_test_selected = x_test_numeric.iloc[selected_indices]
    print(f"x_test after selection: {x_test_selected.shape}")

    threshold = int(threshold)
    if drop_values:
        x_train_to_explain = np.where(
            np.sum(x_train_np, axis=0, keepdims=True) >= threshold,
            x_train_np,
            0,
        )

        x_background = np.where(
            np.sum(x_train_np, axis=0, keepdims=True) >= threshold / 2,
            x_train_np,
            0,
        )
    else:
        # Use the entire dataset without dropping values
        x_train_to_explain = x_train_np
        x_background = x_train_np
    x_background = shap.sample(x_background, 100)

    # Create SHAP explainer and compute SHAP values
    # explainer = shap.DeepExplainer(model, x_train_to_explain, x_background)
    explainer = shap.KernelExplainer(model.predict, x_background)

    shap_values = explainer.shap_values(x_test_selected)
    explanation = shap.Explanation(
        values=shap_values, base_values=explainer.expected_value, data=x_test_selected
    )
    if len(shap_values.shape) == 3:
        # Reshape (flatten) the SHAP values by merging features and classes into one dimension
        num_samples, num_features, num_classes = shap_values.shape
        shap_values = shap_values.reshape(num_samples, -1)  # Merge the last two dimensions
    else:
        shap_values = shap_values
    # Save SHAP values if requested
    if save_values:
        with open(path.join(outputDir, f"shap_values-threshold-{threshold}-{target}.pickle"), "wb") as pkl:
            pickle.dump(shap_values, pkl)
        with open(path.join(outputDir, f"shap_values-threshold-{threshold}-{target}.csv"), "w") as csv_file:
            np.savetxt(csv_file, shap_values)

    return shap_values, explanation



def shap_plots(shap_values,explanation, opts, target, plot_type=["bar", "waterfall"]):
    if "bar" in plot_type:
        shap.plots.bar(shap_values, max_display=10, show=False)
        plt.savefig(
            path.join(opts.outputDir, f"shap_bar-{target}.pdf"),
            format="pdf",
            dpi=1500,
            bbox_inches="tight",
        )
        plt.clf()
        plt.cla()
        plt.close()

    if "waterfall" in plot_type:
        shap.plots.waterfall(
            explanation,
            max_display=15,
            show=False,
        )
        plt.savefig(
            path.join(opts.outputDir, f"shap_waterfall-{target}.pdf"),
            format="pdf",
            dpi=1500,
            bbox_inches="tight",
        )
        plt.clf()
        plt.cla()
        plt.close()

    if "heatmap" in plot_type:
        shap.plots.heatmap(shap_values, max_display=10, show=False)
        plt.savefig(
            path.join(opts.outputDir, f"shap_heatmap-{target}.pdf"),
            format="pdf",
            dpi=1500,
            bbox_inches="tight",
        )
        plt.clf()
        plt.cla()

    if "force" in plot_type:
        shap.plots.force(
            explanation,
            matplotlib=True,
            show=False,
        )
        plt.savefig(
            path.join(opts.outputDir, f"shap_force-{target}.pdf"),
            format="pdf",
            dpi=1500,
            bbox_inches="tight",
        )
        plt.clf()
        plt.cla()
        plt.close()
