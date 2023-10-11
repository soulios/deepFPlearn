import pickle
from os import path
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import shap


# Initialise JavaScript
# shap.initjs()
tf.compat.v1.disable_v2_behavior()

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
    y_pred = model.predict(x_test[:1000])
    print(y_pred)
    # Get the unique indices where either y_test is 1 or y_pred is 1
    selected_indices = np.where((np.round(y_pred) == 1))[0]
    print(selected_indices)
    x_test = x_test[selected_indices]
    print(x_test, x_test.shape)
    if drop_values:
        # x_train_to_explain = np.multiply(x_train, 0, where=np.sum(x_train, axis=0, dtype=np.int32, keepdims=True) < threshold)

        x_train_to_explain = np.where(
            np.sum(x_train, axis=0, dtype=np.int8, keepdims=True) >= threshold,
            x_train,
            0,
        )

        x_background = np.where(
            np.sum(x_train, axis=0, dtype=np.int8, keepdims=True) >= threshold/2,
            x_train,
            0,
        )
        x_include_labels = np.nonzero(x_train_to_explain)
        # print(x_train[x_include_labels])
        # print(model.predict)
        # print(x_train_to_explain)
        # print(f"{type(x_train_to_explain)}\t{len(x_train_to_explain)}")
        # print(f"{type(x_include_labels)}\t{len(x_include_labels[2])}")
        explainer = shap.DeepExplainer(model, x_train_to_explain, x_background)
        shap_values = explainer.shap_values(x_test)

        if save_values:
            with open(
                path.join(
                    outputDir, f"shap_values-threshold-{threshold}-{target}.pickle"
                ),
                "wb",
            ) as pkl:
                pickle.dump(shap_values, pkl)
            with open(
                path.join(outputDir, f"shap_values-threshold-{threshold}-{target}.csv"),
                "wb",
            ) as csv:
                np.savetxt(csv, shap_values)
            with open(
                path.join(
                    outputDir, f"shap_indices-threshold-{threshold}-{target}.csv"
                ),
                "wb",
            ) as idx:
                np.savetxt(idx, x_include_labels, fmt="%d", delimiter=";")
    else:
        explainer = shap.Explainer(model, np.array(x_train[:10]), max_evals="auto")
        shap_values = explainer(np.array(x_test[:10]))

        if save_values:
            with open(
                path.join(outputDir, f"shap_values-full-{target}.pickle"), "wb"
            ) as pkl:
                pickle.dump(shap_values, pkl)
            with open(
                path.join(outputDir, f"shap_values-full-{target}.csv"), "wb"
            ) as csv:
                np.savetxt(csv, shap_values)

    return shap_values


def shap_plots(shap_values, opts, target, plot_type=["bar", "waterfall"]):
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
        shap.plots._waterfall.waterfall_legacy(
            shap_values.base_values[0][0],
            shap_values.values[0],
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
            shap_values.base_values[0],
            shap_values.values[0],
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
