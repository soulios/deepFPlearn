import numpy as np
import math
import matplotlib.pyplot as plt
import shap
import pickle
from os import path

from dfpl import options

# Initialise JavaScript
shap.initjs()

def shap_explain(x_train, model, target, outputDir: str, drop_values=False, threshold=100, save_values=False):
    
    if drop_values:
        x_train_to_explain = np.multiply(x_train, 0, where=np.sum(x_train, axis=0, dtype=np.int32, keepdims=True) < threshold)
        # masker = shap.maskers.Independent(x_train, max_samples=100)
        # explainer = shap.Explainer(model, masker=shap.maskers.Independent(data=x_train, max_samples=100), max_evals=4097)
        explainer = shap.explainers.Permutation(model.predict, x_train, max_evals=max_it)
        shap_values = explainer(x_train_to_explain)

        if save_values:
            with open(path.join(outputDir, f"shap_values-{target}-drop.pickle"), "wb") as pkl:
                pickle.dump(shap_values, pkl)
    else:
        explainer = shap.explainers.Permutation(model.predict, x_train, max_evals=2113)
        shap_values = explainer(x_train)


        if save_values:
            with open(path.join(outputDir, f"shap_values-{target}-full.pickle"), "wb") as pkl:
                pickle.dump(shap_values, pkl)

    return shap_values


def shap_plots(shap_values, opts, target, plot_type=["bar", "waterfall"]):


    if "bar" in plot_type:
        shap.plots.bar(shap_values, max_display=10, show=False)
        plt.savefig(path.join(opts.outputDir, f"shap_bar-{target}.pdf"), format='pdf', dpi=1500, bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close()

    if "waterfall" in plot_type:
        shap.plots._waterfall.waterfall_legacy(shap_values.base_values[0][0], shap_values.values[0], max_display=15, show=False)
        plt.savefig(path.join(opts.outputDir, f"shap_waterfall-{target}.pdf"), format='pdf', dpi=1500, bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close()

    if "heatmap" in plot_type:
        shap.plots.heatmap(shap_values, max_display=10, show=False)
        plt.savefig(path.join(opts.outputDir, f"shap_heatmap-{target}.pdf"), format='pdf', dpi=1500, bbox_inches='tight')
        plt.clf()
        plt.cla()

    if "force" in plot_type:
        shap.plots.force(shap_values.base_values[0], shap_values.values[0], matplotlib=True, show=False)
        plt.savefig(path.join(opts.outputDir, f"shap_force-{target}.pdf"), format='pdf', dpi=1500, bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close()

