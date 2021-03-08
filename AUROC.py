from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

def AUROC_plot(true, scores, title = 'None', color='k', fname = 'None.pdf'):

    list_true =true
    list_scores =scores

    x_margin = 0.05
    y_margin = 0.05
    zoom = 12

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 2
    lw = 2

    f, a = plt.subplots(1, figsize=(3,3))

    fpr, tpr, _ = sklearn.metrics.roc_curve(list_true, list_scores)
    roc_auc = auc(fpr, tpr)

    a.plot(fpr, tpr, color=color,
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)

    a.set_xlim([-0.05, 1.0])
    a.set_ylim([0.0, 1.05])
    a.set_xlabel('FPR')
    a.set_ylabel('TPR')
    a.set_xticks([0, 0.5, 1])
    a.set_yticks([0, 0.5, 1])
    a.set_title(title)



    axins = zoomed_inset_axes(a, zoom, loc=4, borderpad=2) 
    axins.plot(fpr, tpr, color=color, lw=lw*2)
    x1, x2, y1, y2 = -0.002, 0+x_margin, 1.00-y_margin, 1.002 # specify the limits
    axins.set_xlim(x1, x2) # apply the x-limits
    axins.set_ylim(y1, y2) # apply the y-limits
    axins.set_xticks([0.00, x2])
    axins.set_yticks([y1, 1.00])
    axins.xaxis.set_visible('False')
    axins.yaxis.set_visible('False')
    mark_inset(a, axins, loc1=1, loc2=3, fc="none", ec="0.3")
    f.savefig(fname, dpi=270, format='pdf')
    print(fname, 'saved')
    
#AUROC_plot(y_true, y_scores)
#plt.show()




def metric_with_ci_mc(y_true, y_pred, n_bootstraps = 1000):
    # AUROC, Sensitivity and Specificity calc.
    # n=1000 bootstrapping default.

    def bootstrapping(metric_fn, y_true = y_true, y_pred = y_pred):
        rng_seed = 42  # control reproducibility
        bootstrapped_scores = []
        bootstrapped_roc = []

        rng = np.random.RandomState(rng_seed)
        while len(bootstrapped_scores) < n_bootstraps:
            # bootstrap by sampling with replacement on the prediction indices
            indices = rng.randint(0, len(y_pred), len(y_pred))
            if len(np.unique(y_true[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue
            score = metric_fn(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)
            #print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

        unsorted_scores = np.array(bootstrapped_scores)
        argsort = unsorted_scores.argsort()

        # Computing the lower and upper bound of the 90% confidence interval
        # You can change the bounds percentiles to 0.025 and 0.975 to get
        # a 95% confidence interval instead.

        confidence_lower = unsorted_scores[argsort[int(0.025 * len(argsort))]]
        confidence_upper = unsorted_scores[argsort[int(0.975 * len(argsort))]]
        return [metric_fn(y_true, y_pred), confidence_lower, confidence_upper]
    
