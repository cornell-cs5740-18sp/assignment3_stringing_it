import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix
import itertools




def warn(*args, **kwargs): pass
import warnings; warnings.warn = warn


parser = ArgumentParser()

parser.add_argument("-p", "--predicted", dest = "pred_path",
    required = True, help = "path to your model's predicted labels file")

parser.add_argument("-d", "--development", dest = "dev_path",
    required = True, help = "path to the development labels file")

parser.add_argument("-c", "--confusion", dest = "show_confusion",
    action = "store_true", help = "show confusion matrix")

args = parser.parse_args()


pred = pd.read_csv(args.pred_path, index_col = "id")
dev  = pd.read_csv(args.dev_path,  index_col = "id")

pred.columns = ["predicted"]
dev.columns  = ["actual"]

data = dev.join(pred)

print("Accuracy: ", accuracy_score(data.actual, data.predicted))
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix

if args.show_confusion:

    data["count"] = 1
    counts = data.groupby(["actual", "predicted"]).count().reset_index()
    confusion = counts[counts.actual != counts.predicted].reset_index(drop = True)

    print("Confusion Matrix:")

    if confusion.empty: print("None!")
    else: print(confusion)

    df = pd.DataFrame(confusion)
    df.to_csv('confusion_matrix.csv')
    cm = ConfusionMatrix(confusion['predicted'],confusion['actual'])
    cm.plot(normalized=True)
    plt.show()
    # cnf_matrix = confusion_matrix(confusion['predicted'],confusion['actual'])
    # np.set_printoptions(precision=2)
    #
    # # Plot non-normalized confusion matrix
    # plt.figure()
    # class_names = list(set(pd.read_csv('data/dev_y.csv')['tag']))
    # plot_confusion_matrix(cnf_matrix, classes=class_names,
    #                       title='Confusion matrix, without normalization')
    #
    # # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                       title='Normalized confusion matrix')
    #
    # plt.show()
