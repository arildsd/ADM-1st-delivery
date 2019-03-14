import numpy as np
import pandas
import graphviz
from sklearn import tree
import pre_processing as pre
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.utils.multiclass import unique_labels

# GLOBAL CONSTANTS
RANDOM_SEED = 799
TRAIN_FRAC = 0.6  # The fraction of the data used for training
WRITE_FILES = True


def evaluate(model, df, prints=True):
    X = df[["is_gender_female", "is_gender_male", "is_race_group A", "is_race_group B", "is_race_group C",
            "is_race_group D", "is_race_group E", "is_parent_education_associate's degree",
            "is_parent_education_bachelor's degree", "is_parent_education_high school",
            "is_parent_education_master's degree", "is_parent_education_some college",
            "is_parent_education_some high school", "is_lunch_free/reduced", "is_lunch_standard",
            "is_prepared_completed", "is_prepared_none"]]
    y = df["student performance"]

    predictions = model.predict(X)
    conf_matrix = confusion_matrix(y, predictions)
    acc = np.sum(predictions == y) / len(predictions)
    if prints:
        print("Confusion matrix:\n", conf_matrix)
        print(f"Accuracy: {acc}")

    return acc, conf_matrix

def plot_confusion_matrix(conf_matrix, title=None, classes=None, cmap=plt.cm.Blues):

    fig, ax = plt.subplots()
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(conf_matrix.shape[1]),
           yticks=np.arange(conf_matrix.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

training_accs = []
test_accs = []
for leaf_nodes in range(2, 128):
    file_prefix = f"../output/max_leaf_nodes_{leaf_nodes}/"

    # Get data
    df = pre.pre_process()

    # Features for the training set
    X = df[["is_gender_female", "is_gender_male", "is_race_group A", "is_race_group B", "is_race_group C",
                     "is_race_group D", "is_race_group E", "is_parent_education_associate's degree",
                     "is_parent_education_bachelor's degree", "is_parent_education_high school",
                     "is_parent_education_master's degree", "is_parent_education_some college",
                     "is_parent_education_some high school", "is_lunch_free/reduced", "is_lunch_standard",
                     "is_prepared_completed", "is_prepared_none"]]
    features = X.keys()
    X = np.array(X)
    # Targets for the training set
    y = df["student performance"]
    y = np.array(y)

    # Hold one out cross validation
    kf = KFold(n_splits=len(df), shuffle=True, random_state=RANDOM_SEED)
    average_test_acc = 0
    average_training_acc = 0
    counter = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        decision_tree = tree.DecisionTreeClassifier(max_leaf_nodes=leaf_nodes, random_state=RANDOM_SEED)
        decision_tree.fit(X_train, y_train)

        average_training_acc += decision_tree.score(X_train, y_train)
        average_test_acc += decision_tree.score(X_test, y_test)
        counter += 1
    average_training_acc /= counter
    average_test_acc /= counter
    training_accs.append(average_training_acc)
    test_accs.append(average_test_acc)

    # Write results to files
    decision_tree_export = tree.export_graphviz(decision_tree, feature_names=features,
                                                class_names=["Bad", "Average", "Good"])
    graph = graphviz.Source(decision_tree_export)
    if WRITE_FILES: graph.render(file_prefix+"tree_graph")

    print(f"Training accuracy: {average_training_acc}\nTest accuracy: {average_test_acc}")

    #acc, conf_matrix = evaluate(decision_tree, test)
    #plot_confusion_matrix(conf_matrix, classes=["Bad", "Average", "Good"])
    #plt.show()
    #plt.savefig(file_prefix+"confusion_matrix")
    if WRITE_FILES:
        log_file = open(file_prefix+"log.txt", "w")
        log_file.write(f"Training accuracy: {average_training_acc}\nTest accuracy: {average_test_acc}")
        log_file.close()

print("Training accs:", training_accs)
print("Test accs:", test_accs)
