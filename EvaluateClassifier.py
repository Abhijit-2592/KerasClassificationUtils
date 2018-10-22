# @Author: abhijit
# @Date:   25-06-2018
# @Email:  balaabhijit5@gmail.com
# @Last modified by:   abhijit
# @Last modified time: 25-06-2018
from keras.preprocessing.image import ImageDataGenerator
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
from keras.models import model_from_json
import os
from itertools import cycle
from keras.utils import to_categorical
###############################################################################################################################

# this is an internal function and do not call this externally. Call only the make_cm_generator


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Do not call this function from outside.

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#########################################################################################


def custom_threshold(predictions, bias_index, probability):
    """ Utility function to threshold with a bias towards a particular class.

    Keyword Arguments
    predictions -- numpy ndarray: Predictions from model.predict or model.predic_generator
    bias_index --int: The index in the numpy ndarray for the class to be biased for
    probability --float: The probability to threshold the bias_class at.
    Output
    Returns a numpy array of predicted classes with the given bias
    """
    classes = []
    for prediction in predictions:
        if prediction[bias_index] >= probability:
            temp_class = bias_index
        else:
            prediction[bias_index] = 0  # make the prediction as 0
            temp_class = np.argmax(prediction)
        classes.append(temp_class)
    classes = np.array(classes)
    return(classes)
##########################################################################################################


def make_cm_generator(images_path,
                      model_path,
                      weights_path,
                      img_shape,
                      save_path=None,
                      batch_size=16,
                      preprocessing_function=None,
                      normalize=False,
                      bias_tuple=(),  # eg ('dog',0.3) i.e bias at 30% probability for dog class
                      class_name_dict=None,  # eg {'dog': 0, 'cat': 1}
                      fig_size=(8, 8)
                      ):
    """ Generates confusion matrix and classification report using keras generator

    NOTE: Can be used for any number of test images and any number of classes (not limited to binary) because it uses a generator

    Keyword arguments:
    images_path --str: full path to the parent directory containing sub-directory(classes) of images
    model_path --str: full path to a keras model (.json file) (No default)
    weights_path --str: full path to the weights file (.hdf5 file) (No default)
    img_shape --tuple: image shape to input to the model (eg : (224,224,3)) (No default)
    save_path --str: Optional path to save the created Confusion Matrix Dataframe as .pkl (Default None)
    batch_size --int: The batch_size to use for prediction (Default 16)
    preprocessing_function --function: The preprocessing function to use before prediction (Default None)
    normalize -- boolean: whether to normalize the confusion matrix (Default False)
    bias_tuple -- tuple: Optional argument if you want to be biased towards a particular class (Default () 'empty')
                         Example: ('dog',0.3) i.e Infer as 'dog' class
                                  if the probability is 0.3 or greater in the prediction
    class_name_dict -- dict: Optional class name dictionary to use. eg {'dog': 0, 'cat': 1}

    Output:
    Returns a Dataframe in the following format. and saves the same as a pickle file

    ----------------------------------------------
    | Component name  |  True value | Prediction |
    ----------------------------------------------
    |   string        | class_name  | class_name |
    ----------------------------------------------

    """

    # loading model and weights
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights
    loaded_model.load_weights(weights_path)
    print('loaded model from disk')
    # generator for data
    test_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
    test_generator = test_datagen.flow_from_directory(images_path,
                                                      target_size=(img_shape[0], img_shape[1]),
                                                      class_mode='categorical',
                                                      batch_size=batch_size,
                                                      shuffle=False,  # to get ordered result
                                                      )
    # this is an important step else there is a difference in result between predict and predict_generator
    test_generator.reset()  # reset to start with sample 0
    nb_samples = test_generator.samples
    if not class_name_dict:
        class_name_dict = test_generator.class_indices  # eg {'dog': 0, 'cat': 1}
    print(class_name_dict)
    # predict
    predictions = loaded_model.predict_generator(test_generator,
                                                 steps=nb_samples // batch_size,
                                                 max_queue_size=10,
                                                 workers=1,
                                                 use_multiprocessing=False,
                                                 verbose=1)
    probabilities = predictions.copy()
    if len(bias_tuple) == 0:
        print("Thresholding the classes using np.argmax (NO bias threshold)")
        classes_predicted = np.argmax(predictions, axis=1)
    elif len(bias_tuple) == 2:
        print("Thresholding the classes with the given bias")
        bias_index = class_name_dict[bias_tuple[0]]
        classes_predicted = custom_threshold(predictions, bias_index, bias_tuple[1])
    else:
        raise ValueError("The given bias tuple {} is not supported. The supported format is ('class_name',probability)".format(bias_tuple))

    # reverse the classs_name_dictionary (values and keys)
    class_name_dict_reverse = dict(map(reversed, class_name_dict.items()))  # eg {0: 'dog', 1: 'cat'}
    # create a dataframe
    classes_GT = test_generator.classes
    Predictions = []
    GT = []
    Filenames = []
    for t, p, name in zip(classes_GT, classes_predicted, test_generator.filenames):
        Predictions.append(class_name_dict_reverse[p])
        GT.append(class_name_dict_reverse[t])
        Filenames.append(images_path + '/' + name)  # full path
    df = pd.DataFrame({"Component name": Filenames, "True value": GT, "Prediction": Predictions, "Pred Prob": list(probabilities)})
    # generate classification report and confusion matrix
    target_names = []
    for key in range(0, len(class_name_dict_reverse)):
        temp = "class{}: {}".format(key, class_name_dict_reverse[key])
        target_names.append(temp)

    class_names = np.array(target_names)
    print('\n The classification report is printed below \n')
    print(classification_report(y_true=classes_GT, y_pred=classes_predicted, target_names=target_names))
    print('\n The confusion matrix generated using scikit learn library is printed below \n')
    cm = confusion_matrix(y_true=classes_GT, y_pred=classes_predicted)
    plt.figure(figsize=fig_size)
    plot_confusion_matrix(cm, classes=class_names, normalize=normalize,
                          title='Confusion matrix')
    plt.show()
    # save the dataframe
    if save_path:
        df.to_pickle(save_path)
        print("The Confusion matrix is returned and is saved in the path {}".format(save_path))
    else:
        print("The Confusion matrix df is returned")
    return(df)

############################################################################################################################


def plot_roc(images_path,
             model_path,
             weights_path,
             img_shape,
             required_class,
             batch_size=16,
             preprocessing_function=None):
    """ Plots the ROC curve and prints the Area under the Curve

    Keyword Arguments
    images_path --str: full path to the parent directory containing sub-directory(classes) of images
    model_path --str: full path to a keras model (.json file) (No default)
    weights_path --str: full path to the weights file (.hdf5 file) (No default)
    img_shape --tuple: image shape to input to the model (eg : (224,224,3)) (No default)
    required_class --str: The name of the required class on which to generate ROC (example "dog")
    batch_size --int: The batch_size to use for prediction (Default 16)
    preprocessing_function --function: The preprocessing function to use before prediction (Default None)

    Output
    Plots the ROC graph and prints the AUC
    Returns (fpr,tpr,thresholds) which can be passed to evaluate_threshold to find the custom threshold
    """

    # loading model and weights
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights
    loaded_model.load_weights(weights_path)
    print('loaded model from disk')
    # generator for data
    test_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
    test_generator = test_datagen.flow_from_directory(images_path,
                                                      target_size=(img_shape[0], img_shape[1]),
                                                      class_mode='categorical',
                                                      batch_size=batch_size,
                                                      shuffle=False,  # to get ordered result
                                                      )
    # this is an important step else there is a difference in result between predict and predict_generator
    test_generator.reset()  # reset to start with sample 0
    nb_samples = test_generator.samples
    class_name_dict = test_generator.class_indices  # eg {'dog': 0, 'cat': 1}

    keys = class_name_dict.keys()

    if required_class not in keys:
        raise ValueError("The required class {} is not in the class_indices, The class indices are {}".format(required_class, keys))

    print(class_name_dict)
    # binarize the GT wrt to the required class i.r required class is set to 1
    y_true = []
    for name in test_generator.filenames:
        if os.path.dirname(name) == required_class:
            y_true.append(1)
        else:
            y_true.append(0)
    # predict
    predictions = loaded_model.predict_generator(test_generator,
                                                 steps=nb_samples // batch_size,
                                                 max_queue_size=10,
                                                 workers=1,
                                                 use_multiprocessing=False,
                                                 verbose=1)
    # return(predictions,test_generator)

    class_idx = class_name_dict[required_class]
    # get the predicted probabilities for the required class
    print("The ROC curve for the class : {} is \n".format(required_class))
    Y_pred_prob = predictions[:, class_idx]  # prob of that particular class
    fpr, tpr, thresholds = roc_curve(y_true, Y_pred_prob)  # pass the prob (2nd argument) not the class
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve for {}'.format(required_class))
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    auc = roc_auc_score(y_true, Y_pred_prob)
    print("The AUC score for {} is : {}".format(required_class, auc))
    return(fpr, tpr, thresholds)

##############################################################################################################################################


def evaluate_threshold(fpr, tpr, thresholds, custom_threshold):
    """ Utility function to find the corresponding specificity and sensitivity from a custom threshold.

    Keyword arguments
    fpr
    tpr
    thresholds -- all 3 are outputs from plot_roc()
    custom_threshold --int : The threshold for which you want to find the points in the ROC curve

    OUTPUT:
        Prints the Sensitivity and Specificity.

    NOTE:
        This function is useful because Threshold is not given in ROC curve.
        Thus pass arbitary custom_thresholds to find which threshold corresponds to the points in the ROC curve

    """
    print('Sensitivity:', tpr[thresholds > custom_threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > custom_threshold][-1])

###############################################################################################################################


def plot_prediction_histogram(images_path,
                              model_path,
                              weights_path,
                              img_shape,
                              required_class,
                              class_name_dict={},
                              batch_size=16,
                              preprocessing_function=None,
                              plotting_module="matplotlib"):
    """ Function to plot the histogram of predicted probabilities of a given class

    images_path --str: full path to the parent directory containing sub-directory(classes) of images
    model_path --str: full path to a keras model (.json file) (No default)
    weights_path --str: full path to the weights file (.hdf5 file) (No default)
    img_shape --tuple: image shape to input to the model (eg : (224,224,3)) (No default)
    required_class --str: The name of the required class on which to generate ROC (example "dog")
    class_name_dict --dict: Dictionary mapping of classes (default {})
    batch_size --int: The batch_size to use for prediction (Default 16)
    preprocessing_function --function: The preprocessing function to use before prediction (Default None)
    plotting_module --str: The plotting module to use. Either of 'matplotlib' or 'bokeh' (Default matplotlib)

    Output:
    Plots the histogram of the predicted probabilites of the required class

    """
    if len(class_name_dict) == 0:
        raise ValueError("Provide the class_name_dict")

    # loading model and weights
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights
    loaded_model.load_weights(weights_path)
    print('loaded model from disk')
    # generator for data
    test_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
    test_generator = test_datagen.flow_from_directory(images_path,
                                                      target_size=(img_shape[0], img_shape[1]),
                                                      class_mode='categorical',
                                                      batch_size=batch_size,
                                                      shuffle=False,  # to get ordered result
                                                      )
    # this is an important step else there is a difference in result between predict and predict_generator
    test_generator.reset()  # reset to start with sample 0
    nb_samples = test_generator.samples
    class_name_dict = test_generator.class_indices  # eg {'dog': 0, 'cat': 1}
    # predict
    predictions = loaded_model.predict_generator(test_generator,
                                                 steps=nb_samples // batch_size,
                                                 max_queue_size=10,
                                                 workers=1,
                                                 use_multiprocessing=False,
                                                 verbose=1)
    class_idx = class_name_dict[required_class]
    hist_data = predictions[:, class_idx]

    if plotting_module == "matplotlib":
        plt.hist(hist_data, bins=10)
        plt.xlim(0, 1)
        plt.title('Histogram of predicted probabilities')
        plt.xlabel('Predicted probability of {}'.format(required_class))
        plt.ylabel('Frequency')

    elif plotting_module == "bokeh":
        df = pd.DataFrame({"Predicted probability of {}".format(required_class): hist_data})
        from bokeh.io import output_notebook
        import warnings
        warnings.filterwarnings('ignore')
        from bokeh.charts import Histogram, show
        output_notebook()
        p = Histogram(df, "Predicted probability of {}".format(required_class), title="Histogram of predicted probabilities")
        show(p)

    else:
        raise ValueError("Only bokeh and matplotlib are supported")

#################################################################################################################


def make_cm_generator_ensemble(images_path,
                               models_path_list=[],
                               img_shape_list=[],
                               save_path=None,
                               batch_size=16,
                               preprocessing_function_list=[],
                               ensemble_method="avg",
                               weights=None,
                               bias_tuple=(),  # eg ('dog',0.3) i.e bias at 30% probability for dog class
                               normalize=False
                               ):
    """ Creates Confusion matrix for model ensembles

    NOTE: The supported ensemble methods are:
    1. 'max' Where the final predictions are the maximum prediction probabilities of the given models.
    2. 'avg' Where the final predictions are the average of the given models (also supports weighted average)

    Keyword Arguments
    images_path --str: Full path to the folder containing sub-directories(class_names) with images (No default)

    models_path_list --list of tuples of str: Full path to the models and their corresponding weights as list of tuples in
                                              a specific order [(model_path.json,weights_path.hdf5),...] (default [])

    img_shape_list --list of tuples: The image input shapes for the given models in order eg:[(224,224),(224,224),...] (default [])

    save_path --str: Optional path to save the created Confusion Matrix Dataframe as .pkl (Default None)

    batch_size --int: The batch size for predicting using models

    preprocessing_functions_list --list of functions: Must contain The preprocessing_functions used in order.
                                                      If no preprocessing function is used make that element as None
                                                      Example: [function1,None,...] (Default [])

    ensemble_method --str: One of 'avg' or 'max' to be used

    weights --list of float: Optional list of weights for the model for weighted Average Ensemble
                             Set this only when ensemble_method = 'avg'

    bias_tuple --tuple: Optional argument if you want to be biased towards a particular class (Default () 'empty')
                         Example: ('dog',0.3) i.e Infer as 'dog' class
                                  if the probability is 0.3 or greater in the final predictions

    normalize --boolean: Whether to normalize the Confusion Matrix (default False)

    Output

    Returns The Confusion Matrix dataframe

    """

    supported_ensemble_methods = ["avg", "max"]

    # error handling
    if (len(models_path_list) != len(img_shape_list)) and (len(models_path_list) != len(preprocessing_function_list)):
        raise ValueError("ALL THE LISTS MUST BE OF EQUAL LENGTH !!! CHECK THE INPUTS!!!")

    if weights is not None:
        if ensemble_method == "max":
            raise ValueError("weights cannot be applied when ensemble_method is 'max'. Change ensemble_method to 'avg' ")

        elif ensemble_method == "avg":
            if len(weights) != len(models_path_list):
                raise ValueError("The number of weights must match the number of models provided")

            if sum(weights) != 1:
                raise ValueError("The weights given for the models must add upto one. Check the values of the weights")

    if ensemble_method not in supported_ensemble_methods:
        raise ValueError("Only 'avg' and 'max' ensemble methods are supported")

    predictions_list = []
    for model_path, img_shape, preprocessing_function in zip(models_path_list, img_shape_list, preprocessing_function_list):
        # loading model and weights
        # here model path is a tuple (.json,.hdf5)
        print("\nWorking with model {}".format(os.path.basename(model_path[0])))
        json_file = open(model_path[0], 'r')  # 0 is .json
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights
        loaded_model.load_weights(model_path[1])  # 1 is .hdf5
        print('loaded model and its corresponding weights from disk')
        test_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
        test_generator = test_datagen.flow_from_directory(images_path,
                                                          target_size=(img_shape[0], img_shape[1]),
                                                          class_mode='categorical',
                                                          batch_size=batch_size,
                                                          shuffle=False,  # to get ordered result
                                                          )
        # this is an important step else there is a difference in result between predict and predict_generator
        test_generator.reset()  # reset to start with sample 0
        nb_samples = test_generator.samples
        class_name_dict = test_generator.class_indices  # eg {'dog': 0, 'cat': 1}
        print(class_name_dict)
        # predict
        print("Predicting using model {}".format(os.path.basename(model_path[0])))
        temp_predictions = loaded_model.predict_generator(test_generator,
                                                          steps=nb_samples // batch_size,
                                                          max_queue_size=10,
                                                          workers=1,
                                                          use_multiprocessing=False,
                                                          verbose=1)
        predictions_list.append(temp_predictions)

    # Stack the predictions to create a depth dimension
    stacked_predictions = np.stack(predictions_list, axis=2)  # 2 refers to the depth axis

    if ensemble_method == "max":
        print("\nEnsemble is created using MAX POOLING of the predicted probabilities !!! ")
        predictions = np.max(stacked_predictions, axis=2)

    elif ensemble_method == "avg":
        print("\nEnsemble is created using AVERAGE POOLING of the predicted probabilities !!!")
        predictions = np.average(stacked_predictions, axis=2, weights=weights)

    probabilities = predictions.copy()

    if len(bias_tuple) == 0:
        print("\nThresholding the classes using np.argmax (NO bias threshold)")
        classes_predicted = np.argmax(predictions, axis=1)
    elif len(bias_tuple) == 2:
        print("\nThresholding the classes with the given bias")
        bias_index = class_name_dict[bias_tuple[0]]
        classes_predicted = custom_threshold(predictions, bias_index, bias_tuple[1])
    else:
        raise ValueError("The given bias tuple {} is not supported. The supported format is ('class_name',probability)".format(bias_tuple))

    # reverse the classs_name_dictionary (values and keys)
    class_name_dict_reverse = dict(map(reversed, class_name_dict.items()))  # eg {0: 'dog', 1: 'cat'}
    # create a dataframe
    classes_GT = test_generator.classes
    Predictions = []
    GT = []
    Filenames = []
    for t, p, name in zip(classes_GT, classes_predicted, test_generator.filenames):
        Predictions.append(class_name_dict_reverse[p])
        GT.append(class_name_dict_reverse[t])
        Filenames.append(images_path + '/' + name)  # full path
    df = pd.DataFrame({"Component name": Filenames, "True value": GT, "Prediction": Predictions, "Pred Prob": list(probabilities)})
    # generate classification report and confusion matrix
    target_names = []
    for key in range(0, len(class_name_dict_reverse)):
        temp = "class{}: {}".format(key, class_name_dict_reverse[key])
        target_names.append(temp)

    class_names = np.array(target_names)
    print('\n The classification report is printed below \n')
    print(classification_report(y_true=classes_GT, y_pred=classes_predicted, target_names=target_names))
    print('\n The confusion matrix generated using scikit learn library is printed below \n')
    cm = confusion_matrix(y_true=classes_GT, y_pred=classes_predicted)
    plt.figure()
    plot_confusion_matrix(cm, classes=class_names, normalize=normalize,
                          title='Confusion matrix')
    plt.show()
    # save the dataframe
    if save_path:
        df.to_pickle(save_path)
        print("The Confusion matrix is saved in the path {}".format(save_path))
    print("The confusion matrix Df is returned")
    return(df)
#################################################################################################################


def plot_PR_curve_multiclass(images_path,
                             model_path,
                             weights_path,
                             img_shape,
                             batch_size=16,
                             preprocessing_function=None,
                             average_type="micro"):
    """ Function to plot The Precision-Recall curve for multiclass using averaging

    Keyword Arguments:
    images_path --str: full path to the parent directory containing sub-directory(classes) of images
    model_path --str: full path to a keras model (.json file) (No default)
    weights_path --str: full path to the weights file (.hdf5 file) (No default)
    img_shape --tuple: image shape to input to the model (eg : (224,224,3)) (No default)
    batch_size --int: The batch_size to use for prediction (Default 16)
    preprocessing_function --function: The preprocessing function to use before prediction (Default None)
    average_type --str: The type of average to use one of 'micro' or 'macro' (Default 'micro')
    NOTE: It is recommended to use 'micro' averaging than 'macro' averaging

    Output
    plots the Precision Recall Curve using matplotlib
    """

    average_types = ["micro", "macro"]
    if average_type not in average_types:
        raise ValueError("Only 'micro' and 'macro' averages are supported as of now")

    # loading model and weights
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights
    loaded_model.load_weights(weights_path)
    print('loaded model from disk')
    # generator for data
    test_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
    test_generator = test_datagen.flow_from_directory(images_path,
                                                      target_size=(img_shape[0], img_shape[1]),
                                                      class_mode='categorical',
                                                      batch_size=batch_size,
                                                      shuffle=False,  # to get ordered result
                                                      )
    # return(test_generator)
    # this is an important step else there is a difference in result between predict and predict_generator
    test_generator.reset()  # reset to start with sample 0
    nb_samples = test_generator.samples
    class_name_dict = test_generator.class_indices  # eg {'dog': 0, 'cat': 1}
    class_name_dict_reverse = dict(map(reversed, class_name_dict.items()))

    print(class_name_dict)

    y_score = loaded_model.predict_generator(test_generator,
                                             steps=nb_samples // batch_size,
                                             max_queue_size=10,
                                             workers=1,
                                             use_multiprocessing=False,
                                             verbose=1)
    Y_test = to_categorical(test_generator.classes)  # hot encoded GT

    # The below code is mostly from scikit example with minor modifications
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(class_name_dict)):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision[average_type], recall[average_type], _ = precision_recall_curve(Y_test.ravel(),
                                                                              y_score.ravel())
    average_precision[average_type] = average_precision_score(Y_test, y_score,
                                                              average=average_type)

    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=len(class_name_dict))
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall[average_type], precision[average_type], color='gold', lw=2)
    lines.append(l)
    labels.append('{}-average Precision-recall (area = {})'.format(average_type, average_precision[average_type]))

    for i, color in zip(range(len(class_name_dict)), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {} (area = {})'.format(class_name_dict_reverse[i], average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.show()


############################################################################

def plot_PR_curve_singleclass(images_path,
                              model_path,
                              weights_path,
                              img_shape,
                              required_class,
                              batch_size=16,
                              preprocessing_function=None,
                              ):
    """ Function to plot The Precision-Recall curve for a single-class

    NOTE: Here a single class means: Any multiclass can be considered as a binary-class problem

    Keyword Arguments:
    images_path --str: full path to the parent directory containing sub-directory(classes) of images
    model_path --str: full path to a keras model (.json file) (No default)
    weights_path --str: full path to the weights file (.hdf5 file) (No default)
    required_class --str: The required class for which P-R curve must be generated
    img_shape --tuple: image shape to input to the model (eg : (224,224,3)) (No default)
    batch_size --int: The batch_size to use for prediction (Default 16)
    preprocessing_function --function: The preprocessing function to use before prediction (Default None)


    Output
    plots the Precision Recall Curve using matplotlib
    Returns (precision,recall,thresholds)
    """

    # loading model and weights
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights
    loaded_model.load_weights(weights_path)
    print('loaded model from disk')
    # generator for data
    test_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
    test_generator = test_datagen.flow_from_directory(images_path,
                                                      target_size=(img_shape[0], img_shape[1]),
                                                      class_mode='categorical',
                                                      batch_size=batch_size,
                                                      shuffle=False,  # to get ordered result
                                                      )
    # return(test_generator)
    # this is an important step else there is a difference in result between predict and predict_generator
    test_generator.reset()  # reset to start with sample 0
    nb_samples = test_generator.samples
    class_name_dict = test_generator.class_indices  # eg {'dog': 0, 'cat': 1}

    keys = class_name_dict.keys()

    if required_class not in keys:
        raise ValueError("The required class {} is not in the class_indices, The class indices are {}".format(required_class, keys))

    class_idx = class_name_dict[required_class]
    print(class_name_dict)

    y_score = loaded_model.predict_generator(test_generator,
                                             steps=nb_samples // batch_size,
                                             max_queue_size=10,
                                             workers=1,
                                             use_multiprocessing=False,
                                             verbose=1)
    Y_test = to_categorical(test_generator.classes)  # hot encoded GT

    average_precision = average_precision_score(Y_test[:, class_idx], y_score[:, class_idx])

    precision, recall, thresholds = precision_recall_curve(Y_test[:, class_idx], y_score[:, class_idx])

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve for class: {} and its AUC={}'.format(required_class, average_precision))
    plt.show()
    return(precision, recall, thresholds)
