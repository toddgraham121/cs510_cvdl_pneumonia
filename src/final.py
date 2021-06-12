import numpy as np
from datetime import datetime
import keras
from keras.applications import VGG16, ResNet152
from keras.callbacks import EarlyStopping
from keras import layers, models
import tensorflow as tf
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
import cv2 as cv
from pickle import dump

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from skimage import exposure
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow.python.ops.numpy_ops.np_config as np_conf

# tf.python.ops.numpy_ops.\
np_conf.enable_numpy_behavior()

imageSize = 150
batchSize = 32


class Disagreement:
    def __init__(self, filename=None, actual=None, vgg_pred=None, res_pred=None):
        self.name = filename
        self.actual = actual
        self.vgg = vgg_pred
        self.res = res_pred


def loadData():
    trainDatagen = ImageDataGenerator(rescale=1. / 255,
                                      rotation_range=30,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      width_shift_range=0.1,
                                      height_shift_range=0.1)

    testDatagen = ImageDataGenerator(
        rescale=1. / 255)

    valDatagen = ImageDataGenerator(rescale=1. / 255,
                                    rotation_range=30,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1)

    trainData = trainDatagen.flow_from_directory(
        '../preprocessed_data/train', class_mode='binary', shuffle=True, batch_size=batchSize,
        target_size=(imageSize, imageSize))

    testData = testDatagen.flow_from_directory(
        '../preprocessed_data/test', class_mode='binary', shuffle=False, batch_size=batchSize,
        target_size=(imageSize, imageSize))

    valData = valDatagen.flow_from_directory(
        '../preprocessed_data/val', class_mode='binary', shuffle=True, batch_size=batchSize,
        target_size=(imageSize, imageSize))

    return [trainData, testData, valData]


def _load_image_(named: str = '') -> np.array:
    filepath = '../preprocessed_data/' + named
    photo = keras.preprocessing.image.load_img(
        filepath, target_size=(imageSize, imageSize))
    photo_as_array = keras.preprocessing.image.img_to_array(photo)
    return photo_as_array


def _load_and_preprocess_image_(named: str = '') -> np.array:
    path = '../preprocessed_data/' + named
    image = cv.imread(path, 0)
    equalizedImage = cv.equalizeHist(image)
    return equalizedImage


def _save_preprocessed_image(image, filename: str) -> None:
    cv.imwrite(filename, image)


def _save_(image_array: np.array, with_filename: str) -> None:
    image = keras.preprocessing.image.array_to_img(image_array)
    image.save(with_filename)


def _create_(folder):
    if not os.path.isdir(folder):
        print("Creating folder: " + folder)
        os.makedirs(folder)


def preprocess_data():
    # iterate over each file in the data folder and for each file preprocess and save:
    for subdir, dirs, files in os.walk('../data'):
        for file in files:
            new_subdir = f'{subdir.replace("../data/", "")}'
            filepath = os.path.join(subdir, file)
            if filepath.endswith('.jpeg'):
                preprocessed_image = _load_and_preprocess_image_(filepath)
                new_dir = '../preprocessed_data/' + new_subdir
                _create_(new_dir)
                _save_preprocessed_image(
                    preprocessed_image, f'{new_dir}/{file}')


def save_gradcam_for_image_(named: str = '', saliency_maps: list = [], alpha: float = 0.4):
    image = _load_image_(named)

    i = 1
    for heatmap in saliency_maps:
        # rescale range 0-255
        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")  # color for heatmap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # create heatmap image
        jet_heatmap = np.expand_dims(jet_heatmap, 1)
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # create the images
        superimposed_img = jet_heatmap * alpha + image

        # create dated folder for results
        results_folder = '../results/' + datetime.now().strftime(
            '_%Y-%m-%d_%H-%M-%S/')
        _create_(results_folder)
        _create_(results_folder + 'test')
        _create_(results_folder + 'test/NORMAL')
        _create_(results_folder + 'test/PNEUMONIA')

        filename_combined = named[:-5] + '_combined_prediction_' + str(i) + '.jpg'
        filename_heatmap = named[:-5] + '_heatmap_prediction_' + str(i) + '.jpg'

        _save_(superimposed_img, results_folder + filename_combined)
        _save_(jet_heatmap, results_folder + filename_heatmap)

        i += 1


def make_saliency_map_for_gradcam(image: np.array, model, last_conv_layer_name: str, place: int = 1) -> np.array:
    # model = keras.layers.Concatenate([model])
    gradient_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(
            last_conv_layer_name).output, model.output]
    )
    # gradient_model = tf.keras.models.Model(
    #     [model.inputs], [model.layers[0].get_layer(
    #         last_conv_layer_name).output, model.output]
    # )

    image = np.expand_dims(image, 0)
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = gradient_model(image)
        print('preds:', preds)
        print('preds.shape:', preds.shape)
        print('last_conv_layer_output:', last_conv_layer_output)
        sorted_preds = np.argsort(preds[0])
        # print(sorted_preds)
        index = sorted_preds[-1 * place]
        # print('Classification rank:', place, 'Index:',
        #       index, 'Value:', preds[0][index])
        class_channel = preds[:, index]
        print('class_channel:', class_channel)
        print('index', index)

    gradients = tape.gradient(class_channel, last_conv_layer_output)
    print('gradients', gradients)
    # pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1))
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    # print('first:', last_conv_layer_output.shape, 'second:', pooled_gradients[..., tf.newaxis].shape)
    # last_conv_layer_output.reshape(last_conv_layer_output.shape[0], 1)
    # pooled_gradients.reshape(1, last_conv_layer_output.shape[0])
    print('first:', last_conv_layer_output.shape, 'second:', pooled_gradients[..., tf.newaxis].shape)
    saliency_map = last_conv_layer_output @ pooled_gradients[..., tf.newaxis]
    saliency_map = tf.squeeze(saliency_map)
    # saliency_map = tf.squeeze(last_conv_layer_output)
    # For visualization normalize the between 0 & 1
    saliency_map = tf.maximum(saliency_map, 0) / \
                   tf.math.reduce_max(saliency_map)

    return saliency_map.numpy()


def grad_cam_saliency(disagreements: list = [], models: list = []):
    # disagreements.append('test/normal/IM-0025-0001.jpeg')
    # if not models:
    #     models = load_trained_models()
    # vgg16 = keras.applications.VGG16(include_top=True,
    #                                  weights="imagenet",
    #                                  input_tensor=None,
    #                                  input_shape=None,
    #                                  pooling=None,
    #                                  classes=1000,
    #                                  classifier_activation="softmax")
    # vgg16.summary()
    # for i in range(1, 6):
    #     image_name = 'gc' + str(i) + '.jpg'
    #     image = _load_and_preprocess_image_(image_name)
    #     features = vgg16.predict(image)
    #     # print(keras.applications.vgg16.decode_predictions(features, top=3))
    #
    #     saliency_maps = []
    #     for place in range(1, 4):
    #         saliency_maps.append(make_saliency_map_for_gradcam(image, vgg16, 'block5_conv3', place))
    # model_names = ['VGG16', 'ResNet152']
    saliency_maps = []
    print('disagreements:', disagreements)
    for filename in disagreements:
        filename = 'test/' + filename.name
        image = _load_image_(filename)
        saliency_maps.append(make_saliency_map_for_gradcam(image, models[0], 'block5_conv3'))
        # saliency_maps.append(make_saliency_map_for_gradcam(image, models[1], 'conv5_block3_3_conv'))
        save_gradcam_for_image_(filename, saliency_maps)


def down_load_and_modify_models() -> list:
    # Adjustable based on our preprocessing
    image_shape = (imageSize, imageSize, 3)
    models_to_train = []

    pre_model_one = VGG16(
        weights='imagenet', include_top=False, input_shape=image_shape, classes=2)
    pre_model_two = ResNet152(
        weights='imagenet', include_top=False, input_shape=image_shape, classes=2)

    for pre_model in [pre_model_one, pre_model_two]:
        pre_model.trainable = False
        model = models.Sequential()
        model.add(pre_model)
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation='sigmoid'))

        # model.summary()
        models_to_train.append(model)

    return models_to_train


def down_load_and_re_modify_models(trained_model0, trained_model1) -> list:
    # Adjustable based on our preprocessing
    image_shape = (imageSize, imageSize, 3)
    models_to_train = []

    input_layer = layers.Input(shape=image_shape)

    pre_model_one = VGG16(
        weights='imagenet', include_top=False, input_shape=image_shape, classes=2, input_tensor=input_layer)
    test_model0 = keras.models.Sequential()
    # test_model0.add(pre_model_one)
    for layer in pre_model_one.layers:
        test_model0.add(layer)
    for layer in trained_model0.layers[1:]:
        test_model0.add(layer)
    test_model0.compile(optimizer=keras.optimizers.Adam(0.0001),
              loss='binary_crossentropy', metrics=['accuracy'])
    test_model0.summary()
    models_to_train.append(test_model0)

    # models_to_train.append(trained_model1)

    # pre_model_two = ResNet152(
    #     weights='imagenet', include_top=False, input_shape=image_shape, classes=2, input_tensor=input_layer)
    # test_model1 = keras.models.Sequential()
    # # test_model1.add(pre_model_two)
    # for layer in pre_model_two.layers:
    #     if layer is list:
    #         layer = keras.layers.merge(layer.layers)
    #     test_model1.add(layer)
    # for layer in trained_model1.layers[1:]:
    #     test_model1.add(layer)
    # test_model1.compile(optimizer=keras.optimizers.Adam(0.0001),
    #               loss='binary_crossentropy', metrics=['accuracy'])
    # test_model1.summary()
    # models_to_train.append(test_model1)

    # pre_model_one.trainable = False
    # model = keras.Sequential()
    # for layer in pre_model_one.layers:
    #     model.add(layer)
    # for layer in trained_model0.layers[1:]:
    #     model.add(layer)

    # model.compile(optimizer=keras.optimizers.Adam(0.0001),
    #               loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()
    # models_to_train.append(model)
    #
    # pre_model_two.trainable = False
    # model = keras.Sequential()
    # for layer in pre_model_two.layers:
    #     model.add(layer)
    # for layer in trained_model1.layers[1:]:
    #     model.add(layer)
    #
    # model.compile(optimizer=keras.optimizers.Adam(0.0001),
    #               loss='binary_crossentropy', metrics=['accuracy'])
    #
    # model.summary()
    # models_to_train.append(model)

    return models_to_train


def train_(model, data, modelType):
    earlyStop = [EarlyStopping(patience=10, verbose=1)]

    model.compile(optimizer=keras.optimizers.Adam(0.0001),
                  loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(data[0],
                        steps_per_epoch=len(data[0]),
                        epochs=40,
                        validation_data=data[1],
                        validation_steps=len(data[1]),
                        callbacks=earlyStop)

    results_folder = '../results/' + datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')
    model.save(f'{results_folder}/{modelType}')


def test_(model, data, model_name) -> dict:
    predictions = model.predict(data[1], batch_size=32)
    extracted_predictions = []
    for p in predictions:
        extracted_predictions.append(p[0])
    results = dict(zip(data[1].filenames, extracted_predictions))

    test_score = model.evaluate(data[1])
    print("==== " + model_name + " Metrics ====")
    print("    Test Loss: " + str(test_score[0]))
    print("    Test Accuracy: " + str(test_score[1]))
    return results


def determine_disagreements(results0, results1, data) -> list:
    disagreements = []

    # Make a dict of filenames : true labels
    true_labels = dict(zip(data[1].filenames, data[1].labels))
    for filename in results0.keys():
        if results0[filename] >= 0.5:
            c0 = 1
        else:
            c0 = 0
        if results1[filename] >= 0.5:
            c1 = 1
        else:
            c1 = 0
        if c0 != c1:
            # Make a new Disagreement obj
            dis = Disagreement(filename, true_labels[filename], c0, c1)
            disagreements.append(dis)
    return disagreements


def save_confusion_matrices(data, results, model_name):
    true_labels = data[1].labels
    predictions = list(results.values())
    predicted_labels = []

    for probability in predictions:
        if probability >= 0.5:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)

    cm = confusion_matrix(true_labels, predicted_labels)
    cm_display = ConfusionMatrixDisplay(cm)
    cm_display.plot()
    _create_('../results/confusion_matrices')
    plt.savefig('../results/confusion_matrices/' + model_name + datetime.now().strftime('_%Y-%m-%d_%H-%M-%S'))


def save_trained_models(models, results_folder):
    for i, model in enumerate(models):
        name = 'VGG16' if i == 0 else 'ResNet152'
        dump(model, open(f'{results_folder}/model_{name}.pkl', 'wb'))


def load_trained_models(results_folder):
    modelsToLoad = []
    for model in ['VGG16', 'ResNet152']:
        modelsToLoad.append(models.load_model(f'{results_folder}/{model}'))

    return modelsToLoad


if __name__ == '__main__':
    # preprocess_data()  # comment out this line after first run
    data = loadData()
    # models = down_load_and_modify_models()

    trained_models = load_trained_models(
        '../results/_2021-06-08_22-59-26')

    test_models = down_load_and_re_modify_models(trained_models[0], trained_models[1])

    print('first model:')
    # test_models[0].layers[0].summary()
    test_models[0].summary()
    # print('second model:')
    # # test_models[1].layers[0].summary()
    # test_models[1].summary()

    # train_(models[0], data, 'VGG16')
    # train_(models[1], data, 'ResNet152')

    test_results_0 = test_(test_models[0], data, 'VGG16')
    test_results_1 = test_(trained_models[1], data, 'ResNet152')
    disagreements = determine_disagreements(test_results_0, test_results_1, data)
    grad_cam_saliency(disagreements, test_models)
    # save_accuracy_graphs()
    # save_confusion_matrices(data, test_results_0, 'VGG16')
    # save_confusion_matrices(data, test_results_1, 'ResNet152')
