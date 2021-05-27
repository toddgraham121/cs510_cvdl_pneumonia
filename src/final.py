import numpy as np
from datetime import datetime
import keras
from keras.applications import VGG16, ResNet152
from keras import layers, models
import tensorflow as tf
import matplotlib.cm as cm
import os
from pickle import dump


def _load_image_(named: str = '') -> np.array:
    filepath = '../data/' + named
    photo = keras.preprocessing.image.load_img(
        filepath, target_size=(224, 224))
    photo_as_array = keras.preprocessing.image.img_to_array(photo)
    return photo_as_array


def _load_and_preprocess_image_(named: str = '') -> np.array:
    photo_as_array = _load_image_(named)
    photo_as_array = np.expand_dims(photo_as_array, axis=0)
    photo_as_array = keras.applications.vgg16.preprocess_input(
        photo_as_array)  # we probably do not want to do this step
    # histogram normalization here
    return photo_as_array


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
            filepath = os.path.join(subdir, file)
            preprocessed_image_array = _load_and_preprocess_image_(file)
            _save_(preprocessed_image_array,
                   '../preprocessed_data/' + filepath)


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
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # create the images
        superimposed_img = jet_heatmap * alpha + image

        # create dated folder for results
        results_folder = '../results/' + datetime.now().strftime(
            '_%Y-%m-%d_%H-%M-%S')
        _create_(results_folder)

        filename_combined = named[:-4] + '_combined_prediction_' + str(i)
        filename_heatmap = named[:-4] + '_heatmap_prediction_' + str(i)

        _save_(superimposed_img, results_folder + filename_combined)
        _save_(jet_heatmap, results_folder + filename_heatmap)

        i += 1


def make_saliency_map_for_gradcam(image: np.array, model, last_conv_layer_name: str, place: int) -> np.array:
    gradient_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(
            last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = gradient_model(image)
        # print(preds)
        # print(preds.shape)
        sorted_preds = np.argsort(preds[0])
        # print(sorted_preds)
        index = sorted_preds[-1 * place]
        print('Classification rank:', place, 'Index:',
              index, 'Value:', preds[0][index])
        class_channel = preds[:, index]

    gradients = tape.gradient(class_channel, last_conv_layer_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    saliency_map = last_conv_layer_output @ pooled_gradients[..., tf.newaxis]
    saliency_map = tf.squeeze(saliency_map)
    # For visualization normalize the between 0 & 1
    saliency_map = tf.maximum(saliency_map, 0) / \
        tf.math.reduce_max(saliency_map)

    return saliency_map.numpy()


def grad_cam_saliency(disagreements, models):
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

    saliency_maps = []
    for filename, model in disagreements.items():
        image = _load_image_(filename)
        saliency_maps.append(make_saliency_map_for_gradcam(
            image, models[model], 'block5_conv3'))
        save_gradcam_for_image_(filename, saliency_maps)


def down_load_and_modify_models() -> list:
    # Adjustable based on our preprocessing
    image_shape = (150, 150, 3)
    models_to_train = []

    pre_model_one = VGG16(
        weights='imagenet', include_top=False, input_shape=image_shape)
    pre_model_two = ResNet152(
        weights='imagenet', include_top=False, input_shape=image_shape)

    for pre_model in [pre_model_one, pre_model_two]:
        pre_model.trainable = False
        model = models.Sequential()
        model.add(pre_model)
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.summary()
        models_to_train.append(model)

    return models_to_train


def train_(model):
    # TODO: implement,
    pass


def test_(model) -> dict:
    # TODO: implement, return dictionary of filename to tuple of classification and whether it was correct
    pass


def determine_disagreements(results0, results1) -> list:
    # TODO: compare dictionaries and return collection of disagreementsm (Dictionary of filename to which model was correct (0 or 1)
    pass


def save_trained_models(models, results_folder):
    for i, model in enumerate(models):
        name = 'VGG16' if i == 0 else 'ResNet152'
        dump(model, open(f'{results_folder}/model_{name}.pkl', 'wb'))


if __name__ == '__main__':
    # preprocess_data()  # comment out this line after first run
    models = down_load_and_modify_models()
    # train_(models[0])
    # train_(models[0])
    # test_results_0 = test_(models[0])
    # test_results_1 = test_(models[1])
    # disagreements = determine_disagreements(test_results_0, test_results_1)
    # grad_cam_saliency(disagreements, models)
    # save_accuracy_graphs()
    # save_confusion_matrices()
