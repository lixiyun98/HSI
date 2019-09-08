import keras
from keras.utils import plot_model
from sklearn.decomposition import PCA

def general_model3():
    input_data = keras.Input(shape=(11, 11, 220))
    low_conv1 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input_data)
    low_conv2 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(low_conv1)
    low_conv3 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(low_conv2)
    low_conv4 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(low_conv3)
    low_conv5 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(low_conv4)
    low_conv6 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(low_conv5)
    low_conv7 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(low_conv6)
    low_conv8 = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(low_conv7)
    low_conv9 = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(low_conv8)
    low_conv10 = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(low_conv9)
    low_conv11 = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(low_conv10)
    low_conv12 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(low_conv11)
    output = keras.layers.Flatten()(low_conv12)
    output = keras.layers.Dense(units=128, activation='relu')(output)
    output = keras.layers.Dense(units=16, activation='relu')(output)
    model = keras.Model(inputs=input_data, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def general_model2():
    input_data = keras.Input(shape=(11, 11, 220))
    low_conv1 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_data)
    low_conv2 = keras.layers.Conv2D(64, (1, 1), padding='same', activation='relu')(low_conv1)
    low_conv3 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(low_conv2)

    def low_level_inception(inputs, filters):
        low_conv1 = keras.layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(inputs)
        low_conv2 = keras.layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(inputs)
        low_conv2 = keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(low_conv2)
        low_conv3 = keras.layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(inputs)
        low_conv3 = keras.layers.Conv2D(filters, (3, 3), padding='same',activation='relu')(low_conv3)
        low_conv3 = keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(low_conv3)
        output = keras.layers.concatenate([low_conv1, low_conv2, low_conv3])
        return output
    output = low_level_inception(low_conv3, 64)
    output = keras.layers.Flatten()(output)
    output = keras.layers.Dense(units=128, activation='relu')(output)
    output = keras.layers.Dense(units=16, activation='relu')(output)
    model = keras.Model(inputs=input_data, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def general_model():
    input_data = keras.Input(shape=(11, 11, 220))


    def low_level_inception(inputs, filters):
        low_conv1 = keras.layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(inputs)
        low_conv1 = keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(low_conv1)
        low_conv2 = keras.layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(inputs)
        low_conv2 = keras.layers.Conv2D(filters, (5, 5), padding='same', activation='relu')(low_conv2)
        low_conv3 = keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
        low_conv3 = keras.layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(low_conv3)
        output = keras.layers.concatenate([low_conv1, low_conv2, low_conv3])
        return output

    def mid_level_inception(inputs, filters):
        mid_conv1 = keras.layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(inputs)
        mid_conv1 = keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(mid_conv1)
        mid_conv2 = keras.layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(inputs)
        mid_conv2 = keras.layers.Conv2D(filters, (5, 5), padding='same', activation='relu')(mid_conv2)
        mid_conv3 = keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
        mid_conv3 = keras.layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(mid_conv3)
        output = keras.layers.concatenate([mid_conv1, mid_conv2, mid_conv3])
        return output

    def high_level_inception(inputs, filters):
        high_conv1 = keras.layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(inputs)
        high_conv1 = keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(high_conv1)
        high_conv2 = keras.layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(inputs)
        high_conv2 = keras.layers.Conv2D(filters, (5, 5), padding='same', activation='relu')(high_conv2)
        high_conv3 = keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
        high_conv3 = keras.layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(high_conv3)
        output = keras.layers.concatenate([high_conv1, high_conv2, high_conv3])
        return output

    out1 = low_level_inception(input_data, 64)
    out1 = low_level_inception(out1, 64)
    out1 = low_level_inception(out1, 64)

    out2 = mid_level_inception(out1, 256)
    out2 = mid_level_inception(out2, 256)
    out2 = mid_level_inception(out2, 256)

    out3 = high_level_inception(out2, 1024)
    out3 = high_level_inception(out3, 1024)
    out3 = high_level_inception(out3, 1024)

    output = keras.layers.concatenate([out1, out2, out3])
    output = keras.layers.Flatten()(output)
    output = keras.layers.Dense(units=1024, activation='relu')(output)
    output = keras.layers.Dense(units=256, activation='relu')(output)
    output = keras.layers.Dense(units=64, activation='relu')(output)
    output = keras.layers.Dense(units=16, activation='relu')(output)
    output = keras.layers.Dense(units=1, activation='relu')(output)
    model = keras.Model(inputs=input_data, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_image(model):
    plot_model(model, to_file='modelt.png', show_shapes=True, show_layer_names=True)


def PCA_model(input_data):
    pca = PCA(n_components=20)
    pca.fit(input_data)
    x = pca.transform(input_data)
    print(x)
    pass

model=general_model3()
get_image(model)
