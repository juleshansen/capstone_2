import numpy as np
import pandas as pd
import tensorflow as tf

archive = pd.read_csv('data/fer2013.csv')
archive['image'] = archive['pixels'].apply(lambda x: np.array(x.split()).astype(int).reshape(48, 48))

x_train = np.stack(archive.image.to_numpy()).reshape(-1, 48, 48, 1)
y_train = archive.emotion.to_numpy()
# x_test = np.stack(archive.image[archive.Usage == 'PublicTest'].to_numpy()).reshape(-1, 48, 48, 1)/255
# y_test = archive.emotion[archive.Usage == 'PublicTest'].to_numpy()

x_train_3channel = np.zeros((x_train.shape[0], 48, 48, 3))
# x_test_3channel = np.zeros((x_test.shape[0], 48, 48, 3))
for idx, image in enumerate(x_train):
    x_train_3channel[idx] = np.dstack((image, image, image))
# for idx, image in enumerate(x_test):
#     x_test_3channel[idx] = np.dstack((image, image, image))

densenet = tf.keras.applications.DenseNet121(
    include_top=False,
    input_shape=(48, 48, 3),
    weights='imagenet')
for layer in densenet.layers:
    layer.trainable = False
flat1 = tf.keras.layers.Flatten()(densenet.layers[-1].output)
dropout1 = tf.keras.layers.Dropout(0.5)(flat1)
class1 = tf.keras.layers.Dense(1024, activation='relu')(dropout1)
dropout2 = tf.keras.layers.Dropout(0.5)(class1)
output = tf.keras.layers.Dense(7, activation='softmax')(dropout2)

model = tf.keras.models.Model(inputs=densenet.inputs, outputs=output)
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_3channel, y_train, validation_split=0.1, epochs=100)
model.save('model/densenet_0.4')
