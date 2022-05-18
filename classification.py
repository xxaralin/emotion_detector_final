from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os

num_classes = 7
img_rows,img_cols = 48,48
batch_size = 64
nb_train_samples = 24176
nb_validation_samples = 3006
epochs=25
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

train_data_dir = r'C:\ai_project\face-expression-recognition-dataset\train/'
validation_data_dir = r'C:\ai_project\face-expression-recognition-dataset\test/'

def count_exp(path, set_):
    dict_ = {}
    for expression in os.listdir(path):
        dir_= path + expression
        dict_[expression] = len(os.listdir(dir_))
      
    df = pd.DataFrame(dict_, index=[set_])
    return df
train_count = count_exp(train_data_dir, 'train')
test_count = count_exp(validation_data_dir, 'test')

print(train_count)
print(test_count)


train_count.transpose().plot(kind='bar')
plt.show()
test_count.transpose().plot(kind='bar')
plt.show()

train_datagen = ImageDataGenerator(
					rescale=1./255,
				)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode='grayscale',
					target_size=(img_rows,img_cols),
					batch_size=batch_size,
					class_mode='categorical',
					shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
                    validation_data_dir,
                    color_mode='grayscale',
                    target_size=(img_rows,img_cols),
                    batch_size=batch_size,
                    class_mode='categorical',
                    shuffle=True)

labels = list(validation_generator.class_indices.keys())

model = Sequential()

# Block-1

model.add(Conv2D(32,(3,3),padding='same',activation='relu',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-2 

model.add(Conv2D(64,(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-3

model.add(Conv2D(128,(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-4 

model.add(Conv2D(256,(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-5

model.add(Flatten())
model.add(Dense(64,activation='relu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-6

model.add(Dense(64,activation='relu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-7

model.add(Dense(num_classes,activation='relu',kernel_initializer='he_normal'))
model.add(Activation('softmax'))


print(model.summary())

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint('Emotion_little_vgg.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]


print(model.summary())

from tensorflow.keras.optimizers import RMSprop,SGD,Adam


model.compile(optimizer = Adam(learning_rate=0.001), 
            loss = 'categorical_crossentropy', 
            metrics = ["accuracy"])


from sklearn.metrics import classification_report, confusion_matrix

history=model.fit(
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                #callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples//batch_size)

plot_model_history(history)


preds=model.predict(validation_generator)
y_pred = np.argmax(preds,axis=1)
y_actual = validation_generator.classes
cm = confusion_matrix(y_actual, y_pred)
print(cm)

print(classification_report(y_actual, y_pred, target_names=labels))

plt.figure(figsize=(8,8))
plt.imshow(cm, interpolation='nearest')
plt.colorbar()

tick_mark = np.arange(len(labels))
_ = plt.xticks(tick_mark, labels, rotation=90)
_ = plt.yticks(tick_mark, labels)
plt.show()
