import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

IMG_SIZE = 10

#loading both models to obtain predictions
new_model_gray = tf.keras.models.load_model('dog_cat_cfr_gray.model')
new_model_rgb = tf.keras.models.load_model('dog_cat_cfr_rgb.model')

#loading features and labels
X_gray = pickle.load(open("X_gray.pickle", "rb"))
X_rgb = pickle.load(open("X_rgb.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

#normalizing features, also to convert them into float data
X_gray = X_gray/255.0
X_rgb = X_rgb/255.0

#generating predictions with each model
predictions_gray = new_model_gray.predict(X_gray)
predictions_rgb = new_model_rgb.predict(X_rgb)

for i in range(2):
    plt.imshow(X_rgb[i])
    if predictions_gray[i] >= .5:
        if predictions_rgb[i] >= .5:
            plt.title('gray: Cat   //   rgb: Cat')
        else:
            plt.title('gray: Cat   //   rgb: Dog')
    else:
        if predictions_rgb[i] >= .5:
            plt.title('gray: Dog   //   rgb: Cat')
        else:
            plt.title('gray: Dog   //   rgb: Dog')

    plt.show()

#counting the number of different decisions
k=0
for j in range(len(predictions_gray)):
    if predictions_gray[j] >= .5 and predictions_rgb[j] < .5:
        #print the number of the picture where both models differ in their predictions
        print(j)
        k+=1
    elif predictions_gray[j] <= .5 and predictions_rgb[j] > .5:
        #print the number of the picture where both models differ in their predictions
        print(j)
        k+=1

print('Number of differences between both models:', k)
