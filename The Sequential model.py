import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#Define Sequential model with 3 layers
model = keras.Sequential(
    [
        layers.Dense(2,activation="relu", name = "layer1"),
        layers.Dense(3,activation="relu", name = "layer2"),
        layers.Dense(4,name = "layer3"),
    ]
)

#Call model on a test input

x = tf.ones((3,3))
y = model (x)

print (model.layers)

#can also create a Sequential model like this 

model1 = keras.Sequential()
model1.add(layers.Dense(2,activation="relu"))
model1.add(layers.Dense(3,activation="relu"))
model1.add(layers.Dense(4))

#pop()method to remove layers
model1.pop()

#keras accepts a name argument
model2=keras.Sequential(name = "my_sequential")
model.add(layers.Dense(2,activation="relu",name="layer1"))
model.add(layers.Dense(3,activation="relu",name="layer2"))
model.add(layers.Dense(4,name="layer3"))

layer = layers.Dense(3)
layer.weights 

#Call layer on a test input
x = tf.ones((1,4))
y = layer(x)
layer.weights # Now it has weights, of shape(4,3) and (3,)

# The weights are created when the model first sees some input data:
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(4),
    ]
)  # No weights at this stage!

# At this point, you can't do this:
# model.weights

# You also can't do this:
# model.summary()

# Call the model on a test input
x = tf.ones((1, 4))
y = model(x)
print("Number of weights after calling the model:", len(model.weights))  # 6

#Once a model is "built", you can call its summary() method to display its contents:
model.summary()

# In this case, you should start your model by passing an Input object to your model
model = keras.Sequential()
model.add(keras.Input(shape=(4,)))
model.add(layers.Dense(2, activation="relu"))

model.summary()

# A simple alternative is to just pass an input_shape argument to your first layer:
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu", input_shape=(4,)))

model.summary()

# A common debugging workflow: add() + summary()

model = keras.Sequential()
model.add(keras.Input(shape=(250,250,3)))  # 250x250 RGB images
model.add(layers.Conv2D(32,5,strides=2,activation="relu"))
model.add(layers.Conv2D(32,3,activation="relu"))
model.add(layers.MaxPooling2D(3))

# Can you guess what the current output shape is at this point? Probably not.
# Let's just print it:
model.summary()

# The answer was: (40, 40, 32), so we can keep downsampling...

model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(2))

# And now?
model.summary()

# Now that we have 4x4 feature maps, time to apply global max pooling.
model.add(layers.GlobalMaxPooling2D())

# Finally, we add a classification layer.
model.add(layers.Dense(10))