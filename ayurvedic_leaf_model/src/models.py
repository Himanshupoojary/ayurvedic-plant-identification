import tensorflow as tf


def get_backbone(name, input_shape=(224, 224, 3)):
    name = name.lower()

    if name == "mobilenetv2":
        return tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
    if name == "resnet50":
        return tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
    if name == "efficientnetb0":
        return tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )

    raise ValueError(f"Unknown model: {name}")


def build_model(name, num_classes):
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )

    backbone = get_backbone(name)
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3), name="plant_part_image")
    x = augmentation(inputs)

    # Keras application models use their own expected preprocessing.
    if name == "mobilenetv2":
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    elif name == "resnet50":
        x = tf.keras.applications.resnet50.preprocess_input(x)
    elif name == "efficientnetb0":
        # EfficientNet includes its input rescaling/preprocessing.
        x = x

    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="plant_prediction"
    )(x)

    model = tf.keras.Model(inputs, outputs, name=f"{name}_plant_part_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
