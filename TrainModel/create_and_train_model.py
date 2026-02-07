from keras._tf_keras.keras import models, layers


class CreateAndTrainModel:
    def __init__(self, input_shape, train_xy:tuple, val_xy:tuple, **kwargs):
        self.x_images_train, self.y_labels_train = train_xy
        self.x_images_val, self.y_labels_val = val_xy
        self.batch_size = kwargs.get('batch_size', 4)

        self.input_shape = input_shape
        self.activation_2D = 'relu'
        self.activation_dense = 'sigmoid'
        self.padding_2D = 'same'
        self.kernel_size_2D = (3, 3)
        self.max_pool_2D = (2, 2)
        self.filters = 16

        self.model = None
        self.epochs = kwargs.get('epochs', 10)
        self.model_output_path = kwargs.get('output', 'dummy_model.keras')

    def create_model(self):
        """Create the model architecture."""
        activation_and_padding_2d = {'kernel_size':self.kernel_size_2D,
                                     'activation': self.activation_2D,
                                     'padding': self.padding_2D}
        self.model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Conv2D(self.filters, **activation_and_padding_2d),
            layers.MaxPooling2D(self.max_pool_2D),
            layers.Conv2D(self.filters * 2, **activation_and_padding_2d),
            layers.MaxPooling2D(self.max_pool_2D),
            layers.Conv2D(self.filters * 4, **activation_and_padding_2d),
            layers.GlobalAveragePooling2D(),
            # Output layer: 10 detections with 5 values each
            layers.Dense(10 * 5, activation=self.activation_dense),
            layers.Reshape((10, 5))
        ])

        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['accuracy']
        )

        return self.model

    def train_model(self):
        # Train the model
        print(f"Training model for {self.epochs} epochs...")
        history = self.model.fit(
            self.x_images_train, self.y_labels_train,
            validation_data=(self.x_images_val, self.y_labels_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1
        )
        return history

    def save_model(self, **kwargs):
        save_path = kwargs.get('save_path', self.model_output_path)
        # Save the model
        print(f"Saving model to {save_path}...")
        self.model.save(save_path)
        print("Training complete!")
