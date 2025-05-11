from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_model(train_generator=None):
    kernel_initializer = tf.keras.initializers.he_normal()
    bias_initializer = tf.keras.initializers.Zeros()
    
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', 
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer,
                     input_shape=(CONFIG['img_size'], CONFIG['img_size'], 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu',
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid', dtype='float32')
    ])
    
    if CONFIG['use_learning_rate_scheduler'] and train_generator is not None:
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=CONFIG['learning_rate'],
            decay_steps=CONFIG['lr_decay_epochs'] * (len(train_generator) // CONFIG['batch_size']),
            decay_rate=CONFIG['lr_decay_factor']
        )
    else:
        learning_rate = CONFIG['learning_rate']
    
    optimizer_config = {
        'learning_rate': learning_rate
    }
    
    if CONFIG['use_gradient_clipping']:
        optimizer_config['clipnorm'] = CONFIG['max_gradient_norm']
    
    optimizer = tf.keras.optimizers.Adam(**optimizer_config)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    return model