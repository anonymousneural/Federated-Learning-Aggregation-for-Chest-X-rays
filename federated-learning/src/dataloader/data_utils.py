def load_and_preprocess_data():
    img_size = (CONFIG['img_size'], CONFIG['img_size'])
    batch_size = CONFIG['batch_size']
    max_train_samples = 1200
    max_val_samples = 200
    max_test_samples = 400

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.85, 1.15]
    )

    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(CONFIG['dataset_path'], 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(CONFIG['dataset_path'], 'val'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        os.path.join(CONFIG['dataset_path'], 'test'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    if train_generator.samples > max_train_samples:
        train_x, train_y = next(iter(train_datagen.flow_from_directory(
            os.path.join(CONFIG['dataset_path'], 'train'),
            target_size=img_size,
            batch_size=max_train_samples,
            class_mode='binary',
            shuffle=True
        )))
        train_generator = train_datagen.flow(train_x, train_y, batch_size=batch_size)
        train_generator.samples = max_train_samples
        train_generator.n = max_train_samples

    if val_generator.samples > max_val_samples:
        val_x, val_y = next(iter(val_datagen.flow_from_directory(
            os.path.join(CONFIG['dataset_path'], 'val'),
            target_size=img_size,
            batch_size=max_val_samples,
            class_mode='binary',
            shuffle=True
        )))
        val_generator = val_datagen.flow(val_x, val_y, batch_size=batch_size)
        val_generator.samples = max_val_samples
        val_generator.n = max_val_samples

    if test_generator.samples > max_test_samples:
        test_x, test_y = next(iter(test_datagen.flow_from_directory(
            os.path.join(CONFIG['dataset_path'], 'test'),
            target_size=img_size,
            batch_size=max_test_samples,
            class_mode='binary',
            shuffle=True
        )))
        test_generator = test_datagen.flow(test_x, test_y, batch_size=batch_size)
        test_generator.samples = max_test_samples
        test_generator.n = max_test_samples

    if hasattr(train_generator, 'classes'):
        class_0_count = np.sum(train_generator.classes == 0)
        class_1_count = np.sum(train_generator.classes == 1)
        total = class_0_count + class_1_count

        CONFIG['class_weights'] = {
            0: total / (2 * class_0_count) if class_0_count > 0 else 1.0,
            1: total / (2 * class_1_count) if class_1_count > 0 else 1.0
        }

    return train_generator, val_generator, test_generator

def create_non_iid_client_data(train_generator, non_iid_factor):
    X_train = []
    y_train = []
    max_batches = min(len(train_generator), 15)

    for i in range(max_batches):
        X_batch, y_batch = train_generator[i]
        X_train.append(X_batch)
        y_train.append(y_batch)

    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)

    class_0_indices = np.where(y_train == 0)[0]
    class_1_indices = np.where(y_train == 1)[0]

    n_clients = CONFIG['num_clients']
    split_points = np.linspace(0, 1, n_clients + 1)

    if non_iid_factor > 0:
        split_points = split_points ** (1 / (1 + non_iid_factor))

    client_datasets = []

    for i in range(n_clients):
        start_0 = int(len(class_0_indices) * split_points[i])
        end_0 = int(len(class_0_indices) * split_points[i + 1])
        start_1 = int(len(class_1_indices) * split_points[i])
        end_1 = int(len(class_1_indices) * split_points[i + 1])

        client_indices = np.concatenate([
            class_0_indices[start_0:end_0],
            class_1_indices[start_1:end_1]
        ])

        client_X = X_train[client_indices]
        client_y = y_train[client_indices]

        client_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )

        client_generator = client_datagen.flow(
            client_X, client_y,
            batch_size=CONFIG['batch_size']
        )

        client_datasets.append(client_generator)

    return client_datasets