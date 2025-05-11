def clear_memory():
    gc.collect()
    tf.keras.backend.clear_session()
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
            )
    except:
        pass