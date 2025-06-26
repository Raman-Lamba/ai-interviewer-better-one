import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print("GPU Details:")
        print(gpu)
        details = tf.config.experimental.get_device_details(gpu)
        for k, v in details.items():
            print(f"{k}: {v}")
else:
    print("❌ No GPU found.")
