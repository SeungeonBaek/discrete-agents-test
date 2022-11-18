def server_setup():
    try:
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        import tensorflow as tf
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError as e:
                print(e)
    except:
        pass
        
    