import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

def setup_device():
    print("=== Device Configuration ===")
    print(f"TensorFlow version: {tf.__version__}")

    try:
        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            print("GPU/MPS devices found:")
            for gpu in gpus:
                print(f"  - {gpu}")

            try:
                tf.config.set_soft_device_placement(True)

                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        print(f"Memory growth enabled for {gpu}")
                    except:
                        print(f"  Memory growth setting not available for {gpu}")
                
                # Test if MPS is actually working by creating a small tensor
                with tf.device('/GPU:0'):
                    test_tensor = tf.constant([1.0, 2.0, 3.0])
                    result = tf.reduce_sum(test_tensor)
                    print(f"MPS test successful: {result.numpy()}")

                print("GPU/MPS device configured successfully")
                device = "MPS"
            except Exception as e:
                print(f"GPU/MPS setup failed, falling back to CPU: {e}")
                device = "CPU"
        else:
            import platform
            if platform.system() == 'Darwin':
                try:
                    import tensorflow_metal
                    print("tensorflow-metal found but no MPS devices detected")
                except ImportError:
                    print("No MPS support: tensorflow-metal not installed")
                    print("  Install with: pip install tensorflow-metal")

            print("No GPU/MPS devices found, using CPU")
            device = "CPU"

    except Exception as e:
        print(f"Device detection failed: {e}")
        print("Falling back to CPU")
        device = "CPU"

    print(f"Using device: {device}")
    print("================================\n")

    return device

def configure_tensorflow():
    try:
        if len(tf.config.list_physical_devices('GPU')) > 0:
            print("Mixed precision disabled for MPS compatibility")
        else:
            tf.config.optimizer.set_experimental_options({
                'auto_mixed_precision': True
            })
            print("Mixed precision enabled")
    except:
        print("Mixed precision not available")

    device = setup_device()

    return device

if __name__ == "__main__":
    configure_tensorflow()
