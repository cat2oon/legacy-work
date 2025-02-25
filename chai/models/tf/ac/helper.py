"""
    Tensorflow 2.x commons
    GPU 메모리 정보 가져와서 ratio 버전도 구현할 것
"""

def check_tensorflow_use_gpu(cuda_only=False, min_cuda_compute_capability=None):
    import tensorflow as tf
    return tf.test.is_gpu_available(cuda_only, min_cuda_compute_capability)

def set_gpu_memory_growth_mode(gpu_id=0):
    import tensorflow as tf
    # tf.config.gpu.set_per_process_memory_growth(True) --> TF 1.x
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        return
    try:
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)
    
def set_gpu_memory_alloc(memory_in_mb=1024):
    import tensorflow as tf
    # tf.config.gpu.set_per_process_memory_fraction(ratio) --> TF 1.x
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        return
    # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_in_mb)])
    except RuntimeError as e:
        # 프로그램 시작시에 가장 장치가 설정되어야만 합니다
        print(e)