
def set_warning_level(level='3'):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = level