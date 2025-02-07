DEBUG = False
TIME_PROFILING = False
AMP_CACHE_SIZE = 4096
GRAD_CACHE_SIZE = 128
TAG_OFFSET = 10**6

def set_debug(debug):
    global DEBUG
    DEBUG = debug