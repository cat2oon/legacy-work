import logging

# format='%(msecs)dms %(module)s[%(lineno)d] %(message)s',
logging.basicConfig(format='%(asctime)s %(module)s [%(lineno)d] %(message)s',
                    datefmt='%H:%M:%S', level=logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)


# def get_logger(name=__file__, level=logging.INFO):
#     logger = logging.getLogger(name)
#
#     if getattr(logger, '__init_done__', None):
#         logger.setLevel(level)
#         return logger
#
#     logger.__init_done__ = True
#     logger.propagate = False
#     logger.setLevel(level)
#
#     return logger
