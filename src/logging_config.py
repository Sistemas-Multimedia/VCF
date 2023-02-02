'''Configure the logging output.'''

import logging
#FORMAT = "(%(levelname)s) %(module)s: %(message)s"
#logging.basicConfig(format=FORMAT, level=logging.INFO)
FORMAT = "(%(levelname)s) %(module)s %(funcName)s %(lineno)d: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
