import torch
import os
import numpy as np
import logging
import sys


def error_handling():
    return f"{sys.exc_info()[0]}. {sys.exc_info()[1]}, line: {sys.exc_info()[2].tb_lineno}"


LOGGER_NAME = __package__
LOGLEVEL = os.environ.get("LOGLEVEL", "INFO")

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(filename)-12s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)

log = logging.getLogger(LOGGER_NAME)
log.setLevel(os.environ.get("LOGLEVEL", f"{LOGLEVEL}"))
log.addHandler(handler)


PRECISION = 32
DEVICE_ID = 0

PRECISION_CPU = {16: torch.HalfTensor,
                 32: torch.FloatTensor,
                 64: torch.DoubleTensor}

PRECISION_CUDA = {16: torch.cuda.HalfTensor,
                  32: torch.cuda.FloatTensor,
                  64: torch.cuda.DoubleTensor}

DEVICE = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    log.info("Using GPU: CUDA Available")
    torch.cuda.set_device(DEVICE_ID)
    tensor = PRECISION_CUDA[PRECISION]
else:
    log.info("Using CPU")
    tensor = PRECISION_CPU[PRECISION]

pi = tensor([np.pi]).to(DEVICE)  # torch.Size([1])
LOG_2_PI = torch.log(2 * pi)
