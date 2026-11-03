from __future__ import annotations

import queue
import re
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.prompt import Confirm

from kokoro_onnx import Kokoro

