from msgpack import Unpacker
import numpy as np
from logging import getLogger
from datetime import datetime, timedelta, timezone
from .signal import SignalFile, Coordinates, AntennaType

HAS_OVERFLOW = 1
HAS_PPS_SYNC = 2


class SignalFileV3(SignalFile):
  log = getLogger('SignalFileV3')
  SignalFile.PARSER[3] = lambda: SignalFileV3()

  format_version = 3

  def _load_from_io(self, source):
    up = Unpacker(source)

    self.begin_time = datetime.strptime(up.unpack(), '%Y%m%d%H%M%S')
    self.begin_time += timedelta(microseconds=int(up.unpack()))
    self.begin_time = self.begin_time.replace(tzinfo=timezone.utc)
    self.coordinates = Coordinates(
      longitude=float(up.unpack()),
      latitude=float(up.unpack()),
      altitude=float(up.unpack())
    )
    self.sample_rate = int(up.unpack())
    self.power_resitance = int(up.unpack())
    self.adc_bits = int(up.unpack())
    assert self.adc_bits < 32, 'Invalid adc bits'

    self.adc_reference_v = float(up.unpack())
    assert 0 < self.adc_reference_v < 5, 'Invalid adc reference voltage'

    flags = int(up.unpack())
    assert 0 <= flags < 255, 'Invalid signal flags'
    self.has_overflow = bool(flags & HAS_OVERFLOW)
    self.is_pps_sync = bool(flags & HAS_PPS_SYNC)
    self.antenna_type = AntennaType(int(up.unpack()))

    self.samples = np.array(up.unpack(), dtype=int)
    assert self.samples.size > 0, 'Invalid signal length'
    return up


class SignalFileV31(SignalFileV3):
  log = getLogger('SignalFileV31')
  SignalFile.PARSER[4] = lambda: SignalFileV31()

  format_version = 4

  def _load_from_io(self, source):
    up = super()._load_from_io(source)
    self.threshold = int(up.unpack())
