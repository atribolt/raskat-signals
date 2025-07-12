import struct
import numpy as np
from logging import getLogger
from datetime import datetime, timedelta, timezone
from .signal import SignalFile, Coordinates


class SignalFileV1(SignalFile):
  log = getLogger('SignalFileV1')
  SignalFile.PARSER[1] = lambda: SignalFileV1()

  format_version = 1

  def _load_from_io(self, source):
    log = SignalFileV1.log

    self.begin_time = datetime.strptime(source.read(14).decode('ascii'), '%Y%m%d%H%M%S')
    self.begin_time += timedelta(microseconds=int.from_bytes(source.read(4), 'little'))
    self.begin_time = self.begin_time.replace(tzinfo=timezone.utc)

    longitude = int.from_bytes(source.read(2), 'little')
    longitude += int.from_bytes(source.read(4), 'little') / 1_000_000
    latitude = int.from_bytes(source.read(2), 'little')
    latitude += int.from_bytes(source.read(4), 'little') / 1_000_000
    altitude = int.from_bytes(source.read(2), 'little')
    altitude += int.from_bytes(source.read(4), 'little') / 1_000_000
    self.coordinates = Coordinates(longitude, latitude, altitude)

    self.sample_rate = int.from_bytes(source.read(4), 'little')

    samples_count = int.from_bytes(source.read(4), 'little')
    self.has_overflow = False

    OVERFLOW_BIT = 14
    OVERFLOW_MASK = 1 << OVERFLOW_BIT
    SAMPLE_SIZE_BYTES = 2 * samples_count
    V1_OFFSET = 1000

    self.samples = np.array(struct.unpack('<' + 'h' * samples_count, source.read(SAMPLE_SIZE_BYTES)), np.int16)
    if np.any(self.samples < 0):
      log.warning('samples has negative values, it is old format')
      self.samples += V1_OFFSET

    overflows = self.samples & OVERFLOW_MASK
    self.has_overflow = np.any(overflows)
    self.samples &= ~overflows
    self.samples -= V1_OFFSET
