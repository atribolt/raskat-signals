import struct
import numpy as np
from logging import getLogger
from datetime import datetime, timedelta, timezone
from .signal import SignalFile, Coordinates


HAS_OVERFLOW = 1
HAS_PPS_SYNC = 2


class SignalFileV2(SignalFile):
  log = getLogger('SignalFileV2')
  SignalFile.PARSER[2] = lambda: SignalFileV2()

  format_version = 2

  def _load_from_io(self, source):
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
    
    flags = source.read(1)[0]
    self.has_overflow = flags & HAS_OVERFLOW
    self.is_pps_sync = flags & HAS_PPS_SYNC
    
    samples_count = int.from_bytes(source.read(4), 'little')

    SAMPLE_SIZE_BYTES = 2 * samples_count
    self.samples = np.array(struct.unpack('<' + 'h' * samples_count, source.read(SAMPLE_SIZE_BYTES)), np.int16)
