import io
import numpy as np
from logging import getLogger
from datetime import datetime, timedelta
from typing import NamedTuple
from enum import IntEnum


class Coordinates(NamedTuple):
  longitude: float
  latitude: float
  altitude: float


class AntennaType(IntEnum):
  Unknown = 0
  Miniwip = 1
  Magnetic = 2


class SignalFile:
  PARSER = dict()
  PRINT_SAMPLES = True
  PRINT_MAX_SAMPLES = 1000
  
  format_version: int = 0
  begin_time: datetime
  sample_rate: int
  coordinates: Coordinates
  samples: np.ndarray
  has_overflow: bool
  is_pps_sync: bool = False
  power_resitance: int = 1000  # Om
  adc_bits: int = 10
  adc_reference_v: float = 3.3
  antenna_type: AntennaType = AntennaType.Unknown

  def _load_from_io(self, source: io.IOBase):
    raise NotImplementedError('Unknown format version')
  
  @property
  def interval(self) -> timedelta:
    return timedelta(seconds=1) / self.sample_rate

  @property
  def adc_resolution(self) -> int:
    return 2 ** self.adc_bits - 1

  @property
  def samples_voltage(self) -> np.ndarray:
    return (self.samples * self.adc_reference_v) / self.adc_resolution - (self.adc_reference_v / 2)

  @property
  def end_time(self) -> datetime:
    return self.begin_time + self.interval * len(self.samples)

  @staticmethod
  def create(source):
    log = getLogger('SignalFile::create')

    log.debug('start parse source...')
    format_version = int.from_bytes(source.read(2), 'little')

    log.debug(f'version detect: v{format_version}')
    sig_type = SignalFile.PARSER.get(format_version, SignalFile)
    if sig_type is SignalFile:
      log.error(f'version v{format_version} has no parsers')
      raise RuntimeError(f'version v{format_version} has no parsers')

    log.debug(f'parser found for v{format_version}')
    impl = sig_type()
    impl._load_from_io(source)
    return impl
  
  @property
  def longitude(self):
    return self.coordinates.longitude
  
  @property
  def latitude(self):
    return self.coordinates.latitude
  
  @property
  def altitude(self):
    return self.coordinates.altitude
  
  def __str__(self):
    s = ''
    return f'v{self.format_version}, {self.begin_time}, {self.sample_rate}Hz, {self.coordinates}, ' \
           f' adc resistance: {self.power_resitance} Om, adc bits: {self.adc_bits}, ' \
           f'adc vref: {self.adc_reference_v:.2f}V, ' \
           f'overflow: {self.has_overflow}, signal length: {self.samples.size}'
  
  def __repr__(self):
    return f'SignalFile({self.__str__()})'
