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
  threshold: int = 0

  def _load_from_io(self, source: io.IOBase):
    raise NotImplementedError('Unknown format version')

  def tojson(self) -> str:
    import json
    return json.dumps({
      'version': self.format_version,
      'begin_time': self.begin_time.isoformat(),
      'sample_rate': self.sample_rate,
      'lon': self.longitude,
      'lat': self.latitude,
      'alt': self.altitude,
      'samples': self.samples_voltage.tolist(),
      'overflow': self.has_overflow,
      'threshold': self.threshold_voltage
    })

  @property
  def threshold_voltage(self) -> float:
    return (self.threshold * self.adc_reference_v) / self.adc_resolution - (self.adc_reference_v / 2)

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

  def characteristics(self, **kwargs):
    """
    Извлечение характеристик сигнала

    :keyword lf_min_freq: Минимальная частота в герцах для спектрального анализа (по умолчанию 1 кГц)
    :keyword lf_max_freq: Максимальная частота в герцах для спектрального анализа (по умолчанию 50 кГц)

    :return: Характеристика сигнала
    """

    result = {}

    sig = self.samples_voltage
    abs_sig = np.abs(sig)
    abs_sig_max = np.max(abs_sig)

    sig /= abs_sig_max
    abs_sig /= abs_sig_max

    abs_sig_max = 1

    threshold = 0.7 * abs_sig_max
    above_threshold = abs_sig > threshold
    result['impulse_duration'] = np.sum(above_threshold) / self.sample_rate

    diff_sig = np.diff(sig)
    result['max_rise_rate'] = np.max(diff_sig) if len(diff_sig) > 0 else 0
    result['max_fall_rate'] = np.min(diff_sig) if len(diff_sig) > 0 else 0
    result['peak_to_rms'] = abs_sig_max / (np.sqrt(np.mean(sig ** 2)) + 1e-10)

    fft_resolution = sig.size / self.sample_rate
    min_freq = int(kwargs.get('lf_min_freq', 1_000))
    max_freq = int(kwargs.get('lf_max_freq', 50_000))

    fft = np.fft.fft(sig)
    magnitude = np.abs(fft)
    magnitude_sum = float(np.sum(magnitude))

    lf_mag = fft[int(min_freq * fft_resolution):int(max_freq * fft_resolution)]
    hf_mag = fft[int(max_freq * fft_resolution):]

    result['lf_energy_ratio'] = (np.sum(lf_mag) / magnitude_sum).real
    result['hf_energy_ratio'] = (np.sum(hf_mag) / magnitude_sum).real

    result['envelope_mean'] = np.mean(abs_sig)
    result['envelope_std'] = np.std(abs_sig)

    zero_crossings = np.where(np.diff(np.sign(diff_sig)))[0]
    result['extremum_count'] = len(zero_crossings)

    autocorr = np.correlate(sig, sig, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    result['autocorr_peak_ratio'] = autocorr[1] / autocorr[0] if autocorr[0] > 0 else 0
    return result

  def __str__(self):
    s = ''
    return f'v{self.format_version}, {self.begin_time}, {self.sample_rate}Hz, {self.coordinates}, ' \
           f' adc resistance: {self.power_resitance} Om, adc bits: {self.adc_bits}, ' \
           f'adc vref: {self.adc_reference_v:.2f}V, ' \
           f'overflow: {self.has_overflow}, signal length: {self.samples.size / self.sample_rate}s'
  
  def __repr__(self):
    return f'SignalFile({self.__str__()})'
