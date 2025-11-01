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

  def __str__(self):
    s = ''
    return f'v{self.format_version}, {self.begin_time}, {self.sample_rate}Hz, {self.coordinates}, ' \
           f' adc resistance: {self.power_resitance} Om, adc bits: {self.adc_bits}, ' \
           f'adc vref: {self.adc_reference_v:.2f}V, ' \
           f'overflow: {self.has_overflow}, signal length: {self.samples.size / self.sample_rate}s'
  
  def __repr__(self):
    return f'SignalFile({self.__str__()})'


class SignalFeatures(NamedTuple):
  impusle_duration: float
  max_rise_rate: float
  max_fall_rate: float
  peak_to_rms: float
  lf_energy_ratio: float
  hf_energy_ratio: float
  envelope_mean: float
  envelope_std: float
  autocorr_peak_ratio: float


def characteristics(signal: np.ndarray, **kwargs) -> SignalFeatures:
  """
  Извлечение характеристик сигнала

  :keyword k_threshold: Коэфициент порога для отсеивания пика (по умолчанию 0.5)
  :keyword sample_rate: Частота дискретизации сигнала (по умолчанию 500 кГц)
  :keyword lf_min_freq: Минимальная частота в герцах для спектрального анализа низкочастотной составляющей (по умолчанию 3 кГц)
  :keyword lf_max_freq: Максимальная частота в герцах для спектрального анализа низкочастотной составляющей (по умолчанию 50 кГц)
  :keyword hf_max_freq: Максимальная частота в герцах для спектрального анализа высокочастотной составляющей (по умолчанию 100 кГц)

  :return: Характеристика сигнала
  """

  lf_min_freq = int(kwargs.get('lf_min_freq', 3_000))
  lf_max_freq = int(kwargs.get('lf_max_freq', 50_000))
  hf_max_freq = int(kwargs.get('hf_max_freq', 100_000))
  sample_rate = int(kwargs.get('sample_rate', 500_000))
  k_threshold = float(kwargs.get('k_threshold', 0.5))

  result = {}

  sig = signal - np.mean(signal)
  abs_sig = np.abs(sig)
  abs_sig_max = np.max(abs_sig)

  threshold = k_threshold * abs_sig_max
  above_threshold = abs_sig > threshold
  impulse_duration = float(np.sum(above_threshold) / sample_rate)

  diff_sig = np.diff(sig)
  max_rise_rate = float(np.max(diff_sig) if len(diff_sig) > 0 else 0)
  max_fall_rate = float(np.min(diff_sig) if len(diff_sig) > 0 else 0)
  peak_to_rms = float(abs_sig_max / (np.sqrt(np.mean(sig ** 2)) + 1e-10))

  fft_resolution = sig.size / sample_rate

  fft = np.fft.fft(sig).real
  magnitude = fft ** 2
  magnitude_sum = float(np.sum(magnitude))

  lf_mag = magnitude[int(lf_min_freq * fft_resolution):int(lf_max_freq * fft_resolution)]
  hf_mag = magnitude[int(lf_max_freq * fft_resolution):int(hf_max_freq * fft_resolution)]

  lf_energy_ratio = float(np.sum(lf_mag) / magnitude_sum)
  hf_energy_ratio = float(np.sum(hf_mag) / magnitude_sum)

  envelope_mean = float(np.mean(abs_sig))
  envelope_std = float(np.std(abs_sig))

  autocorr = np.correlate(sig, sig, mode='full')
  autocorr = autocorr[len(autocorr)//2:]
  autocorr_peak_ratio = float(autocorr[1] / autocorr[0] if autocorr[0] > 0 else 0)

  return SignalFeatures(
    impusle_duration=impulse_duration,
    max_rise_rate=max_rise_rate,
    max_fall_rate=max_fall_rate,
    peak_to_rms=peak_to_rms,
    lf_energy_ratio=lf_energy_ratio,
    hf_energy_ratio=hf_energy_ratio,
    envelope_mean=envelope_mean,
    envelope_std=envelope_std,
    autocorr_peak_ratio=autocorr_peak_ratio
  )
