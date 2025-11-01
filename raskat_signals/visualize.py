import json
import pprint
import numpy as np
import raskat_signals as rs
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path
import scipy.signal as ss
from dataclasses import dataclass


@dataclass
class FilterConfig:
  frequency: int
  order: int


def show_signal(file: Path, lpf: FilterConfig = None, hpf: FilterConfig = None, autocor: bool = False):
  with file.open('rb') as f:
    sig = rs.SignalFile.create(f)

  print('Show: ', file)
  print('\t', sig)
  print(json.dumps(rs.signal.characteristics(sig.samples_voltage)._asdict(), indent=2))

  filters = []

  if hpf:
    filters.append(ss.butter(hpf.order, hpf.frequency, 'high', False, fs=sig.sample_rate, output='sos'))

  if lpf:
    filters.append(ss.butter(lpf.order, lpf.frequency, 'low', False, fs=sig.sample_rate, output='sos'))

  orig_sig = signal = sig.samples_voltage - np.mean(sig.samples_voltage)
  for filt in filters:
    signal = ss.sosfiltfilt(filt, signal)
    print('Characteristics after filter:')
    print(json.dumps(rs.signal.characteristics(sig.samples_voltage)._asdict(), indent=2))

  fft = np.fft.fft(signal).real ** 2
  mg = (fft ** 2) / fft.size / sig.power_resitance
  fft = 10 * np.log10(mg)
  freq = np.fft.fftfreq(signal.size, 1 / sig.sample_rate)[:fft.size // 2]
  fig, (ax0, ax1) = plt.subplots(2, 1, height_ratios=[5, 2])

  if autocor:
    autocor = np.correlate(signal, signal, 'same')
    ax0.plot(autocor / np.max(autocor), ls='-.', lw=0.4, label='Autocorrelation')

  ax0.plot(orig_sig, ls=':' if lpf or hpf else '-', lw=0.5 if (lpf or hpf) else 1, label='Origin signal')
  ax0: plt.Axes
  ax0.axhline(sig.threshold_voltage, 0, signal.size, ls='--', lw=0.5, color='black', label='Threshold Max')
  ax0.axhline(-sig.threshold_voltage, 0, signal.size, ls='--', lw=0.5, color='black', label='Threshold Min')
  if lpf or hpf:
    ax0.plot(signal, lw=1, label='Filtered signal')

  ax1.plot(freq, fft[:fft.size//2], label='Spectrogram')

  ax0.legend()
  ax1.legend()
  fig.canvas.manager.set_window_title(str(sig))
  plt.show()


def main():
  parser = ArgumentParser('raskat-signal')
  parser.add_argument('-L', '--low-pass', help='Apply low pass filter with you frequency', default=None, type=int)
  parser.add_argument('--low-pass-order', help='Low pass filter order', default=10, type=int)
  parser.add_argument('-H', '--high-pass', help='Apply high pass filter with you frequency', default=None, type=int)
  parser.add_argument('--high-pass-order', help='High pass filter order', default=10, type=int)
  parser.add_argument('--autocor', help='Show autocorrelation for signal', action='store_true', default=False)
  parser.add_argument(dest='signal', nargs='+', help='Dir with signals or signal file', type=Path)
  args = parser.parse_args()

  signals: list[Path] = args.signal

  lpf = None
  if args.low_pass:
    lpf = FilterConfig(args.low_pass, args.low_pass_order)

  hpf = None
  if args.high_pass:
    hpf = FilterConfig(args.high_pass, args.high_pass_order)

  try:
    for path in signals:
      if not path.exists():
        print(f'Path is not exists: {path}')
        continue

      if path.is_dir():
        signal = path.rglob('*.sig')
      else:
        signal = [path]

      for sign in signal:
        show_signal(sign, lpf, hpf, autocor=args.autocor)
  except KeyboardInterrupt:
    pass


if __name__ == '__main__':
  main()
