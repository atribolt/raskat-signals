import sys
import numpy as np
import raskat_signals as rs
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path


def show_signal(file: Path):
  with file.open('rb') as f:
    sig = rs.SignalFile.create(f)
  s = sig.samples_voltage

  print('Show: ', file)
  print('\t', sig)

  fft = np.fft.fft(s).real
  mg = (fft ** 2) / fft.size / sig.power_resitance
  fft = 10 * np.log10(mg)
  freq = np.fft.fftfreq(s.size, 1 / sig.sample_rate)[:fft.size // 2]
  fig, (ax0, ax1) = plt.subplots(2, 1)
  ax0.plot(s)
  ax1.plot(freq, fft[:fft.size//2])
  plt.show()


def main():
  parser = ArgumentParser('raskat-signal')
  parser.add_argument('-s', '--signal', help='Dir with signals or signal file', type=Path, required=True)
  args = parser.parse_args()

  signal: Path = args.signal

  if not signal.exists():
    print('Signal file is not exists')
    sys.exit(1)

  if signal.is_dir():
    signals = signal.glob('*.sig')
  else:
    signals = [signal]

  for file in signals:
    show_signal(file)


if __name__ == '__main__':
  main()
