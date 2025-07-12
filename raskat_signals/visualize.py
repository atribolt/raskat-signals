import sys
import numpy as np
import raskat_signals as rs
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path


def show_signal(file: Path):
  print('Show: ', file)
  with file.open('rb') as f:
    sig = rs.SignalFile.create(f)
  s = (sig.samples * 3.3) / 2**10
  fft = np.fft.fft(s)
  mg  = (fft ** 2) / fft.size / 1000
  fft = 10 * np.log10(mg * 1000)
  freq = np.fft.fftfreq(s.size, 1 / sig.sample_rate)[:fft.size // 2]
  fig, (ax0, ax1) = plt.subplots(2, 1)
  ax0.plot(s)
  ax1.plot(freq, fft[:fft.size// 2])
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
    signal = [signal]

  
  for file in Path(sys.argv[1]).glob('*.sig'):
    show_signal(file)
