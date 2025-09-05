import raskat_signals as rs
from argparse import ArgumentParser
from pathlib import Path


def dump_json(file: Path, out: Path, prefix: str = ''):
  with file.open('rb') as f:
    sig = rs.SignalFile.create(f)
    print('Dump json: ', sig)

    dumpfile = out / (prefix + sig.begin_time.isoformat() + '.json')
    with dumpfile.open('w') as outf:
      outf.write(sig.tojson())


def main():
  parser = ArgumentParser('raskat-dump')
  parser.add_argument('-p', '--prefix', help='Prefix for out files', type=str, default='')
  parser.add_argument('-o', '--output', help='Output dir', type=Path, default=Path('.'))
  parser.add_argument(dest='signal', nargs='+', help='Dir with signals or signal file', type=Path)
  args = parser.parse_args()

  signals: list[Path] = args.signal
  outdir: Path = args.output
  fprefix: str = args.prefix

  if not outdir.exists():
    outdir.mkdir(parents=True, exist_ok=True)

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
        dump_json(sign, out=outdir, prefix=fprefix)

  except KeyboardInterrupt:
    pass


if __name__ == '__main__':
  main()
