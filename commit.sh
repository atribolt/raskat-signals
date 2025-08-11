#!/bin/bash -e

TWEAK=$(( $(git rev-list --count HEAD .) + 1 ))

sed -r -i "s|^TWEAK.+$|TWEAK = ${TWEAK}|" raskat_signals/version.py

git add "raskat_signals/version.py"
git commit
