#!/usr/bin/env bash

set -euo pipefail

FILENAME=YACCLAB_dataset3D.zip

mkdir -p 3d
cd 3d
wget http://imagelab.ing.unimore.it/files/$FILENAME
unzip $FILENAME
rm $FILENAME
