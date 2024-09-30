#!/usr/bin/env bash

set -euo pipefail

FILENAME=YACCLAB_dataset_new.zip

mkdir -p 2d
cd 2d
wget http://imagelab.ing.unimore.it/files/$FILENAME
unzip $FILENAME
rm $FILENAME
