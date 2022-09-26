#!/bin/bash

CURRENT="$PWD"
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir -p "$ROOT/THIRD_PARTY"
cd "$ROOT/THIRD_PARTY"
git clone 'https://github.com/artetxem/undreamt.git' undreamt
cd undreamt
cd ..
git clone 'https://github.com/artetxem/vecmap.git'
git clone 'https://github.com/rsennrich/subword-nmt.git'
