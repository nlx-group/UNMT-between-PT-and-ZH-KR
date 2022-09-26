#!/bin/bash

CURRENT="$PWD"
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir -p "$CURRENT/SCRIPTS"
cd "$ROOT/SCRIPTS"
git clone 'https://github.com/ymoslem/MT-Preparation.git'
