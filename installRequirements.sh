#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
pushd "$script_dir"

merged_requirements=$(mktemp)
find . -name 'requirements.txt' -exec cat {} + >> $merged_requirements
awk '!visited[$0]++' $merged_requirements > $merged_requirements.tmp && mv $merged_requirements.tmp $merged_requirements

pip install -r $merged_requirements

rm $merged_requirements

popd
