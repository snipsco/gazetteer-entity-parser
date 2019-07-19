#!/bin/bash

set -e

NEW_VERSION=$1

if [[ -z ${NEW_VERSION} ]]; then
    echo "Usage: $0 NEW_VERSION"
    exit 1
fi

SPLIT_VERSION=( ${NEW_VERSION//./ } )
if [[ ${#SPLIT_VERSION[@]} -ne 3 ]]; then
  echo "Version number is invalid (must be of the form x.y.z)"
  exit 1
fi

perl -p -i -e "s/^version = \".*\"\$/version = \"$NEW_VERSION\"/g" Cargo.toml
