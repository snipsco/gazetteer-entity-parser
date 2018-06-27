#!/bin/bash

set -e

NEW_VERSION=$1

if [ -z $NEW_VERSION ]; then
    echo "Usage: $0 NEW_VERSION"
    exit 1
fi

SPLIT_VERSION=( ${NEW_VERSION//./ } )
if [ ${#SPLIT_VERSION[@]} -ne 3 ]; then
  echo "Version number is invalid (must be of the form x.y.z)"
  exit 1
fi

let SPLIT_VERSION[1]+=1
NEXT_NEW_VERSION="${SPLIT_VERSION[0]}.${SPLIT_VERSION[1]}.${SPLIT_VERSION[2]}"

echo "REAL VERSION $NEW_VERSION"
echo "NEXT VERSION $NEXT_NEW_VERSION"

perl -p -i -e "s/^version = \".*\"\$/version = \"$NEW_VERSION\"/g" Cargo.toml

cargo update \
    -p gazetteer-entity-parser 

