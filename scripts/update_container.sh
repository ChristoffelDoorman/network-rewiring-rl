#! /bin/bash

echo "<<UPDATING DOCKER IMAGE...>>"
build_targets='manager'
docker-compose -f $RN_SOURCE_DIR/docker-compose.yml build --force-rm $build_targets  && docker image prune -f
