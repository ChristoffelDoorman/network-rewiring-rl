#! /bin/bash

if ! [[ $1 = "up" || $1 = "stop" || $1 = "down" ]]; then
         echo "illegal first argument. must be one of [up, stop, down]."
         exit 1
fi


compose_command=$1

if [[ $compose_command == "up" ]]; then
	compose_command="$compose_command -d"
fi
if [[ ${compose_command:0:2} == "up" || $compose_command == "stop" ]]; then
  compose_command="$compose_command manager"
fi

docker-compose -f $RN_SOURCE_DIR/docker-compose.yml $compose_command
