#! /bin/bash

chgrp -R $RN_GNAME $RN_SOURCE_DIR
chmod -R g+rwx $RN_SOURCE_DIR

mkdir -p $RN_EXPERIMENT_DATA_DIR
mkdir -p $RN_EXPERIMENT_DATA_DIR/real_world_graphs/raw_data

chgrp -R $RN_GNAME $RN_EXPERIMENT_DATA_DIR
chmod -R g+rwx $RN_EXPERIMENT_DATA_DIR

chgrp -R $RN_GNAME $RN_APP_DATA_DIR
chmod -R g+rwx $RN_APP_DATA_DIR