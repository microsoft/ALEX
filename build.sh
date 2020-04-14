#!/usr/bin/env bash
if [[ "$#" -ne 0 && $1 == "debug" ]]
then
    mkdir -p build_debug;
    cd build_debug;
    cmake -DCMAKE_BUILD_TYPE=Debug ..;
else
    mkdir -p build;
    cd build;
    cmake -DCMAKE_BUILD_TYPE=Release ..;
fi
make;
cd ..;