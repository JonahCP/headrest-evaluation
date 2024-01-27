#!/bin/bash
i=1
testname="${1}SSVEP$i"
while [[ -e $testname.txt ]] ; do
    let i++
    testname="${1}SSVEP$i"
done

cl_rpc openxdf "$testname.gdf" "$testname.log" ""
python stimulus-ssvep.py
cl_rpc closexdf