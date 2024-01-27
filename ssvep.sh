#!/bin/bash
i=1
testname="${1}SSVEP$i"
while [[ -e $testname.txt ]] ; do
    let i++
    testname="${1}SSVEP$i"
done

cl_rpc openxdf "$testname.gdf" "$testname.log" ""
echo "Recording started for $testname.gdf"
echo "Beginning ERP stimulus"
python stimulus-ssvep.py
echo "Closing recording"
cl_rpc closexdf