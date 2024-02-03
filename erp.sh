#!/bin/bash
i=1
testname="${1}ERP$i"
while [[ -e $testname.gdf ]] ; do
    let i++
    testname="${1}ERP$i"
done

cl_rpc openxdf "$testname.gdf" "$testname.log" ""
echo "Recording started for $testname.gdf"
echo "Beginning ERP stimulus"
python pygame-erp.py
echo "Closing recording"
cl_rpc closexdf