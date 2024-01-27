#!/bin/bash
i=1
testname="${1}ERP$i"
while [[ -e $testname.txt ]] ; do
    let i++
    testname="${1}ERP$i"
done

cl_rpc openxdf "$testname.gdf" "$testname.log" ""
python stimulus-erp.py
cl_rpc closexdf