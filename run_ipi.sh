#!/bin/bash

rm output.checkpoint start_*.xyz walker-*_output.md walker-*trajectory*.extxyz RESTART

python server.py --input_xml=input.xml --nwalkers=8 --start_xyz=labeled.xyz &

sleep 3s
python client.py --xyz labeled.xyz --model_path highest_acc.pth --device cuda --mode hills --address socket0 --height 0.01 --sigma 0.4 --frequency 20 --hills hills &
python client.py --xyz labeled.xyz --model_path highest_acc.pth --device cuda --mode hills --address socket0 --height 0.01 --sigma 0.4 --frequency 20 --hills hills &

#for i in {0..1}
#do
#    python client.py &
#done

wait

python server.py --cleanup
