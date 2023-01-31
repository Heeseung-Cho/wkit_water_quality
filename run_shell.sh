#!/bin/sh

for target in "2-MIB" "Geosmin"
do
  python main.py  --data_path C001팔당2취 --target $target --train_epochs 100
done

for target in "TOC" "blue-green_algae" "diatomeae" "2-MIB" "Geosmin" "synedra" "Mn"
do
  python main.py --data_path D001청주정 --target $target --train_epochs 100
done

for target in "blue-green_algae" "diatomeae" "2-MIB" "Geosmin"
do
  python main.py --data_path D002구미 --target $target --train_epochs 100
done

for target in "diatomeae" "2-MIB" "Geosmin"
do
  python main.py --data_path D003고령 --target $target --train_epochs 100
done
 
#for target in "Mn"
#do
#  python main.py --data_path D004주암 --target $target --train_epochs 100
#done
 
for target in "blue-green_algae" "diatomeae" "2-MIB" "Geosmin"
do
  python main.py --data_path E003본포 --target $target --train_epochs 100
done

for target in "2-MIB" "Geosmin"
do
  python main.py --data_path E004남강 --target $target --train_epochs 100
done