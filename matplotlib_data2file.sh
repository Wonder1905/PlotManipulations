#!/bin/bash

find PlotsLegend/train/ -name "*jpg*" > PlotsLegend/matplots_img_train.txt
find PlotsLegend/test/ -name "*jpg*" > PlotsLegend/matplots_img_test.txt
sed -i -e 's/^/\/workspace\/FigureEdit\//' PlotsLegend/matplots_img_train.txt
sed -i -e 's/^/\/workspace\/FigureEdit\//' PlotsLegend/matplots_img_test.txt
