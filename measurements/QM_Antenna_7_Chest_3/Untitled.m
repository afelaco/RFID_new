clc
clear all
close all

sobj = sparameters('Data\Measurement\QM_Antenna_7_Chest_new\1_1.s2p');

S = sobj.Parameters(:,:,587);

S = 20.*log10(S);