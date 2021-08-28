# uav-deep-q
This is the source code using deep Q learning for finding the UAVs' positions to maximize the number of served users in the pre-defined target area


First step, run DQL_2agents.py to find the UAV's position that maximizes the total number of served users. 

Second step, run main_UAV_optimization.m to find the optimal transmit beamforming of UAVs, transmit power of MBS, and UAV-user association according to the channels associated to the UAV's positions in the previous step.
