#!/bin/bash

uv run track_seva.py --bag data/10deg_20260804_161328.mcap --topic_prefix /wide_angle_camera_front --output 10deg_20260804_161328_front.txt
uv run track_seva.py --bag data/10deg_20260804_161328.mcap --topic_prefix /wide_angle_camera_rear --output 10deg_20260804_161328_rear.txt


uv run track_seva.py --bag data/20degtrick_20260804_172149.mcap --topic_prefix /wide_angle_camera_front --output 20degtrick_20260804_172149_front.txt
uv run track_seva.py --bag data/20degtrick_20260804_172149.mcap --topic_prefix /wide_angle_camera_rear --output 20degtrick_20260804_172149_rear.txt