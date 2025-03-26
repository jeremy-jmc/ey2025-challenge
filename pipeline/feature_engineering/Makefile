run:
	nohup python3 focal_radius_buffers.py > output_1.log 2>&1 && \
	nohup python3 landsat.py >> output_2.log 2>&1 && \
	nohup python3 sentinel2_bands.py >> output_3.log 2>&1 && \
	nohup python3 sentinel2_focal_buffers_bands.py >> output_4.log 2>&1 && \
	nohup python3 sentinel2_focal_buffers.py >> output_5.log 2>&1 && \
	nohup python3 ny_mesonet.py >> output_6.log 2>&1 && \
	nohup python3 kml.py >> output_7.log 2>&1 && \
	nohup python3 join_data.py >> output_8.log 2>&1 &
ls_with_mb_size:
	watch -n 1 ls -lhtR --block-size=M ../data/processed/