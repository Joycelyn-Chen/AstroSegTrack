# Energy analysis
- These steps will only work on `compute2.idsl`

## Environement setup
- Activate Joycelyn's environment: `conda activate sam_env`

## Execution
- Navigate to the root directory of this project: `cd /home/joy0921/Desktop/AstroSegTrack`

## Calculating masked energy
- execute: `python analysis/masked_energy.py --hdf5_root /home/joy0921/Desktop/Dataset/SB230/HDF5 --start_time_Myr 209 --end_time_Myr 211 --dataset_root ../Dataset/SB230`
- The results will pop up on the screen, and will also be stored here: `/home/joy0921/Desktop/Dataset/SB230`
