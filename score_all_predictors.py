import anomaly_detection
import argparse
from collections import defaultdict
import os
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch RNN Anomaly Detection Model')
parser.add_argument('--prediction_window_size', type=int, default=1,
                    help='prediction_window_size')
parser.add_argument('--data', type=str, default='IMS_bearings_test2',
                    help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
parser.add_argument('--filename', type=str, default='combinedfiles_s0.pkl',
                    help='filename of the dataset')
parser.add_argument('--save_fig', default=True,
                    help='save results as figures')
parser.add_argument('--compensate', action='store_true',
                    help='compensate anomaly score using anomaly score esimation')
parser.add_argument('--beta', type=float, default=1.0,
                    help='beta value for f-beta score')
parser.add_argument('--model', type=str, default="LSTM")
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--use_SVR', action='store_true')
parser.add_argument('--learning_rate', type=float, default=0.0002)
parser.add_argument('--save_path', type=str, default='save')
                    
args = parser.parse_args()

files = defaultdict()
files['IMS_bearings_test1']=("combinedfiles_s4.pkl",
							"combinedfiles_s5.pkl",
							"combinedfiles_s6.pkl",
							"combinedfiles_s7.pkl")
files['IMS_bearings_test2']=(["combinedfiles_s0.pkl"])
files['IMS_bearings_test3']=(["combinedfiles_s2.pkl"])
files['FEMTO_bearings_1_1']=("combinedfiles_accel_s0.pkl",
							"combinedfiles_accel_s1.pkl")
files['FEMTO_bearings_1_2']=("combinedfiles_accel_s0.pkl",
							"combinedfiles_accel_s1.pkl")
files['FEMTO_bearings_2_1']=("combinedfiles_accel_s0.pkl",
							"combinedfiles_accel_s1.pkl")
files['FEMTO_bearings_2_2']=("combinedfiles_accel_s0.pkl",
							"combinedfiles_accel_s1.pkl")
files['Bearing2_3']=("Bearing2_3combinedfiles_accel_s0.pkl",
				 "Bearing2_3combinedfiles_accel_s1.pkl")
files['FEMTO_bearings_3_1']=("combinedfiles_accel_s0.pkl",
							"combinedfiles_accel_s1.pkl")
files['FEMTO_bearings_3_2']=("combinedfiles_accel_s0.pkl",
							"combinedfiles_accel_s1.pkl")
files['Bearing1_3']=("Bearing1_3combinedfiles_accel_s0.pkl",
					"Bearing1_3combinedfiles_accel_s1.pkl")
files['Bearing1_4']=("Bearing1_4combinedfiles_accel_s0.pkl",
					"Bearing1_4combinedfiles_accel_s1.pkl")
files['Bearing1_5']=("Bearing1_5combinedfiles_accel_s0.pkl",
					"Bearing1_5combinedfiles_accel_s1.pkl")
files['Bearing1_6']=("Bearing1_6combinedfiles_accel_s0.pkl",
					"Bearing1_6combinedfiles_accel_s1.pkl")
files['Bearing1_7']=("Bearing1_7combinedfiles_accel_s0.pkl",
					"Bearing1_7combinedfiles_accel_s1.pkl")
files['Bearing2_4']=("Bearing2_4combinedfiles_accel_s0.pkl",
					"Bearing2_4combinedfiles_accel_s1.pkl")
files['Bearing2_5']=("Bearing2_5combinedfiles_accel_s1.pkl",
					"Bearing2_5combinedfiles_accel_s0.pkl")
files['Bearing2_6']=("Bearing2_6combinedfiles_accel_s0.pkl",
					"Bearing2_6combinedfiles_accel_s1.pkl")
files['Bearing2_7']=("Bearing2_7combinedfiles_accel_s0.pkl",
					"Bearing2_7combinedfiles_accel_s1.pkl")
files['Bearing3_3']=("Bearing3_3combinedfiles_accel_s0.pkl",
					"Bearing3_3combinedfiles_accel_s1.pkl")
               
#CAUSED ERRORS:
# files['Bearing1_4']=(["Bearing1_4combinedfiles_accel_s1.pkl"])
# files['Bearing2_3']=("Bearing2_3combinedfiles_accel_s0.pkl",
				# "Bearing2_3combinedfiles_accel_s1.pkl")
# files['Bearing2_5']=(["Bearing2_5combinedfiles_accel_s1.pkl"])
               
for dir, filelist in files.items():
    for f in filelist:
        args.data = dir
        args.filename = f
        if not os.path.exists('dataset/'+dir+'/labeled/test/'+f):
            print("file does not exist:" + dir + "/" + f)
        else:
            try:
                print("detecting " + dir + "/" + f)
                anomaly_detection.main(args)
            except RuntimeError as err:
                print("EXCEPTION OCCURRED")
                print(err)
                with open("errlog_ad.txt",'a+') as f:
                    f.write(str(err))
                    f.write("\n\n")
        
print("scoring of all files completed")