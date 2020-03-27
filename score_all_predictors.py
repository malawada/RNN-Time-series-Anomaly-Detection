import anomaly_detection
import argparse


parser = argparse.ArgumentParser(description='PyTorch RNN Anomaly Detection Model')
parser.add_argument('--prediction_window_size', type=int, default=10,
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
                    
args = parser.parse_args()

files = defaultdict()
files['IMS_bearings_test1']=("combinedfiles_s4.pkl",
                        "combinedfiles_s5.pkl",
                        "combinedfiles_s6.pkl",
                        "combinedfiles_s7.pkl")
files['IMS_bearings_test2']=("combinedfiles_s0.pkl")
files['IMS_bearings_test3']=("combinedfiles_s2.pkl")
files['FEMTO_bearings_1_1']=("combinedfiles_accel_s0.pkl",
				"combinedfiles_accel_s1.pkl")
files['FEMTO_bearings_1_2']=("combinedfiles_accel_s0.pkl",
				"combinedfiles_accel_s1.pkl")
files['FEMTO_bearings_2_1']=("combinedfiles_accel_s0.pkl",
				"combinedfiles_accel_s1.pkl")
files['FEMTO_bearings_2_2']=("combinedfiles_accel_s0.pkl",
				"combinedfiles_accel_s1.pkl")
files['FEMTO_bearings_3_1']=("combinedfiles_accel_s0.pkl",
				"combinedfiles_accel_s1.pkl")
files['FEMTO_bearings_3_2']=("combinedfiles_accel_s0.pkl",
				"combinedfiles_accel_s1.pkl")
files['Bearing_1_3']=("Bearing_1_3combinedfiles_accel_s0.pkl",
				"Bearing_1_3combinedfiles_accel_s1.pkl")
files['Bearing_1_4']=("Bearing_1_4combinedfiles_accel_s0.pkl",
				"Bearing_1_4combinedfiles_accel_s1.pkl")
files['Bearing_1_5']=("Bearing_1_5combinedfiles_accel_s0.pkl",
				"Bearing_1_5combinedfiles_accel_s1.pkl")
files['Bearing_1_6']=("Bearing_1_6combinedfiles_accel_s0.pkl",
				"Bearing_1_6combinedfiles_accel_s1.pkl")
files['Bearing_1_7']=("Bearing_1_7combinedfiles_accel_s0.pkl",
				"Bearing_1_7combinedfiles_accel_s1.pkl")
files['Bearing_2_3']=("Bearing_2_3combinedfiles_accel_s0.pkl",
				"Bearing_2_3combinedfiles_accel_s1.pkl")
files['Bearing_2_4']=("Bearing_2_4combinedfiles_accel_s0.pkl",
				"Bearing_2_4combinedfiles_accel_s1.pkl")
files['Bearing_2_5']=("Bearing_2_5combinedfiles_accel_s0.pkl",
				"Bearing_2_5combinedfiles_accel_s1.pkl")
files['Bearing_2_6']=("Bearing_2_6combinedfiles_accel_s0.pkl",
				"Bearing_2_6combinedfiles_accel_s1.pkl")
files['Bearing_2_7']=("Bearing_2_7combinedfiles_accel_s0.pkl",
				"Bearing_2_7combinedfiles_accel_s1.pkl")
files['Bearing_3_3']=("Bearing_3_3combinedfiles_accel_s0.pkl",
				"Bearing_3_3combinedfiles_accel_s1.pkl")
               
               
for dir, filelist in files.items():
    for f in filelist:
        args.data = dir
        args.filename = f
        
        try:
            anomaly_detection.main(args)
        except RuntimeError as err:
            print("EXCEPTION OCCURRED")
            print(err)
            with open("errlog_ad.txt",'a+') as f:
                f.write(err)
                f.write("\n\n")
        
print("scoring of all files completed")