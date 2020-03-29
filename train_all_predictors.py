from collections import defaultdict
import argparse
import train_predictor
import os

parser = argparse.ArgumentParser(description='PyTorch RNN Prediction Model on Time-series Dataset')
parser.add_argument('--data', type=str, default='IMS_bearings_test2',
                    help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
parser.add_argument('--filename', type=str, default='combinedfiles_s0.pkl',
                    help='filename of the dataset')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, SRU)')
parser.add_argument('--augment', type=bool, default=True,
                    help='augment')
parser.add_argument('--emsize', type=int, default=128,
                    help='size of rnn input features')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--res_connection', action='store_true',
                    help='residual connection')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay')
parser.add_argument('--clip', type=float, default=10,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=500,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=64, metavar='N',
                    help='eval_batch size')
parser.add_argument('--bptt', type=int, default=50,
                    help='sequence length')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.7,
                    help='teacher forcing ratio (deprecated)')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights (deprecated)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--device', type=str, default='cuda',
                    help='cuda or cpu')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='report interval')
parser.add_argument('--save_interval', type=int, default=10, metavar='N',
                    help='save interval')
parser.add_argument('--save_fig', default=True,
                    help='save figure')
parser.add_argument('--resume','-r',
                    help='use checkpoint model parameters as initial parameters (default: True)',
                    default=False)
parser.add_argument('--pretrained','-p',
                    help='use checkpoint model parameters and do not train anymore (default: False)',
                    action="store_true")
parser.add_argument('--prediction_window_size', type=int, default=10,
                    help='prediction_window_size')
args = parser.parse_args()


files = defaultdict()
# files['IMS_bearings_test1']=("combinedfiles_s4.pkl",
                        # "combinedfiles_s5.pkl",
                        # "combinedfiles_s6.pkl",
                        # "combinedfiles_s7.pkl")
# files['IMS_bearings_test2']=(["combinedfiles_s0.pkl"])
# files['IMS_bearings_test3']=(["combinedfiles_s2.pkl"])
# files['FEMTO_bearings_1_1']=("combinedfiles_accel_s0.pkl",
				# "combinedfiles_accel_s1.pkl")
# files['FEMTO_bearings_1_2']=("combinedfiles_accel_s0.pkl",
				# "combinedfiles_accel_s1.pkl")
# files['FEMTO_bearings_2_1']=("combinedfiles_accel_s0.pkl",
				# "combinedfiles_accel_s1.pkl")
# files['FEMTO_bearings_2_2']=("combinedfiles_accel_s0.pkl",
				# "combinedfiles_accel_s1.pkl")
# files['FEMTO_bearings_3_1']=("combinedfiles_accel_s0.pkl",
				# "combinedfiles_accel_s1.pkl")
# files['FEMTO_bearings_3_2']=("combinedfiles_accel_s0.pkl",
				# "combinedfiles_accel_s1.pkl")
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
files['Bearing2_3']=("Bearing2_3combinedfiles_accel_s0.pkl",
				"Bearing2_3combinedfiles_accel_s1.pkl")
files['Bearing2_4']=("Bearing2_4combinedfiles_accel_s0.pkl",
				"Bearing2_4combinedfiles_accel_s1.pkl")
files['Bearing2_5']=("Bearing2_5combinedfiles_accel_s0.pkl",
				"Bearing2_5combinedfiles_accel_s1.pkl")
files['Bearing2_6']=("Bearing2_6combinedfiles_accel_s0.pkl",
				"Bearing2_6combinedfiles_accel_s1.pkl")
files['Bearing2_7']=("Bearing2_7combinedfiles_accel_s0.pkl",
				"Bearing2_7combinedfiles_accel_s1.pkl")
files['Bearing3_3']=("Bearing3_3combinedfiles_accel_s0.pkl",
				"Bearing3_3combinedfiles_accel_s1.pkl")
               

for dir, filelist in files.items():
    for f in filelist:
        args.data = dir
        args.filename = f
        if(os.path.exists('dataset/'+dir+'/labeled/train/'+f)):
            try:
                train_predictor.main(args)
            except RuntimeError as err:
                print("EXCEPTION OCCURRED")
                print(err)
                with open("errlog.txt",'a+') as f:
                    f.write(err)
                    f.write("\n\n")
        else:
            print("File does not exist! filename:" + dir + '/' + f)
        
print("training of all files completed")
