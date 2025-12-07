'''
Basic setup
'''

INDIR = './'
OUTDIR = './'
FEATURES = ['pm10_lag1','temp','rh']
TARGET = ['pm25']
INPUT_DIM = len(FEATURES)
OUTPUT_DIM = len(TARGET)
HIDDEN_DIM = 5
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-1
MOMENTUM = 0.8
SCALE = True #no-op
SEED = 42
N_FOLDS = 5
COLORS = ['red','#7680b5','#44bb66','orange','brown','black']
COLORS_AUX = [color for color in COLORS for _ in (0, 1)]
METRICS = ['MAE','RMSE',r'$R^2$']
