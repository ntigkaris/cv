'''
author:     Alexandros E. Ntigkaris
python:     3.9.2
numpy:      1.21.6
pandas:     1.3.5
matplotlib: 3.3.4
sklearn:    1.0.2
torch:      1.13.0+cu116
pickle:     4.0.0
tqdm:       4.62.2
logging:    0.4.9.6
'''

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import logging

from utils.config import *
from utils.functions import make_preprocessing, predict_holdout
from utils.functions import make_NeuralNetwork, make_LinearRegression

logging.basicConfig(force=True,level=logging.INFO,format='%(asctime)s %(levelname)-8s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
plt.style.use('ggplot')

if __name__ == '__main__':
    df,h_df,h_scaler = make_preprocessing()
    
    total_scores = np.full((N_FOLDS,2,3),fill_value=np.nan) # (n_folds,n_models,n_metrics)
    
    for f in tqdm(range(N_FOLDS)):
        total_scores[f,:,:] = [
                                make_NeuralNetwork(df,f),
                                make_LinearRegression(df,f),
                               ]
    train_score = np.mean(total_scores,axis=0)
    
    logging.info(f'[ANN] mae: {train_score[0,0]:.4f} rmse: {train_score[0,1]:.4f} r2: {train_score[0,2]:.4f}')
    logging.info(f'[LNR] mae: {train_score[1,0]:.4f} rmse: {train_score[1,1]:.4f} r2: {train_score[1,2]:.4f}')
    
    h_scores = predict_holdout(h_df,h_scaler)
    h_score = np.mean(h_scores,axis=0)
    
    logging.info(f'[ANN] mae: {h_score[0,0]:.4f} rmse: {h_score[0,1]:.4f} IoA: {h_score[0,2]:.4f}')
    logging.info(f'[LNR] mae: {h_score[1,0]:.4f} rmse: {h_score[1,1]:.4f} IoA: {h_score[1,2]:.4f}')
    
    plt.figure(figsize=(18,5))
    
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.bar(
                range(12),
                np.concatenate([total_scores[:,:,i].flatten(),train_score[:,i].flatten()],axis=0),
                color=COLORS_AUX,
                )
        plt.title(METRICS[i])
        plt.xticks([])
    
    plt.savefig(OUTDIR+'training.png',dpi=300)
    
    plt.figure(figsize=(15,5))
    
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.bar(
                range(6),
                np.concatenate([h_scores[:,0,i].flatten(),h_score[0,i].flatten()],axis=0),
                color=COLORS,
                )
        plt.title(METRICS[i]) if (i != 2) else plt.title('IoA')
        plt.yticks(np.arange(0,13,1)) if (i != 2) else plt.yticks(np.arange(0,1,0.1))
        plt.xticks([])
    
    plt.savefig(OUTDIR+'inference.png',dpi=300)

