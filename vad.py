# -*- coding: utf-8 -*-
"""
Created on Tue May  1 20:43:28 2018

@author: eesungkim
"""

import numpy as np
import scipy.io.wavfile as wav
import numpy.testing
import scipy.special as sp
import matplotlib.pyplot as plt
from utils.estnoise_ms import * 

def VAD(signal, sr, nFFT=512, win_length=0.025, hop_length=0.01):
    """Voice Activity Detector
    Parameters
    ----------
    signal     : audio time series

    sr    : sampling rate of `signal`

    nFFT     : length of the FFT window

    n_mfcc: int > 0 [scalar]
        number of MFCCs to return

    win_length : window size in sec

    hop_length : hop size in sec

    Returns
    -------
    probRatio     : frame-based voice activity probability sequence
    """
    signal=signal.astype('float')

    maxPosteriorSNR= 1000   
    minPosteriorSNR= 0.0001    

    win_length_sample = round(win_length*sr)
    hop_length_sample = round(hop_length*sr)    

    # the variance of the speech; lambda_x(k)
    _stft = stft(signal, n_fft=nFFT, win_length=win_length_sample, hop_length=hop_length_sample)
    pSpectrum = np.abs(_stft) ** 2                     
    
    # estimate the variance of the noise using minimum statistics noise PSD estimation ; lambda_d(k). 
    estNoise = estnoisem(pSpectrum,hop_length)     
    estNoise = estNoise
    
    aPosterioriSNR=pSpectrum/estNoise                    
    aPosterioriSNR=aPosterioriSNR
    aPosterioriSNR[aPosterioriSNR > maxPosteriorSNR] = maxPosteriorSNR
    aPosterioriSNR[aPosterioriSNR < minPosteriorSNR] = minPosteriorSNR

    a01=hop_length/0.05     # a01=P(signallence->speech)  hop_length/mean signallence length (50 ms)
    a00=1-a01               # a00=P(signallence->signallence)
    a10=hop_length/0.1      # a10=P(speech->signallence) hop/mean talkspurt length (100 ms)
    a11=1-a10               # a11=P(speech->speech)

    b01=a01/a00
    b10=a11-a10*a01/a00
  
    smoothFactorDD=0.99
    previousGainedaPosSNR=1 
    (nFrames,nFFT2) = pSpectrum.shape                
    probRatio=np.zeros((nFrames,1))
    logGamma_frame=0                          
    for i in range(nFrames):                         
        aPosterioriSNR_frame = aPosterioriSNR[i,:]                  
        
        #operator [2](52)
        oper=aPosterioriSNR_frame-1
        oper[oper < 0] = 0 
        smoothed_a_priori_SNR = smoothFactorDD * previousGainedaPosSNR + (1-smoothFactorDD) * oper
        
        #V for MMSE estimate ([2](8)) 
        V=0.5*smoothed_a_priori_SNR*aPosterioriSNR_frame/(1+smoothed_a_priori_SNR)            
        
        #geometric mean of log likelihood ratios for individual frequency band  [1](4)
        logLRforFreqBins=2*V-np.log(smoothed_a_priori_SNR+1)              
        #logLRforFreqBins=np.exp(smoothed_a_priori_SNR*aPosterioriSNR_frame/(1+smoothed_a_priori_SNR))/(1+smoothed_a_priori_SNR)
        gMeanLogLRT=np.mean(logLRforFreqBins)       
        logGamma_frame=np.log(a10/a01) + gMeanLogLRT + np.log(b01+b10/( a10+a00*np.exp(-logGamma_frame) ) )
        probRatio[i]=1/(1+np.exp(-logGamma_frame))
        
        #Calculate Gain function which results from the MMSE [2](7). 
        gainFunc=(0.277+2*V)/aPosterioriSNR_frame          
        if any(V<0.5):
            VV=V[V<0.5]
            gainFunc[V<0.5]=np.sqrt(2*np.pi)*np.sqrt(VV)*((0.5+VV)*sp.iv(0,VV)+VV*sp.iv(1,VV)) / (aPosterioriSNR_frame[V<0.5]*np.exp(VV)) # [2](7)
        previousGainedaPosSNR = (gainFunc**2) * aPosterioriSNR_frame

    return probRatio

if __name__ == '__main__':
    path='C:/Users/eesungkim/FAJW0_SX93_2.wav';
    (sr, signal) = wav.read(path)

    vad=VAD(signal, sr, nFFT=512, win_length=0.025, hop_length=0.01)
          
    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.title('Time Signal')
    
    plt.subplot(2, 1, 2)
    plt.plot(vad)
    plt.xlabel('frame')
    plt.ylabel('Prob')

    plt.tight_layout()
    plt.show()
    

