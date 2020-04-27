# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:09:22 2020

@author: Toshiba
"""

from numpy import sqrt
from numpy.random import rand, randn
import numpy as np
from binom import binom
from PossibleCodewords import PossibleCodewords
from CodewordsIntoBits import CodewordsIntoBits
from BitsIntoCodewords import BitsIntoCodewords


## General Parameters
Frames = 100  # Number of Frames of length N input symbols                                                                                                                                           # Number of OFDM frames(symbols)
M = 4  # M-QAM constellation
NFFT = 72  # for 6 users
N = 12  # number of total subcarriers
CP = NFFT / 8  # length CP
n = 4  # number of subcarriers per group g
g = N / n  # number of groups
k = 1  # number of active subcarriers per group
# g     = 18
# n     = N/g
# alphaa  = M/(M+1)
# k = round(alphaa*n)
K = k * g  # total number of active carriers
m1 = np.floor(np.log2(binom(n, k))) * g  # number of bits for the carriers indices
m2 = k * np.log2(M) * g  # number of bits for a constellation M-ary
m = m1 + m2  # total number of bits for an OFDM block
norms = np.sqrt(1 ** 2 + np.sqrt(2) ** 2 + 0 ** 2 + np.sqrt(10) ** 2 + 0 ** 2 + np.sqrt(
    42) ** 2)  # normalization of the constellation

## Channel parameters
Fade = 1  # = 1 if channel effect is considered, 0 otherwise
v = 1  # Number of paths for Rayleigh channel

## BER parameters
EbNo_L = 0
EbNo_U = 25
EbNo_dB = np.linspace(EbNo_L, EbNo_U, 10)
EbNo_lin = 10 ** (EbNo_dB / 10)
ber = np.zeros(np.size(EbNo_lin))
SNR_BER_db = EbNo_dB + 10 * np.log10(m / (N + CP))
SNR_BER = 10 ** (SNR_BER_db / 10)

## Possible Codeword
Codewords = PossibleCodewords(M, n, k)
C = len(Codewords)
## Generation of binary data
Binary = np.array([0, 1])  # Binary data that take 0 or 1
np.random.standard_normal(12345)  # seeding the Matlab random number generator

for j in range(1, len(EbNo_dB)):
    j
    ## Generation of codewords (symbols) from random bits
    InputBinaryS = [[np.random.choice([1, 2]) for _ in range(Frames*m)]]
    symbols = BitsIntoCodewords(InputBinaryS, m, m1, N, M, n, k, g, Frames)
    ## Channel
    impulse_response = np.zeros(NFFT, Frames)
    if Fade == 1:
        impulse_response[0:v - 1, ] = (1 / sqrt(2 * v)) * complex(randn(v, Frames),
                                                                  randn(v, Frames))  # Rayleigh variance = 1
        fading_coeffs = np.fft(impulse_response)
    else:
        fading_coeffs = np.ones(NFFT, Frames)

    ## DFT
    TxNFFT = np.zeros(NFFT, Frames)
    TxDFT = (1 / np.sqrt(N)) * np.fft(symbols)
    ## Interleaving
    #TxDFTInterleaved = Interleaving(TxDFT,g) # interleaving
    TxDFTInterleaved = TxDFT  # no intelreaving
    ## Subcarrier mapping (Access mode)
    TxNFFT[0:N - 1, 0:Frames - 1] = TxDFTInterleaved  # Localized-FDMA
    #TxNFFT(1:NFFT/N:NFFT,:) = TxDFT # interleaved-FDMA
    ## IFFT
    Tx = (NFFT / np.sqrt(K)) * np.ifft(TxNFFT)  # (sqrt(NFFT))*
    ## CP and Channel
    TxCP = [Tx[end - CP + 1:, ] * Tx]  # cyclic prefix
    for fr in range(Frames - 1):
        TxCPChannel[:, fr - 1] = filter(impulse_response[:, fr - 1], 1, TxCP[:, fr - 1])

    ## GENERATE AND ADD AWGN
    P_signal = np.mean(np.mean(abs(TxCPChannel) ** 2))
    P_noise = P_signal * 10 ^ (-SNR_BER_db(j) / 10)
    noise_norm = np.sqrt(0.5) * complex(randn(NFFT + CP, Frames), randn(NFFT + CP, Frames))
    En = sqrt(P_noise) * noise_norm
    TxCPChannelNoise = TxCPChannel + En
    # "--------------------------------------------------------------------------------------------------------------------------------------------------"
    ## Remove CP
    TxChannelNoise = TxCPChannelNoise[CP:, ]
    ## Channel Equalization
    mmse = np.conj(fading_coeffs) / ((abs(fading_coeffs) ** 2) + (P_noise / P_signal))
    Rx = (sqrt(K) / NFFT) * np.fft(
        TxChannelNoise) * mmse  # ./fading_coeffs # sqrt(NFFT))*#.*  ./ ??????????????????????????
    ## Subcarrier demapping
    RxIDFT = Rx[0:N - 1, 0:Frames - 1]  # Localized FDMA
    #     RxIDFT = Rx(1:NFFT/N:NFFT,:)  # Interleaved FDMA
    ## Deinterleaving
    #     RxIDFTDeinterleaved = DeInterleaving(RxIDFT,g) # interleaving
    RxIDFTDeinterleaved = RxIDFT  # no interleaving
    ## IDFT
    symbols_est = (sqrt(N)) * np.ifft(RxIDFTDeinterleaved)
    ## CodeWord Detection
    Data_out = np.zeros(g, Frames)  # zero recovered data matrix
    for fr in range(Frames - 1):
        for sub_block in range(g):
            ##################### ML detection routine ###################
            Detect_output = np.zeros(C, 1)  # zeroise Detect_output array
            # cycle through codeword set
            for count in range(C):
                code_sent = Codewords[count, :].transport()
                F = symbols_est[(sub_block - 1) * n + 1:sub_block * n, fr]
                H = fading_coeffs[(sub_block - 1) * n + 1:sub_block * n, fr]
                #                 Detect_output(count,1) =  sum(abs(F - code_sent.*H).^2) #if no equalization (only for OFDM)
                Detect_output[count, 1] = sum(abs(F - code_sent) ** 2)

            # Determine Data_sent from minimum
            [Minimum, Data_out[sub_block, fr]] = min(Detect_output)

    ## Demapping
    Data_out_reshape = np.reshape(Data_out, [], 1)
    symbols_estimated = np.reshape(Codewords[Data_out_reshape, :].tansport(), [], Frames)
    OutputBinaryp = CodewordsIntoBits(symbols_estimated, M, n, k, g, Frames)
    OutputBinary = np.reshape(OutputBinaryp, 1, [])
    ## BER Calculation
    Nbre_error = len(np.find(np.reshape(InputBinaryS, 1, []) - OutputBinary != 0))  # Number of errors
    ber[j] = Nbre_error / (Frames * m)  # number of errors per number of total bit
# end  SNR












