

from numpy.random import rand, randn
import numpy as np
from binom import binom
from PossibleCodewords import PossibleCodewords
from CodewordsIntoBits import CodewordsIntoBits
from BitsIntoCodewords import BitsIntoCodewords
from scipy.fftpack import fft, ifft


## General Parameters
Frames = 10  # Number of Frames of length N input symbols                                                                                                                                           # Number of OFDM frames(symbols)
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
## Generation of binary data
#Binary = np.array([0, 1])  # Binary data that take 0 or 1
np.random.standard_normal(12345)  # seeding the Matlab random number generator
InputBinaryS=[]
for j in range(1, len(EbNo_dB)):
    j
    ## Generation of codewords (symbols) from random bits
    InputBinaryS = [np.random.choice([1, 0]) for _ in range(int(m))for _ in range(int(Frames))]
    symbols = BitsIntoCodewords(InputBinaryS, m, m1, N, M, n, k, g, Frames)
    ## Channel
    impulse_response = np.zeros((NFFT, Frames),dtype=complex)
    if Fade == 1:
        impulse_response[0:v, :] = (1 / np.sqrt(2 * v)) *(randn(v, Frames)+ 1j*randn(v, Frames))  # Rayleigh variance = 1
        fading_coeffs = fft(impulse_response)
    else:
        fading_coeffs = np.ones(NFFT, Frames)

    ## DFT
    TxNFFT = np.zeros((NFFT*Frames),dtype=complex) # frames*NFFT 0
#    print("TxNFFT",TxNFFT)  #good
#    print(len(TxNFFT))  #good
    TxDFT = (1 / np.sqrt(N)) * fft(symbols) #60 complexes
#    print("TxDFT",TxDFT)
#    print(len(TxDFT))
    ## Interleaving
    #     TxDFTInterleaved = Interleaving(TxDFT,g) # interleaving
    TxDFTInterleaved = TxDFT  # no intelreaving
    ## Subcarrier mapping (Access mode)

    # Localized-FDMA
    TxNFFT[0:int(N/2),0:Frames] = np.reshape(TxDFTInterleaved, (-1, Frames))
    #TxNFFT(1:NFFT/N:NFFT,:) = TxDFT # interleaved-FDMA
    ## IFFT
    Tx = (NFFT / np.sqrt(K)) * ifft(TxNFFT)  # (sqrt(NFFT))*
    ## CP and Channel
    TxCP =np.concatenate((Tx[:],Tx[int(-CP):]))
#    print("CP to ADD", Tx[int(-CP):])
#    print(TxCP)
    #TxCP = [Tx[len(Tx) - int(CP) + 1:, ] * Tx]  # cyclic prefix
    TxCPChannel=[]
    for fr in range(int(Frames - 1)):
        TxCPChannel[:, int(fr - 1)] = filter(impulse_response[:, int(fr - 1)], TxCP[:, int(fr - 1)])
    ## GENERATE AND ADD AWGN
