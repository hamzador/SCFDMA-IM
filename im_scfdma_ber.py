from numpy import sqrt
from numpy.random import rand, randn
import numpy as np
from binom import binom
from PossibleCodewords import PossibleCodewords
from CodewordsIntoBits import CodewordsIntoBits
from BitsIntoCodewords import BitsIntoCodewords

# % General Parameters
Frames = 100  # Number of Frames of length N input symbols                                                                                                                                       ;    % Number of OFDM frames(symbols)
M = 4  # M-QAM constellation
NFFT = 72
N = 12  # number of total subcarriers
CP = NFFT / 8  # length CP
n = 4  # number of subcarriers per group g
g = N / n  # number of groups
k = 1  # number of active subcarriers per group
# g     = 18;
# n     = N/g;
# alphaa  = M/(M+1);
# k = round(alphaa*n);
K = k * g  # total number of active carriers
m1 = np.floor(np.log2(np.nchoosek(n, k))) * g  # number of bits for the carriers indices
m2 = k * np.log2(M) * g  # number of bits for a constellation M-ary
m = m1 + m2  # total number of bits for an OFDM block
norms = np.mcat([1, sqrt(2), 0, sqrt(10), 0, sqrt(42)])  # normalization of the constellation

# % Channel parameters
Fade = 1  # = 1 if channel effect is considered, 0 otherwise
v = 1  # Number of paths for Rayleigh channel

# % BER parameters
EbNo_L = 0
EbNo_U = 25
EbNo_dB = np.linspace(EbNo_L, EbNo_U, 10)
EbNo_lin = 10. ** (EbNo_dB / 10)
ber = np.zeros(np.size(EbNo_lin))
SNR_BER_db = EbNo_dB + 10 * np.log10(m / (N + CP))
SNR_BER = 10. ** (SNR_BER_db / np.eldiv / 10)

# % Possible Codeword
Codewords = PossibleCodewords(M, n, k)
C = np.length(Codewords)
# % Generation of binary data
Binary = np.mcat([0, 1])  # Binary data that take 0 or 1
randn(np.mstring('state'), 12345)  # seeding the Matlab random number generator

for j in range[len(EbNo_dB)]:
    j()
    # % Generation of codewords (symbols) from random bits
    InputBinaryS = Binary(np.randsrc(1, Frames * m, np.mslice[1:2]))
    symbols = BitsIntoCodewords(InputBinaryS, m, m1, N, M, n, k, g, Frames)
    # % Channel
    impulse_response = np.zeros(NFFT, Frames)
    if Fade == 1:
        impulse_response(np.mslice[1:v], np.mslice[:]).lvalue = (1 / sqrt(2 * v)) * complex(randn(v, Frames), randn(v,
                                                                                                                    Frames))  # Rayleigh variance = 1
        fading_coeffs = np.fft(impulse_response)
    else:
        fading_coeffs = np.ones(NFFT, Frames)

    # % DFT
    TxNFFT = np.zeros(NFFT, Frames)
    TxDFT = (1 / np.sqrt(N)) * np.fft(symbols)
    # % Interleaving
    #     TxDFTInterleaved = Interleaving(TxDFT,g); % interleaving
    TxDFTInterleaved = TxDFT  # no intelreaving
    # % Subcarrier mapping (Access mode)
    TxNFFT[1:N, 1:Frames] = TxDFTInterleaved  # Localized-FDMA
    #     TxNFFT(1:NFFT/N:NFFT,:) = TxDFT; % interleaved-FDMA
    # % IFFT
    Tx = (NFFT / sqrt(K)) * np.ifft(TxNFFT)  # (sqrt(NFFT))*
    # % CP and Channel
    TxCP = np.mcat([Tx[np.mslice[- CP + 1:-1], np.mslice[:]], np.OMPCSEMI, Tx])  # cyclic prefix
    for fr in np.mslice[1:Frames]:
        TxCPChannel[:, fr] = filter[impulse_response[:, fr], 1, TxCP(np.mslice[:], fr)]
    # end
    # % GENERATE AND ADD AWGN
    P_signal = np.mean(np.mean(abs(TxCPChannel) ** np.elpow ** 2))
    P_noise = P_signal * 10 ** (-SNR_BER_db(j) / 10)
    noise_norm = sqrt(0.5) * complex(randn(NFFT + CP, Frames), randn(NFFT + CP, Frames))
    En = sqrt(P_noise) * noise_norm
    TxCPChannelNoise = TxCPChannel + En
    # % Remove CP
    TxChannelNoise = TxCPChannelNoise(np.mslice[CP + 1:end], np.mslice[:])
    # % Channel Equalization
    mmse = np.conj(fading_coeffs) / np.eldiv / ((abs(fading_coeffs) ** np.elpow ** 2) + (P_noise / P_signal))
    Rx = (sqrt(K) / NFFT) * np.fft(TxChannelNoise) * np.elmul * mmse  # ./fading_coeffs; % sqrt(NFFT))*
    # % Subcarrier demapping
    RxIDFT = Rx(np.mslice[1:N], np.mslice[1:Frames])  # Localized FDMA
    #     RxIDFT = Rx(1:NFFT/N:NFFT,:);  % Interleaved FDMA
    # % Deinterleaving
    #     RxIDFTDeinterleaved = DeInterleaving(RxIDFT,g); % interleaving
    RxIDFTDeinterleaved = RxIDFT  # no interleaving
    # % IDFT
    symbols_est = (sqrt(N)) * np.ifft(RxIDFTDeinterleaved)
    # % CodeWord Detection
    Data_out = np.zeros(g, Frames)  # zero recovered data matrix
    for fr in np.mslice[1:Frames]:
        for sub_block in np.mslice[1:g]:
            # %%%%%%%%%%%%%%%%%%%% ML detection routine %%%%%%%%%%%%%%%%%%%
            Detect_output = np.zeros(C, 1)  # zeroise Detect_output array
            # cycle through codeword set
            for count in np.mslice[1:C]:
                code_sent = Codewords(count, np.mslice[:]).T
                F = symbols_est(np.mslice[(sub_block - 1) * n + 1:sub_block * n], fr)
                H = fading_coeffs(np.mslice[(sub_block - 1) * n + 1:sub_block * n], fr)
                # Detect_output(count,1) =  sum(abs(F - code_sent.*H).^2); %if no equalization (only for OFDM)
                Detect_output(count, 1).lvalue = sum(abs(F - code_sent) ** np.elpow ** 2)
            # end
            # Determine Data_sent from minimum
            Minimum = min(Detect_output)
        # end
    # end
    # % Demapping
    Data_out_reshape = np.reshape(Data_out, np.mcat([]), 1)
    symbols_estimated = np.reshape(Codewords(Data_out_reshape, np.mslice[:]).T, np.mcat([]), Frames)
    OutputBinaryp = CodewordsIntoBits(symbols_estimated, M, n, k, g, Frames)
    OutputBinary = np.reshape(OutputBinaryp, 1, np.mcat([]))
    # % BER Calculation
    Nbre_error = np.length(np.find(np.reshape(InputBinaryS, 1, np.mcat([])) - OutputBinary != 0))  # Number of errors
    ber(j).lvalue = Nbre_error / (Frames * m)  # number of errors per number of total bit
# end SNR

# %% BER Plot
np.semilogy(EbNo_dB, ber, np.mstring('d-b'), np.mstring('linewidth'), 2)
np.hold(np.mstring('on'))
np.grid(np.mstring('on'))
np.hold(np.mstring('on'))
np.xlim(np.mcat([0, 25]))
np.ylim(np.mcat([1e-6, 1]))
np.xlabel(np.mstring('Eb/N0 in dB'))
np.ylabel(np.mstring('BER'))
