% Orthogonal Matching Pursuit - based channel estimation for OFDM
clear all
close all
clc
preamble_len = 64; % length of the training sequence
chan_len = 30; % number of channel taps
dominant_taps = 5; % dominant taps (number of nonzero taps)
fade_var_1D = 0.5; % 1D fade variance
SNR_dB = 40; % SNR per bit (dB)
FFT_len = 1024; %length of the FFT/IFFT
num_bit = 2*FFT_len; % number of data bits
cp_len = chan_len-1; % length of the cyclic prefix
num_frames = 10^3; % number of frames
% SNR parameters
SNR = 10^(0.1*SNR_dB);
noise_var_1D = 0.5*2*(2*fade_var_1D*chan_len)/(2*FFT_len*SNR);

% ------------- preamble generation ---------------------------------
preamble_data = randi([0 1],1,2*preamble_len); % source
preamble_qpsk = 1-2*preamble_data(1:2:end)+1i*(1-2*preamble_data(2:2:end));
preamble_qpsk = sqrt(preamble_len/FFT_len)*preamble_qpsk;
preamble_qpsk_ifft = ifft(preamble_qpsk);

% -------------- sensing matrix --------------------------------------
sensing_matrix = zeros(preamble_len-chan_len+1,chan_len); %(26) in paper
for i1 = 1:preamble_len-chan_len+1
sensing_matrix(i1,:) = preamble_qpsk_ifft(chan_len+i1-1:-1:i1);  
end
tic()
C_BER = 0; % bit errors in each frame
for i1=1:num_frames
% --------------transmitter-----------------------------------------------
%Source
data = randi([0 1],1,num_bit); % data
% QPSK mapping 
mod_sig = 1-2*data(1:2:end) + 1i*(1-2*data(2:2:end));
% IFFT operation
T_qpsk_sig = ifft(mod_sig); % T stands for time domain
% inserting cyclic prefix and preamble
T_trans_sig = [preamble_qpsk_ifft T_qpsk_sig(end-cp_len+1:end) T_qpsk_sig]; 

%---------------- channel--------------------------------------------------
%--------------------------------------------------------------------------
%                            CHANNEL   
% Rayleigh channel
fade_chan_active_taps = sqrt(fade_var_1D)*randn(1,dominant_taps) + 1i*sqrt(fade_var_1D)*randn(1,dominant_taps);     
fade_chan = zeros(1,chan_len);
fade_chan(1:dominant_taps) = fade_chan_active_taps;
intr_map = randperm(chan_len); % random permutation
fade_chan = fade_chan(intr_map);

% AWGN
white_noise = sqrt(noise_var_1D)*randn(1,FFT_len + cp_len + preamble_len + chan_len - 1) ...
    + 1i*sqrt(noise_var_1D)*randn(1,FFT_len + cp_len + preamble_len + chan_len - 1); 

% Channel output
chan_op = conv(T_trans_sig,fade_chan) + white_noise; % chan_op stands for channel output

%------------------------ RECEIVER ----------------------------------------
% Channel estimation using OMP
measurement_vec = chan_op(chan_len:preamble_len).'; % column vector
index = [];
[dummy,index(1)] = max(sensing_matrix'*measurement_vec);
A = sensing_matrix(:,index);
x = A\measurement_vec; %least squares solution
residue = [];
residue(:,1) = measurement_vec - A*x;
for i1= 2:dominant_taps
[dummy,index(i1)] = max(sensing_matrix'*residue(:,i1-1));
A = [A sensing_matrix(:,index(i1))];
x = A\measurement_vec;
residue(:,i1) = measurement_vec - A*x;
end
len = length(x);
x = [x;zeros(chan_len-len,1)];
est_fade_chan = zeros(chan_len,1);
for i1 = 1:length(index)
    est_fade_chan(index(i1)) = x(i1);
end
est_fade_chan = est_fade_chan.'; % now a row vector
% -------- data detection ----------------------
est_freq_response = fft(est_fade_chan,FFT_len);
% discarding preamble
chan_op(1:preamble_len) = [];
% discarding cyclic prefix and transient samples
chan_op(1:cp_len) = [];
T_REC_SIG_NO_CP = chan_op(1:FFT_len);
% PERFORMING THE FFT
F_REC_SIG_NO_CP = fft(T_REC_SIG_NO_CP);
% ML DETECTION
QPSK_SYM = [1+1i 1-1i -1+1i -1-1i];
QPSK_SYM1 = QPSK_SYM(1)*ones(1,FFT_len);
QPSK_SYM2 = QPSK_SYM(2)*ones(1,FFT_len);
QPSK_SYM3 = QPSK_SYM(3)*ones(1,FFT_len);
QPSK_SYM4 = QPSK_SYM(4)*ones(1,FFT_len);
DIST = zeros(4,FFT_len);
DIST(1,:)=(abs(F_REC_SIG_NO_CP - est_freq_response.*QPSK_SYM1)).^2; 
DIST(2,:)=(abs(F_REC_SIG_NO_CP - est_freq_response.*QPSK_SYM2)).^2;
DIST(3,:)=(abs(F_REC_SIG_NO_CP - est_freq_response.*QPSK_SYM3)).^2;
DIST(4,:)=(abs(F_REC_SIG_NO_CP - est_freq_response.*QPSK_SYM4)).^2; 
% COMPARING EUCLIDEAN DISTANCE
[~,INDICES] = min(DIST,[],1);
% MAPPING INDICES TO QPSK SYMBOLS
DEC_QPSK_MAP_SYM = QPSK_SYM(INDICES);
% DEMAPPING QPSK SYMBOLS TO BITS
dec_data = zeros(1,num_bit);
dec_data(1:2:end) = real(DEC_QPSK_MAP_SYM)<0;
dec_data(2:2:end) = imag(DEC_QPSK_MAP_SYM)<0;
% CALCULATING BIT ERRORS IN EACH FRAME
C_BER = C_BER + nnz(data-dec_data);
end
toc()
% bit error rate
BER = C_BER/(num_bit*num_frames)