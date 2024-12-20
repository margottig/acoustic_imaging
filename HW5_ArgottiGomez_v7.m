close all
clear;
clc

load("HW5_DataSrc.mat"); % Load data from .mat file
StRec_noise = P.SignalReceiverWithNoise;       % Noisy Signal at the receiver
StRec_clear = P.SignalReceiverWithoutNoise;   % Clean Signal at the receiver
[nt, nRec] = size(StRec_clear);               % Get number of samples and sensors
idx_sensors=1:nRec;

% Time vector and Signal windowing Γ and segmentation into L “snapshots”
t = P.t;                % Time vector
dt = t(2) - t(1);       % Time step
Tsnap = P.TimeSnapShot; % Size of Time snapshot
ntsnap = floor(Tsnap / dt);         % Size of snapshot
nsnap = floor(t(end) / Tsnap) - 4;  % Number of snapshots (25 snapshots)
tshot = t(1:ntsnap);                % Time vector valid for all snapshots

% Frequency vector
fs = 1 / dt;             % Sample frequency
df = fs / ntsnap;        % Frequency resolution
% Frequency vector for fftshifted data
fshot = (-ntsnap / 2 : ntsnap / 2 - 1) * (fs / ntsnap); % Frequency vector
% Frequency bound for computing coherent beamforming output and CSDM
fmin =  P.fmin; %500000           % Minimum frequency
fmax = P.fmax;    %2000000;        % Maximum frequency
c0 = P.c0;                % Sound speed
k = 2 * pi * fshot / c0;  % Wavenumber vector

% Frequency indices and range selection     
[~, fmin_index] = min(abs(fshot - fmin));
[~, fmax_index] = min(abs(fshot - fmax));
freq_indices = fmin_index:fmax_index;
freq_range = fshot(freq_indices);
Nf = length(freq_indices);   % Nf frequencies

% ------------------------------------------------------------
%% ----- BEAMFORMER PARAMETERS
% ------------------------------------------------------------
% Receiver array coordinates
XRec = P.XRec;
YRec = P.YRec;
ZRec = P.ZRec;
Zsrc = P.Zsrc;
% Bounds of the search region/steering locations / Define step sizes and mesh
XsBound = P.XsBound;
YsBound = P.YsBound;
% Step size for meshing
dx = 0.2e-3;
dy = 0.2e-3;
% Array output in frequency domain
rps = [XRec', YRec', ZRec'];  % Receiver positions (nRec x 3)
rangeXs = XsBound(1):dx:XsBound(2);
rangeYs = YsBound(1):dy:YsBound(2);
% Create grid of steering locations
[Xgrid, Ygrid] = ndgrid(rangeXs, rangeYs); % Grid of points
Ngrid = numel(Xgrid);
Xgrid_vec = Xgrid(:); % Ngrid x 1
Ygrid_vec = Ygrid(:); % Ngrid x 1
Zgrid_vec = ones(Ngrid, 1) * Zsrc; % Ngrid x 1
% Steering locations (Ngrid x 3)
rm = [Xgrid_vec, Ygrid_vec, Zgrid_vec];
% Compute distances between steering locations and receivers
D = pdist2(rm, rps); % Ngrid x nRec


% ---------------------------------------------------
%% SNAPSHOT CONSTRUCTION
% ----------------------------------------------------
xtRec_clear = StRec_clear; % nt x nRec Signal
xtRec_noise= StRec_noise; %StRec;% nt x nRec  Signal
% Initialization of a matrix containing time series for each receiver and snapshots (3D matrix)
StShot_clear = zeros(ntsnap, nRec, nsnap);
StShot_noise = zeros(ntsnap, nRec, nsnap); 
% Initialize K_shot_wn_noise
K_shot_clear = zeros(nRec, nRec, ntsnap); % nRec x nRec x ntsnap
K_shot_noise = zeros(nRec, nRec, ntsnap);
% Initialize K_ matrix all snapshots
K_matrix_clear = zeros(nRec, nRec, ntsnap,nsnap);
K_matrix_noise = zeros(nRec, nRec, ntsnap,nsnap);

CSDM_clear = zeros(nRec, nRec, length(fshot)); % Initialize CSDM
CSDM_clear_normalized = zeros(nRec, nRec, length(fshot)); % Initialize CSDM

%------------------------------
%% 2) Plot the received signal with respect to time and index of sensor (use imagesc function from matlab)
%--------------------------------------------
figure;
subplot(2,1,1)
imagesc(t,idx_sensors,StRec_noise')
col = colorbar;
ylabel('# Rec');
xlabel('Time (s)');
title(['Noisy signal plotted in time for each receiver']);
set(gca, 'FontSize', 12, 'fontname', 'Arial', 'LineWidth', 1);
colormap(parula);
set(gcf, 'color', [1 1 1]);
subplot(2,1,2)
imagesc(t,idx_sensors,StRec_clear')
col = colorbar;
ylabel('# Rec');
xlabel('Time (s)');
title(['Clean signal plotted in time for each receiver']);
set(gca, 'FontSize', 12, 'fontname', 'Arial', 'LineWidth', 1);
colormap(parula);
set(gcf, 'color', [1 1 1]);


%---------------------------------------------------------------
%% 3) Estimate the cross-spectral density matrix (CSDM) following the procedure described in lecture 5 for frequencies between fmin and fmax (see parameters in .mat file)
% -------------------------------------------------------
% Loop over each snapshot to pick up portions of the signal
for nn = 1:nsnap
    idx_start = (nn - 1) * ntsnap + 1;
    idx_end = nn * ntsnap;
    StShot_clear(:, :, nn) = xtRec_clear(idx_start:idx_end, :); % ntsnap x nRec x nsnap
    StShot_noise(:, :, nn) = xtRec_noise(idx_start:idx_end, :); % ntsnap x nRec x nsnap
    % Transform to frequency domain
    SfShot_clear = fftshift(fft(StShot_clear(:, :, nn), ntsnap, 1), 1); % ntsnap x nRec
    SfShot_noise = fftshift(fft(StShot_noise(:, :, nn), ntsnap, 1), 1); % ntsnap x nRec
    
    % For each frequency
    for ii = 1:Nf %ntsnap
        idx = freq_indices(ii); 
        % data matrix construction
        d_clear = SfShot_clear(idx, :).'; % Data at current frequency (nRec x 1)
        d_noise = SfShot_noise(idx, :).'; % Data at current frequency (nRec x 1)
        d_n_clear = d_clear ./ norm(d_clear);             % Normalization
        d_n_noise = d_noise ./ norm(d_noise);             % Normalization
        
        % Cross-spectral density matrix construction (nRec x nRec)
        K_clear = d_n_clear * d_n_clear';   % Normalized for beamformer output
        K_noise = d_n_noise * d_n_noise';   % Normalized for beamformer output
       
        K_shot_clear(:, :, idx) = K_clear; %K_shot_clear(:, :, idx) + 
        K_shot_noise(:, :, idx) = K_noise;  % K_shot_noise(:, :, idx)
    end
    K_matrix_clear(:,:,:,nn) = K_shot_clear;
    K_matrix_noise(:,:,:,nn) = K_shot_noise;
end

% Average over snapshots
K_shot_clear_avg = mean(K_matrix_clear,4); %K_shot_clear / nsnap; % nRec x nRec x ntsnap
K_shot_noise_avg = mean(K_matrix_noise,4); %K_shot_noise / nsnap; % nRec x nRec x ntsnap


%-----------------------------------------
%% 3.2) Compute the Fourier transform of each time serie and the corresponding frequency vector. Plot on a same figure an example of a time serie for a single snapshot and receiver and the corresponding frequency spectrum (Make sure to label axes and title the plots) . Based on the observation of your plots, what can you say about the signal? Is it a broadband or narrowband signal?
% --------------------------------------------------------------
%% Plotting  one snapshot 
%----------------------------------------------------
%Choose receiver 10 and the first snapshot to be plotted in time domain and
%frequency domain. Make sure to properly define the new frequency vector
% (here, it is called fshot).
nn=1;
rr=10;
figure;
subplot(211);
plot(tshot, squeeze(StShot_clear(:,rr,nn)),tshot, squeeze(StShot_noise(:,rr,nn))); % function squeeze allows to "squeeze the 3D matrix into a 1d vector here.
xlabel('Time(s)');
ylabel('Amplitude');
legend('Noise free signal','Noisy signal');
subplot(212);
plot(fshot./1000, squeeze(abs(SfShot_clear(:,rr,nn))),fshot./1000, squeeze(abs(SfShot_noise(:,rr,nn))));
xlabel('Frequency(KHz)');
ylabel('Magnitude');
legend('Noise free signal','Noisy signal');
sgtitle('Time serie and spectrum for snapshot #1 and receiver #10');
xlim([0 5000]);


%-----------------------------------------
%% 3.3) Compute the CSDM for each snapshot and each frequency and then the expected value of the CSDM when average over all snapshots . 
% Plot (imagesc) the CSDM for a single snapshot and then the average CSDM at frequency f ≈ 900kHz. 
% Compare both CSDM (for a single snapshot and the average one) and explain/comment the differences.
% ----------------------------------------------
frequency_interest = 236; 
CSDM_avg_clear = abs(K_shot_clear_avg(:,:,frequency_interest));
CSDM_avg_noise = abs(K_shot_noise_avg(:,:,frequency_interest));
CSDM_one_snapshot_clear = abs(K_matrix_clear(:,:,frequency_interest,7)); %seventh snapshot
CSDM_one_snapshot_noise = abs(K_matrix_noise(:,:,frequency_interest,7)); %seventh snapshot

%% 3.3 Averaged CSDM vs a CSDM for a single snapshot
figure; % Create a new figure window
% First subplot
subplot(2, 2,1); % 1 row, 2 columns, first plot
imagesc(idx_sensors, idx_sensors, CSDM_avg_clear);
colorbar; %caxis([0 0.009])
xlabel('# Rec');
ylabel('# Rec');
ylabel(colorbar,'dB', 'rot', 0);
title(['Average CSDM |K| at ', num2str(fshot(frequency_interest), '%.2f'), ' Hz - No noise']);

% Second subplot
subplot(2, 2, 2); % 1 row, 2 columns, second plot
imagesc(idx_sensors, idx_sensors, CSDM_one_snapshot_clear);
colorbar; %caxis([-20 0])
ylabel('# Rec');
xlabel('# Rec');
ylabel(colorbar,'dB', 'rot', 0);
title(['CSDM |K| for 7th snapshot at ', num2str(fshot(frequency_interest), '%.2f'), ' Hz - No noise']);

% Third subplot
subplot(2, 2,3); % 1 row, 2 columns, first plot
imagesc(idx_sensors, idx_sensors, CSDM_avg_noise);
colorbar; %caxis([0 0.009])
xlabel('# Rec');
ylabel('# Rec');
ylabel(colorbar,'dB', 'rot', 0);
title(['Average CSDM |K| at ', num2str(fshot(frequency_interest), '%.2f'), ' Hz - Noisy']);

% Fourth subplot
subplot(2, 2,4); % 1 row, 2 columns, first plot
imagesc(idx_sensors, idx_sensors, CSDM_one_snapshot_noise);
colorbar; %caxis([0 0.009])
xlabel('# Rec');
ylabel('# Rec');
ylabel(colorbar,'dB', 'rot', 0);
title(['CSDM |K| for 7th snapshot at ', num2str(fshot(frequency_interest), '%.2f'), ' Hz - Noisy']);

%% 4 BEAMFORMER
%----------------------------------------
%% 4.1) Compute the output of Bartlett beamformer for each steered location and all frequencies between fmin and fmax when using the estimated CSDM average over all snapshots. Then plot the incoherent Bartlett output average over all frequencies in [fmin;fmax] for each steered location. This 2D plot is also called the ambiguity surface
% -----------------------------------------
% Epsilon for MVDR and Diagonal loading factor
ndiag = P.noise_diagonal;
epsilon=10^(-ndiag/10);
CSDM_avg_loaded_clear = K_shot_clear_avg + epsilon * eye(nRec); % Regularization
CSDM_avg_loaded_noise = K_shot_noise_avg + epsilon * eye(nRec); % Regularization

% ------------------------------------------------------------
%% BEAMFORMER and MDVR COMPUTATION  
% ------------------------------------------------------------
% Preallocate Beamformer grid
B_output_clear = zeros(Ngrid, Nf);
B_output_noise = zeros(Ngrid, Nf);
B_output_matrix_clear = zeros(Ngrid, Nf, nsnap);
B_output_matrix_noise = zeros(Ngrid, Nf, nsnap);
% Initialize MVDR matrix all snapshots
MVDR_matrix_clear = zeros(Ngrid, Nf, nsnap);
MVDR_matrix_noise = zeros(Ngrid, Nf, nsnap);

% Compute B and MVDR in frequency domain for each snapshot
for nn = 1:nsnap
    for idx = 1:Nf
        ii = freq_indices(idx);
        w = exp(1i * k(ii) * D);  % Ngrid x nRec
        % Normalize the steering vectors for each grid point
        w_n = w ./sqrt(sum(abs(w).^2, 2)); % Ngrid x nRec - norm(w)
        
        %% MVDR
        % Inverse of the covariance matrix at frequency ii
        MVDR_clear = CSDM_avg_loaded_clear(:, :, ii); % nRec x nRec
        MVDR_noise = CSDM_avg_loaded_noise(:, :, ii); % nRec x nRec
        % Compute the inverse of MCDR
        MVDR_clear_inv = inv(MVDR_clear); % nRec x nRec
        MVDR_noise_inv = inv(MVDR_noise); % nRec x nRec
        % Compute the numerator
        numerator_clear = w_n * MVDR_clear_inv; % Ngrid x nRec
        numerator_noise = w_n * MVDR_noise_inv; % Ngrid x nRec
        % Corrected denominator computation
        denom_clear = sum(numerator_clear .* conj(w_n), 2); % Ngrid x 1
        denom_noise = sum(numerator_noise .* conj(w_n), 2); % Ngrid x 1
        % Compute the MVDR beamformer output for all grid points at frequency ii
        B_MVDR_output_grid_clear(:, idx) = 1 ./ abs(denom_clear); % Ngrid x 1
        B_MVDR_output_grid_noise(:, idx) = 1 ./ abs(denom_noise); % Ngrid x 1
         
        %% Beamformer Output
        % Compute the beamformer output for all grid points at frequency ii
        v_clear = w_n * K_matrix_clear(:, :, ii,nn);  % Ngrid x nRec
        v_noise = w_n * K_matrix_noise(:, :, ii,nn);  % Ngrid x nRec
        B_output_clear(:, idx) = sum(conj(w_n) .* v_clear, 2);  % Ngri..d x 1 
        B_output_noise(:, idx) = sum(conj(w_n) .* v_noise, 2);  % Ngri..d x 1 
    end
    B_output_matrix_clear(:,:,nn) = B_output_clear;
    B_output_matrix_noise(:,:,nn) = B_output_noise;
    MVDR_matrix_clear(:,:,nn) =  B_MVDR_output_grid_clear;
    MVDR_matrix_noise(:,:,nn) = B_MVDR_output_grid_noise;

end

%% Beamformer 
B_output_clear_avg = reshape( mean(B_output_matrix_clear,3), length(rangeXs), length(rangeYs), Nf); 
B_output_noise_avg = reshape( mean(B_output_matrix_noise,3), length(rangeXs), length(rangeYs), Nf); 
B_output_clear_onesnap = reshape(B_output_matrix_clear(:,:,7), length(rangeXs), length(rangeYs), Nf);
B_output_noise_onesnap = reshape(B_output_matrix_noise(:,:,7), length(rangeXs), length(rangeYs), Nf);

%B_output_grid_reshaped_clear = reshape(B_output_clear, length(rangeXs), length(rangeYs), Nf); % Reshape B_output_grid into 3D matrix

%% MVDR
% Average over frequencies
B_MVDR_output_avg_clear = mean(B_MVDR_output_grid_clear, 2); % Ngrid x 1
B_MVDR_output_avg_noise = mean(B_MVDR_output_grid_noise, 2); % Ngrid x 1
% Normalize the output
B_MVDR_output_avg_clear = B_MVDR_output_avg_clear / max(abs(B_MVDR_output_avg_clear));
B_MVDR_output_avg_noise = B_MVDR_output_avg_noise / max(abs(B_MVDR_output_avg_noise));
% Reshape B_MVDR_output_avg into 2D grid
B_MVDR_output_grid_reshaped_clear = reshape(B_MVDR_output_avg_clear, length(rangeXs), length(rangeYs));
B_MVDR_output_grid_reshaped_noise = reshape(B_MVDR_output_avg_noise, length(rangeXs), length(rangeYs));



%% 4.1) Ambiguity surface of Bartlett incoherent
B_onesnap_clear = abs(B_output_clear_onesnap);  % Bartlett beamformer output
B_i_onesnap_clear = mean(B_onesnap_clear, 3);  % Bartlett incoherent beamformer
B_onesnap_noise = abs(B_output_noise_onesnap);  % Bartlett beamformer output
B_i_onesnap_noise = mean(B_onesnap_noise, 3);  % Bartlett incoherent beamformer

B_avg_clear = abs(B_output_clear_avg);  % Bartlett beamformer output
B_i_avg_clear = mean(B_avg_clear, 3);  % Bartlett incoherent beamformer
B_avg_noise = abs(B_output_clear_avg);  % Bartlett beamformer output
B_i_avg_noise = mean(B_avg_noise, 3);  % Bartlett incoherent beamformer

%% 4.2) Ambiguity surface of Bartlett incoherent
figure; % Create a new figure window
% First subplot
A1 = 10*log10(B_i_onesnap_clear.'/max(B_i_onesnap_clear(:)));
subplot(2, 2,1); 
imagesc(rangeXs*1e3, rangeYs*1e3, A1);
colorbar; 
xlabel('X-axis (mm)');
ylabel('Y-axis (mm)');
ylabel(colorbar,'dB', 'rot', 0);
title('Bi for 7th snapshot - No Noise');
% 
% % Second subplot
A2 = 10*log10(B_i_avg_clear.'/max(B_i_avg_clear(:)));
subplot(2, 2,2); 
imagesc(rangeXs*1e3, rangeYs*1e3, A2);
colorbar; 
xlabel('X-axis (mm)');
ylabel('Y-axis (mm)');
ylabel(colorbar,'dB', 'rot', 0);
title('Bi averaged over all snapshots - No Noise');
% 
% % Third subplot
A3 = 10*log10(B_i_onesnap_noise.'/max(B_i_onesnap_noise(:)));
subplot(2, 2,3); 
imagesc(rangeXs*1e3, rangeYs*1e3, A3);
colorbar; 
xlabel('X-axis (mm)');
ylabel('Y-axis (mm)');
ylabel(colorbar,'dB', 'rot', 0);
title('Bi for 7th snapshot - Noisy');
% 
% % Fourth subplot
A4 = 10*log10(B_i_avg_noise.'/max(B_i_avg_noise(:)));
subplot(2, 2,4); 
imagesc(rangeXs*1e3, rangeYs*1e3, A4);
colorbar; 
xlabel('X-axis (mm)');
ylabel('Y-axis (mm)');
ylabel(colorbar,'dB', 'rot', 0);
title('Bi averaged over all snapshots - Noisy');



%% 4.3) Bartlett beamformer for each steered location and at a single frequency when using the CSDM for a single snapshot.

% Extract specific frequency slices from your data
freqs = [1, 6, 14]; % Indices of frequencies you're interested in
% Initialize cell arrays for datasets and titles
datasets = {};
titles = {};
% Collect clear datasets and titles
for idx = 1:length(freqs)
    fi = freqs(idx);
    freqs_interest = (fshot(freq_indices(fi)));
    data_clear = B_onesnap_clear(:, :, fi);
    datasets{end+1} = data_clear;
    titles{end+1} = sprintf('Clear Data at, %.2f Hz', freqs_interest);
end

% Collect noisy datasets and titles
for idx = 1:length(freqs)
    fi = freqs(idx);
    freqs_interest = (fshot(freq_indices(fi)));
    data_noise = B_onesnap_noise(:, :, fi);
    datasets{end+1} = data_noise;
    titles{end+1} = sprintf('Noisy Data at %.2f Hz', freqs_interest);
end
figure;
% Determine the number of subplots
num_subplots = length(datasets);
% Calculate the subplot grid size (e.g., for 6 datasets, use 2 rows x 3 columns)
num_rows = 2;
num_cols = ceil(num_subplots / num_rows);
% Loop over datasets to create subplots
for i = 1:num_subplots
    % Extract and normalize the data
    data = datasets{i};
    A = 10 * log10(data / max(data(:)));
    % Create subplot
    subplot(num_rows, num_cols, i);
    imagesc(rangeXs * 1e3, rangeYs * 1e3, A.');
    colorbar;
    xlabel('X-axis (mm)');
    ylabel('Y-axis (mm)');
    ylabel(colorbar, 'dB', 'rot', 0);
    title(titles{i});
end
% Adjust the layout
sgtitle('Bartlett Beamforming Outputs for Selected Frequencies'); % Overall title


%% 4.4) What would happen if applying the beamforming process with more frequencies, i.e. with a larger frequency bandwidth [fmin; fmax]?
% Just change the fmin and fmax parameters and see the results ;) 
% As expected not good results, more data dispersion

%% 4.5) Compare the results of the incoherent Bartlett beamformer (average result over all frequencies between fmin and fmax) 
% when working with the noise free random signal and the noisy signal. What difference do you see? Can you explain them?
% SEE FIGURE 4


% ---------------------------------------------------
%% MVDR BEAMFORMER
%---------------------------------------------------------

% ------------------------------------------------------------
%% PLOTTING THE MVDR BEAMFORMER OUTPUT
% ------------------------------------------------------------

figure;
subplot(1, 2,1); 
imagesc(rangeXs*1000, rangeYs*1000, 10 * log10(B_MVDR_output_grid_reshaped_clear).');
colorbar;
xlabel('X (mm)');
ylabel('Y (mm)');
title('MVDR Beamformer Output (Averaged over Frequencies) - Clear Signal');
set(gca, 'YDir', 'reverse');
hold on;

subplot(1, 2,2); 
imagesc(rangeXs*1000, rangeYs*1000, 10 * log10(B_MVDR_output_grid_reshaped_noise).');
colorbar;
xlabel('X (mm)');
ylabel('Y (mm)');
title('MVDR Beamformer Output (Averaged over Frequencies) - Noisy');
set(gca, 'YDir', 'reverse');
hold on;




