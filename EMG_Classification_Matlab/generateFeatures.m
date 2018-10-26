function feat = generateFeatures(x,fs)

% Get standard deviation of the signal as feature
stdev = var(x);

% Get total power of the signal
[pxx,f] = periodogram(x,[],[],fs);
tpwr = trapz(f,pxx,1);

% Preprocessing to remove offset and rectify the raw signal
x = abs(detrend(x));

% Find the linear envelope using a low pass filter (8 columns)
lp = LPFilter;
envavg = mean(filter(lp,x));

feat = [stdev, envavg, tpwr];

end