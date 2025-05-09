clear;
clc;

% YAMNet modelini yükle
yamnet_model = audioPretrainedNetwork('yamnet');

% Örnek ses oku
[waveform, fs] = audioread('example.wav');

% Eğer farklı örnekleme oranıysa 16 kHz'e dönüştür
if fs ~= 16000
    waveform = resample(waveform, 16000, fs);
end

% YAMNet ile sınıf tahmini yap
[probs, classNames, scores] = classify(yamnet, waveform, 16000);

% Sonuçları göster
[topProb, idx] = max(probs);
fprintf('Tahmin: %s (%.2f%% güven)\n', classNames(idx), topProb * 100);
