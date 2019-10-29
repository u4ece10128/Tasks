%% ADVANCED TOPICS ON VIDEO PROCESSING 2ND MODULE %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   EXERCISES                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  Exercises 3                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Marco Marcon                                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% modified by Emanuele Plebani for year 2011-2012             %
% e-mail: eplebani@elet.polimi.it                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Noise models
% see part 6 - Image Restoration
%imgae restoration and noise models
clc;
clear all;
close all;

A = im2double(imread('Lena_grayscale.bmp'));
figure; imshow(A); title('Original Lena image');
figure; imhist(A); title('Original Lena histogram');

B = im2double(imread('Circle.jpg')); 
figure; imshow(B); title('Original circle image');
figure; imhist(B); title('Original cirle histogram');

%% Exponential noise generation from an exponential distribution
% pdf: e^(-x)
% cdf: 1-e^(-x)

%noise can be generated using rand
% how to generate a generic random variable (pag 52)
%invert its cdf

r = rand(1,10000);
x = -log(1-r); % lambda = 1
figure; hist(r); title('uniform distribution pdf in input');
figure; hist(x);
figure; h= hist(x); plot(cumsum(h/numel(x)));
title('exponential distribution pdf with unitary lambda');
pause

%% Additive gaussian noise
N_a = 0.1 * randn(size(A));
A_n = A + N_a;
%the noises added above can lead to values greater than 1 or less than zero
%so we clamp the extreme poitns to 0 or 1
% clamp values on [0, 1]:
A_n(A_n > 1) = 1;
A_n(A_n < 0) = 0;

figure; imshow(A_n);title('Lena with gaussian additive noise');
figure; imhist(A_n);
title('Histogram of Lena with additive gaussian noise');

N_b = 0.07 * randn(size(B));
B_n = B + N_b;
B_n(B_n > 1) = 1;
B_n(B_n < 0) = 0;
figure; imshow(B_n); title('Circle image with additive gaussian noise');
figure; imhist(B_n);
title('Circle histogram with additive gaussian noise');

%% Additive Rayleigh noise
% see pdf in slides pag 43: a = 0, sqrt(b/2) = 0.1

N_a = random('rayl', 0.1, size(A,1), size(A,2));
A_n = A + N_a;
A_n(A_n > 1) = 1;
A_n(A_n < 0) = 0;
figure; imshow(A_n); title('Lena image with additive Rayleigh noise');
figure; imhist(A_n);
title('Lena histogram with additive Rayleigh noise');

N_b = random('rayl', 0.07, size(B,1), size(B,2));
B_n = B + N_b;
B_n(B_n > 1) = 1;
B_n(B_n < 0) = 0;
figure; imshow(B_n); title('Circle image with Rayleigh additive noise');
figure; imhist(B_n);
title('Circle histogram with additive Rayleigh noise');

%% Additive Gamma noise
% see pdf in slides pag 44: b = 1.5, 1/a = 0.1
%the imgae histogram tends to increase towards the greater values more than
%the rayliegh 
N_a = random('gam', 1.5, 0.1, size(A,1), size(A,2));  
A_n = A + N_a;
A_n(A_n > 1) = 1;
A_n(A_n < 0) = 0;
figure; imshow(A_n); title('Lena image with additive Gamma noise');
figure; imhist(A_n); title('Lena histogram with additive Gamma noise');

N_b = random('gam', 1.5, 0.07, size(B,1), size(B,2));
B_n = B + N_b;
B_n(B_n > 1) = 1;
B_n(B_n < 0) = 0;
figure; imshow(B_n); title('Circle image with Gamma additive noise');
figure; imhist(B_n); title('Circle histogram with additive Gamma noise');

%% Additive Exponential noise
% see pdf in slides pag 45: 1/a = 0.1

N_a = random('exp', 0.1, size(A,1), size(A,2));
A_n = A + N_a;
A_n(A_n > 1) = 1;
A_n(A_n < 0) = 0;
figure; imshow(A_n); title('Lena image with additive Exponential noise');
figure; imhist(A_n);
title('Lena histogram with additive Exponential noise');

N_b = random('exp', 0.08, size(B,1), size(B,2));
B_n = B + N_b;
B_n(B_n > 1) = 1;
B_n(B_n < 0) = 0;
figure; imshow(B_n); title('Circle image with additive Exponential noise');
figure; imhist(B_n);
title('Circle histogram with additive Exponential noise');

%% Additive uniform noise
% see pdf in slides pag 46: a = -0.25, b = 0.25 

N_a = 0.5 * rand(size(A)) - 0.25;
A_n = A + N_a;
A_n(A_n > 1) = 1;
A_n(A_n < 0) = 0;
figure; imshow(A_n); title('Lena image with additive Uniform noise');
figure; imhist(A_n); title('Lena histogram with additive Uniform noise');

% see pdf in slides: a = -0.125, b = 0.125
N_b = 0.25 * rand(size(B)) - 0.125;
B_n = B + N_b;
B_n(B_n > 1) = 1;
B_n(B_n < 0) = 0;
figure; imshow(B_n); title('Circle image with additive Uniform noise');
figure; imhist(B_n); title('Circle histogram with additive Uniform noise');

%% Salt-and-pepper noise
% density 0.1: proportion of altered pixels (Matlab default: 0.05)

A_n = imnoise(A, 'salt & pepper', 0.1);
A_n(A_n > 1) = 1;
A_n(A_n < 0) = 0;
figure; imshow(A_n); title('Lena image with salt-and-pepper noise');
figure; imhist(A_n); title('Lena histogram with salt-and-pepper noise');

B_n = imnoise(B, 'salt & pepper', 0.1);
B_n(B_n > 1) = 1;
B_n(B_n < 0) = 0;
figure; imshow(B_n); title('Circle image with salt-and-pepper noise');
figure; imhist(B_n); title('Circle histogram with salt-and-pepper noise');

%manual salt and pepper
close all
N = numel(A); %prod(size(A))
p =0.1;
%transforming the random values to binary values, 
%the value is 1 when the value in the matrix is <p
n = rand(1,N)<p;
sum(n)/N;
k = find(n);
R = randi(numel(k),numel(k));
k = k(R);
mid_index = round(numel(k)/2);
A_n = A;
A_n(k(1:mid_index)) = 1;
A_n(k(mid_index+1 : end)) = 0;
figure; imshow(A_n); title('Lena image with salt-and-pepper noise');
figure; imhist(A_n); title('Lena histogram with salt-and-pepper noise');

%% Spatial filtering: average filter vs. median filter

clc;
clear all;
close all;
%how average filter works
C =randn(6);
f_mean = fspecial('average', 3);
C_mean = imfilter(C, f_mean, 'conv');
C_mean(2,2);
v = C(1:3,1:3);mean(v(:));

%median filter
C_median = medfilt2(C,[3,3]);
C_median(2,2);
v = C(1:3,1:3);
median(v(:));

B = im2double(imread('Circle.jpg')); 
figure; imshow(B); title('Original Circle image');
figure; imhist(B); title('Original Circle histogram');
pause

% add salt-and-pepper noise to the image
B_n = imnoise(B, 'salt & pepper', 0.1);
B_n(B_n > 1) = 1;
B_n(B_n < 0) = 0;
figure; imshow(B_n); title('Circle image with salt-and-pepper noise');
figure; imhist(B_n); title('Circle histogram with salt-and-pepper noise');
pause

% filter the noisy image with AVERAGE filter
%as size of the filter increases we obtain blurred image
F_mean_3x3 = fspecial('average', 3);  % average filter 3x3
B_mean_3x3 = imfilter(B_n, F_mean_3x3, 'conv'); 
figure; imshow(B_mean_3x3);
title('Circle image with salt-and-pepper noise after average filter 3x3'); 

F_mean_5x5 = fspecial('average',5);  % average filter 5x5
B_mean_5x5 = imfilter(B_n, F_mean_5x5, 'conv'); 
figure; imshow(B_mean_5x5);
title('Circle image with salt-and-pepper noise after average filter 5x5');
pause

% filter the noisy image with MEDIAN filter 
% Median filter is better to use incase of salt and pepper noise
% becazue , usulayy salt and pepper noise have small density inside , may be
% we have 1 or 2 corrupted pixels by excatly the same number.

B_median_3x3 = medfilt2(B_n,[3 3]);   % median filter 3x3
figure; imshow(B_median_3x3);
figure; imhist(B_median_3x3);
title('Circle image with salt-and-pepper noise after median filter 3x3');

B_median_5x5 = medfilt2(B_n,[5 5]);   % median filter 5x5
figure; imshow(B_median_5x5);
title('Circle image with salt-and-pepper noise after average filter 5x5');
pause


%% Wiener filtering (from theory, step by step)
clc;
clear all;
close all;

f_I = im2double(imread('Lena_grayscale.bmp'));
figure; imshow(f_I); title('Original image');
pause

%we average pixles by neigbourhood pixels placed at an inclination of 11
%degrees and lenth 31

LEN = 31;
THETA = 11;
h_D = fspecial('motion', LEN,THETA); % create PSF
figure; imagesc(h_D); colorbar; axis equal; axis tight; 
title('Degrading filter kernel (31px motion blur, angle 11 degrees)');
colormap gray

f_D = imfilter(f_I, h_D, 'conv', 'full');
% % filtering in frequency domain with 0-padding:
% F_I = fft2(f_I, size(f_I,1)+size(h_D,1)-1, size(f_I,2)+size(h_D,2)-1);
% H_D = fft2(h_D, size(f_I,1)+size(h_D,1)-1, size(f_I,2)+size(h_D,2)-1);
% F_D = F_I .* H_D;
% f_D = ifft2(F_D);
figure; imshow(f_D);
title('Degraded image (31px motion blur, angle 11 degrees)');
pause

% We Use weiner filtering to recover blurred images

% Wiener step-by-step (noiseless case --> inverse filtering)

% % zero padding applied to the images before calling fft2()
% f_D_padded = zeros(size(f_D,1)+size(h_D,1)-1,size(f_D,2)+size(h_D,2)-1);
% h_D_padded = zeros(size(f_D,1)+size(h_D,1)-1,size(f_D,2)+size(h_D,2)-1);
% f_D_padded (1:size(f_D,1), 1:size(f_D,2)) = f_D;
% h_D_padded (1:size(h_D,1), 1:size(h_D,2)) = h_D;
% figure; imshow(f_D_padded); title('zero padded f_D image');
% figure; imagesc(h_D_padded); title('zero padded h_D filter');
% F_D = fft2(f_D_padded);
% H_D = fft2(h_D_padded);

% Zero padding done by Matlab internally in fft2()
% to maintiain same size of both the matrices that participate in
% convolution
% frequency domain of both degraded image and degrading function
F_D = fft2(f_D, size(f_D,1)+size(h_D,1)-1, size(f_D,2)+size(h_D,2)-1);
H_D = fft2(h_D, size(f_D,1)+size(h_D,1)-1, size(f_D,2)+size(h_D,2)-1);

% % Still another way: by knowing that F_D comes from a convolution,
% % a complete zero padding is not necessary
% F_D = fft2(f_D);
% % minimal zero padding
% H_D = fft2(h_D, size(f_I,1)+size(h_D,1)-1, size(f_I,2)+size(h_D,2)-1);

% Restoration filter H_R
H_R = 1./H_D;
F_I_hat = F_D .* H_R;
f_I_hat = ifft2(F_I_hat);
f_I_hat = f_I_hat(1:size(f_I,1),1:size(f_I,2));  
figure;imshow(f_I_hat);
title('Image restored by Wiener filter (h_D known and noiseless case');
pause

% lets see if we have some additive noise
% additive noise
f_n = 0.1*randn(size(f_D));
f_O = f_D + f_n;
f_O(f_O > 1) = 1;
f_O(f_O < 0) = 0;
figure;imshow(f_O);
title('Image degraded by spatial filter and additive noise');
pause

% Wiener step-by-step (additive noise; H_D, noise power spectrum and
% original image known
F_O = fft2(im2double(f_O));
F_I = fft2(f_I, size(f_O,1), size(f_O,2));
F_n = fft2(f_n, size(f_O,1), size(f_O,2));
H_D = fft2(h_D, size(f_O,1), size(f_O,2)); % minimal zero padding
Power_F_I = abs(F_I).^2;
Power_F_n = abs(F_n).^2;
Power_H_D = abs(H_D).^2;
% Weiner's filter formula (pag 13, first equation)
H_R = (conj(H_D).*Power_F_I) ./ ((Power_H_D.*Power_F_I) + Power_F_n);
F_I_hat = F_O .* H_R;
f_I_hat = real(ifft2(F_I_hat));  
% % you can also use the option 'symmetric' in ifft2
% f_I_hat = ifft2(F_I_hat, 'symmetric'); 
f_I_hat = f_I_hat(1:size(f_I,1), 1:size(f_I,2));  
figure; imshow(f_I_hat);
title(['Image restored by Weiner filter (H_D, noise power spectrum ', ...
        'and original image known)']);
pause

% Wiener step-by-step (additive noise, H_D and SNR known)
% noise average power
NP = abs(fft2(f_n)).^2;
NPOW = sum(NP(:))/numel(f_n);
% original image average power
IP = abs(fft2(f_I)).^2;
IPOW = sum(IP(:))/numel(f_I); 
% average noise/signal ratio
NSR = NPOW/IPOW;
H_R = conj(H_D)./(Power_H_D + NSR);
F_I_hat = F_O .* H_R;
f_I_hat = real(ifft2(F_I_hat));  
f_I_hat = f_I_hat(1:size(f_I,1),1:size(f_I,2));  
figure;imshow(f_I_hat);
title('Image restored by Weiner filter (H_D and SNR known)');
pause


%% Crop a portion of an image
clc;
clear all;
close all;
I = imread('peppers.png');%inbuilt image not present in working dir
I = I(10+(1:256), 222+(1:256), :);
figure, imshow(I);
title('Original Image');

%% simulated blurring using a Point Spread Function (PSF)
LEN = 31;
THETA = 11;
PSF = fspecial('motion', LEN, THETA);
figure, imshow( PSF / max(PSF(:)) );
title('PSF');
blurred = imfilter(I, PSF, 'conv', 'circular');
figure, imshow(blurred);
title('Blurred');

%% restoration using Weiner deconvolution
% factor to avoid numerical instability in Matlab:
eps = 0.00001;
wnr1 = deconvwnr(blurred, PSF,eps);
figure, imshow(wnr1);
title('Restored, True PSF');

%% If the exact PSF is not known:
wnr2 = deconvwnr(blurred, fspecial('motion', 2*LEN, THETA), eps);
figure, imshow(wnr2);
title('Restored, "Long" PSF');
%% If the exact blurring angle is not known:
wnr3 = deconvwnr(blurred, fspecial('motion', LEN, 2*THETA), eps);
figure, imshow(wnr3);
title('Restored, Steep');

%% Added noise and blurring
noise = 0.1 * randn(size(I)); % gaussian noise with 0.1 standard deviation
blurredNoisy = imadd(blurred, im2uint8(noise));
figure, imshow(blurredNoisy);
title('Blurred & Noisy');

%% restoration using the known PSF without considering the noise
wnr4 = deconvwnr(blurredNoisy, PSF);
figure, imshow(wnr4);
title('Inverse Filtering of Noisy Data');

NSR = sum(noise(:).^2) / sum(im2double(I(:)).^2);
wnr5 = deconvwnr(blurredNoisy, PSF, NSR);
figure, imshow(wnr5);
title('Restored with NSR');

%% restoration using as estimate of the noise power 1/2 of the real one
wnr6 = deconvwnr(blurredNoisy, PSF, NSR/2);
figure, imshow(wnr6);
title('Restored with NSR/2');

%% improved restoration using the perfect knowledge of the autocorrelation
% autocorrelation function (AFC) is the power spectrum

NP = abs(fftn(noise)).^2;
NPOW = sum(NP(:))/numel(noise); % noise power
NCORR = fftshift(real(ifftn(NP))); % noise ACF, centered
IP = abs(fftn(im2double(I))).^2;
IPOW = sum(IP(:))/numel(I); % original image power
ICORR = fftshift(real(ifftn(IP))); % image ACF, centered

wnr7 = deconvwnr(blurredNoisy,PSF,NCORR,ICORR);
figure, imshow(wnr7);
title('Restored with ACF');

%% Less statistical information: only noise power spectrum and linear (1-D)
%  autocorrelation of the image

ICORR1 = ICORR(:,ceil(size(I,1)/2));
wnr8 = deconvwnr(blurredNoisy,PSF,NPOW,ICORR1);
figure, imshow(wnr8);
title('Restored with NP & 1D-ACF');