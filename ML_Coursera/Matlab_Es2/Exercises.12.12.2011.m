%% ADVANCED TOPICS ON VIDEO PROCESSING 2ND MODULE %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   EXERCISES                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  Exercises 2                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Massimiliano Pierobon                                       %
%                                                             %
% Ph.D. student                                               %
% Politecnico di Milano                                       %
% DEI - Dipartimento di Elettronica e Informazione            %
% ISPG Lab V.le Rimembranze di Lambrate 14,                   %
% 20133 Milano (ITALY)                                        %
% Tel: +39 02 2399 9640                                       %
% Fax: +39 02 2399 9611                                       %
% e-mail: pierobon@elet.polimi.it                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Modified by Daniela Donno for year 2008/09                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% modified by Emanuele Plebani for year 2011-2012             %
% e-mail: eplebani@elet.polimi.it                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Crop an image
clc;
clear all;
close all;

img_rgb = imread('trees.jpg');
imshow(img_rgb);
title('Original image');

% Select the area by dragging the mouse, press the dx button and select
% 'Crop Image' select sub region of an image
img_rgb_cropped = imcrop(img_rgb);
figure;
imshow(img_rgb_cropped);
title('Cropped image');
pause

img_rgb_cropped_spec = imcrop(img_rgb, [100 150 367 178]);
figure;
imshow(img_rgb_cropped_spec);
title('Croppen image with region [100 150 367 178]');
pause

% alternative method: selecting the working image
figure(1);
% Select the area by dragging the mouse, press the dx button and select
% 'Crop Image'
img_truecolor_cropped = imcrop;
figure;
imshow(img_truecolor_cropped);
title('Cropped image');
pause


%% Image histogram
clc;
clear all;
close all;

img_gray = imread('trees_gray.bmp');
figure; imshow(img_gray); title('Grayscale uint8 [0,255] image');
pause
%histogram-show many pixel in th image with gray level k , k = 0....255
figure; imhist(img_gray);
title('Image histogram');
pause
%computing hist with custom fiunction
%img_gray == i-1 -- checks if value of i-1 exist in the image , if yes the
%corresponding pixel value is set to 1 else to zero
%1st sum sums up the resuktabt 0 1 matrix cloumn wise , 2nd sum receives a
%row vector and computes its sum results in number of pixels with value
%i-1
%sum of a matrix gives a row vector with all the coloumn summations
hist  = zeros(1,255);
for i = 1:256
    hist(i) = sum(sum(img_gray == i-1));
end

%% Histogram shifting
clc;
clear all;
close all;

img_gray_compr = imread('TreesGrayCompress.tif');
figure; imshow(img_gray_compr);
title('Grayscale uint8 [0,255] image w/ compressed histogram');
figure; imhist(img_gray_compr);
title('Compressed histogram');
pause

% brighten the image (increase the luminance by 60)
%shifts histogram by 60.5 gray levels on x axis
img_gray_compr_brighter = img_gray_compr + 60.5;
figure; imshow(img_gray_compr_brighter);
title('Immagine histogram translated by 60');
figure; imhist(img_gray_compr_brighter);
title('Histogram translated by 60');
pause

% NB: doing the same with the non-compressed histogram...
img_gray = imread('trees_gray.bmp');
figure; imshow(img_gray); title('Grayscale uint8 [0,255] image');
figure; imhist(img_gray); title('Image histogram');
pause

% translate the histogram...
img_gray_brighter = img_gray + 40.5;
figure
% conversion to clip overflowed values
imshow(uint8(img_gray_brighter));
figure
imhist(uint8(img_gray_brighter));
pause


%% Compression/dilation of an image histogram
clc;
clear all;
close all;

img_gray_double = double(imread('TreesGrayCompress.tif'));
% re-converted to uint8 for visualization
figure; imshow(uint8(img_gray_double));
title('Grayscale uint8 [0,255] image w/ compressed histogram'); 
figure; imhist(uint8(img_gray_double));
title('Compressed histogram');

max = max(img_gray_double(:));
min = min(min(img_gray_double));
% Rescale the bins
img_gray_double = ((img_gray_double - min) / (max - min)) * 255;
figure; imshow(uint8(img_gray_double));
title('Image w/ histogram dilated to the whole range');
figure; imhist(uint8(img_gray_double));
title('Histogram linearly dilated to the whole range');

%CDF
h = imhist(img_gray_mod);
%%normalize histogram
h_norm = h / prod(size(img_gray_mod));
h_norm_cum = cumsum(h_norm);
figure();
plot(h_norm_cum);
title('CDF');

%% Histogram equalization --> see THEORY
clc;
clear all;
close all;

img_gray_compress = imread('TreesGrayCompress.tif');
figure; imshow(img_gray_compress);
title('Grayscale uint8 [0,255] image w/ compressed histogram');
h = imhist(img_gray_compress);
figure; imhist(img_gray_compress);
title('Compressed histogram');
pause

% Step-by-step equalization
% try to mantain a uniform ripartition of bins (lower -> nearer)
%normalize the image and compute CDFs
h_norm = h / numel(img_gray_compress);
h_norm_cum = cumsum(h_norm);
% NB: gray level are in [0, 255] but Matlab indices in [1, 256]
img_gray_double_equal = h_norm_cum(img_gray_compress + 1);
%rescaling images to 255 levels values ranges from 0 to 255
img_gray_equal = im2uint8(img_gray_double_equal); 
h_equal = imhist(img_gray_equal);
figure; imshow(img_gray_equal);
title('Image w/ equalized histogram');
figure; imhist(img_gray_equal); title('Equalized histogram');
pause

% Equalization using the Matlab function histeq()
img_gray_equal_matlab_256 = histeq(img_gray_compress, 256);
figure; imshow(img_gray_equal_matlab_256);
title('Equalized image using histeq()');
figure; imhist(img_gray_equal_matlab_256);
title('Equalized histogram using histeq()');
pause

% Equalization using the Matlab function histeq()
% reducing the bins from 256 to 64
% the histogram now is more uniform, even if we lose color resolution
img_gray_equal_matlab_64 = histeq(img_gray_compress, 64); 
figure; imshow(img_gray_equal_matlab_64);
title('Equalized image using histeq(), 64 bins');
figure; imhist(img_gray_equal_matlab_64);
title('Equalized histogram using histeq(), 64 bins');
pause


%% Fourier transform of a rectangle
clc;
clear all;
close all;
%zeroes is black 1 is white
f = zeros(600,600);
f(100:480,260:340) = 1;
figure; imshow(f); title('Rectangle');

F = fft2(f);

F_magnitude = abs(F);
F_phase = angle(F);
F_power_spectrum = F_magnitude.^2;
figure
imagesc(F_magnitude); colorbar; title('Spectrum modulus');
figure
imagesc(F_phase); colorbar; title('Spectrum phase');
figure
imagesc(F_power_spectrum); colorbar; title('Power spectrum');
figure
imagesc(0.5*log(1+F_magnitude)); colorbar; title('Spectrum log modulus');
figure
imagesc(0.5*log(1+F_power_spectrum)); colorbar;
title('Log power spectrum');
figure
surf(0.5*log(1+F_magnitude(1:100, 1:100))); colorbar;
title('Spectrum log modulus (3D plot)');


%% Fourier transform of a rectangle (shifted to the center)
clc;
clear all;
close all;

f = zeros(600,600);
f(280:320,260:340) = 1;
figure; imshow(f); title('Rectangle');
pause

% Shifted spectrum w/ dc at the image center 
[i, j] = meshgrid( (1:size(f,2)), (1:size(f,1)) );
f_shift = f.*((-1).^(i + j));

F_shift = fft2(f_shift);

% Alternative: use the fftshift() Matlab function
% F = fft2(f);
% F_shift = fftshift(F);

F_magnitude_shift = abs(F_shift);
F_phase_shift = angle(F_shift);
F_power_spectrum_shift = F_magnitude_shift.^2;
figure
imagesc(F_magnitude_shift); colorbar; title('Spectrum modulus');
figure
imagesc(F_phase_shift); colorbar; title('Spectrum phase');
pause

figure
imagesc(F_power_spectrum_shift); colorbar; title('Power spectrum');
figure
imagesc(0.5*log(1+F_magnitude_shift)); colorbar;
title('Spectrum log modulus');
figure
imagesc(0.5*log(1+F_power_spectrum_shift)); colorbar;
title('Log power spectrum');
pause

figure
surf(F_magnitude_shift(250:350, 250:350)); colorbar;
title('Spectrum modulus (3D plot)');
figure
surf(F_phase_shift(250:350, 250:350)); colorbar;
title('Spectrum phase (3D plot)');
figure
surf(0.5*log(1+F_magnitude_shift(250:350, 250:350))); colorbar;
title('Log power spectrum (3D plot)');
pause

clc
clear all
close all

s = 300;
test = zeros(s,s);
f1 =0.05;
for i = 1:s
    test(:,i) = sin(2*pi*f1*i);
end

F = fft2(test);

F_magnitude = abs(F);
F_phase = angle(F);
F_power_spectrum = F_magnitude.^2;
figure
imagesc(F_magnitude); colorbar; title('Spectrum modulus');
figure
imagesc(F_phase); colorbar; title('Spectrum phase');
figure
imagesc(F_power_spectrum); colorbar; title('Power spectrum');


%% Fourier transform (Lena, dc shifted to the center)
clc;
clear all;
close all;

img_gray_lena = im2double(imread('Lena_grayscale.bmp'));
figure; imshow(img_gray_lena); title('Grayscale image');
pause

gray_lena_DFT = fftshift(fft2(img_gray_lena));

gray_lena_DFT_magnitude = abs(gray_lena_DFT);
gray_lena_DFT_phase = angle(gray_lena_DFT);
gray_lena_DFT_power_spectrum = gray_lena_DFT_magnitude.^2;
figure
imagesc(0.5*log(1+gray_lena_DFT_magnitude));
colorbar; axis equal; axis tight;
title('LENA - Spectrum log modulus');
figure
imagesc(gray_lena_DFT_phase);
colorbar; axis equal; axis tight; title('LENA - Spectrum phase');
figure
imagesc(0.5*log(1+gray_lena_DFT_power_spectrum));
colorbar; axis equal; axis tight; title('LENA - Log power spectrum');
pause

figure
surf(0.5*log(1+gray_lena_DFT_magnitude(160:260, 100:200))); colorbar;
title('LENA - Spectrum log modulus (3D plot)');
figure
surf(gray_lena_DFT_phase(160:260, 100:200)); colorbar;
title('LENA - Spectrum phase (3D plot)');
figure
surf(0.5*log(1+gray_lena_DFT_power_spectrum(160:260, 100:200)));
colorbar; title('LENA - Log power spectrum (3D plot)');
pause


%% Fourier transform (Trees, dc shifted to the center)

img_gray_trees = im2double(imread('trees_gray.bmp'));
figure; imshow(img_gray_trees); title('Grayscale image');
pause

gray_trees_DFT = fftshift(fft2(img_gray_trees));

gray_trees_DFT_magnitude = abs(gray_trees_DFT);
gray_trees_DFT_phase = angle(gray_trees_DFT);
gray_trees_DFT_power_spectrum = gray_trees_DFT_magnitude.^2;
figure
imagesc(0.5*log(1+gray_trees_DFT_magnitude));
colorbar; axis equal; axis tight; title('TREES - Spectrum log modulus');
figure
imagesc(gray_trees_DFT_phase); colorbar;
axis equal; axis tight; title('TREES - Specturm phase');
figure
imagesc(0.5*log(1+gray_trees_DFT_power_spectrum));
colorbar; axis equal; axis tight; title('TREES - Log power spectrum');
pause

figure
surf(0.5*log(1+gray_trees_DFT_magnitude(79:179, 125:225)));
colorbar; title('TREES - Spectrum log modulus (3D plot)');
figure
surf(gray_trees_DFT_phase(79:179, 125:225));
colorbar; title('TREES - Spectrum phase (3D plot)');
figure
surf(0.5*log(1+gray_trees_DFT_power_spectrum(79:179, 125:225)));
colorbar; title('TREES - Log power spectrum (3D plot)');
pause

%power spectrum of lena has bright light angling at 30 degrees approx ,
%whic says there are edges in the original image with similar inclination,
%power spectrum gives the information of the original image


%% Inverse Fourier transform (Lena and Trees)

% recovering the images

% using ifftshift()...
gray_trees_DFT = ifftshift(gray_trees_DFT);
img_gray_trees_reconstructed = ifft2(gray_trees_DFT, 'symmetric');
figure
imshow(img_gray_trees_reconstructed); title('Reconstructed Trees image');
pause

% or using manual shifting
gray_lena_DFT_rec = ...
    complex(gray_lena_DFT_magnitude.*cos(gray_lena_DFT_phase), ...
            gray_lena_DFT_magnitude.*sin(gray_lena_DFT_phase));
img_gray_lena_reconstructed = ifft2(gray_lena_DFT_rec, 'symmetric');
[i, j] = meshgrid( (1:size(img_gray_lena_reconstructed,2)), ...
                   (1:size(img_gray_lena_reconstructed,1)) );
img_gray_lena_reconstructed = img_gray_lena_reconstructed.*((-1).^(i+j));
figure
imshow(img_gray_lena_reconstructed); title('Reconstructed Lena image');
pause


%% Phase swap between images
clc;
clear all;
close all;

img_gray_lena = im2double(imread('Lena_grayscale.bmp'));
figure; imshow(img_gray_lena); title('Lena Grayscale');

gray_lena_DFT = fft2(img_gray_lena);
gray_lena_DFT_magnitude = abs(gray_lena_DFT);
gray_lena_DFT_phase = angle(gray_lena_DFT);

img_gray_trees = im2double(imread('trees_gray.bmp'));
img_gray_trees = imresize(img_gray_trees, size(img_gray_lena));
figure; imshow(img_gray_trees);
title('Trees Grayscale resized to Lena dimensions');
pause

gray_trees_DFT = fft2(img_gray_trees);
gray_trees_DFT_magnitude = abs(gray_trees_DFT);
gray_trees_DFT_phase = angle(gray_trees_DFT);

% swap the phases and reconstruct the images
gray_trees_DFT_rec = ...
    complex(gray_trees_DFT_magnitude.*cos(gray_lena_DFT_phase), ...
            gray_trees_DFT_magnitude.*sin(gray_lena_DFT_phase));
img_gray_trees_reconstructed = ifft2(gray_trees_DFT_rec, 'symmetric');
figure
imshow(img_gray_trees_reconstructed);
title('Recontructed image: Trees modulus, Lena phase');
pause

gray_lena_DFT_rec = ...
    complex(gray_lena_DFT_magnitude.*cos(gray_trees_DFT_phase), ...
            gray_lena_DFT_magnitude.*sin(gray_trees_DFT_phase));
img_gray_lena_reconstructed = ifft2(gray_lena_DFT_rec, 'symmetric');
figure
imshow(im2double(img_gray_lena_reconstructed));
title('Recontructed image: Lena modulus, Trees phase');
pause


%% Fourier transform interpretation (using SEM image)
clc;
clear all;
close all;

img_SEM = imread('SEM.jpg');
figure; imshow(img_SEM);
pause

SEM_DFT = fft2(im2double(img_SEM));
SEM_DFT = fftshift(SEM_DFT);

SEM_DFT_power_spectrum = real(SEM_DFT).^2 + imag(SEM_DFT).^2;
figure
imagesc(0.6*log(1+SEM_DFT_power_spectrum));
colorbar; axis equal; axis tight; colormap gray; caxis([0,10])
title('SEM - Log power spectrum');
pause