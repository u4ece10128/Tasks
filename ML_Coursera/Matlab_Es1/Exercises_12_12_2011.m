%% ADVANCED TOPICS ON VIDEO PROCESSING 2ND MODULE %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   EXERCISES                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  Exercises 1                                %
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


%% Reading an image, visualization and saving

% Initialization
clc;
clear all;
close all;

img_rgb = imread('trees.jpg');
figure;
imshow(img_rgb);
pause

% alternative method
figure;
imshow('trees.jpg');

% getimage always return pixel data as double
img_double = getimage;
% Saving an image
imwrite(img_rgb,'trees.bmp', 'bmp');
pause


%% Different representations of images in Matlab
clc;
clear all;
close all;

% Truecolor
img_rgb = imread('trees.jpg');
figure; imshow(img_rgb); title('Truecolor Image');
pause

% ... read color planes
img_rgb_red_plane = img_rgb(:,:,1);
img_rgb_green_plane = img_rgb(:,:,2);
img_rgb_blue_plane = img_rgb(:,:,3);
% ... visualize color planes:
null_image = zeros(size(img_rgb,1), size(img_rgb,2));
img_rgb_red = cat(3, img_rgb(:,:,1), null_image, null_image );
img_rgb_green = cat(3, null_image, img_rgb(:,:,1), null_image );
img_rgb_blue = cat(3, null_image, null_image, img_rgb(:,:,1) );
figure; imshow(img_rgb_red); title('Truecolor Image - RED channel');
figure; imshow(img_rgb_green); title('Truecolor Image - GREEN channel');
figure; imshow(img_rgb_blue); title('Truecolor Image - BLUE channel');
pause

% Indexed (from truecolor, 16 colors map, no dithering)
%quantized form of the image in this case we use 16 levels to reperesent
%the levels
[img_rgb_ind map_rgb_ind] = rgb2ind(img_rgb, 16, 'nodither');
figure; imshow(img_rgb_ind, map_rgb_ind); colorbar; 
title('Indexed 16-color image (false countours and flat colors)');
pause

% Grayscale (from truecolor, as indexed with an inplicit map in gray tones)
img_gray = rgb2gray(img_rgb);
figure; imshow(img_gray); title('Grayscale image');
pause

%Binary (from grayscale with threshold at 40% of maximum luminance)
img_bin = im2bw(img_gray, 0.4);
figure; imshow(img_bin); title('Binary image (threshold 0.4)');
pause


%% Image conversions
clc;
clear all;
close all;

img_rgb = imread('trees.jpg');
figure; imshow(img_rgb); title('Image in Truecolor');
pause

% truecolor --> indexed
[img_ind map_ind] = rgb2ind(img_rgb, 16, 'nodither');
figure; imshow(img_ind, map_ind); 
title('Truecolor --> Indexed conversion (16 colors)');
pause

% truecolor --> grayscale
img_gray = rgb2gray(img_rgb);
figure; imshow(img_gray); title('Truecolor --> Grayscale conversion');
pause

% indexed --> truecolor
[img_ind_rgb] = ind2rgb(img_ind, map_ind);
figure; imshow(img_ind_rgb);
title('Indexed (16 colors) --> Truecolor conversion');
pause

% grayscale --> binary
img_bin = im2bw(img_gray, 0.4);
figure; imshow(img_bin);
title('Grayscale --> Binary conversion (threshold at 0.4)');
pause


%% Coversion to different data representations
clc;
clear all;
close all;
img_gray = imread('Lena_grayscale.bmp');
figure; imshow(img_gray);colorbar; title('uint8 grayscale image [0,255]');
pause

% grayscale uint8 --> grayscale uint16
img_gray_uint16 = im2uint16(img_gray);
figure; imshow(img_gray_uint16); colorbar;
title('uint8 --> uint16 [0,65535] grayscale conversion');
pause

% grayscale uint8 --> grayscale double
% gray values of pixel ranges from 0 to 1, kind of normalizing the pixel
% values
img_gray_double = im2double(img_gray);
figure; imshow(img_gray_double); colorbar;
title('uint8 --> double [0,1] grayscale conversion');
pause

% im2uint8, im2uint16, im2int16, im2single,
% or im2double


%% Image aritmetic (overflow/underflow, rounding)
clc;
clear all;
close all;
%image arithmeti possible only when the images to be operated are of same
%size
img_gray_lena = imread('Lena_grayscale.bmp');
figure; imshow(img_gray_lena); title('Lena Grayscale uint8 [0,255]');
pause

img_gray_trees = imread('trees_gray.bmp');
lena_size = size(img_gray_lena);
img_gray_trees = imresize(img_gray_trees, lena_size(1:2));
figure; imshow(img_gray_trees);
title('Trees uint8 [0,255] grayscale, resized as the previous image');
pause

% addition
img_gray_lena_add = img_gray_lena + img_gray_trees;
figure; imshow(img_gray_lena_add); title('Lena + Trees');
pause

% addition w/ rounding
img_gray_lena_add_05 = img_gray_lena + 0.5;
figure; imshow(img_gray_lena_add_05); title('Lena with rounding');
pause

% subtraction
img_gray_lena_sub_trees = img_gray_lena - img_gray_trees;
figure; imshow(img_gray_lena_sub_trees); title('Lena - Trees');
pause

% multiplication
img_gray_lena_mult_trees = img_gray_lena.*img_gray_trees;
figure; imshow(img_gray_lena_mult_trees); title('Lena x Trees');
pause

%division
img_gray_lena_div_trees = img_gray_lena./img_gray_trees;
figure; imshow(img_gray_lena_div_trees); title('Lena / Trees');
pause


%% Lena dithering example
%addition of noise,number of colors from binary representation to gray
%scale representation
clc;
clear all;
close all;

img_gray_lena = imread('Lena_grayscale.bmp');
figure; imshow(img_gray_lena); title('Lena Grayscale uint8 [0,255]');
pause

% Dithering and thus Grayscale --> Binary conversion
img_gray_lena_dithered = dither(img_gray_lena);
figure; imshow(img_gray_lena_dithered);
title('Grayscale --> Binary conversion w/ dithering');
pause

% From grayscale to binary w/ dithering
img_bin = im2bw(img_gray_lena, 0.5);
figure; imshow(img_bin);
title('Grayscale --> Binary conversion w/out dithering');
pause


%% Improving perceived color depth using dithering --> DITHERING THEORY
%  (Floyd-Steinberg)
clc;
clear all;
close all;

img_rgb = imread('trees.jpg');
figure; imshow(img_rgb); title('Trees Truecolor');
pause

img_gray = imread('trees_gray.bmp');
figure; imshow(img_gray); title('Trees Grayscale');
pause

% From grayscale image to binary w/ dithering
img_gray_dithered = dither(img_gray);
figure; imshow(img_gray_dithered);
title('Grayscale --> Binary conversion w/ dithering');
pause

% Truecolor (RGB) image conversion to indexed
% using minimum variance quantization w/ dithering
[img_rgb_ind_dithered map_rgb_ind_dithered] = rgb2ind(img_rgb,16);
figure; imshow(img_rgb_ind_dithered);%map_rgb_ind_dithered);
title('Truecolor --> Indexed (16 colors) conversion w/ dithering');
pause

% Truecolor (RGB) image conversion to indexed
% using minimum variance quantization w/out dithering
[img_rgb_ind_dithered map_rgb_ind_dithered] = ...
                                          rgb2ind(img_rgb, 16, 'nodither');
figure; imshow(img_rgb_ind_dithered, map_rgb_ind_dithered);
title('Truecolor --> Indexed (16 colors) conversion w/out dithering');
pause


%% Image resizing --> INTERPOLATION THEORY
clc;
clear all;
close all;

img_gray = imread('trees_gray.bmp');
figure; imshow(img_gray); title('Original image (Grayscale)');
pause

img_gray_reduced = imresize(img_gray, 0.25);
figure; imshow(img_gray_reduced); title('Reduced by 1/4');
pause

img_gray_enlarged = imresize(img_gray_reduced, 4, 'nearest');
figure; imshow(img_gray_enlarged);
title('Reduced (1/4) and enlarged (4) using Nearest-Neighbor interp.');
pause

img_gray_enlarged = imresize(img_gray_reduced, 4, 'bilinear');
figure; imshow(img_gray_enlarged);
title('Reduced (1/4) and enlarged (4) using Bilinear interpolation');
pause

img_gray_enlarged = imresize(img_gray_reduced, 4, 'bicubic');
figure; imshow(img_gray_enlarged);
title('Reduced (1/4) and enlarged (4) using Bicubic interpolation');
pause

img_gray_resized = imresize(img_gray, [200 500]);
figure; imshow(img_gray_resized);
title('Resized to specific size using Nearest-Neighbor (default)');
pause

img_gray_resized = imresize(img_gray, [200 500], 'bicubic' );
figure; imshow(img_gray_resized);
title('resized to specific size using bicubic interpolation');
pause


%% Image rotation
clc;
clear all;
close all;

img_gray = imread('trees_gray.bmp');
figure; imshow(img_gray); title('Original image (Grayscale)');
pause

img_gray_rotated = imrotate(img_gray, 35, 'crop'); 
figure; imshow(img_gray_rotated);
title('35� rotation using Nearest-Neighbor interpolation (cropped)');
pause

img_gray_rotated = imrotate(img_gray, 35);
figure; imshow(img_gray_rotated);
title('35� rotation using Nearest-Neighbor interpolation');
pause

img_gray_rotated = imrotate(img_gray, 35, 'bilinear'); 
figure; imshow(img_gray_rotated);
title('35� rotation using Bilinear interpolation');
pause

img_gray_rotated = imrotate(img_gray, 35, 'bicubic'); 
figure; imshow(img_gray_rotated);
title('35� rotation using Bicubic interpolation');
pause
