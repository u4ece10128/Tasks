%% ADVANCED TOPICS ON VIDEO PROCESSING 2ND MODULE %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   EXERCISES                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  Exercises 4                                %
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


%% Spatial filtering (Laplacian) to emphasize/extract contours
clc;
clear all;
close all;

A = imread('trees_gray.bmp'); % eg. Laplacian gives more thin edges
% A = imread('Lena_grayscale.bmp'); % eg. Laplacian emphasize noise
figure
imshow(A); title('Original image');
figure
surf(double(A)); colormap gray; shading interp;
title('Original image in 3D (luminance on z axis)');

% Laplacian operator
% sum of h is zero , central pixel the importance
h = [1 1 1;
    1 -8 1;
    1 1 1];

% In A_h negative values are clipped to 0... only positive allowed!
%since we deal with positive values we see the imgae to be dark close to
%zero pixels
A_h = imfilter(A, h, 'symmetric', 'conv', 'same');
figure
imshow(A_h); title('Isotropic Laplacian (along 45 deg) - only values > 0');
figure
imshow(A-A_h);
title('Image with emphasized contours using only positive values');

% Laplacian operator that incorporate the difference with the image
% now both positive and negative values are considered 
% (double contour problem)
% sum is 1 
h = [-1 -1 -1;
    -1 9 -1;
    -1 -1 -1];
A_h = imfilter(A, h, 'symmetric', 'conv', 'same');
figure
imshow(A_h); title('Immagine with emphasized contours');

f_gaussian = fspecial('gaussian',5,1.5);%when used on high standard deviation gaussian equlas average



%% Contour extraction by spatial filtering (Gradient)
clc;
clear all;
close all;

A = imread('trees_gray.bmp');
% A = imread('Lena_grayscale.bmp');
figure
imshow(A); title('Original image');
figure
surf(double(A)); colormap gray; shading interp;
title('Original image in 3D (luminance on z axis)');

B = zeros(100,100); %black image 100 x 100
for i = 1: size(B)
    B(i,i:end) = 1;
end
B(1:20,:) = 0;
B(90:100,:) = 0;
figure()
imshow(B)

%% Roberts
h_x = [0 0 -1;
       0 1 0;
       0 0 0];
   
h_y = [-1 0 0;
       0 1 0;
       0 0 0];
A_h_x = imfilter(A, h_x, 'symmetric', 'conv', 'same');
A_h_y = imfilter(A, h_y, 'symmetric', 'conv', 'same');
A_h = abs(A_h_x) + abs(A_h_y);
A_h_thr = im2bw(A_h, 0.3);

figure
imshow(A_h_x); title('Roberts h_x');% higlihts variation in X direction 
figure
imshow(A_h_y); title('Roberts h_y');% higlihts variation in Y direction 
% figure
% imshow(max(max(A_h_x))-A_h_x); title('Roberts h_x');
% figure
% imshow(max(max(A_h_y))-A_h_y); title('Roberts h_y');
figure
imshow(A_h); title('Roberts');
figure
imshow(A_h_thr); title('Roberts thr 0.3');

figure()
plot(A_h(10,:));%10nt row of original image

A_h_x = imfilter(B, h_x, 'symmetric', 'conv', 'same');
A_h_y = imfilter(B, h_y, 'symmetric', 'conv', 'same');
figure
imshow(A_h_x); title('Roberts h_x');% higlihts variation in X direction 
figure
imshow(A_h_y); title('Roberts h_y');% higlihts variation in Y direction

%% Prewitt
close all
%East ,West , North, South
h_E = [-1 0 1;
       -1 0 1;
       -1 0 1];
   
h_NE = [0 1 1;
       -1 0 1;
       -1 -1 0];
   
h_N = [1 1 1;
       0 0 0;
       -1 -1 -1];
   
h_NW = [1 1 0;
        1 0 -1;
       0 -1 -1];
   
h_W = [1 0 -1;
       1 0 -1;
       1 0 -1];
   
h_SW = [0 -1 -1;
        1 0 -1;
        1 1 0];
    
h_S = [-1 -1 -1;
        0 0 0;
        1 1 1];
    
h_SE = [-1 -1 0;
       -1 0 1;
        0 1 1];
A_h_E = imfilter(A, h_E, 'symmetric', 'conv', 'same');
A_h_NE = imfilter(A, h_NE, 'symmetric', 'conv', 'same');
A_h_N = imfilter(A, h_N, 'symmetric', 'conv', 'same');
A_h_NW = imfilter(A, h_NW, 'symmetric', 'conv', 'same');
A_h_W = imfilter(A, h_W, 'symmetric', 'conv', 'same');
A_h_SW = imfilter(A, h_SW, 'symmetric', 'conv', 'same');
A_h_S = imfilter(A, h_S, 'symmetric', 'conv', 'same');
A_h_SE = imfilter(A, h_SE, 'symmetric', 'conv', 'same');
A_h = max(A_h_E,A_h_NE);
A_h = max(A_h,A_h_N);
A_h = max(A_h,A_h_NW);
A_h = max(A_h,A_h_W);
A_h = max(A_h,A_h_SW);
A_h = max(A_h,A_h_S);
A_h = max(A_h,A_h_SE);
A_h_thr = im2bw(A_h, 0.3);

figure
imshow(A_h_E); title('Prewitt E');
figure
imshow(A_h_NE); title('Prewitt NE');
figure
imshow(A_h_N); title('Prewitt N');
figure
imshow(A_h_NW); title('Prewitt NW');
figure
imshow(A_h_W); title('Prewitt W');
figure
imshow(A_h_SW); title('Prewitt SW');
figure
imshow(A_h_S); title('Prewitt S');
figure
imshow(A_h_SE); title('Prewitt SE');
figure
pause

imshow(A_h); title('Prewitt');
figure
imshow(A_h_thr); title('Prewitt thr 0.3');


%% Sobel
close all

h_r = [1 0 -1;
       2 0 -2;
       1 0 -1];
   
h_c = [-1 -2 -1;
       0 0 0;
       1 2 1];
   
A_h_r = imfilter(A, h_r, 'symmetric', 'conv', 'same');
A_h_c = imfilter(A, h_c, 'symmetric', 'conv', 'same');
A_h = sqrt(double(A_h_r).^2 + double(A_h_c).^2);
%rescale
A_h_resc = (A_h - min(min(A_h))) / (max(max(A_h)) - min(min(A_h))) * 255;
A_h_resc_thr = im2bw(uint8(A_h_resc), 0.3);
figure
imshow(A_h_r); title('Sobel row');
figure
imshow(A_h_c); title('Sobel column');
figure
imshow(uint8(A_h_resc)); title('Sobel');
figure
imshow(A_h_resc_thr); title('Sobel thr 0.3');

%% Frei-Chen
close all
h_r = [1 0 -1;
       sqrt(2) 0 -sqrt(2);
       1 0 -1];
   
h_c = [-1 -sqrt(2) -1;
       0 0 0;
       1 sqrt(2) 1];
 
A_h_r = imfilter(A, h_r, 'symmetric', 'conv', 'same');
A_h_c = imfilter(A, h_c, 'symmetric', 'conv', 'same');

A_h = sqrt(double(A_h_r).^2 + double(A_h_c).^2);
A_h_resc = (A_h - min(min(A_h))) / (max(max(A_h)) - min(min(A_h))) * 255;
A_h_resc_thr = im2bw(uint8(A_h_resc), 0.3);
figure
imshow(A_h_r); title('Frei-Chen row');
figure
imshow(A_h_c); title('Frei-Chen column');
figure
imshow(uint8(A_h_resc)); title('Frei-Chen');
figure
imshow(A_h_resc_thr); title('Frei-Chen thr 0.3');



%% Contour extraction by spatial filtering (Gradient) - Matlab filters

clc;
clear all;
close all;

A = imread('trees_gray.bmp');
figure
imshow(A);

% Default filters in Matlab...

h = fspecial('prewitt');
% h = fspecial('sobel');

% using symmetric the image is extended with mirrored values
% to avoid contour detection on the border
A_h_H = imfilter(A, h, 'symmetric', 'conv', 'same');
figure
imshow(A_h_H); title('Filtered image using horizontal Prewitt kernel');
A_h_H_thr = im2bw(A_h_H, 0.3);
figure
imshow(A_h_H_thr);
title('Filtered image using horizontal Prewitt kernel with thr 0.3');
pause

A_h_V = imfilter(A, h', 'symmetric', 'conv', 'same');
figure
imshow(A_h_V); title('Filtered image using vertical Prewitt kernel');
A_h_V_thr = im2bw(A_h_V, 0.3);
figure
imshow(A_h_V_thr);
title('Filtered image using vertical Prewitt kernel with thr 0.3');
pause

% Prewitt method
A_h_H_V = max(A_h_H, A_h_V);
figure
imshow(A_h_H);
title('Filtered image using Prewitt method (only V and H kernels)');
A_h_H_V_thr = im2bw(A_h_H_V, 0.3);
figure
imshow(A_h_H_V_thr);
title('Filtered image using Prewitt method with thr 0.3 (V and H)');
pause