%% ADVANCED TOPICS ON VIDEO PROCESSING 2ND MODULE %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   EXERCISES                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  Exercises 5                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Emanuele Plebani                                            %
%                                                             %
% Research associate                                          %
% Politecnico di Milano                                       %
% DEI - Dipartimento di Elettronica e Informazione            %
% ISPG Lab V.le Rimembranze di Lambrate 14,                   %
% 20133 Milano (ITALY)                                        %
% Tel: +39 02 2399 9654                                       %
% e-mail: eplebani@elet.polimi.it                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Harris keypoint detection
clc;
clear all;
close all;

% im = double(imread('bull.bmp'));

im = imread('Harris_giraffe.png');
im = rgb2gray(im);

imshow(im), title('Original image');

sigma = 2;
thresh = 500;
radius = 3;

% Prewitt operator for gradient estimation
dx = [-1 0 1; -1 0 1; -1 0 1]; 	% horizontal gradient
dy = dx';                       % vertical gradient
    
% derivatives
Ix = conv2(im, double(dx), 'same');
Iy = conv2(im, double(dy), 'same');

% Gaussian filter of size 6*sigma (+/- 3sigma) and minimum size 1
% 'fix' rounds torward zero
g = fspecial('gaussian', max(1, fix(6*sigma)), sigma);

% Smoothed squared image derivatives
Ix2 = conv2(Ix.^2, g, 'same');
Iy2 = conv2(Iy.^2, g, 'same'); 
Ixy = conv2(Ix.*Iy, g, 'same');

% Harris measure.
k = 0.04;
%(det(M) - k*(trace(M)).^2)
cim = (Ix2.*Iy2 - Ixy.^2) - k*(Ix2 + Iy2).^2; 

% % Noble  measure.
% cim = (Ix2.*Iy2 - Ixy.^2)./(Ix2 + Iy2 + eps);

% show corner strenght
cim_max = max(cim(:));
cim_min = min(cim(:));
cim_norm = (cim - cim_min) / (cim_max - cim_min);

imshow(cim_norm), colormap('Jet'), title('Corner strenght');

% show thresholded strenght (doesn't work well)
figure
imshow(cim > 500);
title('Thresholded image, w/out nonmaximal suppression');
    
% Nonmaximal suppression and threshold
	
% Extract local maxima by performing a grey scale morphological
% dilation and then finding points in the corner strength image that
% match the dilated image and are also greater than the threshold.

size = 2 * radius + 1;                   % Size of the mask.
mx = ordfilt2(cim, size^2, ones(size));  % Grey-scale dilate.
cim = (cim == mx) & (cim > thresh);      % Find maxima.

figure
imshow(cim);

% Find row,col coords.
[r,c] = find(cim);

figure, imshow(uint8(im));
hold on;  plot(c, r, 'r+'), axis equal, title('Corners');

%% Hough transform - isolated points
clc;
clear all;
close all;

im = zeros(600, 600);
im(30, 30) = 100;
im(30, 570) = 100;
im(570, 30) = 100;
im(570, 570) = 100;
im(300, 300) = 100;

imshow(im), title('Original image');

% Hough transform
[H, T, R] = hough(im);

figure
imshow(H, 'XData', T, 'YData', R, 'InitialMagnification','fit');
title('Hough Transform of isolated points');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;

%% Hough transform - noise
clc;
clear all;
close all;

im = zeros(600, 600);
for i = 1:1000
    im( ceil(rand(1)*600), ceil(rand(1)*600) ) = 1;
end
imshow(im), title('Noise image');

% Hough transform
[H, T, R] = hough(im);

figure
imagesc(T, R, H);
title('Hough Transform of edge image');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, colormap('hot');

%% Hough transform - straight lines
clc;
clear all;
close all;

im = zeros(600,600);
im(100:480, 200:400) = 1;
im = edge(im, 'canny');
figure, imshow(im), title('Rectangle');

% Hough transform
[H, T, R] = hough(im);

figure
imagesc(T, R, H);
title('Hough Transform of edge image');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, colormap('hot');
pause

figure
imagesc(T, R, log(H));
title('Log Hough Transform of edge image');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, colormap('hot');

% Radon transform, continuos, different coordinate center
theta = 0:0.5:180;
[R, xp] = radon(im, theta);

figure
imagesc(theta, xp, R);
title('R_{\theta} (X\prime)');
xlabel('\theta (degrees)')
ylabel('X\prime')
set(gca,'XTick',0:20:180);
colormap(hot), colorbar


%% Hough transform - photo
clc;
clear all;
close all;

im = imread('SEM.jpg');
% im = imread('circuit.tif');
imshow(im), title('Original image');

edges = edge(im, 'prewitt');
imshow(edges), title('Edge image');

% Hough transform
[H, T, R] = hough(edges);

figure
imagesc(T, R, H);
title('Hough Transform of edge image');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, colormap('hot');


%% Radon transform

clc;
clear all;
close all;

I = im2double(imread('barcode.png'));
figure;
imshow(I), title('Original image');

% Radon transform
theta = 0:0.5:180;
[R, xp] = radon(I, theta);

figure
imagesc(theta, xp, R);
title('R_{\theta} (X\prime)');
xlabel('\theta (degrees)')
ylabel('X\prime')
set(gca,'XTick',0:20:180);
colormap(hot), colorbar

% Radon transform of the same image, rotated
I_rot = imrotate(I, 15, 'crop');
figure;
imshow(I_rot), title('Rotated image');

[R, xp] = radon(I_rot, theta);

figure
imagesc(theta, xp, R);
title('R_{\theta} (X\prime)');
xlabel('\theta (degrees)')
ylabel('X\prime')
set(gca,'XTick',0:20:180);
colormap(hot), colorbar

% inverse transform
I_rad = iradon(R, theta);
figure, imshow(I_rad), title('Recovered image');