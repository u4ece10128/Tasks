clc;
close all;
clear all;
%checkboardImage
check_img = im2uint8(checkerboard(100,4,4));

%imshow(check_img);

%params
sigma = 1.5;
thresh = 0.01;
radius = 0.04;

% Prewitt operator for gradient estimation
dx = [-1 0 1; -1 0 1; -1 0 1]; 	% horizontal gradient
dy = dx';                       % vertical gradient

% derivatives
Ix = conv2(check_img, double(dx), 'same');
Iy = conv2(check_img, double(dy), 'same');

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

[r,c] = find(cim);

figure, imshow(uint8(check_img));
hold on;  plot(c, r, 'r+'), axis equal, title('Corners');