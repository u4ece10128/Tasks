% Initialization
clc;
clear all;
close all;

img_gray = imread('trees_gray.bmp');
figure;imshow(img_gray);

img_bw = im2bw(img_gray);
figure;imshow(img_bw);


%finding number of black and white pixels in trees bW image
 img_black_ctr = max(size(find(img_bw == 0)));
 img_white_ctr = max(size(find(img_bw == 1)));
 
 %finding a threshold for gray scale image to match with BW image
 %visulaize histogram of the gray scale image
 img_hist  = zeros(1,255);
 %construction of histogram - gray scale
for i = 1:256
    img_hist(i) = sum(sum(img_gray == i-1));
end

threshold_black = 0;
%the below for loop iterates the number equal to gray scale values and
%loads threshold value when sum of the pixels in the histogram equlas
%desired black
for i = 1:256
    if sum(img_hist(1,1:i)) <= img_black_ctr
        threshold_black = i;
    else
        continue;
    end
end
% Normalize Threshold
threshold_black = threshold_black / 256;

img_noise = imnoise(img_gray,'salt & pepper', threshold_black);
figure; imshow(img_noise);

SE = strel('arbitrary',[0,1,0;1,1,1;0,1,0]);
% Opening should eliminate small details ("salt")
%  
Im_open = imopen(img_noise,SE);
figure, imshow(Im_open);
Im_open_close = imclose(Im_open,SE);
figure, imshow(Im_open_close);

