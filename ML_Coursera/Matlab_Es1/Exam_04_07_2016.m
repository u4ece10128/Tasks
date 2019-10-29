%%Exam_04_07_2016
%Find the black and white thresholds 
 clc
 clear all
 close all
 img_rgb = imread('trees.jpg');
 img_bw = im2bw(imread('trees.jpg'));
 img_gray = rgb2gray(img_rgb);
 figure(1);imshow(img_rgb);title('Trees Color Imgae');
 figure(2);imshow(img_bw);title('Trees Black and white Image');
 figure(3);imshow(img_gray);title('Trees Gray Scale Image');
 
 %finding number of black and white pixels in trees bW image
 img_black_ctr = max(size(find(img_bw == 0)));
 img_white_ctr = max(size(find(img_bw == 1)));
 
 %finding a threshold for gray scale image to match with BW image
 %visulaize histogram of the gray scale image
 img_hist  = zeros(1,255);
for i = 1:256
    img_hist(i) = sum(sum(img_gray == i-1));
end
threshold_black = 0;
for i = 1:256
    if sum(img_hist(1,1:i)) <= img_black_ctr
        threshold_black = i;
    else
        continue;
    end
end


%constructing black and white image with gray scale thresholding
img_gray_noise = imnoise(img_gray,'salt & pepper',0.1);
[I,J] = size(img_gray);
img_gray_est = zeros(I,J);
for i = 1 : I
    for j = 1 : J
        if img_gray(i,j) <= threshold_black
            img_gray_est(i,j) = 0;%black gray
        else
            img_gray_est(i,j) = 1;%whitegray
        end
    end
end
figure(4);imshow(img_gray_est);title('Gray scale estimate to BW image');



 
 
 
 
 
 