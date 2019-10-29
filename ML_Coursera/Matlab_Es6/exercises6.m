%% ADVANCED TOPICS ON VIDEO PROCESSING 2ND MODULE %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   EXERCISES                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  Exercises 6                                %
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


%% Morphological operators on binary images (erosion/dilation)
clc;
clear all;
close all;

A = zeros(600,600);
A(250:350,250:350) = 1;
A = im2bw(A, 0.5);
figure
imshow(A); title('Original image (100x100 square)');

B = strel('square', 80);

A_dil_B = imdilate(A,B);
figure
imshow(A_dil_B);
title('Dilated image using square 80x80 structuring element');

A_er_B = imerode(A,B);
figure
imshow(A_er_B);
title('Eroded image using square 80x80 structuring element');

B_ = strel('rectangle', [100 10]);
A_er_B_ = imerode(A,B_);
figure
imshow(A_er_B_);
title('Eroded image using rect 100x10 structuring element');

%% Morphological operators on real-world binary images
clc;
clear all;
close all;

A = im2bw(imread('Lena_grayscale.bmp'), 0.6);
% A = im2bw(imread('trees_gray.bmp'), 0.5);
figure
imshow(A); title('Original image');


% B = strel('square', 5);
% B = strel('disk', 3, 0);
% B = strel('line', 9, 45);
% B = strel('line', 9, 135);
% B = strel('line', 100, 45);
% B = strel('line', 100, 135);
B = strel('arbitrary', [1 1 1 1; 1 1 1 0; 1 1 0 0; 1 0 0 0]);


A_dil_B = imdilate(A, B);
figure
imshow(A_dil_B); title('Dilated image using structuring element');


A_er_B = imerode(A, B);
figure
imshow(A_er_B); title('Eroded image using structuring element');


A_er_dil_B = imdilate(A_er_B, B);
figure
imshow(A_er_dil_B);
title('Eroded and dilated image using structuring element');

%% Erosion, then dilation to eliminate details under a given size
clc;
clear all;
close all;

A = zeros(600,600);
% rand() gives uniformly distributed between 0 and 1
% for i=1:100
%     % 10x10 square
%     A((1:10) + floor(rand(1)*590), (1:10) + floor(rand(1)*590)) = 1;
% end
% for i=100:160
%     % 30x30 square
%     A((1:30) + floor(rand(1)*590), (1:30) + floor(rand(1)*590)) = 1;
% end

for i=1:150
    w = floor(rand(1) * 40);
    A((1:w) + floor(rand(1)*590), (1:w) + floor(rand(1)*590)) = 1;
end

A = im2bw(A, 0.5);
figure
imshow(A); title('Original image');

B = strel('square', 30);
% B = strel('disk', 15, 0);

A_er_B = imerode(A, B);
figure
imshow(A_er_B);
title('Eroded image using 30x30 square structuring element');

A_er_dil_B = imdilate(A_er_B, B);
figure
imshow(A_er_dil_B);
title('Dilated image using 30x30 square structuring element');


%% Dilation to improve the legibility of a text
clc;
clear all;
close all;

% Here black = 1 (text) and white = 0 (no text)
% color encoding is black = 0, white = 1 so we need the negative image 
A = ~im2bw(imread('Zisserman_page_thr_140_full.jpg'), 0.5);
figure
imshow(~A); title('Low legibility text');

B = strel('arbitrary', [0 1 0; 1 1 1; 0 1 0]);

A_dil_B = imdilate(A, B);
figure
imshow(~A_dil_B); title('Dilated text using default structuring element');

%% Opening and closing for bw image filtering
clc;
clear all;
close all;

A = double(imread('fingerprint.gif'));
imshow(A), title('Noisy fingerprint');

B = strel('arbitrary', [1 1 1; 1 1 1; 1 1 1]);

% opening (erosion followed by dilation)
A_op_B = imopen(A, B);
figure, imshow(A_op_B), title('Fingerprint + opening');

% closing (dilation followed by erosion)
A_cl_B = imclose(A_op_B, B);
figure, imshow(A_cl_B), title('Final filtered fingerprint');

%% Hit-or-miss operator
clc;
clear all;
close all;

A = zeros(600,600);
A(100:500, 100:260) = 1;  % rectangle
A(120:200, 400:480) = 1;  % small square
A(350:449, 420:519) = 1;  % big square
A = im2bw(A, 0.5);
figure
imshow(A); title('Original image');

B = strel('square', 100); % big square template
A_er_B = imerode(A, B);
figure, imshow(A_er_B), title('Image eroded with template');

wb_matrix = ones(120);
wb_matrix(10:110, 10:110) = 0; % local background
WB = strel('arbitrary', wb_matrix);
A_c_er_WB = imerode(~A, WB);
figure, imshow(A_c_er_WB);
title('Complimentary image eroded with template');

hit_image = A_er_B & A_c_er_WB;
figure, imshow(hit_image), title('Template matches in the image');
pause

hit_image = bwhitmiss(A, B, WB);
figure, imshow(hit_image), title('Template matches (matlab function)');

[i, j] = find(hit_image);
fprintf('Final coordinates: %d, %d\n', i, j);

%% Boundary extraction
clc;
clear all;
close all;

im = imread('coins.png');
A = im2bw(im, 0.37);
imshow(A), title('Original image');

% border image
%B = strel('arbitrary', [1 1 1; 1 1 1; 1 1 1]);  % 8-connected
B = strel('arbitrary', [0 1 0; 1 1 1; 0 1 0]);  % 4-connected

A_borders = A - imerode(A, B);
figure, imshow(A_borders), title('Image borders');

A_borders = bwperim(A);
figure, imshow(A_borders), title('Image borders (Matlab function)');

%% Advanced binary operators
clc;
clear all;
close all;

A = imread('circles.png');
figure, imshow(A), title('Original image');

% convex hull
CH = bwconvhull(A);
figure, imshow(CH), title('Convex Hull image');


A = imread('circbw.tif');
figure, imshow(A), title('Original image');

% label connected components
labels = bwlabel(A, 8); % 8-connected regions
figure, imagesc(labels), axis off, title('Connected regions');

% thinning
thin = bwmorph(A, 'thin', Inf);  % thinning operator
figure, imshow(thin), title('Thinning');

% skeletonization
skel = bwmorph(A, 'skel', Inf);   % skeletonization operator
figure, imshow(skel), title('Skeleton');

%% Grayscale dilation and erosion
clc;
clear all;
close all;

A = imread('cameraman.tif');
imshow(A), title('Original')

B = strel('disk', 5);
% B = strel('disk', 3);

% dilation: local maximum
A_dil_B = imdilate(A, B);
figure, imshow(A_dil_B), title('Dilated');

% erosion: local minimum
A_er_B = imerode(A, B);
figure, imshow(A_er_B), title('Eroded');

% morphological gradient
A_morph_grad = A_dil_B - A_er_B;
figure, imshow(A_morph_grad), title('Morphological gradient');