image1 = zeros(100,100);
image2 = zeros(100,100);
[a b] = size(image1);
[c d] = size(image2);
for i = 1 : a
    for j = 1 : b/2
        image1(i,j) = 1;
    end
end

for i = 1 : a
    for j = 1 : b
        if (mod(i,2)== 1 && mod(j,2) == 0)
            image2(i,j) = 1;
        else
            continue;
        end
    end
end
subplot(2,2,1),imshow(image1);
subplot(2,2,2),imshow(image2);
subplot(2,2,3),imhist(image1);
subplot(2,2,4),imhist(image2);
