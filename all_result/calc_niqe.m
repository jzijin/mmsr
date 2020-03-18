clear;
% path = "/home/jzijin/code/bysj/code/mmsr/datasets/test_x4_256/Bic/X4";
path = "/home/jzijin/code/bysj/code/mmsr/results/IncNet_GAN/test_x4_256";
filename = dir(path);
score = 0.0;
index = 0;
for i=1:length(filename)
    if filename(i).isdir
        continue;
    end
    index = index + 1;
    f = [filename(i).folder, '/', filename(i).name];
%     disp(f)
    I = imread(f);
    score = score + niqe(I);
end
disp(score ./ index)
