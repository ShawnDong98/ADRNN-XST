clear;clc;

res_path = './results/ADRNN_XST_simu_KAIST.mat';
load(res_path);

psnr_total=0.0;
ssim_total=0.0;
sam_total=0.0;
psnr_list = zeros(10,1);
ssim_list = zeros(10,1);
sam_list = zeros(10,1);
for i=1:10
    Z = squeeze(pred(i,:,:,:));
    Z = double(Z);
    S = squeeze(truth(i,:,:,:));
    S = double(S);
    
    Z(Z>1.0) = 1.0;
    Z(Z<0.0) = 0.0;

    [psnr, rmse, ergas, sam, uiqi, ssim] = quality_assessment(double(im2uint8(S)), double(im2uint8(Z)), 0, 1);
%     [psnr, rmse, ergas, sam, uiqi, ssim] = quality_assessment(double(S * 255.), double(Z * 255.), 0, 1);

    pred(1,i) = psnr;
    pred(2,i) = rmse;
    pred(3,i) = ergas;
    pred(4,i) = sam;
    pred(5,i) = uiqi;
    pred(6,i) = ssim;
    psnr_list(i,1) = psnr;
    ssim_list(i,1) = ssim;
    sam_list(i, 1) = sam;
    psnr
    ssim
    sam
    
    psnr_total = psnr_total+psnr;
    ssim_total = ssim_total+ssim;
end
psnr = mean(psnr_list);
ssim = mean(ssim_list);
sam = mean(sam_list);
fprintf('The PNSR=%f\n',psnr);
fprintf('The SSIM=%f\n',ssim);
fprintf('The SAM=%f\n',sam);

