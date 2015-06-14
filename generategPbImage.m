function [] = generategPbImage(imgFile)

[path, name, ~] = fileparts(imgFile);

% compute globalPb
addpath(fullfile('/home/berksevilmis/workspace/Edges_Contours/BSR/grouping/lib'));
[gPb_orient, gPb_thin, ~] = globalPb(imgFile);

gPbResponse = zeros(size(gPb_orient,1),size(gPb_orient,2));
for r = 1:size(gPb_orient,1)
    for c = 1:size(gPb_orient,2)
        gPbResponse(r,c) = max(gPb_orient(r,c,:));
    end
end
maxgPbResponse = max(max(gPbResponse));
mingPbResponse = min(min(gPbResponse));
gPbResponse = 255 * (gPbResponse - mingPbResponse) ./ (maxgPbResponse - mingPbResponse);
gPbResponse = uint8(gPbResponse);
imwrite(gPbResponse, ['./data/' name '_gPb' '.png']);

maxgPbThinResponse = max(max(gPb_thin));
mingPbThinResponse = min(min(gPb_thin));
gPbThinResponse = 255 * (gPb_thin - mingPbThinResponse) ./ ...
    (maxgPbThinResponse - mingPbThinResponse);
gPbThinResponse = uint8(gPbThinResponse);
imwrite(gPbThinResponse, ['./data/' name '_gPbThin' '.png']);
end



