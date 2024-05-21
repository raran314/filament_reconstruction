load('volume_image.mat')
volume_image = imdilate(volume_image, strel('sphere', 5));
volshow(volume_image)
%%
volume_image = logical(volume_image);
skel = bwskel(volume_image,'MinBranchLength',5);


branches = bwmorph3(skel, 'branchpoints');
skel_without_branches = skel & ~branches;

% broken segments
cc = bwconncomp(skel_without_branches);
broken_segments_ind = cc.PixelIdxList(cellfun(@numel, cc.PixelIdxList) > 1);
num_broken_segments = numel(broken_segments_ind);

broken_segments = cell(num_broken_segments, 1);
for i = 1:length(broken_segments_ind)
    ind = broken_segments_ind{i};
    [x,y,z] = ind2sub(size(skel), ind);
    broken_segments{i} = [x,y,z];
end

figure
for i = 1:length(broken_segments)
    curve = broken_segments{i};
    plot3(curve(:,1), curve(:,2), curve(:,3), '.','LineWidth', 2);
    hold on
end