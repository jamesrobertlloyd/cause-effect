%% Load

load HFtestrun2_nnet_epoch100

%% Set up set params

layersizes = [size(indata,1) layersizes size(outdata,1)];

%% More setup

numlayers = size(layersizes,2) - 1;

%% Unpack params

M = paramsp;
W = cell( numlayers, 1 );
b = cell( numlayers, 1 );

cur = 0;
for i = 1:numlayers
    W{i} = reshape( M((cur+1):(cur + layersizes(i)*layersizes(i+1)), 1), [layersizes(i+1) layersizes(i)] );

    cur = cur + layersizes(i)*layersizes(i+1);

    b{i} = reshape( M((cur+1):(cur + layersizes(i+1)), 1), [layersizes(i+1) 1] );

    cur = cur + layersizes(i+1);
end

%% Forward prop test

figure;
rand_index = randi(size(indata,2))
y = indata(:,rand_index);
imagesc(reshape(y,20,20))

for i = 1:numlayers

    x = W{i} * y + b{i};

    if strcmp(layertypes{i}, 'logistic')
        y = 1./(1 + exp(-x));
    elseif strcmp(layertypes{i}, 'tanh')
        y = tanh(x);
    elseif strcmp(layertypes{i}, 'linear')
        y = x;
    elseif strcmp(layertypes{i}, 'softmax' )
        tmp = exp(x);
        y = tmp./repmat( sum(tmp), [layersizes(i+1) 1] );   
        tmp = [];
    end
    
    if i == 3
        r = y
    end

end

figure;
imagesc(reshape(y,20,20))
labels(rand_index)

%% What's the basis?

y = r;
y = zeros(size(r));
y(25) = 1;
y = randn(size(r));

for i = 4:numlayers

    x = W{i} * y + b{i};

    if strcmp(layertypes{i}, 'logistic')
        y = 1./(1 + exp(-x));
    elseif strcmp(layertypes{i}, 'tanh')
        y = tanh(x);
    elseif strcmp(layertypes{i}, 'linear')
        y = x;
    elseif strcmp(layertypes{i}, 'softmax' )
        tmp = exp(x);
        y = tmp./repmat( sum(tmp), [layersizes(i+1) 1] );   
        tmp = [];
    end

end

figure;
imagesc(reshape(y,20,20))