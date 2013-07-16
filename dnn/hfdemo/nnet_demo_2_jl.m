%% Load and train

seed = 1234;

randn('state', seed );
rand('twister', seed+1 );

%you will NEVER need more than a few hundred epochs unless you are doing
%something very wrong.  Here 'epoch' means parameter update, not 'pass over
%the training set'.
maxepoch = 100;

numchunks = 4
numchunks_test = 4;

tmp = load('images_30_pit.mat');
all_data = [tmp.train_images', tmp.valid_images'];
data = all_data(:,1:(floor(size(all_data, 2) / (numchunks * 5)) * (numchunks * 5)));
perm = randperm(size(data,2));
indata = data(:,perm(1:(0.8*length(perm))));
intest = data(:,perm((0.8*length(perm) + 1):end));
clear tmp

%perm = randperm(size(indata,2));
%indata = indata( :, perm );

%it's an auto-encoder so output is input
outdata = indata;
outtest = intest;


runName = 'HFtestrun2';

runDesc = ['seed = ' num2str(seed) ', enter anything else you want to remember here' ];

%next try using autodamp = 0 for rho computation.  both for version 6 and
%versions with rho and cg-backtrack computed on the training set


layersizes = [900 300 100 33 5 33 100 300 900];
%layersizes = [200 100 25 100 200];
%Note that the code layer uses linear units
layertypes = {'logistic', 'logistic', 'logistic', 'logistic', 'linear', 'logistic', 'logistic', 'logistic', 'logistic', 'logistic'};
%layertypes = {'logistic', 'logistic', 'linear', 'logistic', 'logistic', 'logistic'};

resumeFile = [];

paramsp = [];
Win = [];
bin = [];
%[Win, bin] = loadPretrainedNet_curves;

mattype = 'gn'; %Gauss-Newton.  The other choices probably won't work for whatever you're doing
%mattype = 'hess';
%mattype = 'empfish';

rms = 0;

hybridmode = 1;

%decay = 1.0;
decay = 0.95;

jacket = 0;
%this enables Jacket mode for the GPU
%jacket = 1;

errtype = 'L2'; %report the L2-norm error (in addition to the quantity actually being optimized, i.e. the log-likelihood)

%standard L_2 weight-decay:
weightcost = 2e-5
%weightcost = 0


nnet_train_2( runName, runDesc, paramsp, Win, bin, resumeFile, maxepoch, indata, outdata, numchunks, intest, outtest, numchunks_test, layersizes, layertypes, mattype, rms, errtype, hybridmode, weightcost, decay, jacket);

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

%% Forward prop all

y = all_data(:,:);

for i = 1:numlayers

    x = W{i} * y + repmat(b{i}, 1, size(y, 2));

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
    
    if i == 5
        r = y;
    end

end

%% Save data

features = r';
csvwrite('auto_pit_30_05.csv', features);