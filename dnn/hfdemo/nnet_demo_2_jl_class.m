seed = 1234;

randn('state', seed );
rand('twister', seed+1 );

%you will NEVER need more than a few hundred epochs unless you are doing
%something very wrong.  Here 'epoch' means parameter update, not 'pass over
%the training set'.
maxepoch = 100;

numchunks = 4
numchunks_test = 4;

tmp = load('images.mat');
data = [tmp.train_images'];%; tmp.valid_images'];
targs = tmp.train_targets';
data = data(:,1:(floor(size(data, 2) / (numchunks * 5)) * (numchunks * 5)));
perm = randperm(size(data,2));
indata = data(:,perm(1:(0.8*length(perm))));
outdata = targs(:,perm(1:(0.8*length(perm))));
intest = data(:,perm((0.8*length(perm) + 1):end));
outtest = targs(:,perm((0.8*length(perm) + 1):end));
clear tmp

%perm = randperm(size(indata,2));
%indata = indata( :, perm );

%it's an auto-encoder so output is input
%outdata = indata;
%outtest = intest;


runName = 'HFtestrun2';

runDesc = ['seed = ' num2str(seed) ', enter anything else you want to remember here' ];

%next try using autodamp = 0 for rho computation.  both for version 6 and
%versions with rho and cg-backtrack computed on the training set


layersizes = [512 512 200 100 50];
%layersizes = [200 100 25 100 200];
%Note that the code layer uses linear units
layertypes = {'logistic', 'logistic', 'logistic', 'logistic', 'logistic', 'softmax'};
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

errtype = 'class'; %report the L2-norm error (in addition to the quantity actually being optimized, i.e. the log-likelihood)

%standard L_2 weight-decay:
weightcost = 2e-5
%weightcost = 0


nnet_train_2( runName, runDesc, paramsp, Win, bin, resumeFile, maxepoch, indata, outdata, numchunks, intest, outtest, numchunks_test, layersizes, layertypes, mattype, rms, errtype, hybridmode, weightcost, decay, jacket);
