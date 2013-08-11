% This needs to be in the /code subdirectory of the additive noise
% source code
% compile it as per instruction

function pf = run(x,y)
    
    cd '/home/ole/src/cause_effect_sample_code/additive-noise/code/gpml-matlab-v3.0-2010-07-23'
    startup
    cd ..

    addpath('fasthsic');
    addpath('dags');
    addpath('experiments');

    fprintf('----------\n');
    format long

    % do transpose
    y=y.'
    x=x.'

    % forward model
    %fprintf('Fitting forward model...\n');
    yf = fit_gp(x,y);
    [pf hf] = fasthsic(x, yf - y);
    %fprintf('  p-value for independence: %e\n\n',pf);
    A = sortrows([x y yf]); x = A(:,1); y = A(:,2); yf = A(:,3);



