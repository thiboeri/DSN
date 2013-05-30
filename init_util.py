import numpy, os, sys, cPickle
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
import PIL.Image
from collections import OrderedDict
from image_tiler import *
import time
import pylearn.io.filetensor as ft
from likelihood_estimation_parzen import * 


cast32      = lambda x : numpy.cast['float32'](x)
trunc       = lambda x : str(x)[:8]
logit       = lambda p : numpy.log(p / (1 - p) )
binarize    = lambda x : cast32(x >= 0.5)
sigmoid     = lambda x : cast32(1. / (1 + numpy.exp(-x)))

def SaltAndPepper(X, rate=0.3):
    # Salt and pepper noise
    
    drop = numpy.arange(X.shape[1])
    numpy.random.shuffle(drop)
    sep = int(len(drop)*rate)
    drop = drop[:sep]
    X[:, drop[:sep/2]]=0
    X[:, drop[sep/2:]]=1
    return X

def get_shared_weights(n_in, n_out, interval, name):
    #val = numpy.random.normal(0, sigma_sqr, size=(n_in, n_out))
    val = numpy.random.uniform(-interval, interval, size=(n_in, n_out))
    val = cast32(val)
    val = theano.shared(value = val, name = name)
    return val

def get_shared_bias(n, name, offset = 0):
    val = numpy.zeros(n) - offset
    val = cast32(val)
    val = theano.shared(value = val, name = name)
    return val

def load_mnist(path):
    data = cPickle.load(open(os.path.join(path,'mnist.pkl'), 'r'))
    return data

def load_tfd(path):
    '''
    import scipy.io as io
    data = io.loadmat(os.path.join(path, 'TFD_48x48.mat'))
    X = cast32(data['images'])/cast32(255)
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
    labels  = data['labs_ex'].flatten()
    labeled = labels != -1
    unlabeled   =   labels == -1  
    train_X =   X[unlabeled]
    valid_X =   X[unlabeled][:100] # Stuf
    test_X  =   X[labeled]
    del data
    '''

    import pylearn.io.filetensor as io
    F   =   open(os.join.path(path, 'TFD_48x48.ft'), 'r')
    train_X =  ft.read(F)
    train_Y =  ft.read(F)
    valid_X =  ft.read(F)
    valid_Y =  ft.read(F)
    test_X  =  ft.read(F)
    test_Y  =  ft.read(F)
    
    return (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y)

    #return (train_X, labels[unlabeled]), (valid_X, labels[unlabeled][:100]), (test_X, labels[labeled])


def experiment(state, channel):
    print 'LOADING MODEL CONFIG'
    config_path =   '/'+os.path.join(*state.model_path.split('/'))
    print state.model_path

    if 'config' in os.listdir(config_path):
        
        config_file = open(os.path.join(config_path, 'config'), 'r')
        config      =   config_file.readlines()
        try:
            config_vals =   config[0].split('(')[1:][0].split(')')[:-1][0].split(', ')
        except:
            config_vals =   config[0][3:-1].replace(': ','=').replace("'","").split(', ')
            config_vals =   filter(lambda x:not 'jobman' in x and not '/' in x and not ':' in x and not 'experiment' in x, config_vals)
        
        for CV in config_vals:
            print CV
            try:
                exec('state.'+CV) in globals(), locals()
            except:
                exec('state.'+CV.split('=')[0]+"='"+CV.split('=')[1]+"'") in globals(), locals()
    else:
        import pdb; pdb.set_trace()

    # LOAD DATA
    if 'mnist' in state.data_path:
        (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = load_mnist(state.data_path)
        train_X = numpy.concatenate((train_X, valid_X))
    elif 'TFD' in state.data_path:
        (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = load_tfd(state.data_path)
    
    N_input =   train_X.shape[1]
    root_N_input = numpy.sqrt(N_input)


    #train_X = binarize(train_X)
    #valid_X = binarize(valid_X)
    #test_X = binarize(test_X)
    numpy.random.seed(1)
    numpy.random.shuffle(train_X)
    train_X = theano.shared(train_X)
    valid_X = theano.shared(valid_X)
    test_X  = theano.shared(test_X)
    # shuffle Y also if necessary

    # THEANO VARIABLES
    X       = T.fmatrix()
    index   = T.lscalar()
    MRG = RNG_MRG.MRG_RandomStreams(1)
    
    # SPECS
    K               =   state.K
    N               =   state.N
    layer_sizes     =   [N_input] + [state.hidden_size] * K
    learning_rate   =   theano.shared(cast32(state.learning_rate))
    annealing       =   cast32(state.annealing)
    momentum        =   theano.shared(cast32(state.momentum))

    # PARAMETERS
    # weights

    weights_list    =   [get_shared_weights(layer_sizes[i], layer_sizes[i+1], numpy.sqrt(6. / (layer_sizes[i] + layer_sizes[i+1] )), 'W') for i in range(K)]
    bias_list       =   [get_shared_bias(layer_sizes[i], 'b') for i in range(K + 1)]


    # LOAD PARAMS
    print 'Loading model params...',
    print 'Loading last epoch...',
    param_files =   filter(lambda x: x.endswith('ft'), os.listdir(config_path))
    max_epoch   =   numpy.argmax([int(x.split('_')[-1].split('.')[0]) for x in param_files])

    params_to_load  =   os.path.join(config_path, param_files[max_epoch])
    F   =   open(params_to_load, 'r')

    n_params = len(weights_list) + len(bias_list)
    print param_files[max_epoch]

    for i in range(0, len(weights_list)):
        weights_list[i].set_value(ft.read(F))

    for i in range(len(bias_list)):
        bias_list[i].set_value(ft.read(F))


    print 'Model parameters loaded!!'

        # functions
    def dropout(IN, p = 0.5):
        noise   =   MRG.binomial(p = p, n = 1, size = IN.shape, dtype='float32')
        OUT     =   (IN * noise) / cast32(p)
        return OUT

    def add_gaussian_noise(IN, std = 1):
        print 'GAUSSIAN NOISE : ', std
        noise   =   MRG.normal(avg  = 0, std  = std, size = IN.shape, dtype='float32')
        OUT     =   IN + noise
        return OUT

    def corrupt_input(IN, p = 0.5):
        # salt and pepper? masking?
        noise   =   MRG.binomial(p = p, n = 1, size = IN.shape, dtype='float32')
        IN      =   IN * noise
        return IN

    def salt_and_pepper(IN, p = 0.2):
        # salt and pepper noise
        print 'DAE uses salt and pepper noise'
        a = MRG.binomial(size=IN.shape, n=1,
                              p = 1 - p,
                              dtype='float32')
        b = MRG.binomial(size=IN.shape, n=1,
                              p = 0.5,
                              dtype='float32')
        c = T.eq(a,0) * b
        return IN * a + c

    def update_odd_layers(hiddens, noisy):
        for i in range(1, K+1, 2):
            print i
            if noisy:
                simple_update_layer(hiddens, None, i)
            else:
                simple_update_layer(hiddens, None, i, mul_noise = False, add_noise = False)

    # we can append the reconstruction at each step
    def update_even_layers(hiddens, p_X_chain, autoregression, noisy):
        for i in range(0, K+1, 2):
            print i
            if noisy:
                simple_update_layer(hiddens, p_X_chain, i, autoregression)
            else:
                simple_update_layer(hiddens, p_X_chain, i, autoregression, mul_noise = False, add_noise = False)

    def simple_update_layer(hiddens, p_X_chain, i, autoregression=False, mul_noise=True, add_noise=True):
        # Compute the dot product, whatever layer
        post_act_noise  =   0
        if i == 0:
            hiddens[i]  =   T.dot(hiddens[i+1], weights_list[i].T) + bias_list[i]           

        elif i == K:
            hiddens[i]  =   T.dot(hiddens[i-1], weights_list[i-1]) + bias_list[i]
            # TODO compute d h_i / d h_(i-1)

            # derivee de h[i] par rapport a h[i-1]
            # W is what transpose...

            if state.scaled_noise:
            
                # to remove this, remove the post_act_noise variable initialisation and the following block
                # and put back post activation noise like it was (just normal calling of the function)
                W   =   weights_list[i-1]
                hn  =   T.tanh(hiddens[i])
                ww  =   T.dot(W.T, W)
                s   =   (cast32(1) - hn**2)
                jj  =   ww * s.dimshuffle(0, 'x', 1) * s.dimshuffle(0, 1, 'x')
                scale_noise =   lambda alpha : (alpha.dimshuffle(0, 1, 'x') * jj).sum(1)

                print 'SCALED_NOISE!!!, Last layer : set add_noise to False, add its own scaled noise'
                add_noise   =   False

                #pre_act_noise   =   MRG.normal(avg  = 0, std  = std, size = hn.shape, dtype='float32')
                post_act_noise  =   MRG.normal(avg  = 0, std  = state.hidden_add_noise_sigma, size = hn.shape, dtype='float32')

                #pre_act_noise   =   scale_noise(pre_act_noise)
                post_act_noise  =   scale_noise(post_act_noise)

                #hiddens[i]      +=  pre_act_noise


        else:
            # next layer        :   layers[i+1], assigned weights : W_i
            # previous layer    :   layers[i-1], assigned weights : W_(i-1)
            hiddens[i]  =   T.dot(hiddens[i+1], weights_list[i].T) + T.dot(hiddens[i-1], weights_list[i-1]) + bias_list[i]

        # Add pre-activation noise if NOT input layer
        if i==1 and state.noiseless_h1:
            print '>>NO noise in first layer'
            add_noise   =   False


        # pre activation noise            
        if i != 0 and add_noise and not state.scaled_noise:
            print 'Adding pre-activation gaussian noise'
            hiddens[i]  =   add_gaussian_noise(hiddens[i], state.hidden_add_noise_sigma)
      
      
      
       
        # ACTIVATION!
        if i == 0:
            print 'Sigmoid units'
            hiddens[i]  =   T.nnet.sigmoid(hiddens[i])
        else:
            print 'Hidden units'
            hiddens[i]  =   hidden_activation(hiddens[i])

        # post activation noise            
        if i != 0 and add_noise:
            print 'Adding post-activation gaussian noise'
            if state.scaled_noise:
                hiddens[i]  +=  post_act_noise
            else:
                hiddens[i]  =   add_gaussian_noise(hiddens[i], state.hidden_add_noise_sigma)


        # POST ACTIVATION NOISE 
        if i != 0 and mul_noise and state.hidden_dropout:
            # dropout if hidden
            print 'Dropping out', state.hidden_dropout
            hiddens[i]  =   dropout(hiddens[i], state.hidden_dropout)
        elif i == 0:
            # if input layer -> append p(X|...)
            p_X_chain.append(hiddens[i])
            
            # sample from p(X|...)
            if state.input_sampling:
                print 'Sampling from input'
                sampled     =   MRG.binomial(p = hiddens[i], size=hiddens[i].shape, dtype='float32')
            else:
                print '>>NO input sampling'
                sampled     =   hiddens[i]
            # add noise
            sampled     =   salt_and_pepper(sampled, state.input_salt_and_pepper)
            
            # set input layer
            hiddens[i]  =   sampled

    def update_layers(hiddens, p_X_chain, autoregression, noisy = True):
        print 'odd layer update'
        update_odd_layers(hiddens, noisy)
        print
        print 'even layer update'
        update_even_layers(hiddens, p_X_chain, autoregression, noisy)

 
    ''' F PROP '''
    #X = T.fmatrix()
    if state.act == 'sigmoid':
        print 'Using sigmoid activation'
        hidden_activation = T.nnet.sigmoid
    elif state.act == 'rectifier':
        print 'Using rectifier activation'
        hidden_activation = lambda x : T.maximum(cast32(0), x)
    elif state.act == 'tanh':
        hidden_activation = lambda x : T.tanh(x)    
   
    
    ''' Corrupt X '''
    X_corrupt   = salt_and_pepper(X, state.input_salt_and_pepper)

    f_noise = theano.function(inputs = [X], outputs = salt_and_pepper(X, state.input_salt_and_pepper))

    ''' Commented for now (unless we need more denoising stuff)
    #############
    # Denoise some numbers  :   show number, noisy number, reconstructed number
    #############
    import random as R
    R.seed(1)
    random_idx      =   numpy.array(R.sample(range(len(test_X.get_value())), 100))
    numbers         =   test_X.get_value()[random_idx]
    
    noisy_numbers   =   f_noise(test_X.get_value()[random_idx])

    # Recompile the graph without noise for reconstruction function
    hiddens_R     = [X]
    p_X_chain_R   = []

    for w,b in zip(weights_list, bias_list[1:]):
        # init with zeros
        hiddens_R.append(T.zeros_like(T.dot(hiddens_R[-1], w)))

    # The layer update scheme
    for i in range(2 * N * K):
        update_layers(hiddens_R, p_X_chain_R, noisy=False, autoregression=state.autoregression)

    f_recon = theano.function(inputs = [X], outputs = p_X_chain_R[-1]) 

    '''
    
    ##################################
    # Sampling, round 2 motherf***** #
    ##################################
    
    # the input to the sampling function
    network_state_input     =   [X] + [T.fmatrix() for i in range(K)]
    'first input will be a noisy number and zeros at the hidden layer, is this correc?'
   
    # "Output" state of the network (noisy)
    # initialized with input, then we apply updates
    #network_state_output    =   network_state_input
    # WTFFFF why is it not the same? fucking python list = list not the same as list = list(list) ???
    network_state_output    =   [X] + network_state_input[1:]
    

    visible_pX_chain        =   []

    #for i in range(2 * N * K):
    #    update_layers(network_state_output, visible_pX_chain, noisy=True, autoregression=False)

    # ONE update
    update_layers(network_state_output, visible_pX_chain, noisy=True, autoregression=False)

    # WHY IS THERE A WARNING????
    # because the first odd layers are not used -> directly computed FROM THE EVEN layers
    f_sample2   =   theano.function(inputs = network_state_input, outputs = network_state_output + visible_pX_chain, on_unused_input='warn')

    def sampling_wrapper(NSI):
        out             =   f_sample2(*NSI)
        NSO             =   out[:len(network_state_output)]
        vis_pX_chain    =   out[len(network_state_output):]
        return NSO, vis_pX_chain

    def sample_some_numbers(n_digits = 400):
        to_sample = time.time()
        # The network's initial state
        #init_vis    =   test_X.get_value()[:1]
        init_vis    =   test_X[:1]

        noisy_init_vis  =   f_noise(init_vis)

        network_state   =   [[noisy_init_vis] + [numpy.zeros((1,len(b.get_value())), dtype='float32') for b in bias_list[1:]]]

        visible_chain   =   [init_vis]

        noisy_h0_chain  =   [noisy_init_vis]

        for i in range(n_digits - 1):
           
            # feed the last state into the network, compute new state, and obtain visible units expectation chain 
            net_state_out, vis_pX_chain =   sampling_wrapper(network_state[-1])

            # append to the visible chain
            visible_chain   +=  vis_pX_chain

            # append state output to the network state chain
            network_state.append(net_state_out)
            
            noisy_h0_chain.append(net_state_out[0])

        print 'Took ' + str(time.time() - to_sample) + ' to sample ' + str(n_digits) + ' digits'
        return numpy.vstack(visible_chain), numpy.vstack(noisy_h0_chain)
    
    def plot_samples(epoch_number):
        V, H0 = sample_some_numbers()
        img_samples =   PIL.Image.fromarray(tile_raster_images(V, (root_N_input,root_N_input), (20,20)))
        
        fname       =   'samples_epoch_'+str(epoch_number)+'.png'
        img_samples.save(fname) 

    def save_params(n, params):
        fname   =   'params_epoch_'+str(n)+'.ft'
        f       =   open(fname, 'w')
        
        for p in params:
            ft.write(f, p.get_value(borrow=True))
       
        f.close() 


    def plot_one_digit(digit):
        plot_one    =   PIL.Image.fromarray(tile_raster_images(digit, (root_N_input,root_N_input), (1,1)))
        fname       =   'one_digit.png'
        plot_one.save(fname)
        os.system('eog one_digit.png')

    def inpainting(digit):
        # The network's initial state

        # NOISE INIT
        init_vis    =   cast32(numpy.random.uniform(size=digit.shape))

        #noisy_init_vis  =   f_noise(init_vis)
        #noisy_init_vis  =   cast32(numpy.random.uniform(size=init_vis.shape))

        # INDEXES FOR VISIBLE AND NOISY PART
        noise_idx = (numpy.arange(N_input) % root_N_input < (root_N_input/2))
        fixed_idx = (numpy.arange(N_input) % root_N_input > (root_N_input/2))
        # function to re-init the visible to the same noise

        # FUNCTION TO RESET HALF VISIBLE TO DIGIT
        def reset_vis(V):
            V[0][fixed_idx] =   digit[0][fixed_idx]
            return V
        
        # INIT DIGIT : NOISE and RESET HALF TO DIGIT
        init_vis = reset_vis(init_vis)

        network_state   =   [[init_vis] + [numpy.zeros((1,len(b.get_value())), dtype='float32') for b in bias_list[1:]]]

        visible_chain   =   [init_vis]

        noisy_h0_chain  =   [init_vis]

        for i in range(49):
           
            # feed the last state into the network, compute new state, and obtain visible units expectation chain 
            net_state_out, vis_pX_chain =   sampling_wrapper(network_state[-1])


            # reset half the digit
            net_state_out[0] = reset_vis(net_state_out[0])
            vis_pX_chain[0]  = reset_vis(vis_pX_chain[0])

            # append to the visible chain
            visible_chain   +=  vis_pX_chain

            # append state output to the network state chain
            network_state.append(net_state_out)
            
            noisy_h0_chain.append(net_state_out[0])

        return numpy.vstack(visible_chain), numpy.vstack(noisy_h0_chain)
 

    #V_inpaint, H_inpaint = inpainting(test_X.get_value()[:1])
    #plot_one    =   PIL.Image.fromarray(tile_raster_images(V_inpaint, (root_N_input,root_N_input), (1,50)))
    #fname       =   'test.png'
    #plot_one.save(fname)
    #os.system('eog test.png')
                                   
   
    #get all digits, and do it a couple of times
    test_X  =   test_X.get_value()
    #test_Y  =   test_Y.get_value()

    numpy.random.seed(1)
    test_idx    =   numpy.arange(len(test_Y))


    for Iter in range(10):

        numpy.random.shuffle(test_idx)
        test_X = test_X[test_idx]
        test_Y = test_Y[test_idx]
    
        digit_idx = [(test_Y==i).argmax() for i in range(10)]
        inpaint_list = []

        for idx in digit_idx:
            DIGIT = test_X[idx:idx+1] 
            V_inpaint, H_inpaint = inpainting(DIGIT)
            inpaint_list.append(V_inpaint)

        INPAINTING  =   numpy.vstack(inpaint_list)

        plot_inpainting =   PIL.Image.fromarray(tile_raster_images(INPAINTING, (root_N_input,root_N_input), (10,50)))

        fname   =   'inpainting_'+str(Iter)+'.png'

        plot_inpainting.save(fname)

        if False and __name__ ==  "__main__":
            os.system('eog inpainting.png')



    # PARZEN 
    # Generating 10000 samples
    samples, _ = sample_some_numbers(n_digits=10000) 
    
    Mean, Std   =   main(state.sigma_parzen, samples, test_X)

    #plot_samples(999)
    #sample_numbers(counter, [])

    if __name__ == '__main__':
        return Mean, Std
        #import ipdb; ipdb.set_trace()
    
    return channel.COMPLETE

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='.')
    args = parser.parse_args()
       
   
    
    args.K          =   2
    args.N          =   1
    args.n_epoch    =   500
    args.batch_size =   100

    #args.hidden_add_noise_sigma =   1e-10
    args.scaled_noise           =   False
    args.hidden_add_noise_sigma =   2.0
    args.hidden_dropout         =   0
    args.input_salt_and_pepper  =   0.3

    args.learning_rate  =   0.25
    args.momentum       =   0.5
    args.annealing      =   0.99

    args.hidden_size    =   1500

    args.input_sampling =   True
    args.noiseless_h1   =   True

    args.vis_init       =   False

    #args.act            =   'rectifier'
    args.act            =   'tanh'

    args.autoregression =   False

    args.data_path      =   '/data/lisa/data/mnist/'
    #args.data_path      =   '/data/lisa/data/faces/TFD'

    #args.model_path     =   '/data/lisa/exp/thiboeri/repos/DSN/saved_models/3layer_1500tanh_prepostadd2_SP0.4/'

    #models_to_evaluate  =   '/data/lisa/exp/thiboeri/repos/DSN/saved_models/'
    models_to_evaluate  =   args.path
    
    print 'Evaluating models in path : ', models_to_evaluate
    #subfolders  =   [o for o in os.listdir(models_to_evaluate) if os.path.isdir(o)]


    #models = filter(lambda x:numpy.any(['params' in P for P in os.listdir(os.path.join(models_to_evaluate, x))]), subfolders)
    #models = filter(lambda x:numpy.any(['params' in P for P in os.listdir(os.path.join(models_to_evaluate, x))]), os.listdir(models_to_evaluate))

    models = os.listdir(models_to_evaluate)
    models = filter(lambda x:('jobman' in  x) or ('layer' in x), models)


    print models
    print
    print 'Starting Loop'
    print
   
    
    for model in models:
        #try:
            args.model_path = os.path.join(models_to_evaluate, model)
            
            print 'XXXX', args.model_path
            args.sigma_parzen   =   0.2
            (M,S)   =   experiment(args, None)
            fname   =   'parzen_ll'
            f       =   open(os.path.join(models_to_evaluate, fname), 'w')
            f.write(str([M, S]))
            f.close()
        #except IOError:
        #    import pdb; pdb.set_trace()
        #except OSError:
        #    print 'not a directory'
    #experiment(args, None)

