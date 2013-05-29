import numpy, os, sys, cPickle
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
import PIL.Image
from collections import OrderedDict
from image_tiler import *
import time
import pylearn.io.filetensor as ft

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

def experiment(state, channel):
    print 'LOADING MODEL CONFIG'
    import pdb; pdb.set_trace()

    print state
    f = open('config', 'w')
    f.write(str(state))
    f.close()
    # LOAD DATA
    (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = load_mnist(state.data_path)

    train_X = numpy.concatenate((train_X, valid_X))

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
    layer_sizes     =   [784] + [state.hidden_size] * K
    learning_rate   =   theano.shared(cast32(state.learning_rate))
    annealing       =   cast32(state.annealing)
    momentum        =   theano.shared(cast32(state.momentum))

    # PARAMETERS
    # weights

    weights_list    =   [get_shared_weights(layer_sizes[i], layer_sizes[i+1], numpy.sqrt(6. / (layer_sizes[i] + layer_sizes[i+1] )), 'W') for i in range(K)]
    bias_list       =   [get_shared_bias(layer_sizes[i], 'b') for i in range(K + 1)]


    if __name__ == '__main__':
        if len(sys.argv) > 1:
            PARAMS  =   cPickle.load(open(sys.argv[1], 'r'))
            [p.set_value(lp.get_value(borrow=False)) for lp, p in zip(PARAMS[:weights_list], weights_list)]
            [p.set_value(lp.get_value(borrow=False)) for lp, p in zip(PARAMS[weights_list:], bias_list)]


    # This guy has to be lower diagonal!
    # Proper init?
    #V               =   get_shared_weights(layer_sizes[0], layer_sizes[0], 1e-5, 'V')
    V               =   theano.shared(cast32(numpy.zeros((784,784))))
   
    
    # lower diagonal matrix with 1's under the main diagonal, will be used for masking
    # Upper diagonal actually, because of parametrisation : X * V -> X_i = 
    
    dim = layer_sizes[0]
    lower_diag_I    =   cast32((numpy.tril(numpy.ones((dim, dim))).T + 1) % 2).T
    V.set_value(lower_diag_I * V.get_value())

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
            
            if autoregression:
                print 'First layer auto-regressor'
                hiddens[i] = hiddens[i] + T.dot(X, V)

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

    ''' hidden layer init '''
    
    hiddens     = [X_corrupt]
    p_X_chain   = [] 

    print "Hidden units initialization"
    for w,b in zip(weights_list, bias_list[1:]):
        # no noise, basic f prop to
        # init f prop
        #hiddens.append(hidden_activation(T.dot(hiddens[-1], w) + b))

        # init with zeros
        print "Init hidden units at zero before creating the graph"
        hiddens.append(T.zeros_like(T.dot(hiddens[-1], w)))

    # The layer update scheme
    print "Building the graph :", 2*N*K,"updates"
    for i in range(2 * N * K):
        update_layers(hiddens, p_X_chain, autoregression = state.autoregression)
    

    # COST AND GRADIENTS    

    print 'Cost w.r.t p(X|...) at every step in the graph'
    #COST        =   T.mean(T.nnet.binary_crossentropy(reconstruction, X))
    COST        =   [T.mean(T.nnet.binary_crossentropy(rX, X)) for rX in p_X_chain]
    show_COST   =   COST[-1] 
    COST        =   numpy.sum(COST)

    params          =   weights_list + bias_list
    if state.autoregression:
        params      +=  [V]

    gradient        =   T.grad(COST, params)
    if state.autoregression:
        gradient[-1]=(gradient[-1] * lower_diag_I)

    gradient_buffer =   [theano.shared(numpy.zeros(x.get_value().shape, dtype='float32')) for x in params]
    
    m_gradient      =   [momentum * gb + (cast32(1) - momentum) * g for (gb, g) in zip(gradient_buffer, gradient)]
    g_updates       =   [(p, p - learning_rate * mg) for (p, mg) in zip(params, m_gradient)]
    b_updates       =   zip(gradient_buffer, m_gradient)

    updates         =   g_updates + b_updates
    #if state.autoregression:
    #    updates     +=  [(V, V * lower_diag_I)]
    updates         =   OrderedDict(g_updates + b_updates)
    
    #updates     =   OrderedDict([(p, p - learning_rate * g) for (p, g) in zip(params, gradient)])


    f_cost      =   theano.function(inputs = [X], outputs = show_COST)
    
    indexed_batch   = train_X[index * state.batch_size : (index+1) * state.batch_size]
    sampled_batch   = MRG.binomial(p = indexed_batch, size = indexed_batch.shape, dtype='float32')
    
    f_learn     =   theano.function(inputs  = [index], 
                                    updates = updates, 
                                    givens  = {X : indexed_batch},
                                    outputs = show_COST)
    
    f_test      =   theano.function(inputs  =   [X],
                                    outputs =   [X_corrupt] + hiddens[0] + p_X_chain,
                                    on_unused_input = 'warn')


    #############
    # Denoise some numbers  :   show number, noisy number, reconstructed number
    #############
    import random as R
    R.seed(1)
    random_idx      =   numpy.array(R.sample(range(len(test_X.get_value())), 100))
    numbers         =   test_X.get_value()[random_idx]
    
    f_noise = theano.function(inputs = [X], outputs = salt_and_pepper(X, state.input_salt_and_pepper))
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

    def sample_some_numbers():
        # The network's initial state
        init_vis    =   test_X.get_value()[:1]

        noisy_init_vis  =   f_noise(init_vis)

        network_state   =   [[noisy_init_vis] + [numpy.zeros((1,len(b.get_value())), dtype='float32') for b in bias_list[1:]]]

        visible_chain   =   [init_vis]

        noisy_h0_chain  =   [noisy_init_vis]

        for i in range(399):
           
            # feed the last state into the network, compute new state, and obtain visible units expectation chain 
            net_state_out, vis_pX_chain =   sampling_wrapper(network_state[-1])

            # append to the visible chain
            visible_chain   +=  vis_pX_chain

            # append state output to the network state chain
            network_state.append(net_state_out)
            
            noisy_h0_chain.append(net_state_out[0])

        return numpy.vstack(visible_chain), numpy.vstack(noisy_h0_chain)
    
    def plot_samples(epoch_number):
        to_sample = time.time()
        V, H0 = sample_some_numbers()
        img_samples =   PIL.Image.fromarray(tile_raster_images(V, (28,28), (20,20)))
        
        fname       =   'samples_epoch_'+str(epoch_number)+'.png'
        img_samples.save(fname) 
        print 'Took ' + str(time.time() - to_sample) + ' to sample 400 numbers'

    def save_params(n, params):
        fname   =   'params_epoch_'+str(n)+'.ft'
        f       =   open(fname, 'w')
        
        for p in params:
            ft.write(f, p.get_value(borrow=True))
       
        f.close() 



    plot_samples(counter)
    #sample_numbers(counter, [])

    if __name__ == '__main__':
        import ipdb; ipdb.set_trace()
    
    return channel.COMPLETE

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
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

    args.model_path     =   '/data/lisa/exp/thiboeri/repos/DSN/saved_models/2layer_1000tanh_prepostadd2_SP0.3/params_epoch_250.ft'

    experiment(args, None)
