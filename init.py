import numpy, os, sys, cPickle
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
import PIL.Image
from collections import OrderedDict
from image_tiler import *
import time

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
    print state
    # LOAD DATA
    (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = load_mnist(state.data_path)

    train_X = numpy.concatenate((train_X, valid_X))

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
    layer_sizes     =   [784] + [state.hidden_size] * K
    learning_rate   =   theano.shared(cast32(state.learning_rate))
    annealing       =   cast32(state.annealing)
    momentum        =   theano.shared(cast32(state.momentum))


    # PARAMETERS
    # weights

    weights_list    =   [get_shared_weights(layer_sizes[i], layer_sizes[i+1], numpy.sqrt(6. / (layer_sizes[i] + layer_sizes[i+1] )), 'W') for i in range(K)]
    bias_list       =   [get_shared_bias(layer_sizes[i], 'b') for i in range(K + 1)]


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
        if i == 0:
            hiddens[i]  =   T.dot(hiddens[i+1], weights_list[i].T) + bias_list[i]           
            
            if autoregression:
                print 'First layer auto-regressor'
                hiddens[i] = hiddens[i] + T.dot(X, V)

        elif i == K:
            hiddens[i]  =   T.dot(hiddens[i-1], weights_list[i-1]) + bias_list[i]

        else:
            # next layer        :   layers[i+1], assigned weights : W_i
            # previous layer    :   layers[i-1], assigned weights : W_(i-1)
            hiddens[i]  =   T.dot(hiddens[i+1], weights_list[i].T) + T.dot(hiddens[i-1], weights_list[i-1]) + bias_list[i]

        # Add pre-activation noise if NOT input layer
        if i==1 and state.noiseless_h1:
            print '>>NO noise in first layer'
            add_noise   =   False
            
        if i != 0 and add_noise:
            print 'Adding pre-activation gaussian noise'
            hiddens[i]  =   add_gaussian_noise(hiddens[i], state.hidden_add_noise_sigma)
       
        # ACTIVATION!
        if i == 0:
            print 'Sigmoid units'
            hiddens[i]  =   T.nnet.sigmoid(hiddens[i])
        else:
            print 'Hidden units'
            hiddens[i]  =   hidden_activation(hiddens[i])
       
        # POST ACTIVATION NOISE 
        if i != 0 and mul_noise:
            # dropout if hidden
            print 'Dropping out'
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
    
    # Denoise some numbers  :   show number, noisy number, reconstructed number
    import random as R
    R.seed(1)
    random_idx      =   numpy.array(R.sample(range(len(test_X.get_value())), 100))
    numbers         =   test_X.get_value()[random_idx]
    
    f_noise = theano.function(inputs = [X], outputs = salt_and_pepper(X, state.input_salt_and_pepper))
    noisy_numbers   =   f_noise(test_X.get_value()[random_idx])

    '''
    H  =   X
    for w,b in zip(weights_list, bias_list[1:]):
        H   =   hidden_activation(T.dot(H, w) + b)

    for i in range(K-1,-1,-1):
        H   =   T.dot(H, weights_list[i].T) + bias_list[i]
        if i != 0:
            H   =   hidden_activation(H)
        else:
            H   =   T.nnet.sigmoid(H) 

    # reconstruction function
    f_recon = theano.function(inputs = [X], outputs = H)
    '''

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

    # TRAINING
    n_epoch     =   state.n_epoch
    batch_size  =   state.batch_size
    STOP        =   False
    counter     =   0

    train_costs =   []
    valid_costs =   []
    test_costs  =   []
    
    if state.vis_init:
        bias_list[0].set_value(logit(numpy.clip(0.9,0.001,train_X.get_value().mean(axis=0))))

    while not STOP:
        counter     +=  1
        t = time.time()
        print counter,'\t',

        #train
        train_cost  =   []
        for i in range(len(train_X.get_value(borrow=True)) / batch_size):
            #train_cost.append(f_learn(train_X[i * batch_size : (i+1) * batch_size]))
            #training_idx = numpy.array(range(i*batch_size, (i+1)*batch_size), dtype='int32')
            train_cost.append(f_learn(i))
        train_cost = numpy.mean(train_cost) 
        train_costs.append(train_cost)
        print 'Train : ',trunc(train_cost), '\t',


        #valid
        valid_cost  =   []
        for i in range(len(valid_X.get_value(borrow=True)) / 100):
            valid_cost.append(f_cost(valid_X.get_value()[i * 100 : (i+1) * batch_size]))
        valid_cost = numpy.mean(valid_cost)
        valid_costs.append(valid_cost)
        print 'Valid : ', trunc(valid_cost), '\t',

        #test
        test_cost  =   []
        for i in range(len(test_X.get_value(borrow=True)) / 100):
            test_cost.append(f_cost(test_X.get_value()[i * 100 : (i+1) * batch_size]))
        test_cost = numpy.mean(test_cost)
        test_costs.append(test_cost)
        print 'Test  : ', trunc(test_cost), '\t',

        if counter >= n_epoch:
            STOP = True

        print 'time : ', trunc(time.time() - t),

        print 'MeanVisB : ', trunc(bias_list[0].get_value().mean()),
        
        print 'W : ', [trunc(abs(w.get_value(borrow=True)).mean()) for w in weights_list]

        # Checking reconstruction
        reconstructed   =   f_recon(noisy_numbers) 
        # Concatenate stuff
        stacked         =   numpy.vstack([numpy.vstack([numbers[i*10 : (i+1)*10], noisy_numbers[i*10 : (i+1)*10], reconstructed[i*10 : (i+1)*10]]) for i in range(10)])
        
        number_reconstruction   =   PIL.Image.fromarray(tile_raster_images(stacked, (28,28), (10,30)))
        epoch_number    =   reduce(lambda x,y : x + y, ['_'] * (3-len(str(counter)))) + str(counter)
        number_reconstruction.save('number_reconstruction'+str(counter)+'.png')
     
        # ANNEAL!
        new_lr = learning_rate.get_value() * annealing
        learning_rate.set_value(new_lr)

    # Save
   
    state.train_costs = train_costs
    state.valid_costs = valid_costs
    state.test_costs = test_costs

    cPickle.dump(params, open('params.pkl', 'w'))
    
    # Sample some numbers   :   start a chain, save the chain output (whole chain? start with noise)


    ''' SHITTY WAY OF DOING IT!!!!!!!!!!!!!!!!! COPIED EVERYTHING, AND PASTE!!!! '''

    ''' Corrupt X '''
    X = T.fmatrix()
    X_corrupt   = salt_and_pepper(X, state.input_salt_and_pepper)

    hiddens_input   =   [T.fmatrix() for i in range(K)]

    ''' hidden layer init '''
    ''' Here the hiddens are given as input also, to keep the chain going '''     
    hiddens     = [X_corrupt] + hiddens_input
    p_X_chain   = [] 
    
    # ONE update, without autoregression (this will be done in python -> with one update we only have time to produce the next
    # output layer. So we just get the inverse sigmoid, which gives us Wx + b, and the rest is iteratively constructing the new p_X
    # using the autoregression weights
    update_layers(hiddens, p_X_chain, noisy=True, autoregression=False)
    
    #for i in range(2 * N * K):
    #    update_layers(hiddens, p_X_chain)

    f_sample    =   theano.function(inputs = [X] + hiddens_input, outputs = p_X_chain+hiddens, on_unused_input='warn')


    # call f_sample:
    # get h1, get p_X_chain[0]
    # loop to compute x[i]

    numpy.random.seed(1)
    
    noise_init  =   numpy.random.uniform
    
    print 'Generating samples...',

    t = time.time()
    init        =   cast32(numpy.random.uniform(size=(1,784)))
    #init        =   test_X.get_value()[:1]
    zeros       =   [numpy.zeros((1,len(b.get_value())), dtype='float32') for b in bias_list[1:]]
    
    samples     =   [[init] + zeros]
    output      =   [init]

    for i in range(399):
        network_state   =   f_sample(*samples[-1])
       
        
        p_X             =   network_state[0]

        if state.autoregression:
            x_init      =   logit(p_X).flatten()

            for i in range(784):
                x_init[i]   +=  numpy.dot(sigmoid(x_init), V.get_value().T[i])

            p_X  =   numpy.array([sigmoid(x_init)])
            # I think it fucks up...
            p_X  =   numpy.clip(0.999,0.001, p_X)
            #network_state[0] = p_X
            network_state[1]    =   f_noise(p_X)
            network_state       =   network_state[1:]

        output.append(p_X)
        samples.append(network_state) 
        
    
    x_chain =   numpy.vstack(output)

    '''
    # chainnnn : f_noise -> f_sample on last sample
    for i in range(100-1):
        #noisy [x]
        _input  =   samples[-1][-len(hiddens):]

        #sample
        sample  = f_sample(*_input)

        # save output
        output  =   output + sample[:len(hiddens)]
        # append to sample list
        samples.append(sample)

    x_chain = [init] + reduce(lambda x,y : x+y, [x[:len(p_X_chain)] for x in samples[1:]])

    x_chain = numpy.vstack(x_chain)

    chain_length = len(x_chain)

    # we want, say 400 samples
    if chain_length < 400:
        missing = numpy.zeros((400 - chain_length, 784))
        x_chain = numpy.vstack((x_chain, missing))
    else:
        x_chain = x_chain[:400]
    
    #[samples.append(f_sample(f_noise(samples[-1]))) for i in range(100 - 1)]
    #samples = numpy.vstack(samples)

    '''
    #plot
    img_samples =   PIL.Image.fromarray(tile_raster_images(x_chain, (28,28), (20,20)))
    img_samples.save('img_samples.png')
    print 'took ', time.time() - t, ' seconds'

    if __name__ == '__main__':
        os.system('eog img_samples.png')

    # Sample from the model -> show the chain

    if __name__ == '__main__':
        import ipdb; ipdb.set_trace()
    
    return channel.COMPLETE

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    args.K          =   2
    args.N          =   2
    args.n_epoch    =   40
    args.batch_size =   100

    args.hidden_add_noise_sigma =   0.25
    args.hidden_dropout         =   0.5
    args.input_salt_and_pepper  =   0.3

    args.learning_rate  =   0.1
    args.momentum       =   0.9
    args.annealing      =   0.95

    args.hidden_size    =   1000

    args.input_sampling =   False
    args.noiseless_h1   =   False

    args.vis_init       =   False

    args.act            =   'rectifier'

    args.autoregression =   True

    args.data_path      =   '/data/lisa/data/mnist/'

    experiment(args, None)
