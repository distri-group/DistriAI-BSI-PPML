
import math
import re

from compiler import mpc_math, util
from compiler.types import *
from compiler.types import _unreduced_squant
from compiler.library import *
from compiler.util import is_zero, tree_reduce
from compiler.comparison import CarryOutRawLE
from compiler.GC.types import sbitint
from functools import reduce

class SGD(Optimizer):
    """ Stochastic gradient descent.

    :param layers: layers of linear graph
    :param n_epochs: number of epochs for training
    :param report_loss: disclose and print loss
    """
    def __init__(self, layers, n_epochs=1, debug=False, report_loss=None):
        super(SGD, self).__init__(report_loss=report_loss)
        self.momentum = 0.9
        self.layers = layers
        self.n_epochs = n_epochs
        self.nablas = []
        self.delta_thetas = []
        for layer in layers:
            self.nablas.extend(layer.nablas())
            for theta in layer.thetas():
                self.delta_thetas.append(theta.same_shape())
        self.set_learning_rate(0.01)
        self.debug = debug
        print_both('Using SGD')

    @_no_mem_warnings
    def reset(self, X_by_label=None):
        """ Reset layer parameters and optionally balance training data by public labels.

        :param X_by_label: if provided, this parameter is used to set the training data 
                           by public labels for the purpose of balancing the dataset.
        """
        self.X_by_label = X_by_label
        if X_by_label is not None:
            for label, X in enumerate(X_by_label):
                @for_range_multithread(self.n_threads, 1, len(X))
                def _(i):
                    j = i + label * len(X_by_label[0])
                    self.layers[0].X[j] = X[i]
                    self.layers[-1].Y[j] = label
        for y in self.delta_thetas:
            y.assign_all(0)
        super(SGD, self).reset()

    def _update(self, i_epoch, i_batch, batch):
        for nabla, theta, delta_theta in zip(self.nablas, self.thetas,
                                             self.delta_thetas):
            @multithread(self.n_threads, nabla.total_size())
            def _(base, size):
                old = delta_theta.get_vector(base, size)
                red_old = self.momentum * old
                rate = self.gamma.expand_to_vector(size)
                nabla_vector = nabla.get_vector(base, size)
                log_batch_size = math.log(len(batch), 2)
                # divide by len(batch) by truncation
                # increased rate if len(batch) is not a power of two
                pre_trunc = nabla_vector.v * rate.v
                k = max(nabla_vector.k, rate.k) + rate.f
                m = rate.f + int(log_batch_size)
                if self.early_division:
                    v = pre_trunc
                else:
                    v = pre_trunc.round(k, m, signed=True,
                                        nearest=sfix.round_nearest)
                new = nabla_vector._new(v)
                diff = red_old - new
                delta_theta.assign_vector(diff, base)
                theta.assign_vector(theta.get_vector(base, size) +
                                    delta_theta.get_vector(base, size), base)
            if self.print_update_average:
                vec = abs(delta_theta.get_vector().reveal())
                print_ln('update average: %s (%s)',
                         sum(vec) * cfix(1 / len(vec)), len(vec))
            if self.debug:
                limit = int(self.debug)
                d = delta_theta.get_vector().reveal()
                aa = [cfix.Array(len(d.v)) for i in range(3)]
                a = aa[0]
                a.assign(d)
                @for_range(len(a))
                def _(i):
                    x = a[i]
                    print_ln_if((x > limit) + (x < -limit),
                                'update epoch=%s %s index=%s %s',
                                i_epoch.read(), str(delta_theta), i, x)
                a = aa[1]
                a.assign(nabla.get_vector().reveal())
                @for_range(len(a))
                def _(i):
                    x = a[i]
                    print_ln_if((x > len(batch) * limit) + (x < -len(batch) * limit),
                                'nabla epoch=%s %s index=%s %s',
                                i_epoch.read(), str(nabla), i, x)
                a = aa[2]
                a.assign(theta.get_vector().reveal())
                @for_range(len(a))
                def _(i):
                    x = a[i]
                    print_ln_if((x > limit) + (x < -limit),
                                'theta epoch=%s %s index=%s %s',
                                i_epoch.read(), str(theta), i, x)
            if self.print_random_update:
                print_ln('update')
                l = min(100, nabla.total_size())
                if l < 100:
                    index = 0
                else:
                    index = regint.get_random(64) % (nabla.total_size() - l)
                print_ln('%s at %s: nabla=%s update=%s theta=%s', str(theta),
                         index, nabla.to_array().get_vector(index, l).reveal(),
                         delta_theta.to_array().get_vector(index, l).reveal(),
                         theta.to_array().get_vector(index, l).reveal())
        self.gamma.imul(1 - 10 ** - 6)

def apply_padding(input_shape, kernel_size, strides, padding):
    """
    Calculate the output shape of a convolution operation given the input shape, kernel size, strides, and padding.

    Parameters:
    input_shape (tuple): The shape of the input volume as (height, width).
    kernel_size (tuple): The size of the convolution kernel as (height, width).
    strides (tuple): The strides of the convolution operation as (stride_height, stride_width).
    padding (int, tuple, or str): The padding added to all four sides of the input. If an integer, the same value
                                  is used for all spatial dimensions. If a tuple, it must contain two integers
                                  (padding_height, padding_width). If 'valid', no padding is added (only valid
                                  convolutions). If 'same', padding is added to keep the output shape the same
                                  as the input shape (padding may be added).

    Returns:
    tuple: The output shape of the convolution operation as (output_height, output_width).

    Raises:
    Exception: If the padding type is not recognized or if the resulting output dimensions are not positive.
    """
    if isinstance(padding, int):
        padding = [padding, padding]
    if isinstance(padding, (tuple, list)):
        input_shape = [x + sum(padding) for x in input_shape]
        padding = 'valid'
    if padding.lower() == 'valid':
        res = (input_shape[0] - kernel_size[0] + 1) // strides[0], \
            (input_shape[1] - kernel_size[1] + 1) // strides[1],
        assert min(res) > 0, (input_shape, kernel_size, strides, padding)
        return res
    elif padding.lower() == 'same':
        return (input_shape[0]) // strides[0], \
            (input_shape[1]) // strides[1],
    else:
        raise Exception('invalid padding: %s' % padding)

class keras:
    class layers:
        Flatten = lambda *args, **kwargs: ('flatten', args, kwargs)
        Dense = lambda *args, **kwargs: ('dense', args, kwargs)

        def Conv2D(filters, kernel_size, strides=(1, 1), padding='valid',
                   activation=None, input_shape=None):
            return 'conv2d', {'filters': filters, 'kernel_size': kernel_size,
                              'strides': strides, 'padding': padding,
                              'activation': activation}

        def MaxPooling2D(pool_size=2, strides=None, padding='valid'):
            return 'maxpool', {'pool_size': pool_size, 'strides': strides,
                               'padding': padding}

        def AveragePooling2D(pool_size=2, strides=None, padding='valid'):
            return 'avgpool', {'filter_size': pool_size, 'strides': strides,
                               'padding': padding}

        def Dropout(rate):
            l = math.log(rate, 2)
            if int(l) != l:
                raise Exception('rate needs to be a power of two')
            return 'dropout', rate

        def Activation(activation):
            assert(activation == 'relu')
            return activation,

        def BatchNormalization():
            return 'batchnorm',

    class optimizers:
        SGD = lambda *args, **kwargs: ('sgd', args, kwargs)
        Adam = lambda *args, **kwargs: ('adam', args, kwargs)

    class models:
        class Sequential:
            def __init__(self, layers):
                self.layers = layers
                self.optimizer = None
                self.opt = None

            def compile(self, optimizer):
                self.optimizer = optimizer

            def compile_by_args(self, program):
                if 'adam' in program.args:
                    self.optimizer = 'adam', [], {}
                elif 'amsgrad' in program.args:
                    self.optimizer = 'adam', [], {'amsgrad': True}
                elif 'amsgradprec' in program.args:
                    self.optimizer = 'adam', [], {'amsgrad': True,
                                                  'approx': False}
                else:
                    self.optimizer = 'sgd', [], {}

            @property
            def trainable_variables(self):
                if self.opt == None:
                    raise Exception('need to run build() or fit() first')
                return list(self.opt.thetas)

            def summary(self):
                self.opt.summary()

            def build(self, input_shape, batch_size=128):
                data_input_shape = input_shape
                if self.opt != None and \
                   input_shape == self.opt.layers[0]._X.sizes and \
                   batch_size <= self.batch_size and \
                   type(self.opt).__name__.lower() == self.optimizer[0]:
                    return
                if self.optimizer == None:
                    self.optimizer = 'inference', [], {}
                if input_shape == None:
                    raise Exception('must specify number of samples')
                Layer.back_batch_size = batch_size
                layers = []
                for i, layer in enumerate(self.layers):
                    name = layer[0]
                    if name == 'dense':
                        if len(layers) == 0:
                            N = input_shape[0]
                            n_units = reduce(operator.mul, input_shape[1:])
                        else:
                            N = batch_size
                            n_units = reduce(operator.mul,
                                             layers[-1].Y.sizes[1:])
                        if i == len(self.layers) - 1:
                            activation = layer[2].get('activation', None)
                            if activation in ('softmax', 'sigmoid'):
                                layer[2].pop('activation', None)
                            if activation == 'softmax' and layer[1][0] == 1:
                                raise CompilerError(
                                    'softmax requires more than one output neuron')
                        layers.append(Dense(N, n_units, layer[1][0],
                                            **layer[2]))
                        input_shape = layers[-1].Y.sizes
                    elif name == 'conv2d':
                        input_shape = list(input_shape) + \
                            [1] * (4 - len(input_shape))
                        print (layer[1])
                        kernel_size = layer[1]['kernel_size']
                        filters = layer[1]['filters']
                        strides = layer[1]['strides']
                        padding = layer[1]['padding']
                        layers.append(easyConv2d(
                            input_shape, batch_size, filters, kernel_size,
                            strides, padding))
                        output_shape = layers[-1].Y.sizes
                        input_shape = output_shape
                        print('conv output shape', output_shape)
                    elif name == 'maxpool':
                        pool_size = layer[1]['pool_size']
                        strides = layer[1]['strides']
                        padding = layer[1]['padding']
                        layers.append(easyMaxPool(input_shape, pool_size,
                                                  strides, padding))
                        input_shape = layers[-1].Y.sizes
                    elif name == 'avgpool':
                        layers.append(FixAveragePool2d(input_shape, None, **layer[1]))
                        input_shape = layers[-1].Y.sizes
                    elif name == 'dropout':
                        layers.append(Dropout(batch_size, reduce(
                            operator.mul, layers[-1].Y.sizes[1:]),
                                              alpha=layer[1]))
                        input_shape = layers[-1].Y.sizes
                    elif name == 'flatten':
                        pass
                    elif name == 'relu':
                        layers.append(Relu(layers[-1].Y.sizes))
                    elif name == 'batchnorm':
                        input_shape = layers[-1].Y.sizes
                        layers.append(BatchNorm(layers[-1].Y.sizes))
                    else:
                        raise Exception(layer[0] + ' not supported')
                if layers[-1].d_out == 1:
                    layers.append(Output(data_input_shape[0]))
                else:
                    layers.append(
                        MultiOutput(data_input_shape[0], layers[-1].d_out))
                if self.optimizer[1]:
                    raise Exception('use keyword arguments for optimizer')
                opt = self.optimizer[0]
                opts = self.optimizer[2]
                if opt == 'sgd':
                    opt = SGD(layers, 1)
                    momentum = opts.pop('momentum', None)
                    if momentum != None:
                        opt.momentum = momentum
                elif opt == 'adam':
                    opt = Adam(layers, amsgrad=opts.pop('amsgrad', None),
                               approx=opts.pop('approx', True))
                    beta1 = opts.pop('beta_1', None)
                    beta2 = opts.pop('beta_2', None)
                    epsilon = opts.pop('epsilon', None)
                    if beta1 != None:
                        opt.beta1 = beta1
                    if beta2:
                        opt.beta2 = beta2
                    if epsilon:
                        if epsilon < opt.epsilon:
                            print('WARNING: epsilon smaller than default might '
                                  'cause overflows')
                        opt.epsilon = epsilon
                elif opt == 'inference':
                    opt = Optimizer()
                    opt.layers = layers
                else:
                    raise Exception(opt + ' not supported')
                lr = opts.pop('learning_rate', None)
                if lr != None:
                    opt.set_learning_rate(lr)
                if opts:
                    raise Exception(opts + ' not supported')
                self.batch_size = batch_size
                self.opt = opt

            def fit(self, x, y, batch_size, epochs=1, validation_data=None):
                print("###########begin model's fit")
                assert len(x) == len(y)
                self.build(x.sizes, batch_size)
                if x.total_size() != self.opt.layers[0]._X.total_size():
                    raise Exception('sample data size mismatch')
                if y.total_size() != self.opt.layers[-1].Y.total_size():
                    print (y, self.opt.layers[-1].Y)
                    raise Exception('label size mismatch')
                if validation_data == None:
                    validation_data = None, None
                else:
                    if len(validation_data[0]) != len(validation_data[1]):
                        raise Exception('test set size mismatch')
                self.opt.layers[0]._X.address = x.address
                self.opt.layers[-1].Y.address = y.address
                self.opt.run_by_args(get_program(), epochs, batch_size,
                                     validation_data[0], validation_data[1],
                                     batch_size)
                return self.opt

            def predict(self, x, batch_size=None):
                if self.opt == None:
                    raise Exception('need to run fit() or build() first')
                if batch_size != None:
                    batch_size = min(batch_size, self.batch_size)
                return self.opt.eval(x, batch_size=batch_size)

class SGDLogistic(OneLayerSGD):
    """ Logistic regression using SGD.

    :param n_epochs: number of epochs
    :param batch_size: batch size
    :param program: program object to use command-line options from (default is
      not to use any)

    """
    print_accuracy = True

    def init(self, X):
        dense = Dense(*X.sizes, 1)
        if self.program:
            sigmoid = Output.from_args(X.sizes[0], self.program)
            self.opt = Optimizer.from_args(self.program, [dense, sigmoid])
        else:
            sigmoid = Output(X.sizes[0])
            self.opt = SGD([dense, sigmoid], 1)

    def predict(self, X):
        """ Use model to predict labels.

        :param X: sample data with row-wise samples (sfix matrix)
        :returns: sint array

        """
        return self.opt.eval(X, top=True)

    def predict_proba(self, X):
        """ Use model for probility estimates.

        :param X: sample data with row-wise samples (sfix matrix)
        :returns: sfix array

        """
        return super(SGDLogistic, self).predict(X)

class SGDLinear(OneLayerSGD):
    """ Linear regression using SGD.

    :param n_epochs: number of epochs
    :param batch_size: batch size
    :param program: program object to use command-line options from (default is
      not to use any)

    """
    print_accuracy = False

    def init(self, X):
        dense = Dense(*X.sizes, 1)
        output = LinearOutput(X.sizes[0])
        if self.program:
            self.opt = Optimizer.from_args(self.program, [dense, output])
        else:
            self.opt = SGD([dense, output], 1)

def solve_linear(A, b, n_iterations, progress=False, n_threads=None,
                 stop=False, already_symmetric=False, precond=False):
    """ Iterative linear solution approximation for :math:`Ax=b`.

    :param progress: print some information on the progress (implies revealing)
    :param n_threads: number of threads to use
    :param stop: whether to stop when converged (implies revealing)

    """
    assert len(b) == A.sizes[0]
    x = sfix.Array(A.sizes[1])
    x.assign_vector(sfix.get_random(-1, 1, size=len(x)))
    if already_symmetric:
        AtA = A
        r = Array.create_from(b - AtA * x)
    else:
        AtA = sfix.Matrix(len(x), len(x))
        A.trans_mul_to(A, AtA, n_threads=n_threads)
        r = Array.create_from(A.transpose() * b - AtA * x)
    if precond:
        return solve_linear_diag_precond(AtA, b, x, r, n_iterations,
                                         progress, stop)
    v = sfix.Array(A.sizes[1])
    v.assign_all(0)
    Av = sfix.Array(len(x))
    @for_range(n_iterations)
    def _(i):
        v[:] = r - sfix.dot_product(r, Av) / sfix.dot_product(v, Av) * v
        Av[:] = AtA * v
        v_norm = sfix.dot_product(v, Av)
        vr = sfix.dot_product(v, r)
        alpha = (v_norm == 0).if_else(0, vr / v_norm)
        x[:] = x + alpha * v
        r[:] = r - alpha * Av
        if progress:
            print_ln('%s alpha=%s vr=%s v_norm=%s', i, alpha.reveal(),
                     vr.reveal(), v_norm.reveal())
        if stop:
            return (alpha > 0).reveal()
    if not already_symmetric:
        AtA.delete()
    return x

def solve_linear_diag_precond(A, b, x, r, n_iterations, progress=False,
                              stop=False):
    """
    Solve a linear system Ax = b using the Conjugate Gradient method with
    diagonal preconditioning.

    Parameters:
    - A: A matrix representing the system.
    - b: The right-hand side vector of the linear system.
    - x: The initial guess for the solution vector.
    - r: The initial residual vector, which should be b - Ax.
    - n_iterations: The number of iterations to perform.
    - progress: A boolean indicating whether to print progress information.
    - stop: A boolean indicating whether to stop if the step size (alpha) is positive.

    Returns:
    - x: The approximate solution to the linear system after n_iterations.
    """
    m = 1 / A.diag()
    mr = Array.create_from(m * r[:])
    d = Array.create_from(mr)
    @for_range(n_iterations)
    def _(i):
        Ad = A * d
        d_norm = sfix.dot_product(d, Ad)
        alpha = (d_norm == 0).if_else(0, sfix.dot_product(r, mr) / d_norm)
        x[:] = x[:] + alpha * d[:]
        r_norm = sfix.dot_product(r, mr)
        r[:] = r[:] - alpha * Ad
        tmp = m * r[:]
        beta = (r_norm == 0).if_else(0, sfix.dot_product(r, tmp) / r_norm)
        mr[:] = tmp
        d[:] = tmp + beta * d
        if progress:
            print_ln('%s alpha=%s beta=%s r_norm=%s d_norm=%s', i,
                     alpha.reveal(), beta.reveal(), r_norm.reveal(),
                     d_norm.reveal())
        if stop:
            return (alpha > 0).reveal()
    return x

def mr(A, n_iterations, stop=False):
    """ Iterative matrix inverse approximation.

    :param A: matrix to invert
    :param n_iterations: maximum number of iterations
    :param stop: whether to stop when converged (implies revealing)

    """
    assert len(A.sizes) == 2
    assert A.sizes[0] == A.sizes[1]
    M = A.same_shape()
    n = A.sizes[0]
    @for_range(n)
    def _(i):
        e = sfix.Array(n)
        e.assign_all(0)
        e[i] = 1
        M[i] = solve_linear(A, e, n_iterations, stop=stop)
    return M.transpose()

def var(x):
    """ Variance. """
    mean = MemValue(type(x[0])(0))
    @for_range_opt(len(x))
    def _(i):
        mean.iadd(x[i])
    mean /= len(x)
    res = MemValue(type(x[0])(0))
    @for_range_opt(len(x))
    def _(i):
        res.iadd((x[i] - mean.read()) ** 2)
    return res.read()

def cholesky(A, reveal_diagonal=False):
    """ Cholesky decomposition.

    :returns: lower triangular matrix

    """
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    L = A.same_shape()
    L.assign_all(0)
    diag_inv = A.value_type.Array(A.shape[0])
    @for_range(A.shape[0])
    def _(i):
        @for_range(i + 1)
        def _(j):
            sum = sfix.dot_product(L[i], L[j])

            @if_e(i == j)
            def _():
                L[i][j] = mpc_math.sqrt(A[i][i] - sum)
                diag_inv[i] = 1 / L[i][j]
                if reveal_diagonal:
                    print_ln('L[%s][%s] = %s = sqrt(%s - %s)', i, j,
                             L[i][j].reveal(), A[i][j].reveal(), sum.reveal())
            @else_
            def _():
                L[i][j] = (diag_inv[j] * (A[i][j] - sum))
    return L

def solve_lower(A, b):
    """ Linear solver where :py:obj:`A` is lower triangular quadratic. """
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert len(b) == A.shape[0]
    b = Array.create_from(b)
    res = sfix.Array(len(b))
    @for_range(len(b))
    def _(i):
        res[i] = b[i] / A[i][i]
        b[:] -= res[i] * A.get_column(i)
    return res

def solve_upper(A, b):
    """ Linear solver where :py:obj:`A` is upper triangular quadratic. """
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert len(b) == A.shape[0]
    b = Array.create_from(b)
    res = sfix.Array(len(b))
    @for_range(len(b) - 1, -1, -1)
    def _(i):
        res[i] = b[i] / A[i][i]
        b[:] -= res[i] * A.get_column(i)
    return res

def solve_cholesky(A, b, debug=False):
    """ Linear solver using Cholesky decomposition. """
    L = cholesky(A, reveal_diagonal=debug)
    if debug:
        Optimizer.stat('L', L)
    x = solve_lower(L, b)
    if debug:
        Optimizer.stat('intermediate', x)
    return solve_upper(L.transpose(), x)
