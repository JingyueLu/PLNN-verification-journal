from torch import nn 
import torch
from plnn.modules import View, Flatten
from torch.nn.parameter import Parameter
from plnn.model import simplify_network
from exp_utils.mnist_utils import flatten_layers
import random
import copy
import json

'''
This file contains all model structures we have considered
'''

## original kw small model
## 14x14x16 (3136) --> 7x7x32 (1568) --> 100 --> 10 ----(4804 ReLUs)
def mnist_model(): 
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

## 14*14*8 (1568) --> 14*14*8 (1568) --> 14*14*8 (1568) --> 392 --> 100 (5196 ReLUs)
def mnist_model_deep():
    model = nn.Sequential(
        nn.Conv2d(1, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

# first medium size model 14x14x4 (784) --> 7x7x8 (392) --> 50 --> 10 ----(1226 ReLUs)
# robust error 0.068
def mnist_model_m1():
    model = nn.Sequential(
        nn.Conv2d(1, 4, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*7*7,50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    return model


# increase the mini model by increasing the number of channels
## 8x8x8 (512) --> 4x4x16 (256) --> 50 (50) --> 10 (818)
def mini_mnist_model_m1():
    model = nn.Sequential(
        nn.Conv2d(1, 8, 2, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 2, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(4*4*16,50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    return model


# without the extra 50-10 layer (originally, directly 128-10, robust error is around 0.221)
## 8x8x4 (256) --> 4x4x8 (128) --> 50 --> 10 ---- (434 ReLUs)
def mini_mnist_model(): 
    model = nn.Sequential(
        nn.Conv2d(1, 4, 2, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 8, 2, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*4*4,50),
        nn.ReLU(),
        nn.Linear(50,10),
    )
    return model

#### CIFAR

# 32*32*32 (32768) --> 32*16*16 (8192) --> 64*16*16 (16384) --> 64*8*8 (4096) --> 512 --> 512 
# 54272 ReLUs
def large_cifar_model(): 
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*8*8,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model

# 16*16*16 (4096) --> 32*8*8 (2048) --> 100 
# 6244 ReLUs
# wide model
def cifar_model():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

# 16*16*8 (2048) -->  16*16*8 (2048) --> 16*16*8 (2048) --> 512 --> 100
# 6756 ReLUs
#deep model
def cifar_model_deep():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*8*8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

# 16*16*8 (2048) --> 16*8*8 (1024) --> 100 
# 3172 ReLUs (small model)
def cifar_model_m2():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(16*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

# 16*16*4 (1024) --> 8*8*8 (512) --> 100 
def cifar_model_m1(): 
    model = nn.Sequential(
        nn.Conv2d(3, 4, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*8*8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def max_pool(candi_tot, lb_abs, change_sign=False):
    '''
    diff layer is provided when simplify linear layers are required
    by providing linear layers, we reduce consecutive linear layers
    to one
    '''
    layers = []
    # perform max-pooling
    # max-pooling is performed in terms of paris.
    # Each loop iteration reduces the number of candidates by two
    while candi_tot > 1:
        temp = list(range(0, candi_tot//2))
        even = [2*i for i in temp]
        odd = [i+1 for i in even]
        max_pool_layer1 = nn.Linear(candi_tot, candi_tot, bias=True)
        weight_mp_1 = torch.eye(candi_tot)
        weight_mp_1[even,odd] = -1
        bias_mp_1 = torch.zeros(candi_tot)
        bias_mp_1[odd] = -lb_abs
        bias_mp_1[-1] = -lb_abs
        #import pdb; pdb.set_trace()
        max_pool_layer1.weight = Parameter(weight_mp_1, requires_grad=False)
        max_pool_layer1.bias = Parameter(bias_mp_1, requires_grad=False)
        layers.append(max_pool_layer1)
        layers.append(nn.ReLU())
        new_candi_tot = (candi_tot+1)//2
        sum_layer = nn.Linear(candi_tot, new_candi_tot, bias=True)
        sum_layer_weight = torch.zeros([new_candi_tot, candi_tot])
        sum_layer_weight[temp,even]=1; sum_layer_weight[temp,odd]=1
        sum_layer_weight[-1][-1] = 1
        sum_layer_bias = torch.zeros(new_candi_tot)
        sum_layer_bias[temp]= lb_abs
        sum_layer_bias[-1] = lb_abs
        if change_sign is True and new_candi_tot==1:
            sum_layer.weight = Parameter(-1*sum_layer_weight, requires_grad=False)
            sum_layer.bias = Parameter(-1*sum_layer_bias, requires_grad=False)
        else:
            sum_layer.weight = Parameter(sum_layer_weight, requires_grad=False)
            sum_layer.bias = Parameter(sum_layer_bias, requires_grad=False)
        layers.append(sum_layer)

        pre_candi_tot = candi_tot
        candi_tot = new_candi_tot

        #import pdb; pdb.set_trace()

    return layers



def add_properties(model, true_label, lb_abs = -1000, ret_ls = False):
    '''
    Input: pre_trained models
    Output: net layers with the mnist verification property added
    '''
    for p in model.parameters():
        p.requires_grad =False

    layers = list(model.children())
    last_layer = layers[-1]
    diff_in = last_layer.out_features
    diff_out = last_layer.out_features-1
    diff_layer = nn.Linear(diff_in,diff_out, bias=True)
    temp_weight_diff = torch.eye(10)
    temp_weight_diff[:,true_label] -= 1
    all_indices = list(range(10))
    all_indices.remove(true_label)
    weight_diff = temp_weight_diff[all_indices]
    bias_diff = torch.zeros(9)
    
    diff_layer.weight = Parameter(weight_diff, requires_grad=False)
    diff_layer.bias = Parameter(bias_diff, requires_grad=False)
    layers.append(diff_layer)
    candi_tot = diff_out
    # since what we are actually interested in is the minium of gt-cls,
    # we revert all the signs of the last layer
    max_pool_layers = max_pool(candi_tot, lb_abs, change_sign=True)
    # simplify linear layers
    simp_required_layers = layers[-2:]+max_pool_layers
    simplified_layers = simplify_network(simp_required_layers)
    final_layers = layers[:-2]+simplified_layers
    if ret_ls is False:
        return final_layers
    else:
        return [layers[:-2], simplified_layers]




def load_cifar_1to1_layers_dc(model):
    '''
    return a dictionary of fixed network layers and 
    final layers incorporating different properties
    '''
    if model=='cifar_kw':
        model_name = '/home/jodie/PLNN/PLNN-verification-journal/models/cifar_small_kw.pth'
        model = cifar_model()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='cifar_kw_m1':
        model_name = '/home/jodie/PLNN/PLNN-verification-journal/models/cifar_m1_kw.pth'
        model = cifar_model_m1()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='cifar_kw_m2':
        model_name = '/home/jodie/PLNN/PLNN-verification-journal/models/cifar_m2_kw.pth'
        model = cifar_model_m2()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='cifar_kw_wide':
        model_name = '/home/jodie/PLNN/PLNN-verification-journal/models/cifar_small_kw.pth'
        model = cifar_model()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='cifar_kw_deep':
        model_name = '/home/jodie/PLNN/PLNN-verification-journal/models/cifar_deep_kw.pth'
        model = cifar_model_deep()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    else:
        raise NotImplementedError

    layers_dc = {}
    
    layers = list(model.children())
    layers_dc['fixed_layers'] = layers[:-1]
    for y_pred in range(10):
        for test in range(10):
            if y_pred == test:
                continue
            else:
                added_prop_layers = add_single_prop(layers, y_pred, test)
                layers_dc[f'pred_{y_pred}_against_{test}'] = added_prop_layers[-1]
    return layers_dc
    


def load_mnist_1to1_layers_dc(model):
    '''
    return a dictionary of fixed network layers and 
    final layers incorporating different properties
    '''
    if model=='kw_m1':
        model_name = '/home/jodie/PLNN/PLNN-verification-journal/models/m1.pth'
        model = mnist_model_m1()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='madry_med':
        model_name = '/home/jodie/PLNN/PLNN-verification-journal/models/madry_sgd_slr_0.001.pth'
        model = mnist_model_m1()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='kw_large':
        model_name = '/home/jodie/PLNN/PLNN-verification-journal/models/small.pth'
        model = mnist_model()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    else:
        raise NotImplementedError

    layers_dc = {}
    
    layers = list(model.children())
    layers_dc['fixed_layers'] = layers[:-1]
    for y_pred in range(10):
        for test in range(10):
            if y_pred == test:
                continue
            else:
                added_prop_layers = add_single_prop(layers, y_pred, test)
                layers_dc[f'pred_{y_pred}_against_{test}'] = added_prop_layers[-1]
    return layers_dc



##### ALL the models we tested on 

def load_med_model(true_label, lb_abs= -1000, model=None):
    if model is None:
        model_name = '/home/jodie/PLNN/PLNN-verification-journal/models/m1.pth'
        model = mnist_model_m1()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])

    added_prop_layers = add_properties(model, true_label, lb_abs)
    for layer in added_prop_layers:
        for p in layer.parameters():
            p.requires_grad = False
    return added_prop_layers

def load_med_exp(idx, lb_abs=-1000, model=None, mnist_test = None, gurobi=False):
    if model is None:
        model_name = '/home/jodie/PLNN/PLNN-verification-journal/models/m1.pth'
        model = mnist_model_m1()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    if mnist_test is None:
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        mnist_test = datasets.MNIST("/home/jodie/PLNN/PLNN-verification-journal/data", train=False, download=True, transform =transforms.ToTensor())

    x,y = mnist_test[idx]
    x = x.unsqueeze(0)
    # first check the model is correct at the input
    y_pred = torch.max(model(x)[0], 0)[1].item()
    if  y_pred != y.item(): 
        print('model prediction is incorrect for the given model')
        return None
    else: 
        added_prop_layers = add_properties(model, y.item(), lb_abs)
        for layer in added_prop_layers:
            for p in layer.parameters():
                p.requires_grad = False
        ## load gurobi model
        if gurobi:
            elided_models = make_elided_models(model)
            gurobi_layers = elided_models[y_pred]
            return added_prop_layers,x, gurobi_layers
        else:
            return added_prop_layers, x, None


def load_1to1_exp(model, idx, test = None, mnist_test = None):
    if model=='kw_m1':
        model_name = '/home/jodie/PLNN/PLNN-verification-journal/models/m1.pth'
        model = mnist_model_m1()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='madry_med':
        model_name = '/home/jodie/PLNN/PLNN-verification-journal/models/madry_sgd_slr_0.001.pth'
        model = mnist_model_m1()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='kw_large':
        model_name = '/home/jodie/PLNN/PLNN-verification-journal/models/small.pth'
        model = mnist_model()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    else:
        raise NotImplementedError
    if mnist_test is None:
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        mnist_test = datasets.MNIST("/home/jodie/PLNN/PLNN-verification-journal/data", train=False, download=True, transform =transforms.ToTensor())

    x,y = mnist_test[idx]
    x = x.unsqueeze(0)
    # first check the model is correct at the input
    y_pred = torch.max(model(x)[0], 0)[1].item()
    print('predicted label ', y_pred, ' correct label ', y.item())
    if  y_pred != y.item(): 
        print('model prediction is incorrect for the given model')
        return None
    else: 
        if test ==None:
            choices = list(range(10))
            choices.remove(y_pred)
            test = random.choice(choices)

        print('tested against ',test)
        for p in model.parameters():
            p.requires_grad =False

        layers = list(model.children())
        added_prop_layers = add_single_prop(layers, y_pred, test)
        return x, added_prop_layers, test

    
def load_large_exp(idx, lb_abs=-1000, model=None, mnist_test = None, gurobi=False):
    if model is None:
        model_name = '/home/jodie/PLNN/PLNN-verification-journal/models/small.pth'
        model = mnist_model()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    if mnist_test is None:
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        mnist_test = datasets.MNIST("/home/jodie/PLNN/PLNN-verification-journal/data", train=False, download=True, transform =transforms.ToTensor())

    x,y = mnist_test[idx]
    x = x.unsqueeze(0)
    # first check the model is correct at the input
    y_pred = torch.max(model(x)[0], 0)[1].item()
    if  y_pred != y.item(): 
        print('model prediction is incorrect for the given model')
        return None
    else: 
        added_prop_layers = add_properties(model, y.item(), lb_abs)
        for layer in added_prop_layers:
            for p in layer.parameters():
                p.requires_grad = False
        ## load gurobi model
        if gurobi:
            elided_models = make_elided_models(model)
            gurobi_layers = elided_models[y_pred]
            return added_prop_layers,x, gurobi_layers
        else:
            return added_prop_layers, x, None


def load_mini_exp():
    # load model
    model_name = '/home/jodie/PLNN/PLNN-verification-journal/models/mini.pth'
    model =mini_mnist_model().cpu()
    model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    # load data
    mini_mnist_test = torch.load('/home/jodie/PLNN/PLNN-verification-journal/mini_data/mini_test.pt')
    # load testing eps
    with open('/home/jodie/PLNN/PLNN-verification-journal/models/all_candidate_props_eps.json', 'r') as f:
        all_eps = json.load(f)

    return model, mini_mnist_test, all_eps


def add_single_prop(layers, gt, cls):
    '''
    gt: ground truth lable
    cls: class we want to verify against
    '''
    additional_lin_layer = nn.Linear(10, 1, bias=True)
    lin_weights = additional_lin_layer.weight.data
    lin_weights.fill_(0)
    lin_bias = additional_lin_layer.bias.data
    lin_bias.fill_(0)
    lin_weights[0, cls] = -1
    lin_weights[0, gt] = 1

    #verif_layers2 = flatten_layers(verif_layers1,[1,14,14])
    final_layers = [layers[-1], additional_lin_layer]
    final_layer  = simplify_network(final_layers)
    verif_layers = layers[:-1] + final_layer
    for layer in verif_layers:
        for p in layer.parameters():
            p.requires_grad = False

    return verif_layers

################ Madry reduced MNIST networks

    
################ Madry large CIFAR networks

def load_cifar_madry_exp(idx, model_name,  lb_abs=-1000):
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    # load the model
    model_name = f'/home/jodie/PLNN/PLNN-verification-journal/models/{model_name}'
    model = cifar_model()
    model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    # load the data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    #train = datasets.CIFAR10('./cifardata', train=True, download=True,
    #    transform=transforms.Compose([
    #        transforms.RandomHorizontalFlip(),
    #        transforms.RandomCrop(32, 4),
    #        transforms.ToTensor(),
    #        normalize,
    #    ]))
    test = datasets.CIFAR10('/home/jodie/PLNN/PLNN-verification-journal/cifardata/', train=False,transform=transforms.Compose([transforms.ToTensor(), normalize]))

    x,y = test[idx]
    x = x.unsqueeze(0)
    # first check the model is correct at the input
    y_pred = torch.max(model(x)[0], 0)[1].item()
    if  y_pred != y: 
        print('model prediction is incorrect for the given model')
        return None
    else: 
        added_prop_layers = add_properties(model, y, lb_abs)
        for layer in added_prop_layers:
            for p in layer.parameters():
                p.requires_grad = False
        return added_prop_layers,x 



def load_cifar_1to1_exp(model, idx, test = None, cifar_test = None):
    if model=='cifar_madry':
        model_name = '/home/jodie/PLNN/PLNN-verification-journal/models/cifar_madry_8px.pth'
        model = cifar_model()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='cifar_madry_wide':
        model_name = '/home/jodie/PLNN/PLNN-verification-journal/models/cifar_madry_wide_new.pth'
        model = cifar_model()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='cifar_kw_m1':
        model_name = '/home/jodie/PLNN/PLNN-verification-journal/models/cifar_m1_kw.pth'
        model = cifar_model_m1()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='cifar_kw_m2':
        model_name = '/home/jodie/PLNN/PLNN-verification-journal/models/cifar_m2_kw.pth'
        model = cifar_model_m2()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='cifar_kw_wide':
        model_name = '/home/jodie/PLNN/PLNN-verification-journal/models/cifar_small_kw.pth'
        model = cifar_model()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='cifar_kw_deep':
        model_name = '/home/jodie/PLNN/PLNN-verification-journal/models/cifar_deep_kw.pth'
        model = cifar_model_deep()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    else:
        raise NotImplementedError
    if cifar_test is None:
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.225, 0.225, 0.225])
        cifar_test = datasets.CIFAR10('/home/jodie/PLNN/PLNN-verification-journal/cifardata/', train=False,transform=transforms.Compose([transforms.ToTensor(), normalize]))

    x,y = cifar_test[idx]
    x = x.unsqueeze(0)
    # first check the model is correct at the input
    y_pred = torch.max(model(x)[0], 0)[1].item()
    print('predicted label ', y_pred, ' correct label ', y)
    if  y_pred != y: 
        print('model prediction is incorrect for the given model')
        return None, None, None
    else: 
        if test ==None:
            choices = list(range(10))
            choices.remove(y_pred)
            test = random.choice(choices)

        print('tested against ',test)
        for p in model.parameters():
            p.requires_grad =False

        layers = list(model.children())
        added_prop_layers = add_single_prop(layers, y_pred, test)
        return x, added_prop_layers, test





################### difference model that is used for Gurobi New Encoding



    

def make_elided_models(model, return_error=False):
    """
    Default is to return GT - other
    Set `return_error` to True to get instead something that returns a loss
    (other - GT)

    mono_output=False is an argument I removed
    """
    elided_models = []
    layers = [lay for lay in model]
    assert isinstance(layers[-1], nn.Linear)

    net = layers[:-1]
    last_layer = layers[-1]
    nb_classes = last_layer.out_features

    for gt in range(nb_classes):
        new_layer = nn.Linear(last_layer.in_features,
                              last_layer.out_features-1)

        wrong_weights = last_layer.weight[[f for f in range(last_layer.out_features) if f != gt], :]
        wrong_biases = last_layer.bias[[f for f in range(last_layer.out_features) if f != gt]]

        if return_error:
            new_layer.weight.data.copy_(wrong_weights - last_layer.weight[gt])
            new_layer.bias.data.copy_(wrong_biases - last_layer.bias[gt])
        else:
            new_layer.weight.data.copy_(last_layer.weight[gt] - wrong_weights)
            new_layer.bias.data.copy_(last_layer.bias[gt] - wrong_biases)

        layers = copy.deepcopy(net) + [new_layer]
        # if mono_output and new_layer.out_features != 1:
        #     layers.append(View((1, new_layer.out_features)))
        #     layers.append(nn.MaxPool1d(new_layer.out_features,
        #                                stride=1))
        #     layers.append(View((1,)))
        new_elided_model = nn.Sequential(*layers)
        elided_models.append(new_elided_model)
    return elided_models
