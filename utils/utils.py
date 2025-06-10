import random
import os

from torch import inf
import torch
import numpy as np

from torch.utils.data.sampler import Sampler
import torch.optim as optim

import yaml
from argparse import Namespace

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from collections import OrderedDict
from torch.utils.data.dataloader import default_collate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ordered_yaml():
    """
    yaml orderedDict support
    """
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def pad_batch(data, batch, max_input_len=np.inf, get_mask=True, seed=None):
    """
        input:
        h_node: node features (N,dim)
        batch: batch information from pyg (N,1)
        max_input_len: max node num per batch (1)

        output:
        padded_h_node: batched features (b,n,dim)
        src_padding_mask: (b,n) 标记哪几个节点是有效节点

        num_nodes: 节点数量
        masks: bool mask list [(n,1), (n,1)]
        max_num_nodes
    """
    num_batch = batch[-1] + 1
    num_nodes = []
    masks = []
    for i in range(num_batch):
        mask = batch.eq(i)  # n*1 bool掩码
        masks.append(mask)
        num_node = mask.sum()
        num_nodes.append(num_node)

    max_num_nodes = min(max(num_nodes), max_input_len)
    padded_h_node = data.new(num_batch, max_num_nodes, data.size(-1)).fill_(0)  # b * n * k
    src_padding_mask = data.new(num_batch, max_num_nodes).fill_(0).bool()  # b * n

    for i, mask in enumerate(masks):
        if num_node > max_num_nodes:
            # 随机采样node
            random_indices = torch.randperm(num_node)
            selected_indices = random_indices[:max_num_nodes]
            selected_indices.sort()
            padded_h_node[i, :max_num_nodes] = data[mask][selected_indices]
        else:
            padded_h_node[i, :num_node] = data[mask][:num_node]
        src_padding_mask[i, :num_node] = True  # [b, s]

    if get_mask:
        return padded_h_node, src_padding_mask, num_nodes, masks, max_num_nodes
    return padded_h_node, src_padding_mask


def my_list_collate_fn(batch):
    batch = [x for x in batch if x is not None]
    transposed_batch = list(zip(*batch))
    collated_data = [list(items) for items in transposed_batch]
    return collated_data


def my_default_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return default_collate(batch)


def get_optim(model, args, lr=None):
    if lr is None:
        lr = args.lr
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer


def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)


### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


import copy


def get_custom_exp_code(args):
    # param_code 模型参数 包括学习率 alpha
    # exp_code 数据集参数 包括使用的split，边类型
    param_code = args.model_type



    param_code += '_%s' % args.bag_loss
    args.param_code = param_code
    args.dataset_path = './dataset_csv'

    exp_code = args.project_name
    # exp_code = ''
    exp_code += '_%s' % args.split_dir
    if isinstance(args.data_dir, list):
        for x in args.data_dir:
            exp_code += '_%s' % x.split('/')[-1]
    else:
        exp_code += '_%s' % args.data_dir.split('/')[-1]

    args.exp_code = exp_code

    return args


class Sampler_custom(Sampler):
    def __init__(self, censorship, batch_size):
        # super().__init__()
        event_list = np.where(np.array(censorship) == 0)[0]  # 0事件发生(死亡)
        censor_list = np.where(np.array(censorship) == 1)[0]  # 1存活

        print('init Sampler_custom')
        print('event_list', len(event_list))
        print('censor_list', len(censor_list))

        self.event_list = event_list
        self.censor_list = censor_list
        self.batch_size = batch_size

        self.min_event = 2
        if self.batch_size < 4:
            self.min_event = 1

        self.size = (len(event_list) + len(censor_list)) // batch_size

    def __iter__(self):
        # random.seed(int(time.time()))
        train_batch_sampler = []
        Event_idx = copy.deepcopy(self.event_list)
        Censored_idx = copy.deepcopy(self.censor_list)
        np.random.shuffle(Event_idx)
        np.random.shuffle(Censored_idx)

        Event_idx = list(Event_idx)
        Censored_idx = list(Censored_idx)
        while 1:
            if len(Event_idx) == 0 or (len(Event_idx) + len(Censored_idx)) < self.batch_size:
                break
            event_rate = random.uniform(0.2, 1)
            event_num = max(self.min_event, int(event_rate * self.batch_size))
            if len(Event_idx) < event_num:
                event_num = len(Event_idx)

            censor_num = self.batch_size - event_num
            if len(Censored_idx) < censor_num:
                censor_num = len(Censored_idx)
                event_num = self.batch_size - censor_num

            event_batch_select = random.sample(Event_idx, event_num)
            censor_batch_select = random.sample(Censored_idx, censor_num)

            Event_idx = list(set(Event_idx) - set(event_batch_select))
            Censored_idx = list(set(Censored_idx) - set(censor_batch_select))

            selected_list = event_batch_select + censor_batch_select
            random.shuffle(selected_list)
            train_batch_sampler.append(selected_list)

        random.shuffle(train_batch_sampler)
        self.size = len(train_batch_sampler)

        for idx in range(len(train_batch_sampler)):
            yield from train_batch_sampler[idx]

    def __len__(self):
        return int(self.size)


class Sampler_custom_graph(Sampler):
    def __init__(self, censorship, batch_size):
        event_list = np.where(np.array(censorship) == 0)[0]  # 0事件发生(死亡)
        censor_list = np.where(np.array(censorship) == 1)[0]  # 1存活

        print('init Sampler_custom')
        print('event_list', len(event_list))
        print('censor_list', len(censor_list))

        self.event_list = event_list
        self.censor_list = censor_list
        self.batch_size = batch_size

        self.min_event = 2
        if self.batch_size < 4:
            self.min_event = 1

        self.size = (len(event_list) + len(censor_list)) // batch_size

    def __iter__(self):
        # random.seed(int(time.time()))
        train_batch_sampler = []
        Event_idx = copy.deepcopy(self.event_list)
        Censored_idx = copy.deepcopy(self.censor_list)
        np.random.shuffle(Event_idx)
        np.random.shuffle(Censored_idx)

        Event_idx = list(Event_idx)
        Censored_idx = list(Censored_idx)
        while 1:
            if len(Event_idx) == 0 or (len(Event_idx) + len(Censored_idx)) < self.batch_size:
                break
            event_rate = random.uniform(0.2, 1)
            event_num = max(self.min_event, int(event_rate * self.batch_size))
            if len(Event_idx) < event_num:
                event_num = len(Event_idx)

            censor_num = self.batch_size - event_num
            if len(Censored_idx) < censor_num:
                censor_num = len(Censored_idx)
                event_num = self.batch_size - censor_num

            event_batch_select = random.sample(Event_idx, event_num)
            censor_batch_select = random.sample(Censored_idx, censor_num)

            Event_idx = list(set(Event_idx) - set(event_batch_select))
            Censored_idx = list(set(Censored_idx) - set(censor_batch_select))

            selected_list = event_batch_select + censor_batch_select
            random.shuffle(selected_list)
            train_batch_sampler.append(selected_list)

        random.shuffle(train_batch_sampler)
        self.size = len(train_batch_sampler)
        # print('run custom resample',len(train_batch_sampler))

        return iter(train_batch_sampler)

    def __len__(self):
        return int(self.size)


def get_args(args, added_dict=None, configs_dir='configs'):
    task = args.task
    project_nm = args.project_name
    print('project_name', project_nm)
    print('task', task)
    if '/configs/' not in args.yml_opt_path:
        args.yml_opt_path = f"./{configs_dir}/{task}/{args.yml_opt_path}"
    if '.yml' not in args.yml_opt_path:
        args.yml_opt_path = f"{args.yml_opt_path}.yml"
    print('args.yml_opt_path', args.yml_opt_path, flush=True)

    with open(args.yml_opt_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {args.yml_opt_path}")
    args = Namespace(**config)

    if added_dict is not None:
        for key, value in added_dict.items():
            print(f'added_dict set args {key} to {value}', flush=True)
            setattr(args, key, value)

    args.project_name = project_nm
    print('Current project:', args.project_name)

    args.reg = float(args.reg)
    args.task = task

    args.mode = 'patch'

    if not hasattr(args, 'train_mode'):
        args.train_mode = 'patch'
    if not hasattr(args, 'use_fp16'):
        args.use_fp16 = False
    args.use_fp16 = bool(args.use_fp16)

    if not hasattr(args, 'proto_fold'):
        args.proto_fold = './prototypes'

    if task == 'subtype':
        from datasets_custom.utils_labels import subtype_dict
        assert args.project_name in subtype_dict.keys()
        args.num_classes = len(subtype_dict[args.project_name])
    if args.task == 'survival':
        args.num_classes = 1
    elif args.task == 'staging':
        args.num_classes = 4
    else:
        assert hasattr(args, 'num_classes')
    # average setting for multi-classification metrics
    if not hasattr(args, 'average'):
        args.average = 'macro'

    # assert hasattr(args, 'pooling_method')
    if not hasattr(args, 'label_col'):
        if args.task == 'survival':
            args.label_col = 'survival_months'
        elif args.task == 'subtype':
            args.label_col = 'primary_diagnosis'
        elif args.task == 'staging':
            args.label_col = 'staging'
        else:
            raise NotImplementedError

    args.csv_path = args.csv_path.replace('(project_name)', args.project_name)
    args.split_dir = args.split_dir.replace('(project_name)', args.project_name)
    args.data_dir = args.data_dir.replace('(project_name)', args.project_name)

    if not hasattr(args, 'in_memory'):
        args.in_memory = False
    if not isinstance(args.data_dir, list):
        args.data_dir = [args.data_dir]

    args.pin_memory = True
    ### Creates Experiment Code from argparse + Folder Name to Save Results
    args = get_custom_exp_code(args)
    print("Experiment Name:", args.exp_code)
    args.results_dir = f"{args.results_dir}/{args.task}/{args.mode}_{args.model_type}/{args.which_splits}/" \
                       f"{str(args.param_code)}/{str(args.exp_code)}/"

    args.split_dir = os.path.join('./splits', args.task, args.which_splits, args.split_dir)
    print("split_dir", args.split_dir)
    assert os.path.isdir(args.split_dir)

    return args


from models.models_demo.model_transformer_pamoe import TransformerEncoder
from models.models_demo.model_transmil_pamoe import transmil_pmoe

def get_model(args):
    print('\nInit Model...')
    print('args.model_type', args.model_type)


    if args.model_type.lower() == 'transmil_pamoe':
        model = transmil_pmoe(n_classes=args.num_classes,
                              capacity_factor=args.capacity_factor,
                              num_expert_proto=args.num_expert_proto,
                              num_expert_extra=args.num_expert_extra,
                              prototype_pth=f'{args.proto_fold}/{args.project_name}.pt'
                              )
    elif args.model_type.lower() == 'transformer_pamoe':
        model = TransformerEncoder(
            n_classes=args.num_classes,
            layer_type=args.layer_type,
            input_dim=args.input_dim,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            ff_dim_mult=args.ff_dim_mult,
            dropout=args.dropout,
            use_cls_token=args.use_cls_token,

            num_expert_proto=args.num_expert_proto,
            num_expert_extra=args.num_expert_extra,
            prototype_pth=f'{args.proto_fold}/{args.project_name}.pt',
            capacity_factor=args.capacity_factor,
            drop_zeros=args.drop_zeros,
            pamoe_use_residual=args.pamoe_use_residual
        )

    else:
        raise NotImplementedError

    # print(model)
    print('Done', flush=True)
    return model


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    return total_norm
