import math
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel
import time
import torch
from utils.utils import NativeScalerWithGradNormCount as NativeScaler
from utils.core_funcs_graph import *

from utils.utils_earlystop import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_graph(datasets: tuple, args: Namespace, cur=0, max_node_num=20_0000, bar=True):
    """
        train for a single fold
    """

    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    os.makedirs(writer_dir, exist_ok=True)
    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    if test_split != None:
        save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, f'splits_{cur}.csv'))
    else:
        save_splits((train_split, val_split), ['train', 'val'], os.path.join(args.results_dir, f'splits_{cur}.csv'))

    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    reg_fn = None
    model_type = args.model_type

    model = get_model(args)
    print('model', type(model))

    gpu_ids = args.gpu_ids
    print('gpu_ids', gpu_ids)
    if isinstance(gpu_ids, list):
        torch.cuda.set_device(gpu_ids[0])
    else:
        torch.cuda.set_device(gpu_ids)
        gpu_ids = [gpu_ids]

    model = DataParallel(model, device_ids=gpu_ids)

    loss_fn = get_loss(args)
    print(args.bag_loss)
    print('loss_fn', type(loss_fn))

    if hasattr(args, 'pretrained_pth'):
        print('pretrained_pth:', args.pretrained_pth)
        ags2 = Namespace()
        ags2.yml_opt_path = args.pretrained_pth
        pretrained_pth = f"{get_args(ags2).results_dir}/s_0_maxcindex_checkpoint.pt"
        print('load checkpoint from:', pretrained_pth)
        model.load_state_dict(torch.load(pretrained_pth, map_location='cpu'))

    model = model.to(torch.device('cuda'))
    print('Done!')

    print('\nInit Loaders...')

    print('len dataset')
    print(len(train_split))

    if args.task == 'survival':
        print('Use custom Sampler')
        train_batch_sampler = Sampler_custom_graph(np.array(train_split.censorship), args.batch_size)
        # train_batch_sampler = Sampler_custom_multidataset(train_split.censorship, train_split.dataset, args.batch_size)
        print(type(train_batch_sampler))
        train_loader = DataListLoader(train_split, batch_sampler=train_batch_sampler,
                                      num_workers=args.num_workers, pin_memory=args.pin_memory)
    else:
        train_loader = DataListLoader(train_split, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers,
                                      pin_memory=args.pin_memory, drop_last=True)

    val_censorship = val_split.censorship
    val_event_list = np.where(np.array(val_censorship) == 0)[0]
    val_censor_list = np.where(np.array(val_censorship) == 1)[0]
    print('val event list', len(val_event_list))
    print('val censor list', len(val_censor_list))

    # args.batch_size
    # batch_size = args.batch_size
    batch_size = 1
    val_loader = DataListLoader(val_split, batch_size=batch_size, num_workers=args.num_workers, shuffle=False)
    if test_split != None:
        test_loader = DataListLoader(test_split, batch_size=batch_size, num_workers=args.num_workers,
                                     shuffle=False)
    else:
        test_loader = None

    print('Done!')

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    ## 动态学习率
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5, last_epoch=-1)
    # T_max=5, eta_min=1e-5
    # T_max=3, eta_min=1e-5

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=5e-6, last_epoch=-1)
    print('Done!')

    print('args.batch_size', args.batch_size)

    print('\nSetup EarlyStopping...', end=' ')
    if args.task == 'survival':
        EarlyStopping_fn = EarlyStopping_cindex
    else:
        EarlyStopping_fn = EarlyStopping_bacc
        # EarlyStopping_fn = EarlyStopping_loss
    print('EarlyStopping_fn', EarlyStopping_fn)
    if args.early_stopping:
        # early_stopping = EarlyStopping_fn(warmup=0, patience=15, stop_epoch=25, verbose=True)
        early_stopping = EarlyStopping_fn(warmup=0, patience=5, stop_epoch=25, verbose=True)
    else:
        early_stopping = None

    # loss_scaler
    print('Init loss_scaler ...', end=' ')
    loss_scaler = NativeScaler()
    print('Done!', flush=True)

    print('\nSetup Validation C-Index Monitor...', end=' ')
    print('Done!')

    for epoch in range(1, args.max_epochs + 1):
        st = time.time()
        try:
            train_loop_survival(epoch, model, train_loader, optimizer, args, writer, loss_fn,
                                args.gc, bar=bar, loss_scaler=loss_scaler)
            print(f'Epoch: {epoch}, learning_rate:', optimizer.param_groups[0]['lr'], flush=True)
            if args.testing:
                validate_survival(epoch, model, test_loader, args, early_stopping=None, writer=None,
                                  results_dir=args.results_dir, cur=cur, bar=bar, mode='Test')
        except Exception as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                print('OutOfMemoryError')
            else:
                print(e)
            continue

        stop = validate_survival(epoch, model, val_loader, args, early_stopping, writer,
                                 args.results_dir, cur=cur, bar=bar)
        scheduler.step()
        ed = time.time()
        print(f"Epoch {epoch} cost {ed - st}s", flush=True)
        print()
        if stop:
            print('early stop break', flush=True)
            break
        gc.collect()

    torch.save(model.state_dict(), os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt"))
    model.load_state_dict(
        torch.load(os.path.join(args.results_dir, f"s_{cur}_maxcindex_checkpoint.pt"), map_location='cpu'))
    model = model.to(torch.device('cuda'))

    print('summary_survival val split')
    patient_results, metrics_dict = summary_survival(model, val_loader, bar=bar, mode='val', args=args)
    print('Val: ', end=' ')
    for key, it in metrics_dict.items():
        print(f"{key}: {it:.4f}", end=' ', flush=True)
    print()
    fin_patient_results = {'val': patient_results}

    if test_loader is not None:
        patient_results_tst, metrics_dict_tst = summary_survival(model, val_loader, bar=bar, mode='test', args=args)
        print('Test: ', end=' ')
        for key, it in metrics_dict.items():
            print(f"{key}: {it:.4f}", end=' ', flush=True)
        print()
        fin_patient_results = {'val': patient_results}

        metrics_dict.update(metrics_dict_tst)
        fin_patient_results['test'] = patient_results_tst

    writer.close()
    return fin_patient_results, metrics_dict
