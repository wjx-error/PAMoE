from torch.nn import DataParallel
from utils.core_funcs import *
from torch.utils.data import DataLoader
import time
from torch.cuda import OutOfMemoryError

from utils.utils import my_list_collate_fn
from utils.utils import my_default_collate_fn
from utils.utils import NativeScalerWithGradNormCount as NativeScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(datasets: tuple, args: Namespace, cur=0, bar=True):
    """   
        train for a single fold
    """
    print('Training Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    os.makedirs(writer_dir, exist_ok=True)
    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    print('Init train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    if test_split != None:
        save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, f'splits_{cur}.csv'))
    else:
        save_splits((train_split, val_split), ['train', 'val'], os.path.join(args.results_dir, f'splits_{cur}.csv'))

    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    gpu_ids = args.gpu_ids
    print('gpu_ids', gpu_ids)
    if not isinstance(gpu_ids, list):
        gpu_ids = [gpu_ids]
    torch.cuda.set_device(gpu_ids[0])

    model = get_model(args)
    # print(model)
    print('model', type(model), flush=True)


    model = DataParallel(model, device_ids=gpu_ids)
    # if len(gpu_ids) > 1:
    #     model = DataParallel(model, device_ids=gpu_ids)

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

    model = model.cuda()
    print('Done!')

    print('Init Loaders...')
    my_collate_fn = my_list_collate_fn

    if args.task == 'survival':
        print('batch_size:', args.batch_size, flush=True)
        train_batch_sampler = Sampler_custom(np.array(train_split.censorship), args.batch_size)
        # train_batch_sampler = Sampler_custom_multidataset(train_split.censorship, train_split.dataset, args.batch_size)
        print(type(train_batch_sampler))
        train_loader = DataLoader(
            train_split,
            batch_size=args.batch_size,
            sampler=train_batch_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=True,
            collate_fn=my_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_split, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory,
            drop_last=True, shuffle=True, collate_fn=my_collate_fn
        )
    bs = 1
    val_loader = DataLoader(val_split, batch_size=bs, num_workers=args.num_workers,
                            shuffle=False, collate_fn=my_collate_fn)
    if test_split != None:
        test_loader = DataLoader(test_split, batch_size=bs, num_workers=args.num_workers,
                                 shuffle=False, collate_fn=my_collate_fn)
    else:
        test_loader = None
    print('Done!')

    print('Setup EarlyStopping...')
    if args.task == 'survival':
        EarlyStopping_fn = EarlyStopping_cindex
    else:
        EarlyStopping_fn = EarlyStopping_bacc
        # EarlyStopping_fn = EarlyStopping_loss
    print('EarlyStopping_fn', EarlyStopping_fn)
    if args.early_stopping:
        # early_stopping = EarlyStopping_fn(warmup=0, patience=15, stop_epoch=25, verbose=True)
        early_stopping = EarlyStopping_fn(warmup=0, patience=5, stop_epoch=25, verbose=True)
        # early_stopping = EarlyStopping_fn(warmup=0, patience=0, stop_epoch=0, verbose=True)
    else:
        early_stopping = None
    print('Done!', flush=True)

    # loss_scaler
    print('Init loss_scaler ...', end=' ')
    loss_scaler = NativeScaler()
    print('Done!', flush=True)

    print('Init optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    ## 动态学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5, last_epoch=-1)
    # T_max=5, eta_min=1e-5
    # T_max=3, eta_min=1e-5
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=5e-6, last_epoch=-1)
    print('Done!', flush=True)

    for epoch in range(1, args.max_epochs + 1):
        st = time.time()
        try:
            train_loop_survival(epoch, model, train_loader, optimizer, args, writer, loss_fn,
                                args.gc, loss_scaler=loss_scaler, bar=bar)
            print(f'Epoch: {epoch}, learning_rate:', optimizer.param_groups[0]['lr'])
            stop = validate_survival(epoch, model, val_loader, args, early_stopping=early_stopping,
                                     writer=writer, results_dir=args.results_dir, cur=cur, bar=bar)
            if test_loader is not None:
                validate_survival(epoch, model, test_loader, args, early_stopping=None, writer=None,
                                  results_dir=args.results_dir, cur=cur, bar=bar, mode='test')
        except Exception as e:
            print(f'Epoch {epoch} error', flush=True)
            # print(e)
            if isinstance(e, OutOfMemoryError):
                print('OOM error', flush=True)
            else:
                print(e)
            continue
        scheduler.step()
        ed = time.time()
        print(f"Epoch {epoch} cost {ed - st}s", flush=True)
        print()
        if stop:
            print('early stop break', flush=True)
            break
        gc.collect()

    # torch.save(model.state_dict(), os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt"))
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
