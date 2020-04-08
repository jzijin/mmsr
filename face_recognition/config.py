class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 10177
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    finetune = False

    train_root = '/data/Datasets/webface/CASIA-maxpy-clean-crop-144/'
    train_list = '/data/Datasets/webface/train_data_13938.txt'
    val_list = '/data/Datasets/webface/val_data_13938.txt'

    test_root = '/data1/Datasets/anti-spoofing/test/data_align_256'
    test_list = 'test.txt'

    lfw_root = '/home/jzijin/code/bysj/code/mmsr/datasets/lfw-align-128'
    lfw_test_list = 'lfw_test_pair.txt'

    checkpoints_path = 'checkpoints'
    load_model_path = 'models/resnet18.pth'
    test_model_path = 'resnet18_110.pth'
    save_interval = 3000

    train_batch_size = 32  # batch size
    test_batch_size = 16

    input_shape = (1, 128, 128)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0'
    num_workers = 8  # how many workers for loading data
    print_freq = 10  #100 print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 500000
    lr = 1e-4  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4

    start_epoch = 0
    pretrain_model_F = 'checkpoints/resnet18_1.pth'
    pretrain_model_G = 'checkpoints/232000_G.pth'

    lr_G = 1e-5
    beta1_G = 0.9
    beta2_G = 0.999
    alpha = 8
    lr_step_G = 10
    wd_G = 5e-4

