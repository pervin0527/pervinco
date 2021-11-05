import yaml
from anchor import generate_default_boxes
from voc_data import create_batch_generator
from network import create_ssd

if __name__ == "__main__":
    data_dir = "/data/Datasets/Seeds/VOCtrainval_11-May-2012/VOCdevkit/"
    year = "2012"
    arch = "ssd300"
    batch_size = 32
    num_batches = -1
    num_classes = 21
    pretrained_type = 'base'
    checkpoint_dir = None

    with open('./config.yml') as f:
        cfg = yaml.load(f)

    try:
        config = cfg[arch.upper()]
    except AttributeError:
        raise ValueError('Unknown architecture: {}'.format(args.arch))

    # print(config)
    default_boxes = generate_default_boxes(config)

    # train_generator, valid_generator, info = create_batch_generator(data_dir, year, default_boxes, config['image_size'], batch_size, num_batches, mode='train', augmentation=['flip'])
    train_generator, valid_generator, info = create_batch_generator(data_dir, year, default_boxes, config['image_size'], batch_size, num_batches, mode='train')

    try:
        ssd = create_ssd(num_classes, arch, pretrained_type, checkpoint_dir)
    
    except Exception as e:
        print(e)
        print('The program is exiting...')
        sys.exit()