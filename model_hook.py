import time
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    LlamaForCausalLM, LlamaTokenizer,
    BloomForCausalLM, BloomTokenizerFast,
)

from torch.utils.data import DataLoader, DistributedSampler
from dataset import (
    get_pt_dataset, JsonDatasetSFT,
    DataCollatorForPT, DataCollatorForSFT
)

from main import log_dist

from config import get_deepspeed_config, parse_arguments


def get_tokenizer(args):
    '''
    加载tokebizer
    '''
    # 对LLama系列模型使用LlamaTokenizer类
    if 'llama' in args.model_name_or_path.lower():
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    # 对Bloom系列模型使用BloomTokebizerFast类
    elif 'bloom' in args.model_name_or_path.lower():
        tokenizer = BloomTokenizerFast.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, fast_tokenizer=True)

    # 将分词器的pad_token设置为eos_token，以便正确处理填充（padding）
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_model_common(args):
    '''
    获取并加载模型文件
    '''
    log_dist(f'================ Loading Model =================')
    log_dist(f"loading model from {args.model_name_or_path}")
    tic = time.time()

    # 对LLama系列模型使用LlamaForCausaLLM类
    if 'llama' in args.model_name_or_path.lower():
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    # 对Bloom系统模型使用BloomForCausallm类

    elif 'bloom' in args.model_name_or_path.lower():
        model = BloomForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    log_dist(f'model loaded. costtime={time.time() - tic:.2f}s')
    log_dist(f"model={model}")

    return model


def get_dataloader_common(args):
    '''
    用于创建数据加载器（DataLoader）和数据集
    '''

    tokenizer = get_tokenizer(args)

    log_dist(f"==============Loading dataset==============")
    tic = time.time()
    if args.train_mode == 'pretrain':
        # 对于已被预处理的语料数据，直接使用load_from_disk函数加载即可
        train_dataset = get_pt_dataset(args)
        collator = DataCollatorForPT(pad_token_id=tokenizer.pad_token_id)
    elif args.train_mode == 'sft':
        train_dataset = JsonDatasetSFT(args.data_path, tokenizer, args.max_length)
        collator = DataCollatorForSFT(pad_token_id=tokenizer.pad_token_id)
    else:
        raise ValueError(f"train_mode  {args.train_mode} is not supported")

    # 分布式随机采样，确保分布式训练环境下语料被切分到各块GPU上，即每块GPU只训练整体语料的一个切片
    sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        num_workers=16,  # 指定16个核并行处理
        sampler=sampler,
        collate_fn=collator,
    )

    log_dist(f"Dataset Loaded:{args.data_path} costtime={time.time() - tic:.2f}s")
    log_dist(f"Num samples:{len(train_dataset)}")
    log_dist(f"Num Tokens:{len(train_dataset) * args.max_length / 1e9:.2f}B")
    log_dist(f"Total Steps:{len(train_dataloader)}")

    return {
        "sampler": sampler,
        "train_dataloader": train_dataloader,
        "tokenizer": tokenizer
    }
def get_ds_config(args):
    '''
    用于获取deepspeed的配置参数
    :param args:
    :return:
    '''
    ds_config = get_deepspeed_config(args)  # 获取deepspeed的配置参数，在config.py中定义
    return ds_config


def parse_args():
    '''
    解析命令行参数
    :return:
    '''
    args = parse_arguments()  # 解析命令行参数的函数，在config.py中定义

    log_dist('=========== 参数 ==========')
    for k, v in vars(args).items():
        log_dist(f'{k}={v}')
    log_dist('======================')
    return args


def get_op_lr(args, origin_model, dataloader_dict):
    '''
    获取优化器和学习率
    :param args:
    :param origin_model:
    :param dataloader_dict:
    :return:
    '''
    return None


def before_train(args, model_engine, dataloader_dict):
    '''
    训练开始前执行
    :param args:
    :param model_engine:
    :param dataloader_dict:
    :return:
    '''
    pass


def on_step_end(args, model_engine, dataloader_dict, step_num, epoch_num, outputs):
    '''
    在每个训练step结束时执行
    :param args:
    :param model_engine:
    :param dataloader_dict:
    :param step_num:
    :param epoch_num:
    :param outputs:
    :return:
    '''
    pass


def on_epoch_end(args, model_engine, dataloader_dict, epoch_num):
    '''
    在每个训练周期（epoch）结束时执行
    :param args:
    :param model_engine:
    :param dataloader_dict:
    :param epoch_num:
    :return:
    '''
    pass


def after_train(args, model_engine, dataloader_dict):
    '''
    在整个训练过程结束时执行
    :param args:
    :param model_engine:
    :param dataloader_dict:
    :return:
    '''
    pass