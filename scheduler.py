from torch.optim.lr_scheduler import LambdaLR


def create_scheduler(args, optimizer):
    if 'num_training_steps' not in args:
        args['num_training_steps'] = args['epochs'] * args['step_per_epoch']
    print("### num_training_steps, ", args['num_training_steps'], flush=True)

    if isinstance(args['num_warmup_steps'], float):
        assert 0 <= args['num_warmup_steps'] < 1
        args['num_warmup_steps'] = int(args['num_training_steps'] * args['num_warmup_steps'])
    print("### num_warmup_steps, ", args['num_warmup_steps'], flush=True)

    print('sched:', args.sched, flush=True)

    if args.sched == 'linear':
        def lr_lambda(current_step: int):
            if current_step < args.num_warmup_steps:
                return float(current_step) / float(max(1, args.num_warmup_steps))
            return max(
                0.0, float(args.num_training_steps - current_step) / float(
                    max(1, args.num_training_steps - args.num_warmup_steps))
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    elif args.sched == 'step':
        def lr_lambda(current_step: int):
            if current_step < args.num_warmup_steps:
                return float(current_step) / float(max(1, args.num_warmup_steps))
            elif current_step < args.num_warmup_steps * 4:
                tt = 1
            elif current_step < args.num_warmup_steps * 7:
                tt = 0.5
            else:
                tt = 0.2

            return tt * max(
                0.0, float(args.num_training_steps - current_step) / float(
                    max(1, args.num_training_steps - args.num_warmup_steps))
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    else:
        raise NotImplementedError(f"args.sched == {args.sched}")

    return lr_scheduler

# from torch.optim.lr_scheduler import CosineAnnealingLR

# def create_scheduler(args, optimizer):
#     if 'num_training_steps' not in args:
#         args['num_training_steps'] = args['epochs'] * args['steps_per_epoch']
#     print("### num_training_steps, ", args['num_training_steps'], flush=True)

#     if isinstance(args['num_warmup_steps'], float):
#         assert 0 <= args['num_warmup_steps'] < 1
#         args['num_warmup_steps'] = int(args['num_training_steps'] * args['num_warmup_steps'])
#     print("### num_warmup_steps, ", args['num_warmup_steps'], flush=True)

#     # 判断调度策略，替换为CosineAnnealingLR
#     if args['sched'] == 'cosine':
#         # 使用CosineAnnealingLR
#         lr_scheduler = CosineAnnealingLR(optimizer, T_max=args['epochs'], eta_min=0.01 * args['lr'])
#     else:
#         # 如果需要，这里可以保留其他调度器的逻辑
#         raise NotImplementedError(f"args.sched == {args['sched']} is not implemented. Please use 'cosine'.")

#     return lr_scheduler

