from extremal_perturbation import Perturbation, MaskGenerator
from extremal_perturbation import Perturbation, MaskGenerator
if True:
    if isinstance(areas, float):
        areas = [areas]
    momentum = 0.9
    learning_rate = 0.01
    regul_weight = 300
    device = input.device

    regul_weight_last = max(regul_weight / 2, 1)

    if debug:
        print(
            f"extremal_perturbation:\n"
            f"- target: {target}\n"
            f"- areas: {areas}\n"
            f"- variant: {variant}\n"
            f"- max_iter: {max_iter}\n"
            f"- step/sigma: {step}, {sigma}\n"
            f"- image size: {list(input.shape)}\n"
            f"- reward function: {reward_func.__name__}"
        )

    # Disable gradients for model parameters.
    # TODO(av): undo on leaving the function.
    for p in model.parameters():
        p.requires_grad_(False)

    # Get the perturbation operator.
    # The perturbation can be applied at any layer of the network (depth).
    perturbation = Perturbation(
        input,
        num_levels=num_levels,
        type=perturbation
    ).to(device)

    perturbation_str = '\n  '.join(perturbation.__str__().split('\n'))
    if debug:
        print(f"- {perturbation_str}")

    # Prepare the mask generator.
    shape = perturbation.pyramid.shape[2:]
    mask_generator = MaskGenerator(shape, step, sigma).to(device)
    h, w = mask_generator.shape_in
    pmask = torch.ones(len(areas), 1, h, w).to(device)
    if debug:
        print(f"- mask resolution:\n  {pmask.shape}")

    # Prepare reference area vector.
    max_area = np.prod(mask_generator.shape_out)
    reference = torch.ones(len(areas), max_area).to(device)
    for i, a in enumerate(areas):
        reference[i, :int(max_area * (1 - a))] = 0

    # Initialize optimizer.
    optimizer = optim.SGD([pmask],
                          lr=learning_rate,
                          momentum=momentum,
                          dampening=momentum)
    hist = torch.zeros((len(areas), 2, 0))

    for t in range(max_iter):
        pmask.requires_grad_(True)

        # Generate the mask.
        mask_, mask = mask_generator.generate(pmask)

        # Apply the mask.
        if variant == DELETE_VARIANT:
            x = perturbation.apply(1 - mask_)
        elif variant == PRESERVE_VARIANT:
            x = perturbation.apply(mask_)
        elif variant == DUAL_VARIANT:
            x = torch.cat((
                perturbation.apply(mask_),
                perturbation.apply(1 - mask_),
            ), dim=0)
        else:
            assert False

        # Apply jitter to the masked data.
        if jitter and t % 2 == 0:
            x = torch.flip(x, dims=(3,))

        # Evaluate the model on the masked data.
        y = model(x)

        # Get reward.
        reward = reward_func(y, target, variant=variant)

        # Reshape reward and average over spatial dimensions.
        reward = reward.reshape(len(areas), -1).mean(dim=1)

        # Area regularization.
        mask_sorted = mask.reshape(len(areas), -1).sort(dim=1)[0]
        regul = - ((mask_sorted - reference)**2).mean(dim=1) * regul_weight
        energy = (reward + regul).sum()

        # Gradient step.
        optimizer.zero_grad()
        (- energy).backward()
        optimizer.step()

        pmask.data = pmask.data.clamp(0, 1)

        # Record energy.
        hist = torch.cat(
            (hist,
             torch.cat((
                 reward.detach().cpu().view(-1, 1, 1),
                 regul.detach().cpu().view(-1, 1, 1)
             ), dim=1)), dim=2)

        # Adjust the regulariser/area constraint weight.
        regul_weight *= 1.0035

        # Diagnostics.
        debug_this_iter = debug and (t in (0, max_iter - 1)
                                     or regul_weight / regul_weight_last >= 2)

        if (print_iter is not None and t % print_iter == 0) or debug_this_iter:
            print("[{:04d}/{:04d}]".format(t + 1, max_iter), end="")
            for i, area in enumerate(areas):
                print(" [area:{:.2f} loss:{:.2f} reg:{:.2f}]".format(
                    area,
                    hist[i, 0, -1],
                    hist[i, 1, -1]), end="")
            print()

        if debug_this_iter:
            regul_weight_last = regul_weight
            for i, a in enumerate(areas):
                plt.figure(i, figsize=(20, 6))
                plt.clf()
                ncols = 4 if variant == DUAL_VARIANT else 3
                plt.subplot(1, ncols, 1)
                plt.plot(hist[i, 0].numpy())
                plt.plot(hist[i, 1].numpy())
                plt.plot(hist[i].sum(dim=0).numpy())
                plt.legend(('energy', 'regul', 'both'))
                plt.title(f'target area:{a:.2f}')
                plt.subplot(1, ncols, 2)
                imsc(mask[i], lim=[0, 1])
                plt.title(
                    f"min:{mask[i].min().item():.2f}"
                    f" max:{mask[i].max().item():.2f}"
                    f" area:{mask[i].sum() / mask[i].numel():.2f}")
                plt.subplot(1, ncols, 3)
                imsc(x[i])
                if variant == DUAL_VARIANT:
                    plt.subplot(1, ncols, 4)
                    imsc(x[i + len(areas)])
                plt.pause(0.001)

    mask_ = mask_.detach()

    # Resize saliency map.
    mask_ = resize_saliency(input,
                            mask_,
                            resize,
                            mode=resize_mode)

    # Smooth saliency map.
    if smooth > 0:
        mask_ = imsmooth(
            mask_,
            sigma=smooth * min(mask_.shape[2:]),
            padding_mode='constant'
        )
if True:
    ##########################
    import extremal_perturbation
    areas = [0.1]
    perturbation = extremal_perturbation.BLUR_PERTURBATION
    max_iter = 800
    num_levels=8
    step=7
    sigma=21
    jitter=True
    variant=extremal_perturbation.PRESERVE_VARIANT
    print_iter=None
    debug=False
    reward_func=extremal_perturbation.simple_reward
    resize=False
    resize_mode='bilinear'
    smooth=0
    ##########################
    if isinstance(areas, float):
        areas = [areas]
    momentum = 0.9
    learning_rate = 0.01
    regul_weight = 300
    device = input.device

    regul_weight_last = max(regul_weight / 2, 1)

    if debug:
        print(
            f"extremal_perturbation:\n"
            f"- target: {target}\n"
            f"- areas: {areas}\n"
            f"- variant: {variant}\n"
            f"- max_iter: {max_iter}\n"
            f"- step/sigma: {step}, {sigma}\n"
            f"- image size: {list(input.shape)}\n"
            f"- reward function: {reward_func.__name__}"
        )

    # Disable gradients for model parameters.
    # TODO(av): undo on leaving the function.
    for p in model.parameters():
        p.requires_grad_(False)

    # Get the perturbation operator.
    # The perturbation can be applied at any layer of the network (depth).
    perturbation = Perturbation(
        input,
        num_levels=num_levels,
        type=perturbation
    ).to(device)

    perturbation_str = '\n  '.join(perturbation.__str__().split('\n'))
    if debug:
        print(f"- {perturbation_str}")

    # Prepare the mask generator.
    shape = perturbation.pyramid.shape[2:]
    mask_generator = MaskGenerator(shape, step, sigma).to(device)
    h, w = mask_generator.shape_in
    pmask = torch.ones(len(areas), 1, h, w).to(device)
    if debug:
        print(f"- mask resolution:\n  {pmask.shape}")

    # Prepare reference area vector.
    max_area = np.prod(mask_generator.shape_out)
    reference = torch.ones(len(areas), max_area).to(device)
    for i, a in enumerate(areas):
        reference[i, :int(max_area * (1 - a))] = 0

    # Initialize optimizer.
    optimizer = optim.SGD([pmask],
                          lr=learning_rate,
                          momentum=momentum,
                          dampening=momentum)
    hist = torch.zeros((len(areas), 2, 0))

    for t in range(max_iter):
        pmask.requires_grad_(True)

        # Generate the mask.
        mask_, mask = mask_generator.generate(pmask)

        # Apply the mask.
        if variant == DELETE_VARIANT:
            x = perturbation.apply(1 - mask_)
        elif variant == PRESERVE_VARIANT:
            x = perturbation.apply(mask_)
        elif variant == DUAL_VARIANT:
            x = torch.cat((
                perturbation.apply(mask_),
                perturbation.apply(1 - mask_),
            ), dim=0)
        else:
            assert False

        # Apply jitter to the masked data.
        if jitter and t % 2 == 0:
            x = torch.flip(x, dims=(3,))

        # Evaluate the model on the masked data.
        y = model(x)

        # Get reward.
        reward = reward_func(y, target, variant=variant)

        # Reshape reward and average over spatial dimensions.
        reward = reward.reshape(len(areas), -1).mean(dim=1)

        # Area regularization.
        mask_sorted = mask.reshape(len(areas), -1).sort(dim=1)[0]
        regul = - ((mask_sorted - reference)**2).mean(dim=1) * regul_weight
        energy = (reward + regul).sum()

        # Gradient step.
        optimizer.zero_grad()
        (- energy).backward()
        optimizer.step()

        pmask.data = pmask.data.clamp(0, 1)

        # Record energy.
        hist = torch.cat(
            (hist,
             torch.cat((
                 reward.detach().cpu().view(-1, 1, 1),
                 regul.detach().cpu().view(-1, 1, 1)
             ), dim=1)), dim=2)

        # Adjust the regulariser/area constraint weight.
        regul_weight *= 1.0035

        # Diagnostics.
        debug_this_iter = debug and (t in (0, max_iter - 1)
                                     or regul_weight / regul_weight_last >= 2)

        if (print_iter is not None and t % print_iter == 0) or debug_this_iter:
            print("[{:04d}/{:04d}]".format(t + 1, max_iter), end="")
            for i, area in enumerate(areas):
                print(" [area:{:.2f} loss:{:.2f} reg:{:.2f}]".format(
                    area,
                    hist[i, 0, -1],
                    hist[i, 1, -1]), end="")
            print()

        if debug_this_iter:
            regul_weight_last = regul_weight
            for i, a in enumerate(areas):
                plt.figure(i, figsize=(20, 6))
                plt.clf()
                ncols = 4 if variant == DUAL_VARIANT else 3
                plt.subplot(1, ncols, 1)
                plt.plot(hist[i, 0].numpy())
                plt.plot(hist[i, 1].numpy())
                plt.plot(hist[i].sum(dim=0).numpy())
                plt.legend(('energy', 'regul', 'both'))
                plt.title(f'target area:{a:.2f}')
                plt.subplot(1, ncols, 2)
                imsc(mask[i], lim=[0, 1])
                plt.title(
                    f"min:{mask[i].min().item():.2f}"
                    f" max:{mask[i].max().item():.2f}"
                    f" area:{mask[i].sum() / mask[i].numel():.2f}")
                plt.subplot(1, ncols, 3)
                imsc(x[i])
                if variant == DUAL_VARIANT:
                    plt.subplot(1, ncols, 4)
                    imsc(x[i + len(areas)])
                plt.pause(0.001)

    mask_ = mask_.detach()

    # Resize saliency map.
    mask_ = resize_saliency(input,
                            mask_,
                            resize,
                            mode=resize_mode)

    # Smooth saliency map.
    if smooth > 0:
        mask_ = imsmooth(
            mask_,
            sigma=smooth * min(mask_.shape[2:]),
            padding_mode='constant'
        )
        
if True:
    ##########################
    input = augmentation
    import extremal_perturbation
    areas = [0.1]
    perturbation = extremal_perturbation.BLUR_PERTURBATION
    max_iter = 800
    num_levels=8
    step=7
    sigma=21
    jitter=True
    variant=extremal_perturbation.PRESERVE_VARIANT
    print_iter=None
    debug=False
    reward_func=extremal_perturbation.simple_reward
    resize=False
    resize_mode='bilinear'
    smooth=0
    ##########################
    if isinstance(areas, float):
        areas = [areas]
    momentum = 0.9
    learning_rate = 0.01
    regul_weight = 300
    device = input.device

    regul_weight_last = max(regul_weight / 2, 1)

    if debug:
        print(
            f"extremal_perturbation:\n"
            f"- target: {target}\n"
            f"- areas: {areas}\n"
            f"- variant: {variant}\n"
            f"- max_iter: {max_iter}\n"
            f"- step/sigma: {step}, {sigma}\n"
            f"- image size: {list(input.shape)}\n"
            f"- reward function: {reward_func.__name__}"
        )

    # Disable gradients for model parameters.
    # TODO(av): undo on leaving the function.
    for p in model.parameters():
        p.requires_grad_(False)

    # Get the perturbation operator.
    # The perturbation can be applied at any layer of the network (depth).
    perturbation = Perturbation(
        input,
        num_levels=num_levels,
        type=perturbation
    ).to(device)

    perturbation_str = '\n  '.join(perturbation.__str__().split('\n'))
    if debug:
        print(f"- {perturbation_str}")

    # Prepare the mask generator.
    shape = perturbation.pyramid.shape[2:]
    mask_generator = MaskGenerator(shape, step, sigma).to(device)
    h, w = mask_generator.shape_in
    pmask = torch.ones(len(areas), 1, h, w).to(device)
    if debug:
        print(f"- mask resolution:\n  {pmask.shape}")

    # Prepare reference area vector.
    max_area = np.prod(mask_generator.shape_out)
    reference = torch.ones(len(areas), max_area).to(device)
    for i, a in enumerate(areas):
        reference[i, :int(max_area * (1 - a))] = 0

    # Initialize optimizer.
    optimizer = optim.SGD([pmask],
                          lr=learning_rate,
                          momentum=momentum,
                          dampening=momentum)
    hist = torch.zeros((len(areas), 2, 0))

    for t in range(max_iter):
        pmask.requires_grad_(True)

        # Generate the mask.
        mask_, mask = mask_generator.generate(pmask)

        # Apply the mask.
        if variant == DELETE_VARIANT:
            x = perturbation.apply(1 - mask_)
        elif variant == PRESERVE_VARIANT:
            x = perturbation.apply(mask_)
        elif variant == DUAL_VARIANT:
            x = torch.cat((
                perturbation.apply(mask_),
                perturbation.apply(1 - mask_),
            ), dim=0)
        else:
            assert False

        # Apply jitter to the masked data.
        if jitter and t % 2 == 0:
            x = torch.flip(x, dims=(3,))

        # Evaluate the model on the masked data.
        y = model(x)

        # Get reward.
        reward = reward_func(y, target, variant=variant)

        # Reshape reward and average over spatial dimensions.
        reward = reward.reshape(len(areas), -1).mean(dim=1)

        # Area regularization.
        mask_sorted = mask.reshape(len(areas), -1).sort(dim=1)[0]
        regul = - ((mask_sorted - reference)**2).mean(dim=1) * regul_weight
        energy = (reward + regul).sum()

        # Gradient step.
        optimizer.zero_grad()
        (- energy).backward()
        optimizer.step()

        pmask.data = pmask.data.clamp(0, 1)

        # Record energy.
        hist = torch.cat(
            (hist,
             torch.cat((
                 reward.detach().cpu().view(-1, 1, 1),
                 regul.detach().cpu().view(-1, 1, 1)
             ), dim=1)), dim=2)

        # Adjust the regulariser/area constraint weight.
        regul_weight *= 1.0035

        # Diagnostics.
        debug_this_iter = debug and (t in (0, max_iter - 1)
                                     or regul_weight / regul_weight_last >= 2)

        if (print_iter is not None and t % print_iter == 0) or debug_this_iter:
            print("[{:04d}/{:04d}]".format(t + 1, max_iter), end="")
            for i, area in enumerate(areas):
                print(" [area:{:.2f} loss:{:.2f} reg:{:.2f}]".format(
                    area,
                    hist[i, 0, -1],
                    hist[i, 1, -1]), end="")
            print()

        if debug_this_iter:
            regul_weight_last = regul_weight
            for i, a in enumerate(areas):
                plt.figure(i, figsize=(20, 6))
                plt.clf()
                ncols = 4 if variant == DUAL_VARIANT else 3
                plt.subplot(1, ncols, 1)
                plt.plot(hist[i, 0].numpy())
                plt.plot(hist[i, 1].numpy())
                plt.plot(hist[i].sum(dim=0).numpy())
                plt.legend(('energy', 'regul', 'both'))
                plt.title(f'target area:{a:.2f}')
                plt.subplot(1, ncols, 2)
                imsc(mask[i], lim=[0, 1])
                plt.title(
                    f"min:{mask[i].min().item():.2f}"
                    f" max:{mask[i].max().item():.2f}"
                    f" area:{mask[i].sum() / mask[i].numel():.2f}")
                plt.subplot(1, ncols, 3)
                imsc(x[i])
                if variant == DUAL_VARIANT:
                    plt.subplot(1, ncols, 4)
                    imsc(x[i + len(areas)])
                plt.pause(0.001)

    mask_ = mask_.detach()

    # Resize saliency map.
    mask_ = resize_saliency(input,
                            mask_,
                            resize,
                            mode=resize_mode)

    # Smooth saliency map.
    if smooth > 0:
        mask_ = imsmooth(
            mask_,
            sigma=smooth * min(mask_.shape[2:]),
            padding_mode='constant'
        )
        
if True:
    ##########################
    input = augmentations
    import extremal_perturbation
    areas = [0.1]
    perturbation = extremal_perturbation.BLUR_PERTURBATION
    max_iter = 800
    num_levels=8
    step=7
    sigma=21
    jitter=True
    variant=extremal_perturbation.PRESERVE_VARIANT
    print_iter=None
    debug=False
    reward_func=extremal_perturbation.simple_reward
    resize=False
    resize_mode='bilinear'
    smooth=0
    ##########################
    if isinstance(areas, float):
        areas = [areas]
    momentum = 0.9
    learning_rate = 0.01
    regul_weight = 300
    device = input.device

    regul_weight_last = max(regul_weight / 2, 1)

    if debug:
        print(
            f"extremal_perturbation:\n"
            f"- target: {target}\n"
            f"- areas: {areas}\n"
            f"- variant: {variant}\n"
            f"- max_iter: {max_iter}\n"
            f"- step/sigma: {step}, {sigma}\n"
            f"- image size: {list(input.shape)}\n"
            f"- reward function: {reward_func.__name__}"
        )

    # Disable gradients for model parameters.
    # TODO(av): undo on leaving the function.
    for p in model.parameters():
        p.requires_grad_(False)

    # Get the perturbation operator.
    # The perturbation can be applied at any layer of the network (depth).
    perturbation = Perturbation(
        input,
        num_levels=num_levels,
        type=perturbation
    ).to(device)

    perturbation_str = '\n  '.join(perturbation.__str__().split('\n'))
    if debug:
        print(f"- {perturbation_str}")

    # Prepare the mask generator.
    shape = perturbation.pyramid.shape[2:]
    mask_generator = MaskGenerator(shape, step, sigma).to(device)
    h, w = mask_generator.shape_in
    pmask = torch.ones(len(areas), 1, h, w).to(device)
    if debug:
        print(f"- mask resolution:\n  {pmask.shape}")

    # Prepare reference area vector.
    max_area = np.prod(mask_generator.shape_out)
    reference = torch.ones(len(areas), max_area).to(device)
    for i, a in enumerate(areas):
        reference[i, :int(max_area * (1 - a))] = 0

    # Initialize optimizer.
    optimizer = optim.SGD([pmask],
                          lr=learning_rate,
                          momentum=momentum,
                          dampening=momentum)
    hist = torch.zeros((len(areas), 2, 0))

    for t in range(max_iter):
        pmask.requires_grad_(True)

        # Generate the mask.
        mask_, mask = mask_generator.generate(pmask)

        # Apply the mask.
        if variant == DELETE_VARIANT:
            x = perturbation.apply(1 - mask_)
        elif variant == PRESERVE_VARIANT:
            x = perturbation.apply(mask_)
        elif variant == DUAL_VARIANT:
            x = torch.cat((
                perturbation.apply(mask_),
                perturbation.apply(1 - mask_),
            ), dim=0)
        else:
            assert False

        # Apply jitter to the masked data.
        if jitter and t % 2 == 0:
            x = torch.flip(x, dims=(3,))

        # Evaluate the model on the masked data.
        y = model(x)

        # Get reward.
        reward = reward_func(y, target, variant=variant)

        # Reshape reward and average over spatial dimensions.
        reward = reward.reshape(len(areas), -1).mean(dim=1)

        # Area regularization.
        mask_sorted = mask.reshape(len(areas), -1).sort(dim=1)[0]
        regul = - ((mask_sorted - reference)**2).mean(dim=1) * regul_weight
        energy = (reward + regul).sum()

        # Gradient step.
        optimizer.zero_grad()
        (- energy).backward()
        optimizer.step()

        pmask.data = pmask.data.clamp(0, 1)

        # Record energy.
        hist = torch.cat(
            (hist,
             torch.cat((
                 reward.detach().cpu().view(-1, 1, 1),
                 regul.detach().cpu().view(-1, 1, 1)
             ), dim=1)), dim=2)

        # Adjust the regulariser/area constraint weight.
        regul_weight *= 1.0035

        # Diagnostics.
        debug_this_iter = debug and (t in (0, max_iter - 1)
                                     or regul_weight / regul_weight_last >= 2)

        if (print_iter is not None and t % print_iter == 0) or debug_this_iter:
            print("[{:04d}/{:04d}]".format(t + 1, max_iter), end="")
            for i, area in enumerate(areas):
                print(" [area:{:.2f} loss:{:.2f} reg:{:.2f}]".format(
                    area,
                    hist[i, 0, -1],
                    hist[i, 1, -1]), end="")
            print()

        if debug_this_iter:
            regul_weight_last = regul_weight
            for i, a in enumerate(areas):
                plt.figure(i, figsize=(20, 6))
                plt.clf()
                ncols = 4 if variant == DUAL_VARIANT else 3
                plt.subplot(1, ncols, 1)
                plt.plot(hist[i, 0].numpy())
                plt.plot(hist[i, 1].numpy())
                plt.plot(hist[i].sum(dim=0).numpy())
                plt.legend(('energy', 'regul', 'both'))
                plt.title(f'target area:{a:.2f}')
                plt.subplot(1, ncols, 2)
                imsc(mask[i], lim=[0, 1])
                plt.title(
                    f"min:{mask[i].min().item():.2f}"
                    f" max:{mask[i].max().item():.2f}"
                    f" area:{mask[i].sum() / mask[i].numel():.2f}")
                plt.subplot(1, ncols, 3)
                imsc(x[i])
                if variant == DUAL_VARIANT:
                    plt.subplot(1, ncols, 4)
                    imsc(x[i + len(areas)])
                plt.pause(0.001)

    mask_ = mask_.detach()

    # Resize saliency map.
    mask_ = resize_saliency(input,
                            mask_,
                            resize,
                            mode=resize_mode)

    # Smooth saliency map.
    if smooth > 0:
        mask_ = imsmooth(
            mask_,
            sigma=smooth * min(mask_.shape[2:]),
            padding_mode='constant'
        )
        
if True:
    ##########################
    input = augmentations
    import extremal_perturbation
    areas = [0.1]
    perturbation = extremal_perturbation.BLUR_PERTURBATION
    max_iter = 800
    num_levels=8
    step=7
    sigma=21
    jitter=True
    variant=extremal_perturbation.PRESERVE_VARIANT
    print_iter=None
    debug=False
    reward_func=extremal_perturbation.simple_reward
    resize=False
    resize_mode='bilinear'
    smooth=0
    ##########################
    from torchvision.models import resnet50
    model = resnet50(pretrained=True).to(device)
    model.eval()
    #########################
    if isinstance(areas, float):
        areas = [areas]
    momentum = 0.9
    learning_rate = 0.01
    regul_weight = 300
    device = input.device

    regul_weight_last = max(regul_weight / 2, 1)

    if debug:
        print(
            f"extremal_perturbation:\n"
            f"- target: {target}\n"
            f"- areas: {areas}\n"
            f"- variant: {variant}\n"
            f"- max_iter: {max_iter}\n"
            f"- step/sigma: {step}, {sigma}\n"
            f"- image size: {list(input.shape)}\n"
            f"- reward function: {reward_func.__name__}"
        )

    # Disable gradients for model parameters.
    # TODO(av): undo on leaving the function.
    for p in model.parameters():
        p.requires_grad_(False)

    # Get the perturbation operator.
    # The perturbation can be applied at any layer of the network (depth).
    perturbation = Perturbation(
        input,
        num_levels=num_levels,
        type=perturbation
    ).to(device)

    perturbation_str = '\n  '.join(perturbation.__str__().split('\n'))
    if debug:
        print(f"- {perturbation_str}")

    # Prepare the mask generator.
    shape = perturbation.pyramid.shape[2:]
    mask_generator = MaskGenerator(shape, step, sigma).to(device)
    h, w = mask_generator.shape_in
    pmask = torch.ones(len(areas), 1, h, w).to(device)
    if debug:
        print(f"- mask resolution:\n  {pmask.shape}")

    # Prepare reference area vector.
    max_area = np.prod(mask_generator.shape_out)
    reference = torch.ones(len(areas), max_area).to(device)
    for i, a in enumerate(areas):
        reference[i, :int(max_area * (1 - a))] = 0

    # Initialize optimizer.
    optimizer = optim.SGD([pmask],
                          lr=learning_rate,
                          momentum=momentum,
                          dampening=momentum)
    hist = torch.zeros((len(areas), 2, 0))

    for t in range(max_iter):
        pmask.requires_grad_(True)

        # Generate the mask.
        mask_, mask = mask_generator.generate(pmask)

        # Apply the mask.
        if variant == DELETE_VARIANT:
            x = perturbation.apply(1 - mask_)
        elif variant == PRESERVE_VARIANT:
            x = perturbation.apply(mask_)
        elif variant == DUAL_VARIANT:
            x = torch.cat((
                perturbation.apply(mask_),
                perturbation.apply(1 - mask_),
            ), dim=0)
        else:
            assert False

        # Apply jitter to the masked data.
        if jitter and t % 2 == 0:
            x = torch.flip(x, dims=(3,))

        # Evaluate the model on the masked data.
        y = model(x)

        # Get reward.
        reward = reward_func(y, target, variant=variant)

        # Reshape reward and average over spatial dimensions.
        reward = reward.reshape(len(areas), -1).mean(dim=1)

        # Area regularization.
        mask_sorted = mask.reshape(len(areas), -1).sort(dim=1)[0]
        regul = - ((mask_sorted - reference)**2).mean(dim=1) * regul_weight
        energy = (reward + regul).sum()

        # Gradient step.
        optimizer.zero_grad()
        (- energy).backward()
        optimizer.step()

        pmask.data = pmask.data.clamp(0, 1)

        # Record energy.
        hist = torch.cat(
            (hist,
             torch.cat((
                 reward.detach().cpu().view(-1, 1, 1),
                 regul.detach().cpu().view(-1, 1, 1)
             ), dim=1)), dim=2)

        # Adjust the regulariser/area constraint weight.
        regul_weight *= 1.0035

        # Diagnostics.
        debug_this_iter = debug and (t in (0, max_iter - 1)
                                     or regul_weight / regul_weight_last >= 2)

        if (print_iter is not None and t % print_iter == 0) or debug_this_iter:
            print("[{:04d}/{:04d}]".format(t + 1, max_iter), end="")
            for i, area in enumerate(areas):
                print(" [area:{:.2f} loss:{:.2f} reg:{:.2f}]".format(
                    area,
                    hist[i, 0, -1],
                    hist[i, 1, -1]), end="")
            print()

        if debug_this_iter:
            regul_weight_last = regul_weight
            for i, a in enumerate(areas):
                plt.figure(i, figsize=(20, 6))
                plt.clf()
                ncols = 4 if variant == DUAL_VARIANT else 3
                plt.subplot(1, ncols, 1)
                plt.plot(hist[i, 0].numpy())
                plt.plot(hist[i, 1].numpy())
                plt.plot(hist[i].sum(dim=0).numpy())
                plt.legend(('energy', 'regul', 'both'))
                plt.title(f'target area:{a:.2f}')
                plt.subplot(1, ncols, 2)
                imsc(mask[i], lim=[0, 1])
                plt.title(
                    f"min:{mask[i].min().item():.2f}"
                    f" max:{mask[i].max().item():.2f}"
                    f" area:{mask[i].sum() / mask[i].numel():.2f}")
                plt.subplot(1, ncols, 3)
                imsc(x[i])
                if variant == DUAL_VARIANT:
                    plt.subplot(1, ncols, 4)
                    imsc(x[i + len(areas)])
                plt.pause(0.001)

    mask_ = mask_.detach()

    # Resize saliency map.
    mask_ = resize_saliency(input,
                            mask_,
                            resize,
                            mode=resize_mode)

    # Smooth saliency map.
    if smooth > 0:
        mask_ = imsmooth(
            mask_,
            sigma=smooth * min(mask_.shape[2:]),
            padding_mode='constant'
        )
        
if True:
    ##########################
    input = augmentations
    import extremal_perturbation
    from torch import optim
    areas = [0.1]
    perturbation = extremal_perturbation.BLUR_PERTURBATION
    max_iter = 800
    num_levels=8
    step=7
    sigma=21
    jitter=True
    variant=extremal_perturbation.PRESERVE_VARIANT
    print_iter=None
    debug=False
    reward_func=extremal_perturbation.simple_reward
    resize=False
    resize_mode='bilinear'
    smooth=0
    ##########################
    from torchvision.models import resnet50
    model = resnet50(pretrained=True).to(device)
    model.eval()
    #########################
    if isinstance(areas, float):
        areas = [areas]
    momentum = 0.9
    learning_rate = 0.01
    regul_weight = 300
    device = input.device

    regul_weight_last = max(regul_weight / 2, 1)

    if debug:
        print(
            f"extremal_perturbation:\n"
            f"- target: {target}\n"
            f"- areas: {areas}\n"
            f"- variant: {variant}\n"
            f"- max_iter: {max_iter}\n"
            f"- step/sigma: {step}, {sigma}\n"
            f"- image size: {list(input.shape)}\n"
            f"- reward function: {reward_func.__name__}"
        )

    # Disable gradients for model parameters.
    # TODO(av): undo on leaving the function.
    for p in model.parameters():
        p.requires_grad_(False)

    # Get the perturbation operator.
    # The perturbation can be applied at any layer of the network (depth).
    perturbation = Perturbation(
        input,
        num_levels=num_levels,
        type=perturbation
    ).to(device)

    perturbation_str = '\n  '.join(perturbation.__str__().split('\n'))
    if debug:
        print(f"- {perturbation_str}")

    # Prepare the mask generator.
    shape = perturbation.pyramid.shape[2:]
    mask_generator = MaskGenerator(shape, step, sigma).to(device)
    h, w = mask_generator.shape_in
    pmask = torch.ones(len(areas), 1, h, w).to(device)
    if debug:
        print(f"- mask resolution:\n  {pmask.shape}")

    # Prepare reference area vector.
    max_area = np.prod(mask_generator.shape_out)
    reference = torch.ones(len(areas), max_area).to(device)
    for i, a in enumerate(areas):
        reference[i, :int(max_area * (1 - a))] = 0

    # Initialize optimizer.
    optimizer = optim.SGD([pmask],
                          lr=learning_rate,
                          momentum=momentum,
                          dampening=momentum)
    hist = torch.zeros((len(areas), 2, 0))

    for t in range(max_iter):
        pmask.requires_grad_(True)

        # Generate the mask.
        mask_, mask = mask_generator.generate(pmask)

        # Apply the mask.
        if variant == DELETE_VARIANT:
            x = perturbation.apply(1 - mask_)
        elif variant == PRESERVE_VARIANT:
            x = perturbation.apply(mask_)
        elif variant == DUAL_VARIANT:
            x = torch.cat((
                perturbation.apply(mask_),
                perturbation.apply(1 - mask_),
            ), dim=0)
        else:
            assert False

        # Apply jitter to the masked data.
        if jitter and t % 2 == 0:
            x = torch.flip(x, dims=(3,))

        # Evaluate the model on the masked data.
        y = model(x)

        # Get reward.
        reward = reward_func(y, target, variant=variant)

        # Reshape reward and average over spatial dimensions.
        reward = reward.reshape(len(areas), -1).mean(dim=1)

        # Area regularization.
        mask_sorted = mask.reshape(len(areas), -1).sort(dim=1)[0]
        regul = - ((mask_sorted - reference)**2).mean(dim=1) * regul_weight
        energy = (reward + regul).sum()

        # Gradient step.
        optimizer.zero_grad()
        (- energy).backward()
        optimizer.step()

        pmask.data = pmask.data.clamp(0, 1)

        # Record energy.
        hist = torch.cat(
            (hist,
             torch.cat((
                 reward.detach().cpu().view(-1, 1, 1),
                 regul.detach().cpu().view(-1, 1, 1)
             ), dim=1)), dim=2)

        # Adjust the regulariser/area constraint weight.
        regul_weight *= 1.0035

        # Diagnostics.
        debug_this_iter = debug and (t in (0, max_iter - 1)
                                     or regul_weight / regul_weight_last >= 2)

        if (print_iter is not None and t % print_iter == 0) or debug_this_iter:
            print("[{:04d}/{:04d}]".format(t + 1, max_iter), end="")
            for i, area in enumerate(areas):
                print(" [area:{:.2f} loss:{:.2f} reg:{:.2f}]".format(
                    area,
                    hist[i, 0, -1],
                    hist[i, 1, -1]), end="")
            print()

        if debug_this_iter:
            regul_weight_last = regul_weight
            for i, a in enumerate(areas):
                plt.figure(i, figsize=(20, 6))
                plt.clf()
                ncols = 4 if variant == DUAL_VARIANT else 3
                plt.subplot(1, ncols, 1)
                plt.plot(hist[i, 0].numpy())
                plt.plot(hist[i, 1].numpy())
                plt.plot(hist[i].sum(dim=0).numpy())
                plt.legend(('energy', 'regul', 'both'))
                plt.title(f'target area:{a:.2f}')
                plt.subplot(1, ncols, 2)
                imsc(mask[i], lim=[0, 1])
                plt.title(
                    f"min:{mask[i].min().item():.2f}"
                    f" max:{mask[i].max().item():.2f}"
                    f" area:{mask[i].sum() / mask[i].numel():.2f}")
                plt.subplot(1, ncols, 3)
                imsc(x[i])
                if variant == DUAL_VARIANT:
                    plt.subplot(1, ncols, 4)
                    imsc(x[i + len(areas)])
                plt.pause(0.001)

    mask_ = mask_.detach()

    # Resize saliency map.
    mask_ = resize_saliency(input,
                            mask_,
                            resize,
                            mode=resize_mode)

    # Smooth saliency map.
    if smooth > 0:
        mask_ = imsmooth(
            mask_,
            sigma=smooth * min(mask_.shape[2:]),
            padding_mode='constant'
        )
        
if True:
    ##########################
    input = augmentations
    from extremal_perturbation import *
    import extremal_perturbation
    from torch import optim
    areas = [0.1]
    perturbation = extremal_perturbation.BLUR_PERTURBATION
    max_iter = 800
    num_levels=8
    step=7
    sigma=21
    jitter=True
    variant=extremal_perturbation.PRESERVE_VARIANT
    print_iter=None
    debug=False
    reward_func=extremal_perturbation.simple_reward
    resize=False
    resize_mode='bilinear'
    smooth=0
    ##########################
    from torchvision.models import resnet50
    model = resnet50(pretrained=True).to(device)
    model.eval()
    #########################
    if isinstance(areas, float):
        areas = [areas]
    momentum = 0.9
    learning_rate = 0.01
    regul_weight = 300
    device = input.device

    regul_weight_last = max(regul_weight / 2, 1)

    if debug:
        print(
            f"extremal_perturbation:\n"
            f"- target: {target}\n"
            f"- areas: {areas}\n"
            f"- variant: {variant}\n"
            f"- max_iter: {max_iter}\n"
            f"- step/sigma: {step}, {sigma}\n"
            f"- image size: {list(input.shape)}\n"
            f"- reward function: {reward_func.__name__}"
        )

    # Disable gradients for model parameters.
    # TODO(av): undo on leaving the function.
    for p in model.parameters():
        p.requires_grad_(False)

    # Get the perturbation operator.
    # The perturbation can be applied at any layer of the network (depth).
    perturbation = Perturbation(
        input,
        num_levels=num_levels,
        type=perturbation
    ).to(device)

    perturbation_str = '\n  '.join(perturbation.__str__().split('\n'))
    if debug:
        print(f"- {perturbation_str}")

    # Prepare the mask generator.
    shape = perturbation.pyramid.shape[2:]
    mask_generator = MaskGenerator(shape, step, sigma).to(device)
    h, w = mask_generator.shape_in
    pmask = torch.ones(len(areas), 1, h, w).to(device)
    if debug:
        print(f"- mask resolution:\n  {pmask.shape}")

    # Prepare reference area vector.
    max_area = np.prod(mask_generator.shape_out)
    reference = torch.ones(len(areas), max_area).to(device)
    for i, a in enumerate(areas):
        reference[i, :int(max_area * (1 - a))] = 0

    # Initialize optimizer.
    optimizer = optim.SGD([pmask],
                          lr=learning_rate,
                          momentum=momentum,
                          dampening=momentum)
    hist = torch.zeros((len(areas), 2, 0))

    for t in range(max_iter):
        pmask.requires_grad_(True)

        # Generate the mask.
        mask_, mask = mask_generator.generate(pmask)

        # Apply the mask.
        if variant == DELETE_VARIANT:
            x = perturbation.apply(1 - mask_)
        elif variant == PRESERVE_VARIANT:
            x = perturbation.apply(mask_)
        elif variant == DUAL_VARIANT:
            x = torch.cat((
                perturbation.apply(mask_),
                perturbation.apply(1 - mask_),
            ), dim=0)
        else:
            assert False

        # Apply jitter to the masked data.
        if jitter and t % 2 == 0:
            x = torch.flip(x, dims=(3,))

        # Evaluate the model on the masked data.
        y = model(x)

        # Get reward.
        reward = reward_func(y, target, variant=variant)

        # Reshape reward and average over spatial dimensions.
        reward = reward.reshape(len(areas), -1).mean(dim=1)

        # Area regularization.
        mask_sorted = mask.reshape(len(areas), -1).sort(dim=1)[0]
        regul = - ((mask_sorted - reference)**2).mean(dim=1) * regul_weight
        energy = (reward + regul).sum()

        # Gradient step.
        optimizer.zero_grad()
        (- energy).backward()
        optimizer.step()

        pmask.data = pmask.data.clamp(0, 1)

        # Record energy.
        hist = torch.cat(
            (hist,
             torch.cat((
                 reward.detach().cpu().view(-1, 1, 1),
                 regul.detach().cpu().view(-1, 1, 1)
             ), dim=1)), dim=2)

        # Adjust the regulariser/area constraint weight.
        regul_weight *= 1.0035

        # Diagnostics.
        debug_this_iter = debug and (t in (0, max_iter - 1)
                                     or regul_weight / regul_weight_last >= 2)

        if (print_iter is not None and t % print_iter == 0) or debug_this_iter:
            print("[{:04d}/{:04d}]".format(t + 1, max_iter), end="")
            for i, area in enumerate(areas):
                print(" [area:{:.2f} loss:{:.2f} reg:{:.2f}]".format(
                    area,
                    hist[i, 0, -1],
                    hist[i, 1, -1]), end="")
            print()

        if debug_this_iter:
            regul_weight_last = regul_weight
            for i, a in enumerate(areas):
                plt.figure(i, figsize=(20, 6))
                plt.clf()
                ncols = 4 if variant == DUAL_VARIANT else 3
                plt.subplot(1, ncols, 1)
                plt.plot(hist[i, 0].numpy())
                plt.plot(hist[i, 1].numpy())
                plt.plot(hist[i].sum(dim=0).numpy())
                plt.legend(('energy', 'regul', 'both'))
                plt.title(f'target area:{a:.2f}')
                plt.subplot(1, ncols, 2)
                imsc(mask[i], lim=[0, 1])
                plt.title(
                    f"min:{mask[i].min().item():.2f}"
                    f" max:{mask[i].max().item():.2f}"
                    f" area:{mask[i].sum() / mask[i].numel():.2f}")
                plt.subplot(1, ncols, 3)
                imsc(x[i])
                if variant == DUAL_VARIANT:
                    plt.subplot(1, ncols, 4)
                    imsc(x[i + len(areas)])
                plt.pause(0.001)

    mask_ = mask_.detach()

    # Resize saliency map.
    mask_ = resize_saliency(input,
                            mask_,
                            resize,
                            mode=resize_mode)

    # Smooth saliency map.
    if smooth > 0:
        mask_ = imsmooth(
            mask_,
            sigma=smooth * min(mask_.shape[2:]),
            padding_mode='constant'
        )
        
if True:
    ##########################
    input = augmentations
    from extremal_perturbation import *
    import extremal_perturbation
    from torch import optim
    target = imagenet_target
    areas = [0.1]
    perturbation = extremal_perturbation.BLUR_PERTURBATION
    max_iter = 800
    num_levels=8
    step=7
    sigma=21
    jitter=True
    variant=extremal_perturbation.PRESERVE_VARIANT
    print_iter=None
    debug=False
    reward_func=extremal_perturbation.simple_reward
    resize=False
    resize_mode='bilinear'
    smooth=0
    ##########################
    from torchvision.models import resnet50
    model = resnet50(pretrained=True).to(device)
    model.eval()
    #########################
    if isinstance(areas, float):
        areas = [areas]
    momentum = 0.9
    learning_rate = 0.01
    regul_weight = 300
    device = input.device

    regul_weight_last = max(regul_weight / 2, 1)

    if debug:
        print(
            f"extremal_perturbation:\n"
            f"- target: {target}\n"
            f"- areas: {areas}\n"
            f"- variant: {variant}\n"
            f"- max_iter: {max_iter}\n"
            f"- step/sigma: {step}, {sigma}\n"
            f"- image size: {list(input.shape)}\n"
            f"- reward function: {reward_func.__name__}"
        )

    # Disable gradients for model parameters.
    # TODO(av): undo on leaving the function.
    for p in model.parameters():
        p.requires_grad_(False)

    # Get the perturbation operator.
    # The perturbation can be applied at any layer of the network (depth).
    perturbation = Perturbation(
        input,
        num_levels=num_levels,
        type=perturbation
    ).to(device)

    perturbation_str = '\n  '.join(perturbation.__str__().split('\n'))
    if debug:
        print(f"- {perturbation_str}")

    # Prepare the mask generator.
    shape = perturbation.pyramid.shape[2:]
    mask_generator = MaskGenerator(shape, step, sigma).to(device)
    h, w = mask_generator.shape_in
    pmask = torch.ones(len(areas), 1, h, w).to(device)
    if debug:
        print(f"- mask resolution:\n  {pmask.shape}")

    # Prepare reference area vector.
    max_area = np.prod(mask_generator.shape_out)
    reference = torch.ones(len(areas), max_area).to(device)
    for i, a in enumerate(areas):
        reference[i, :int(max_area * (1 - a))] = 0

    # Initialize optimizer.
    optimizer = optim.SGD([pmask],
                          lr=learning_rate,
                          momentum=momentum,
                          dampening=momentum)
    hist = torch.zeros((len(areas), 2, 0))

    for t in range(max_iter):
        pmask.requires_grad_(True)

        # Generate the mask.
        mask_, mask = mask_generator.generate(pmask)

        # Apply the mask.
        if variant == DELETE_VARIANT:
            x = perturbation.apply(1 - mask_)
        elif variant == PRESERVE_VARIANT:
            x = perturbation.apply(mask_)
        elif variant == DUAL_VARIANT:
            x = torch.cat((
                perturbation.apply(mask_),
                perturbation.apply(1 - mask_),
            ), dim=0)
        else:
            assert False

        # Apply jitter to the masked data.
        if jitter and t % 2 == 0:
            x = torch.flip(x, dims=(3,))

        # Evaluate the model on the masked data.
        y = model(x)

        # Get reward.
        reward = reward_func(y, target, variant=variant)

        # Reshape reward and average over spatial dimensions.
        reward = reward.reshape(len(areas), -1).mean(dim=1)

        # Area regularization.
        mask_sorted = mask.reshape(len(areas), -1).sort(dim=1)[0]
        regul = - ((mask_sorted - reference)**2).mean(dim=1) * regul_weight
        energy = (reward + regul).sum()

        # Gradient step.
        optimizer.zero_grad()
        (- energy).backward()
        optimizer.step()

        pmask.data = pmask.data.clamp(0, 1)

        # Record energy.
        hist = torch.cat(
            (hist,
             torch.cat((
                 reward.detach().cpu().view(-1, 1, 1),
                 regul.detach().cpu().view(-1, 1, 1)
             ), dim=1)), dim=2)

        # Adjust the regulariser/area constraint weight.
        regul_weight *= 1.0035

        # Diagnostics.
        debug_this_iter = debug and (t in (0, max_iter - 1)
                                     or regul_weight / regul_weight_last >= 2)

        if (print_iter is not None and t % print_iter == 0) or debug_this_iter:
            print("[{:04d}/{:04d}]".format(t + 1, max_iter), end="")
            for i, area in enumerate(areas):
                print(" [area:{:.2f} loss:{:.2f} reg:{:.2f}]".format(
                    area,
                    hist[i, 0, -1],
                    hist[i, 1, -1]), end="")
            print()

        if debug_this_iter:
            regul_weight_last = regul_weight
            for i, a in enumerate(areas):
                plt.figure(i, figsize=(20, 6))
                plt.clf()
                ncols = 4 if variant == DUAL_VARIANT else 3
                plt.subplot(1, ncols, 1)
                plt.plot(hist[i, 0].numpy())
                plt.plot(hist[i, 1].numpy())
                plt.plot(hist[i].sum(dim=0).numpy())
                plt.legend(('energy', 'regul', 'both'))
                plt.title(f'target area:{a:.2f}')
                plt.subplot(1, ncols, 2)
                imsc(mask[i], lim=[0, 1])
                plt.title(
                    f"min:{mask[i].min().item():.2f}"
                    f" max:{mask[i].max().item():.2f}"
                    f" area:{mask[i].sum() / mask[i].numel():.2f}")
                plt.subplot(1, ncols, 3)
                imsc(x[i])
                if variant == DUAL_VARIANT:
                    plt.subplot(1, ncols, 4)
                    imsc(x[i + len(areas)])
                plt.pause(0.001)

    mask_ = mask_.detach()

    # Resize saliency map.
    mask_ = resize_saliency(input,
                            mask_,
                            resize,
                            mode=resize_mode)

    # Smooth saliency map.
    if smooth > 0:
        mask_ = imsmooth(
            mask_,
            sigma=smooth * min(mask_.shape[2:]),
            padding_mode='constant'
        )
        
if True:
    ##########################
    input = augmentations
    from extremal_perturbation import *
    import extremal_perturbation
    from torch import optim
    target = imagenet_target
    areas = [0.1]
    perturbation = extremal_perturbation.BLUR_PERTURBATION
    max_iter = 800
    num_levels=8
    step=7
    sigma=21
    jitter=True
    variant=extremal_perturbation.PRESERVE_VARIANT
    print_iter=None
    debug=False
    reward_func=extremal_perturbation.simple_reward
    resize=False
    resize_mode='bilinear'
    smooth=0
    ##########################
    from torchvision.models import resnet50
    model = resnet50(pretrained=True).to(device)
    model.eval()
    #########################
    if isinstance(areas, float):
        areas = [areas]
    momentum = 0.9
    learning_rate = 0.01
    regul_weight = 300
    device = input.device

    regul_weight_last = max(regul_weight / 2, 1)

    if debug:
        print(
            f"extremal_perturbation:\n"
            f"- target: {target}\n"
            f"- areas: {areas}\n"
            f"- variant: {variant}\n"
            f"- max_iter: {max_iter}\n"
            f"- step/sigma: {step}, {sigma}\n"
            f"- image size: {list(input.shape)}\n"
            f"- reward function: {reward_func.__name__}"
        )

    # Disable gradients for model parameters.
    # TODO(av): undo on leaving the function.
    for p in model.parameters():
        p.requires_grad_(False)

    # Get the perturbation operator.
    # The perturbation can be applied at any layer of the network (depth).
    perturbation = Perturbation(
        input,
        num_levels=num_levels,
        type=perturbation
    ).to(device)

    perturbation_str = '\n  '.join(perturbation.__str__().split('\n'))
    if debug:
        print(f"- {perturbation_str}")

    # Prepare the mask generator.
    shape = perturbation.pyramid.shape[2:]
    mask_generator = MaskGenerator(shape, step, sigma).to(device)
    h, w = mask_generator.shape_in
    pmask = torch.ones(len(areas), 1, h, w).to(device)
    if debug:
        print(f"- mask resolution:\n  {pmask.shape}")

    # Prepare reference area vector.
    max_area = np.prod(mask_generator.shape_out)
    reference = torch.ones(len(areas), max_area).to(device)
    for i, a in enumerate(areas):
        reference[i, :int(max_area * (1 - a))] = 0

    # Initialize optimizer.
    optimizer = optim.SGD([pmask],
                          lr=learning_rate,
                          momentum=momentum,
                          dampening=momentum)
    hist = torch.zeros((len(areas), 2, 0))

    for t in range(max_iter):
        pmask.requires_grad_(True)

        # Generate the mask.
        mask_, mask = mask_generator.generate(pmask)

        # Apply the mask.
        if variant == DELETE_VARIANT:
            x = perturbation.apply(1 - mask_)
        elif variant == PRESERVE_VARIANT:
            x = perturbation.apply(mask_)
        elif variant == DUAL_VARIANT:
            x = torch.cat((
                perturbation.apply(mask_),
                perturbation.apply(1 - mask_),
            ), dim=0)
        else:
            assert False

        # Apply jitter to the masked data.
        if jitter and t % 2 == 0:
            x = torch.flip(x, dims=(3,))

        # Evaluate the model on the masked data.
        y = model(x)

        # Get reward.
        reward = reward_func(y, target, variant=variant)

        # Reshape reward and average over spatial dimensions.
        reward = reward.reshape(len(areas), -1).mean(dim=1)

        # Area regularization.
        mask_sorted = mask.reshape(len(areas), -1).sort(dim=1)[0]
        regul = - ((mask_sorted - reference)**2).mean(dim=1) * regul_weight
        energy = (reward + regul).sum()

        # Gradient step.
        optimizer.zero_grad()
        (- energy).backward()
        optimizer.step()

        pmask.data = pmask.data.clamp(0, 1)

        # Record energy.
        hist = torch.cat(
            (hist,
             torch.cat((
                 reward.detach().cpu().view(-1, 1, 1),
                 regul.detach().cpu().view(-1, 1, 1)
             ), dim=1)), dim=2)

        # Adjust the regulariser/area constraint weight.
        regul_weight *= 1.0035

        # Diagnostics.
        debug_this_iter = debug and (t in (0, max_iter - 1)
                                     or regul_weight / regul_weight_last >= 2)

        if (print_iter is not None and t % print_iter == 0) or debug_this_iter:
            print("[{:04d}/{:04d}]".format(t + 1, max_iter), end="")
            for i, area in enumerate(areas):
                print(" [area:{:.2f} loss:{:.2f} reg:{:.2f}]".format(
                    area,
                    hist[i, 0, -1],
                    hist[i, 1, -1]), end="")
            print()

        if debug_this_iter:
            regul_weight_last = regul_weight
            for i, a in enumerate(areas):
                plt.figure(i, figsize=(20, 6))
                plt.clf()
                ncols = 4 if variant == DUAL_VARIANT else 3
                plt.subplot(1, ncols, 1)
                plt.plot(hist[i, 0].numpy())
                plt.plot(hist[i, 1].numpy())
                plt.plot(hist[i].sum(dim=0).numpy())
                plt.legend(('energy', 'regul', 'both'))
                plt.title(f'target area:{a:.2f}')
                plt.subplot(1, ncols, 2)
                imsc(mask[i], lim=[0, 1])
                plt.title(
                    f"min:{mask[i].min().item():.2f}"
                    f" max:{mask[i].max().item():.2f}"
                    f" area:{mask[i].sum() / mask[i].numel():.2f}")
                plt.subplot(1, ncols, 3)
                imsc(x[i])
                if variant == DUAL_VARIANT:
                    plt.subplot(1, ncols, 4)
                    imsc(x[i + len(areas)])
                plt.pause(0.001)

    mask_ = mask_.detach()

    # Resize saliency map.
    mask_ = extremal_perturbation.resize_saliency(input,
                            mask_,
                            resize,
                            mode=resize_mode)

    # Smooth saliency map.
    if smooth > 0:
        mask_ = imsmooth(
            mask_,
            sigma=smooth * min(mask_.shape[2:]),
            padding_mode='constant'
        )
        
cams.shape
mask_.shape
#[Out]# torch.Size([1, 1, 256, 457])
import dutils
dutils.img_save(tensor_to_numpy(mask_[0,0]),'elp_mask.png')
if True:
    ##########################
    from gradcam import normalize_tensor
    input = normalize_tensor(augmentations)
    from extremal_perturbation import *
    import extremal_perturbation
    from torch import optim
    target = imagenet_target
    areas = [0.1]
    perturbation = extremal_perturbation.BLUR_PERTURBATION
    max_iter = 800
    num_levels=8
    step=7
    sigma=21
    jitter=True
    variant=extremal_perturbation.PRESERVE_VARIANT
    print_iter=None
    debug=False
    reward_func=extremal_perturbation.simple_reward
    resize=False
    resize_mode='bilinear'
    smooth=0
    ##########################
    from torchvision.models import resnet50
    model = resnet50(pretrained=True).to(device)
    model.eval()
    #from gradcam import normalize_tensor
    #########################
    if isinstance(areas, float):
        areas = [areas]
    momentum = 0.9
    learning_rate = 0.01
    regul_weight = 300
    device = input.device

    regul_weight_last = max(regul_weight / 2, 1)

    if debug:
        print(
            f"extremal_perturbation:\n"
            f"- target: {target}\n"
            f"- areas: {areas}\n"
            f"- variant: {variant}\n"
            f"- max_iter: {max_iter}\n"
            f"- step/sigma: {step}, {sigma}\n"
            f"- image size: {list(input.shape)}\n"
            f"- reward function: {reward_func.__name__}"
        )

    # Disable gradients for model parameters.
    # TODO(av): undo on leaving the function.
    for p in model.parameters():
        p.requires_grad_(False)

    # Get the perturbation operator.
    # The perturbation can be applied at any layer of the network (depth).
    perturbation = Perturbation(
        input,
        num_levels=num_levels,
        type=perturbation
    ).to(device)

    perturbation_str = '\n  '.join(perturbation.__str__().split('\n'))
    if debug:
        print(f"- {perturbation_str}")

    # Prepare the mask generator.
    shape = perturbation.pyramid.shape[2:]
    mask_generator = MaskGenerator(shape, step, sigma).to(device)
    h, w = mask_generator.shape_in
    pmask = torch.ones(len(areas), 1, h, w).to(device)
    if debug:
        print(f"- mask resolution:\n  {pmask.shape}")

    # Prepare reference area vector.
    max_area = np.prod(mask_generator.shape_out)
    reference = torch.ones(len(areas), max_area).to(device)
    for i, a in enumerate(areas):
        reference[i, :int(max_area * (1 - a))] = 0

    # Initialize optimizer.
    optimizer = optim.SGD([pmask],
                          lr=learning_rate,
                          momentum=momentum,
                          dampening=momentum)
    hist = torch.zeros((len(areas), 2, 0))

    for t in range(max_iter):
        pmask.requires_grad_(True)

        # Generate the mask.
        mask_, mask = mask_generator.generate(pmask)

        # Apply the mask.
        if variant == DELETE_VARIANT:
            x = perturbation.apply(1 - mask_)
        elif variant == PRESERVE_VARIANT:
            x = perturbation.apply(mask_)
        elif variant == DUAL_VARIANT:
            x = torch.cat((
                perturbation.apply(mask_),
                perturbation.apply(1 - mask_),
            ), dim=0)
        else:
            assert False

        # Apply jitter to the masked data.
        if jitter and t % 2 == 0:
            x = torch.flip(x, dims=(3,))

        # Evaluate the model on the masked data.
        y = model(x)

        # Get reward.
        reward = reward_func(y, target, variant=variant)

        # Reshape reward and average over spatial dimensions.
        reward = reward.reshape(len(areas), -1).mean(dim=1)

        # Area regularization.
        mask_sorted = mask.reshape(len(areas), -1).sort(dim=1)[0]
        regul = - ((mask_sorted - reference)**2).mean(dim=1) * regul_weight
        energy = (reward + regul).sum()

        # Gradient step.
        optimizer.zero_grad()
        (- energy).backward()
        optimizer.step()

        pmask.data = pmask.data.clamp(0, 1)

        # Record energy.
        hist = torch.cat(
            (hist,
             torch.cat((
                 reward.detach().cpu().view(-1, 1, 1),
                 regul.detach().cpu().view(-1, 1, 1)
             ), dim=1)), dim=2)

        # Adjust the regulariser/area constraint weight.
        regul_weight *= 1.0035

        # Diagnostics.
        debug_this_iter = debug and (t in (0, max_iter - 1)
                                     or regul_weight / regul_weight_last >= 2)

        if (print_iter is not None and t % print_iter == 0) or debug_this_iter:
            print("[{:04d}/{:04d}]".format(t + 1, max_iter), end="")
            for i, area in enumerate(areas):
                print(" [area:{:.2f} loss:{:.2f} reg:{:.2f}]".format(
                    area,
                    hist[i, 0, -1],
                    hist[i, 1, -1]), end="")
            print()

        if debug_this_iter:
            regul_weight_last = regul_weight
            for i, a in enumerate(areas):
                plt.figure(i, figsize=(20, 6))
                plt.clf()
                ncols = 4 if variant == DUAL_VARIANT else 3
                plt.subplot(1, ncols, 1)
                plt.plot(hist[i, 0].numpy())
                plt.plot(hist[i, 1].numpy())
                plt.plot(hist[i].sum(dim=0).numpy())
                plt.legend(('energy', 'regul', 'both'))
                plt.title(f'target area:{a:.2f}')
                plt.subplot(1, ncols, 2)
                imsc(mask[i], lim=[0, 1])
                plt.title(
                    f"min:{mask[i].min().item():.2f}"
                    f" max:{mask[i].max().item():.2f}"
                    f" area:{mask[i].sum() / mask[i].numel():.2f}")
                plt.subplot(1, ncols, 3)
                imsc(x[i])
                if variant == DUAL_VARIANT:
                    plt.subplot(1, ncols, 4)
                    imsc(x[i + len(areas)])
                plt.pause(0.001)

    mask_ = mask_.detach()

    # Resize saliency map.
    mask_ = extremal_perturbation.resize_saliency(input,
                            mask_,
                            resize,
                            mode=resize_mode)

    # Smooth saliency map.
    if smooth > 0:
        mask_ = imsmooth(
            mask_,
            sigma=smooth * min(mask_.shape[2:]),
            padding_mode='constant'
        )
        
dutils.img_save(tensor_to_numpy(mask_[0,0]),'elp_mask.png')
if True:
    ##########################
    from gradcam import normalize_tensor
    input = normalize_tensor(ref)
    from extremal_perturbation import *
    import extremal_perturbation
    from torch import optim
    target = imagenet_target
    areas = [0.2]
    perturbation = extremal_perturbation.BLUR_PERTURBATION
    max_iter = 800
    num_levels=8
    step=7
    sigma=21
    jitter=True
    variant=extremal_perturbation.PRESERVE_VARIANT
    print_iter=None
    debug=False
    reward_func=extremal_perturbation.simple_reward
    resize=False
    resize_mode='bilinear'
    smooth=0
    ##########################
    from torchvision.models import resnet50
    model = resnet50(pretrained=True).to(device)
    model.eval()
    #from gradcam import normalize_tensor
    #########################
    if isinstance(areas, float):
        areas = [areas]
    momentum = 0.9
    learning_rate = 0.01
    regul_weight = 300
    device = input.device

    regul_weight_last = max(regul_weight / 2, 1)

    if debug:
        print(
            f"extremal_perturbation:\n"
            f"- target: {target}\n"
            f"- areas: {areas}\n"
            f"- variant: {variant}\n"
            f"- max_iter: {max_iter}\n"
            f"- step/sigma: {step}, {sigma}\n"
            f"- image size: {list(input.shape)}\n"
            f"- reward function: {reward_func.__name__}"
        )

    # Disable gradients for model parameters.
    # TODO(av): undo on leaving the function.
    for p in model.parameters():
        p.requires_grad_(False)

    # Get the perturbation operator.
    # The perturbation can be applied at any layer of the network (depth).
    perturbation = Perturbation(
        input,
        num_levels=num_levels,
        type=perturbation
    ).to(device)

    perturbation_str = '\n  '.join(perturbation.__str__().split('\n'))
    if debug:
        print(f"- {perturbation_str}")

    # Prepare the mask generator.
    shape = perturbation.pyramid.shape[2:]
    mask_generator = MaskGenerator(shape, step, sigma).to(device)
    h, w = mask_generator.shape_in
    pmask = torch.ones(len(areas), 1, h, w).to(device)
    if debug:
        print(f"- mask resolution:\n  {pmask.shape}")

    # Prepare reference area vector.
    max_area = np.prod(mask_generator.shape_out)
    reference = torch.ones(len(areas), max_area).to(device)
    for i, a in enumerate(areas):
        reference[i, :int(max_area * (1 - a))] = 0

    # Initialize optimizer.
    optimizer = optim.SGD([pmask],
                          lr=learning_rate,
                          momentum=momentum,
                          dampening=momentum)
    hist = torch.zeros((len(areas), 2, 0))

    for t in range(max_iter):
        pmask.requires_grad_(True)

        # Generate the mask.
        mask_, mask = mask_generator.generate(pmask)

        # Apply the mask.
        if variant == DELETE_VARIANT:
            x = perturbation.apply(1 - mask_)
        elif variant == PRESERVE_VARIANT:
            x = perturbation.apply(mask_)
        elif variant == DUAL_VARIANT:
            x = torch.cat((
                perturbation.apply(mask_),
                perturbation.apply(1 - mask_),
            ), dim=0)
        else:
            assert False

        # Apply jitter to the masked data.
        if jitter and t % 2 == 0:
            x = torch.flip(x, dims=(3,))

        # Evaluate the model on the masked data.
        y = model(x)

        # Get reward.
        reward = reward_func(y, target, variant=variant)

        # Reshape reward and average over spatial dimensions.
        reward = reward.reshape(len(areas), -1).mean(dim=1)

        # Area regularization.
        mask_sorted = mask.reshape(len(areas), -1).sort(dim=1)[0]
        regul = - ((mask_sorted - reference)**2).mean(dim=1) * regul_weight
        energy = (reward + regul).sum()

        # Gradient step.
        optimizer.zero_grad()
        (- energy).backward()
        optimizer.step()

        pmask.data = pmask.data.clamp(0, 1)

        # Record energy.
        hist = torch.cat(
            (hist,
             torch.cat((
                 reward.detach().cpu().view(-1, 1, 1),
                 regul.detach().cpu().view(-1, 1, 1)
             ), dim=1)), dim=2)

        # Adjust the regulariser/area constraint weight.
        regul_weight *= 1.0035

        # Diagnostics.
        debug_this_iter = debug and (t in (0, max_iter - 1)
                                     or regul_weight / regul_weight_last >= 2)

        if (print_iter is not None and t % print_iter == 0) or debug_this_iter:
            print("[{:04d}/{:04d}]".format(t + 1, max_iter), end="")
            for i, area in enumerate(areas):
                print(" [area:{:.2f} loss:{:.2f} reg:{:.2f}]".format(
                    area,
                    hist[i, 0, -1],
                    hist[i, 1, -1]), end="")
            print()

        if debug_this_iter:
            regul_weight_last = regul_weight
            for i, a in enumerate(areas):
                plt.figure(i, figsize=(20, 6))
                plt.clf()
                ncols = 4 if variant == DUAL_VARIANT else 3
                plt.subplot(1, ncols, 1)
                plt.plot(hist[i, 0].numpy())
                plt.plot(hist[i, 1].numpy())
                plt.plot(hist[i].sum(dim=0).numpy())
                plt.legend(('energy', 'regul', 'both'))
                plt.title(f'target area:{a:.2f}')
                plt.subplot(1, ncols, 2)
                imsc(mask[i], lim=[0, 1])
                plt.title(
                    f"min:{mask[i].min().item():.2f}"
                    f" max:{mask[i].max().item():.2f}"
                    f" area:{mask[i].sum() / mask[i].numel():.2f}")
                plt.subplot(1, ncols, 3)
                imsc(x[i])
                if variant == DUAL_VARIANT:
                    plt.subplot(1, ncols, 4)
                    imsc(x[i + len(areas)])
                plt.pause(0.001)

    mask_ = mask_.detach()

    # Resize saliency map.
    mask_ = extremal_perturbation.resize_saliency(input,
                            mask_,
                            resize,
                            mode=resize_mode)

    # Smooth saliency map.
    if smooth > 0:
        mask_ = imsmooth(
            mask_,
            sigma=smooth * min(mask_.shape[2:]),
            padding_mode='constant'
        )
    dutils.img_save(tensor_to_numpy(mask_[0,0]),'elp_mask.png')
    
