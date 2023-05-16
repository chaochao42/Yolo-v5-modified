def spreader_seg(
        cfg,
        im0,
        seg_thres,
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.5,  # NMS IOU threshold
        max_det=1000,
):

    imgsz = cfg.imgsz

    im = letterbox(im0, imgsz, cfg.stride)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to(model.device)
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred, proto = cfg.model(im)[:2]

    pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det, nm=32)

    for i, det in enumerate(pred):
        if len(det):
            masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)


    colors = [(0, 0, 255) for i in range(masks.shape[0])]
    # alpha = 0.5
    alpha = 1
    colors = torch.tensor(colors, device=device, dtype=torch.float32) / 255.0
    colors = colors[:, None, None]
    if len(masks.shape) == 3:
        masks = masks.unsqueeze(3)
    masks_color = masks * (colors * alpha)
    inv_alph_masks = (1 - masks * alpha).cumprod(0)
    mcs = (masks_color * inv_alph_masks).sum(0) * 2
    im_gpu = im[0]
    im_gpu = im_gpu.flip(dims=[0])
    im_gpu = im_gpu.permute(1, 2, 0).contiguous()
    im_gpu = im_gpu * inv_alph_masks[-1] + mcs
    im_mask = (masks_color.sum(0) * 255).byte().cpu().numpy()

    im0_with_mask = scale_image(im_gpu.shape, im_mask, im0.shape)

    im0[:] = im0_with_mask


    return im0