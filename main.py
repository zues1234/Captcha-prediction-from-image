def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append("§")
            else:
                p = encoder.inverse_transform([k])[0]
                temp.append(p)
        tp = "".join(temp).replace("§", "")
        cap_preds.append(remove_duplicates(tp))
    return cap_preds


def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin





def run_training():
    DATA_DIR = "./captcha_images_v2/"
    BATCH_SIZE = 8
    IMAGE_WIDTH = 300
    IMAGE_HEIGHT = 75
    NUM_WORKERS = 8
    EPOCHS = 200
    DEVICE = "cuda"
    
    
    image_files = glob(os.path.join(DATA_DIR, "*.png"))
    targets_orig = [x.split("/")[-1][:-4] for x in image_files]
    targets = [[c for c in x] for x in targets_orig]
    targets_flat = [c for clist in targets for c in clist]

    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)
    targets_enc = [lbl_enc.transform(x) for x in targets]
    targets_enc = np.array(targets_enc)
    targets_enc = targets_enc + 1

    (
        train_imgs,
        test_imgs,
        train_targets,
        test_targets,
        _,
        test_targets_orig,
    ) = model_selection.train_test_split(
        image_files, targets_enc, targets_orig, test_size=0.1, random_state=42
    )

    train_dataset = ClassificationDataset(
        image_paths=train_imgs,
        targets=train_targets,
        resize=(IMAGE_HEIGHT, IMAGE_WIDTH),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )
    test_dataset = ClassificationDataset(
        image_paths=test_imgs,
        targets=test_targets,
        resize=(IMAGE_HEIGHT, IMAGE_WIDTH),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        shuffle=False,
    )

    model = CaptchaModel(num_chars=len(lbl_enc.classes_))
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )
    for epoch in range(EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer)
        valid_preds, test_loss = eval_fn(model, test_loader)
        valid_captcha_preds = []
        for vp in valid_preds:
            current_preds = decode_predictions(vp, lbl_enc)
            valid_captcha_preds.extend(current_preds)
        combined = list(zip(test_targets_orig, valid_captcha_preds))
        print(combined[:10])
        test_dup_rem = [remove_duplicates(c) for c in test_targets_orig]
        accuracy = metrics.accuracy_score(test_dup_rem, valid_captcha_preds)
        print(
            f"Epoch={epoch}, Train Loss={train_loss}, Test Loss={test_loss} Accuracy={accuracy}"
        )
        scheduler.step(test_loss)