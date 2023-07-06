import torch
import torch.nn as nn

## test
def test(model, data, idx_train, idx_val, idx_test, num, logic):
    model.eval()
    embeds = model.embed(data.x, data.edge_index).detach()
    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]

    train_lbls = data.y[idx_train]
    val_lbls = data.y[idx_val]
    test_lbls = data.y[idx_test]

    accs = []
    xent = nn.CrossEntropyLoss()

    for _ in range(num):

        opt = torch.optim.Adam(logic.parameters(), lr=0.01, weight_decay=0.0)
        logic.cuda()

        for _ in range(100):
            logic.train()
            opt.zero_grad()

            logits = logic(train_embs)
            loss = xent(logits, train_lbls)
            loss.backward(retain_graph=True)
            opt.step()

        logits = logic(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)

    maxnum = max(accs)
    minnum = min(accs)
    accs = torch.stack(accs)
    print(accs.mean().item(), accs.std().item(), '###', maxnum.item(), minnum.item())
    return accs.mean().item()