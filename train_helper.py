import numpy as np
import matplotlib.pyplot as plt


def train_model(num_epochs: int, model: object,
                optimizer: object, criterion: object,
                X_train: np.ndarray, Y_train: np.ndarray,
                plot_and_save: bool = 0, fig_name: str = "loss_function"):

    loss_val = []
    epochs = []

    for epoch in range(num_epochs):
        outputs = model(X_train)
        optimizer.zero_grad()

        loss = criterion(outputs, Y_train)
        loss.backward()
        
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
            loss_val.append(loss.item())
            epochs.append(epoch)

    if plot_and_save:
        fig1, ax1 = plt.subplots(figsize=(12,8))
        ax1.plot(epochs, loss_val)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
<<<<<<< HEAD
        fig1.savefig("loss_" + fig_name + ".jpg")
=======
        fig1.show()
        input("Press Enter...")
        fig1.savefig(fig_name + ".jpg")
>>>>>>> 3a3385426f5314dfc7f66420212db4c71a8bb3b1

